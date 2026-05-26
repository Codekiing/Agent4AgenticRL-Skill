"""
Agent RL Demo: rllm + TRL integration on Mac Air

Usage:
    # Natural language:
    python -m rllm_train.train "用 qwen-0.5b 训练数学 agent，64 个问题，2 个 epoch"
    python -m rllm_train.train "quick test with 16 problems"

    # Default config:
    python -m rllm_train.train
"""

import os
import sys
import warnings

os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

warnings.filterwarnings("ignore", message=".*pin_memory.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import torch
from datasets import Dataset
import transformers
transformers.logging.set_verbosity_error()
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from rllm_train.config import TrainingConfig, parse_natural_language
from rllm_train.logger import TrainingLogger
from rllm_train.math_env import CalculateTool, FinishTool, generate_math_problems, MathCalcEnv, score_math_trajectory
from rllm_train.parsers import QwenToolParser
from rllm_train.rollout import make_rllm_rollout_func
from rllm_train.trajectory_writer import TrajectoryWriter
from rllm_train.tool_agent import ToolAgent
from transformers import TrainerCallback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_model_path(model_name: str) -> str:
    if os.path.isdir(model_name):
        return model_name
    basename = model_name.split("/")[-1]
    for candidate in [
        os.path.join(_PROJECT_ROOT, basename),
        os.path.join(_PROJECT_ROOT, "models", basename),
        os.path.join(_PROJECT_ROOT, "model", basename),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return model_name


def build_dataset(problems):
    records = []
    for p in problems:
        records.append({
            "prompt": [
                {"role": "system", "content": "Solve the math problem. Show your reasoning, then write the final numeric answer."},
                {"role": "user", "content": p["question"]},
            ],
            "answer": p["answer"],
        })
    return Dataset.from_list(records)


def load_external_dataset(config):
    from datasets import load_from_disk
    ds = load_from_disk(config.dataset_path)
    if config.num_problems < len(ds):
        ds = ds.shuffle(seed=config.seed).select(range(config.num_problems))
    records = []
    for item in ds:
        question = item.get("problem", item.get("question", ""))
        records.append({
            "prompt": [
                {"role": "system", "content": "Solve the math problem. Show your reasoning, then write the final numeric answer."},
                {"role": "user", "content": question},
            ],
            "answer": item["answer"],
        })
    return Dataset.from_list(records)


def _completion_text(completion):
    if isinstance(completion, list):
        return " ".join(
            msg.get("content", "") for msg in completion if isinstance(msg, dict)
        )
    return str(completion)


def _completion_reward(text, answer):
    parser = QwenToolParser()
    parsed_tool_call = "<tool_call>" in text and "</tool_call>" in text
    successful_calculates = 0
    calculator_errors = 0
    unknown_tools = 0
    finished = False
    final_response = ""
    env = MathCalcEnv({"answer": answer})
    try:
        tool_calls = parser.parse(text)
    except Exception:
        tool_calls = []

    for tool_call in tool_calls:
        if tool_call.name == "calculate":
            args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            result = env._safe_eval(args.get("expression", ""))
            if str(result).startswith("Error:"):
                calculator_errors += 1
            else:
                successful_calculates += 1
        elif tool_call.name == "finish":
            finished = True
            args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            final_response = args.get("response", "")
        else:
            unknown_tools += 1

    if not tool_calls:
        final_response = text
        finished = True

    reward, _ = score_math_trajectory(
        final_response,
        answer,
        parsed_tool_call=bool(tool_calls),
        synthetic_finish=not bool(tool_calls),
        finished=finished,
        steps=1 if finished else 3,
        max_steps=3,
        successful_calculates=successful_calculates,
        calculator_errors=calculator_errors,
        unknown_tools=unknown_tools,
    )
    return reward


def math_reward_fn(prompts, completions, **kwargs):
    answers = kwargs.get("answer", [])
    return [
        _completion_reward(_completion_text(completion), answers[i] if i < len(answers) else None)
        for i, completion in enumerate(completions)
    ]


def deepscaler_reward_fn(prompts, completions, **kwargs):
    return math_reward_fn(prompts, completions, **kwargs)


def main(config: TrainingConfig | None = None):
    if config is None:
        config = TrainingConfig()

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config.to_json(os.path.join(output_dir, "config.json"))

    log_file = os.path.join(output_dir, "training_log.txt")
    log = TrainingLogger(verbose=config.verbose, log_file=log_file)
    log.log_training_start(config)

    class RllmCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                log.update_training_metrics(logs)

    model_path = _resolve_model_path(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32
    else:
        device = "cpu"
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True,
    )
    model.enable_input_require_grads()
    param_count = sum(p.numel() for p in model.parameters())
    log.log_model_loaded(model_path, device, param_count)

    if not config.dataset_path:
        raise ValueError(
            "dataset_path is required. "
            "Provide a path to a HuggingFace dataset directory with 'problem'/'question' and 'answer' fields."
        )
    dataset = load_external_dataset(config)
    if config.dataset == "deepscaler":
        reward_fn = deepscaler_reward_fn
    else:
        reward_fn = math_reward_fn
    log.log_dataset_ready(len(dataset), dataset[0])

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        logging_strategy="steps",
        log_level="info",
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=getattr(config, 'gradient_checkpointing', False),
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=True,
        temperature=config.temperature,
        dataloader_num_workers=0,
    )

    log.log_trainer_ready(config.num_epochs)

    trajectory_writer = TrajectoryWriter(output_dir, enabled=True)

    answer_map = {}
    for item in dataset:
        for msg in item["prompt"]:
            if msg["role"] == "user":
                answer_map[msg["content"]] = item["answer"]
                break

    rollout_func = make_rllm_rollout_func(
        agent_class=ToolAgent,
        agent_args={
            "system_prompt": (
                "You are a tool-using math agent. Do not answer in plain text.\n"
                "For every step, output exactly one or more <tool_call> XML blocks and no prose.\n"
                "Use calculate for arithmetic, fractions, factorial, comb/binomial, log/ln, exp, sqrt, and trig expressions.\n"
                "When you know the final numeric answer, call finish with the answer string.\n"
                "Example final call:\n"
                "<tool_call>\n"
                "{\"name\": \"finish\", \"arguments\": {\"response\": \"42\"}}\n"
                "</tool_call>\n"
            ),
            "tool_map": {"calculate": CalculateTool, "finish": FinishTool},
        },
        env_class=MathCalcEnv,
        max_steps=config.max_agent_steps,
        max_response_length=config.max_response_length,
        max_prompt_length=config.max_prompt_length,
        sampling_params={"temperature": config.temperature, "top_p": config.top_p},
        training_logger=log,
        trajectory_writer=trajectory_writer,
        answer_map=answer_map,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        rollout_func=rollout_func,
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        callbacks=[RllmCallback()],
    )

    trainer.train()

    log.print_training_report(config, None, output_dir)

    if config.save_model:
        save_path = os.path.join(output_dir, "final_model")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        log.log_model_saved(save_path)

    log.close()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = " ".join(sys.argv[1:])
        if arg.endswith(".json") and os.path.isfile(arg):
            cfg = TrainingConfig.from_json(arg)
        else:
            cfg = parse_natural_language(arg)
    else:
        cfg = TrainingConfig()
    main(cfg)
