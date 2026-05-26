---
id: 005-behavioral-metrics
target_section: anomaly-detection
action: append
description: Add behavioral metrics extraction from trajectory JSONL and new circuit breakers C7-C9
status: proposed
source: manual
---

<!-- section:behavioral-metrics -->
### 轨迹行为指标提取

每步训练完成后，从最新 trajectory JSONL 文件中提取 6 个行为指标。这些指标能直接解释 loss=0 和 reward plateau 的根因。

**数据来源**: `rllm_train/output/runs/<run_id>/trajectories/step_XXXX.jsonl`

**提取方式**: 每次检测到新 step 完成后，Read 最新的 step JSONL 文件，逐条记录计算：

```python
import json

def extract_behavioral_metrics(step_file):
    with open(step_file) as f:
        records = [json.loads(line) for line in f]

    total = len(records)
    if total == 0:
        return {}

    # chat_completions 中是否出现 finish 调用
    def has_finish(record):
        for msg in record.get("chat_completions", []):
            content = msg.get("content", "")
            if "finish" in content.lower() or '"name": "finish"' in content:
                return True
        return False

    # chat_completions 中是否出现 calculate 调用
    def has_calculate(record):
        for msg in record.get("chat_completions", []):
            content = msg.get("content", "")
            if "calculate" in content.lower() or '"name": "calculate"' in content:
                return True
        return False

    # response_text 中是否含数字
    def has_numeric_answer(record):
        import re
        text = record.get("response_text", "")
        return bool(re.search(r'-?\d+\.?\d*', text))

    # 是否被截断（达到 max_steps 且 reward=0）
    def is_truncated(record):
        return record.get("num_steps", 0) >= 3 and record.get("reward", 0) == 0

    finish_count = sum(1 for r in records if has_finish(r))
    calculate_count = sum(1 for r in records if has_calculate(r))
    numeric_count = sum(1 for r in records if has_numeric_answer(r))
    trunc_count = sum(1 for r in records if is_truncated(r))
    rewards = [r.get("reward", 0.0) for r in records]
    steps = [r.get("num_steps", 0) for r in records]

    return {
        "finish_rate": finish_count / total,
        "tool_usage_rate": calculate_count / total,
        "truncation_rate": trunc_count / total,
        "answer_coverage": numeric_count / total,
        "reward_variance": (sum((r - sum(rewards)/total)**2 for r in rewards) / total) ** 0.5,
        "avg_agent_steps": sum(steps) / total,
    }
```

**8 个行为指标定义**:

| 指标 | 含义 | 正常范围 | 异常信号 |
|------|------|---------|---------|
| finish_rate | 调用了 finish 的轨迹占比 | > 50% | < 20% = 格式退化 |
| finish_format_rate | 调了 finish 且参数为数字的占比 | > 80% | < 50% = finish 格式错误 |
| tool_usage_rate | 使用了 calculate 的轨迹占比 | > 60% | < 30% = 模型不用工具 |
| truncation_rate | 被截断（达到 max_steps 且 reward=0）的轨迹占比 | < 20% | > 50% = max_completion_length 不够 |
| answer_coverage | 产生了可解析数字答案的轨迹占比 | > 60% | < 30% = 输出格式崩坏 |
| reward_variance | 同一步内各 generation 的 reward 标准差 | > 0.1 | < 0.02 = GRPO 无学习信号 |
| avg_agent_steps | 所有轨迹的平均交互轮次 | 1.5-2.5 | > 2.8 = 模型空转 |
| completion_length_mean | 平均 completion token 数 | < max_completion_length×0.7 | > max_completion_length×0.9 = 即将截断 |

**汇报格式**: 不再追加第二种行为指标行。行为指标必须并入主动轮询的固定单行 schema：

```
Step X/Y | R Z.ZZZ | Rstd Z.ZZZ | Loss L.LLLL | Grad G.GGGG | Ent E.EEEE | Clip C.CC | Len N | Finish P% | FmtOK P% | Tool P% | Ans P% | tok/s T.T | Time S.Ss | ETA ~Mm | Status OK/WARN/STOP
```
<!-- /section:behavioral-metrics -->

<!-- section:circuit-breaker-extension -->
### 行为指标熔断条件（C7-C9）

在现有 C0-C6 熔断基础上，新增 3 个基于行为指标的熔断条件。**每步完成后必须检查**。

- [ ] **C7 — 行为退化（格式遗忘）**: 连续 ≥3 步 finish_rate < 20%。模型忘记了如何调用 finish 提交答案，退化为只调 calculate 或纯文本输出。
  fix_preset: `diagnose`（需要完整分析诊断根因）

- [ ] **C8 — 无学习信号（GRPO 失效）**: 连续 ≥4 步 reward_variance < 0.02。同一步内所有 generation 得到几乎相同的 reward，GRPO 算不出 advantage，梯度为 0。这是 loss=0 的直接根因。
  fix_preset: `entropy_bonus`（lr/2 + temperature+0.2，增大探索多样性）

- [ ] **C9 — 输出崩坏**: 连续 ≥3 步 answer_coverage < 30%。模型输出不含任何可解析的数字答案，reward 函数无法给分。
  fix_preset: `diagnose`（可能是 max_completion_length 截断了 tool call 或模型完全退化）

- [ ] **C11 — 长度上限/截断**: `clipped_ratio >= 0.80` 或 `completion_length_mean >= max_completion_length * 0.90`。连续 2 个已完成 step 命中时，固定输出的 `Status` 必须写 `WARN(length-limit)`；连续 3 个已完成 step 命中时，触发熔断，`Status` 写 `STOP(length-limit)`。
  fix_preset: `increase_max_completion_length`（max_completion_length / max_response_length ×2，优先解决截断，再判断 lr 或探索问题）

**C8 与 C0 的关系**: C0 检测 loss=0（结果），C8 检测 reward_variance≈0（原因）。C8 比 C0 更早触发（reward_variance 趋近 0 先于 loss 完全归零），且能直接定位根因。

**fix_preset 新增映射**:

| 命中 Check | fix_preset | 含义 |
|---|---|---|
| C7 行为退化 | `diagnose` | 不跳过 Phase 5，走完整分析 |
| C8 无学习信号 | `increase_temperature` | temperature += 0.2 |
| C9 输出崩坏 | `diagnose` | 不跳过 Phase 5，走完整分析 |
| C11 长度上限/截断 | `increase_max_completion_length` | max_completion_length / max_response_length ×2 |
<!-- /section:circuit-breaker-extension -->

<!-- section:data-surfacing-extension -->
### 轨迹文件数据表面化

在现有数据表面化准则基础上，增加轨迹文件的定期读取：

| 时机 | 操作 | 工具 | 捕获内容 |
|------|------|------|---------|
| 每步完成时 | Read `trajectories/step_XXXX.jsonl` | Read | 模型行为数据（工具调用、答案、reward 分布） |

**频率**: 每检测到新 step 完成时读取一次（与 tail training_log.txt 同步）。不需要读所有历史 step，只读最新的一个。

**注意**: 轨迹 JSONL 文件可能较大（含完整 chat_completions），Read 时只关注行为指标计算所需字段，不需要逐行人工审阅。
<!-- /section:data-surfacing-extension -->
