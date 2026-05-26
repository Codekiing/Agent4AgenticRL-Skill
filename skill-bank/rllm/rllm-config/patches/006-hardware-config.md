---
id: "006-hardware-config"
target_section: "output"
action: append
description: "config.json 中附加硬件信息字段，供 rllm-run 等下游 skill 读取"
source: "2026-05-22 用户反馈: TrainingConfig 没有 GPU 字段，rllm-run 无法得知硬件配置"
created: "2026-05-22"

depends_on: []
conflicts_with: []

status: active
superseded_by: ""
---

### 硬件信息字段

生成的 config.json 除了 TrainingConfig 标准字段外，附加以下硬件信息字段：

```json
{
  "num_gpus": 4,
  "gpu_type": "A100"
}
```

- `num_gpus`: 从用户输入中提取的 GPU 数量（未指定时为 1）
- `gpu_type`: GPU 型号，从用户输入中提取（A100/H100/4090 等，未指定时为空字符串）
- 这些字段不影响 TrainingConfig 解析（`from_json` 会过滤未知字段），仅供 rllm-run 读取

### GPU 相关预检

当 `num_gpus >= 1` 且 `gpu_type` 非空时，额外执行以下检查：

1. **显存可行性**: 查上方 GPU 显存估算表，检查 per_gpu 显存是否满足模型 + batch + generations 需求
2. **梯度检查点**: 7B 模型自动启用 `gradient_checkpointing: true`（减少 ~30% 显存）
