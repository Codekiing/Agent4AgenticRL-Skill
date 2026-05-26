---
id: "003-active-polling-mandate"
target_section: "monitoring-methods"
action: append
description: >-
  强制要求主动逐步轮询监控，禁止用被动后台等待替代。
  修复 Round 1 中 monitor 在 Step 5 后停止输出的问题。
source: "2026-05-20 R1 训练, 编排层在 Step 5 后停止 tail 轮询改为被动等待"
created: "2026-05-20"

depends_on: ["002-reliability"]
conflicts_with: []

status: active
superseded_by: ""
---

### 主动轮询强制规则 (ACTIVE POLLING MANDATE)

#### 核心规则：Monitor 必须主动轮询，绝不被动等待

**定义**:
- **主动轮询 (Active Polling)**: 编排层每隔固定间隔执行 `tail -30 training_log.txt`，提取并汇报关键指标给用户
- **被动等待 (Passive Waiting)**: 启动一个后台 while 循环或 background task 等待训练完成，期间不做任何 tail 或汇报

**强制规则**:

1. **主动轮询是唯一的监控手段**。编排层 MUST 在训练过程中持续执行:
   ```bash
   tail -30 rllm_train/output/runs/<run_id>/training_log.txt
   ```
   间隔: 每 2 分钟 (120s) 或每观察到新 step 完成时，取先到者。

2. **禁止被动等待**。编排层 MUST NOT:
   - 创建 `while true; do sleep; done` 后台循环来等待训练完成
   - 仅依赖 `TaskOutput` 阻塞等待而不做中间汇报
   - 在任何 step 之后停止 tail 检查
   - 将"等待训练完成"作为唯一的监控策略

3. **每次轮询必须汇报，且格式固定**。每个 tail 检查后，编排层 MUST 向用户输出一行训练状态摘要。无论本轮拿到哪些指标，都必须使用同一个字段顺序；缺失字段统一填 `—`，不得临时增减字段。
   ```
   Step X/Y | R Z.ZZZ | Rstd Z.ZZZ | Loss L.LLLL | Grad G.GGGG | Ent E.EEEE | Clip C.CC | Len N | Finish P% | FmtOK P% | Tool P% | Ans P% | tok/s T.T | Time S.Ss | ETA ~Mm | Status OK/WARN/STOP
   ```
   字段含义固定为：`R=avg_reward`, `Rstd=reward_variance`, `Loss=policy_loss`, `Grad=grad_norm`, `Ent=entropy`, `Clip=clipped_ratio`, `Len=completion_length_mean`, `Finish=finish_rate`, `FmtOK=finish_format_rate`, `Tool=tool_usage_rate`, `Ans=answer_coverage`, `Time=step_time`。

   如果没有新 step 完成，也必须输出同一格式，复用最新已完成 step 的指标，并将 `Status` 写为 `RUN(no-new-step)`；如果还没有任何已完成 step，则所有指标填 `—`。

   `Status` 必须优先暴露长度/截断风险：当 `Clip >= 0.80` 或 `Len >= max_completion_length * 0.90` 时写 `WARN(length-limit)`；连续 3 个已完成 step 命中该条件时写 `STOP(length-limit)` 并触发 `increase_max_completion_length`。

4. **轮询不可中断**。只有以下情况可以停止主动轮询:
   - 训练已完成（检测到 "Training Report" 或后台进程退出）
   - 检测到严重异常（OOM、崩溃）需要停止训练
   - 用户明确要求停止

5. **后台进程是辅助，不是替代**。训练进程可以后台运行（`run_in_background: true`），但后台进程状态检查只是主动轮询的补充信息来源，不能替代 tail 日志检查。

#### 执行模式

正确的监控循环 (伪代码):

```
train_process = Bash("torchrun ...", run_in_background=true)
last_step = 0

while 训练未完成:
    # 1. 主动读取日志
    log_output = Bash("tail -30 training_log.txt")
    current_step = extract_step(log_output)

    # 2. 如果有新 step，立即汇报
    if current_step > last_step:
        向用户汇报训练进度
        last_step = current_step

    # 3. 检查异常
    check_anomalies(log_output)

    # 4. 检查训练是否完成
    if "Training Report" in log_output or 训练进程已退出:
        break

    # 5. 等待下一轮轮询
    ScheduleWakeup(120s) 或 sleep 120s
```

错误的监控模式 (禁止):

```
# 错误: 只启动后台等待，不做中间汇报
train_process = Bash("torchrun ...", run_in_background=true)
TaskOutput(train_process, block=true, timeout=600000)  # 阻塞等待，中间无输出
# 用户在 30 分钟内看不到任何训练状态
```
