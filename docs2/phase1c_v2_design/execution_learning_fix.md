# Phase 1C-v2 执行学习修复

## 目标

Phase 1C-v2 在冻结 Chapter 3 core、Phase 1B 在线语义和 Phase 1C-v1 训练入口的前提下，单独修复发现目标后的执行学习问题。概念方法名仍为 `ch3_bser_rmaddpg_phase1c`，实现版本通过 `bser.phase1c.execution_v2` 区分。

## 信息边界

每个智能体的 actor observation 仍为 28D，action 仍为 3D，集中式 critic 输入仍为 124D。contact、full-hold、success-tail、真实目标距离和预测误差均不拼接到 actor/critic 输入。真实目标位置只允许出现在离线 diagnostics 和显式标注的 oracle diagnostic evaluator 中。

## 奖励修复

v2 不修改 `core/env/uav_env.py`。环境先返回冻结 core 的 tanh reward，随后 `Phase1CExecutionRewardAdapter` 在 wrapper 外层做显式校正：发现前保留 base reward；首次发现步搜索者只保留 `reward_find_event + reward_early_find` 的 tanh 后奖励；后续发现后搜索者奖励清零；executor 保留 base reward并叠加 contact-entry、full-hold counter 增量和首次 terminal success 的 post-tanh bonus。默认系数 0.25/0.20/2.0 仅是 v2 初始训练参数，不代表论文最终最优值。

## Phase-aware replay

每条 transition 额外保存 replay-only metadata，并按 `SUCCESS > HOLD > CONTACT > POST_FOUND > PRE_FOUND` 分类。抽样层目标比例为 pre-found 40%、post-found 30%、contact/hold 20%、success-tail 10%。成功 episode 的末尾 32 条 transition 回标为 success-tail。某层为空时 quota 会重新分配；稀有层不足时默认允许有放回抽样；importance sampling 权重基于实际混合概率计算。

## Checkpoint 隔离

v2 checkpoint schema 为 `bser.phase1c.training_state.v2`。加载器在恢复网络或 replay 前检查 schema，明确拒绝 `bser.phase1c.training_state.v1`，避免 v1 reward/replay 语义与 v2 混用。

## 手工验收

先运行定向测试，再执行 v1 checkpoint deterministic diagnostics，再运行 `run_phase1c_v2_train.ps1 -DryRun`。只有 dry-run 的 replay sample、optimizer update、actor 参数变化、diagnostics 和 checkpoint reload 均通过后，才手工启动 100/300 episode pilot；不得由代码生成流程自动启动正式长训练。
