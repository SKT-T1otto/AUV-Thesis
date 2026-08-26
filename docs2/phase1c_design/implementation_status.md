# Chapter 3 Phase 1C BSER-RMADDPG 历史实现与运行记录

## 1. 状态结论

本文件是 Historical record，保留旧 BSER-RMADDPG Phase 1C 的历史实现与运行记录。它此前被描述为
resume-ready，但不再是当前实验或当前续跑目标。当前工作是 PRRAC 架构实现后的
post-rebuild source repair and verification；尚未运行 PRRAC dry-run 或 pilot，
`performance_passed` 尚未成立，Chapter 4 RCAG 尚未开始。

旧 Phase 1C 的接口骨架、独立 runtime、preflight、训练入口、checkpoint 与恢复机制均已实现。历史实验状态必须与当前 PRRAC 代码实现状态分开描述：

- 实现状态：trainer 和恢复管线已存在。
- 实验状态：**WIP / short training interrupted / resume-ready**。
- 尚未完成：1000 episodes、正常终止汇总、正式曲线、收敛判断和正式对比结论。

当前方法为 `ch3_bser_rmaddpg_phase1c`。BSER 保持高层任务分配与 guidance，现有 waypoint prior 保持导航先验，RMADDPG 保持三维 residual action 生成器。

## 2. 已实现组件

### Integration

- `chapter3_bser/integration/control_context.py`
- `chapter3_bser/integration/rmaddpg_bridge.py`
- `chapter3_bser/integration/guided_env.py`

`BSERControlContextV1` 是不可变控制上下文；bridge 只把 allocation 编译成 tracking target/hold position，不生成速度或动作。`GuidedEnv` 安装 guidance 并在 guidance 更新后刷新 28D observation。

### Runtime 与训练器

- `chapter3_bser/experiments/phase1c_bser_rmaddpg/run_phase1c.py`
- `chapter3_bser/experiments/phase1c_bser_rmaddpg/train_phase1c.py`
- `configs/chapter3/bser_phase1c_train.json`
- `scripts/run_phase1c_train.ps1`
- `scripts/run_phase1c_train.bat`

独立 trainer 支持配置、seed/episode override、dry run、隔离输出、定期 checkpoint 和 `--resume`。Checkpoint 包含 actor、双 critic、target networks、optimizers、replay buffer、已完成 episode、更新计数和 config hash。

### 诊断与日志

- 每 episode 指标：`initial_planner_endpoint_fallback_count`
- Worker 异常：`training/logs/worker_failure_episode_XXXX.json`
- PowerShell 控制台：`training/logs/console_YYYYMMDD_HHMMSS.log`
- Episode 指标：`training/metrics/episode_metrics.csv`

## 3. 接口调用链

```text
belief/task state
  -> OnlineBSERController
  -> BSERControlContextV1
  -> RMADDPGGuidanceBridge
  -> GuidedEnv waypoint prior
  -> RMADDPG residual action
  -> environment step
  -> refreshed planning state and guidance
  -> synchronized 28D next observation
  -> replay transition
```

RMADDPG 仍是唯一动作生成位置。Observation 为每 agent 28D，action 为每 agent 3D，centralized critic 输入为 124；replay state/action/next_state contract 未改变。

## 4. 历史训练配置

| 字段 | 历史值 |
|---|---|
| Method | `ch3_bser_rmaddpg_phase1c` |
| Profile | `M20_MOVING_UNKNOWN_MULTI` |
| Seed | `2729` |
| Episodes planned | `1000` |
| Max steps | `400` |
| Workers | `4` |
| Training update | `true` |
| Checkpoint interval | `50 episodes` |

历史输出隔离在 `outputs/chapter3/phase1c_bser_rmaddpg/training/`。

## 5. 历史中断与修复

短训练在记录 episode 128 后异常退出。最后完整 checkpoint 是 episode 100；episode 101-128 只存在于中断前 CSV，不属于该 checkpoint，恢复后需要重新运行。当前没有正常结束时应生成的完整 `training_summary.json`、`training_log.json` 和正式曲线。

异常为：

```text
RuntimeError: current_pos is not a legal reachable planner point
```

调用链位于 unknown-map reset：初始 obstacle scan 之后，连续空间合法的 agent 位置可能被粗网格判为 `point_invalid` 或 `no_connector`；legacy `initial_search_targets` 随后直接调用 `sample_next_waypoint`，worker 异常再传播到 `ProcessPoolExecutor` 并终止主训练。

当前 `GuidedEnv` 已加入限定在 Phase 1C 的 initial endpoint guard：

- unknown-map reset 时检查初始搜索者 endpoint；
- 不可达时临时使用当前位置作为 legacy initial waypoint；
- reset 返回后仍由 BSER guidance 覆盖；
- 不跳过场景或 episode；
- 不修改 `core/mapping/path_planner.py`；
- 不修改 Phase 1B、reward、动力学、网络或 replay contract。

## 6. Historical recorded verification（历史本地验证）

这些结果来自最近一次本地验证，不是 GitHub CI workflow 结果：

| 验证 | 记录结果 |
|---|---|
| Phase 1C guidance | `3 tests OK` |
| Observation 28D + Phase 1B path tracking | `PASS (2 tests)` |
| Episode 100 checkpoint | 可读取 |
| Actor/critic/optimizer/replay/training counters | 可恢复 |

验证命令：

```powershell
conda run --no-capture-output -n AUV python -B -m unittest tests.test_phase1c_guidance -v
conda run --no-capture-output -n AUV python -B -m unittest tests.test_observation_28d_contract tests.test_phase1b2_path_tracking -v
```

## 7. 历史续跑命令

以下命令只为历史溯源保留，不是当前 PRRAC 工作流；只有用户明确恢复旧实验时才可执行：

```powershell
.\scripts\run_phase1c_train.ps1 `
  -Resume "outputs/chapter3/phase1c_bser_rmaddpg/training/checkpoints/phase1c_episode_0100.pt"
```

该命令不会由 Codex 自动运行。恢复点为 episode 100，不是 episode 128。

## 8. 边界结论

旧 Phase 1C 曾被记录为具备 resume-ready 的工程条件，但它不再是当前实验或当前续跑目标，也不能标记为完成。当前 PRRAC 必须先由用户明确批准实验；只有计划训练正常结束、终态汇总与曲线存在，并完成正式评估后，才能讨论最终性能或对比结论。
