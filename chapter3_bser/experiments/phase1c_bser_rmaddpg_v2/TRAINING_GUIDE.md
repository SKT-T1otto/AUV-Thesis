# Phase 1C-v2 手工运行指南

## 1. 适用范围

本入口对应同一概念方法 `ch3_bser_rmaddpg_phase1c` 的独立实现版本 `bser.phase1c.execution_v2`。它保留每个智能体 28 维观测、3 维动作和 124 维集中式 critic 输入，只增加：

1. 核心环境外部的执行阶段 reward adapter；
2. 仅存于 replay/checkpoint 的 phase metadata；
3. phase-aware prioritized replay；
4. 执行阶段训练诊断。

v2 不读取 v1 checkpoint；v1 与 v2 输出目录、checkpoint schema、配置和启动脚本均独立。

## 2. 运行前检查

在仓库根目录打开 PowerShell，并激活项目环境：

```powershell
conda activate AUV
git status --short
git rev-parse HEAD
```

确认以下冻结文件没有因本次改动产生差异：

```powershell
git diff -- `
  chapter3_bser/experiments/phase1c_bser_rmaddpg/train_phase1c.py `
  configs/chapter3/bser_phase1c_train.json `
  scripts/run_phase1c_train.ps1 `
  scripts/run_phase1c_train.bat `
  core/env/uav_env.py `
  core/env/mission_env.py `
  core/replay/ch3_buffer.py
```

运行定向测试：

```powershell
python -m pytest `
  tests/test_phase1c_execution_diagnostics.py `
  tests/test_phase1c_checkpoint_evaluator.py `
  tests/test_phase1c_v2_reward_protocol.py `
  tests/test_phase1c_v2_execution_reward.py `
  tests/test_phase1c_v2_replay.py `
  tests/test_phase1c_v2_training_smoke.py `
  tests/test_phase1c_v2_isolation.py -q
```

## 3. 先诊断已有 Phase 1C-v1 checkpoint

单 checkpoint：

```powershell
.\scripts\run_phase1c_diagnostic_eval.ps1 `
  -Checkpoint "outputs\chapter3\phase1c_bser_rmaddpg\training\checkpoints\phase1c_episode_0100.pt"
```

多 checkpoint 对比：

```powershell
.\scripts\run_phase1c_diagnostic_eval.ps1 `
  -Checkpoint @(
    "outputs\chapter3\phase1c_bser_rmaddpg\training\checkpoints\phase1c_episode_0100.pt",
    "outputs\chapter3\phase1c_bser_rmaddpg\training\checkpoints\phase1c_episode_0500.pt",
    "outputs\chapter3\phase1c_bser_rmaddpg\training\checkpoints\phase1c_episode_1000.pt"
  ) `
  -ScenarioCount 20 `
  -Overwrite
```

BAT 入口适合单 checkpoint：

```powershell
.\scripts\run_phase1c_diagnostic_eval.bat -Checkpoint "完整checkpoint路径" -Overwrite
```

默认输出：

```text
outputs/chapter3/phase1c_bser_rmaddpg/diagnostic_eval/
├── resolved_diagnostic_eval_config.json
├── evaluation_manifest.json
├── episode_execution_diagnostics.csv
├── checkpoint_execution_summary.csv
└── diagnostic_eval_summary.json
```

该入口只执行确定性 rollout，不调用 replay 或任何 optimizer update。

## 4. v2 dry-run

先执行隔离的短 smoke run：

```powershell
.\scripts\run_phase1c_v2_train.ps1 -DryRun
```

BAT 等价入口：

```powershell
.\scripts\run_phase1c_v2_train.bat -DryRun
```

Dry-run 使用配置中的 `dry_run` 覆盖值，输出到：

```text
outputs/chapter3/phase1c_bser_rmaddpg_v2/training/dry_run/
```

只有 `training_summary.json` 中同时满足以下条件才视为通过：episode/diagnostic 数量完整、`replay_sample_count > 0`、`optimizer_update_count > 0`、actor 参数确实变化、checkpoint 保存后可重新加载校验。

## 5. 正式 v2 训练

```powershell
.\scripts\run_phase1c_v2_train.ps1
```

配置文件：

```text
configs/chapter3/bser_phase1c_v2_train.json
```

默认正式计划为 seed 2729、1000 episodes、每 episode 最多 400 steps、4 workers，并按 100 episodes 保存 checkpoint。不要在同一输出目录重复启动一条全新训练；入口会拒绝覆盖已有结果。

## 6. v2 断点续训

仅允许 v2 schema checkpoint：

```powershell
.\scripts\run_phase1c_v2_train.ps1 `
  -Resume "outputs\chapter3\phase1c_bser_rmaddpg_v2\training\checkpoints\phase1c_v2_episode_0500.pt"
```

恢复时必须使用与 checkpoint 完全一致的 resolved config。`bser.phase1c.training_state.v1` 会被明确拒绝，不能通过改文件名绕过。

## 7. 正式输出

```text
outputs/chapter3/phase1c_bser_rmaddpg_v2/training/
├── resolved_training_config.json
├── checkpoints/
│   ├── phase1c_v2_episode_0100.pt
│   └── checkpoint_list.json
├── logs/
│   ├── training_log.json
│   └── worker_failure_episode_XXXX.json  # 仅异常时生成
└── metrics/
    ├── episode_metrics.csv
    ├── execution_diagnostics.csv
    ├── training_summary.json
    ├── loss_curves.png
    ├── reward_curve.png
    ├── success_found_trend.png
    └── phase_aware_replay_curve.png
```

重点检查：`success_if_found_rate`、发现后到成功的时延、executor 与目标/拦截点距离、contact/hold 统计、post-found reward、replay 四类样本比例、fallback 次数以及 success-tail 标记数量。
