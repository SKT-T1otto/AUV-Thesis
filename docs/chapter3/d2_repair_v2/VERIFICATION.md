# D2 repair v2：最新记录的本地验证

结论为 **`MECHANISM_VERIFIED`**，不是正式训练性能通过，也不是 CI 或完整回归通过。

## 环境

- Windows 10 build 26200；Python 3.10.20，`D:\anaconda\anaconda\envs\AUV\python.exe`。
- Torch 2.11.0+cu126、NumPy 2.2.6，实际 learner 使用 CPU，单线程。
- `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。
- HEAD：`6538c3e6a67e96d56940ab09fcff15cd9dc7a7d8`，分支 `codex/framework-audit-20261018-164056`，未提交修复。

## 实际执行与结果

| 验证 | 实际命令/完整命令记录 | 结果 |
|---|---|---|
| 修复前接口反例 | `python -m docs.chapter3.framework_reaudit_20261018.reproduce.probe_interfaces`；见 before_probes.log | 复现广播、恢复未写槽位、门控旁路、快照别名、step 截断、诊断标志 |
| 定向及真实环境链路 | `python -m tmp.d2_repair_v2_20261018.run_verification targeted` | 27 项通过，exit 0，264.937 秒 |
| 原调度与 checkpoint 最终重跑 | `python -m unittest tests.test_d2_repair_v2_checkpoint -v` | 1 项通过，exit 0，315.701 秒 |
| 修正新来源清单后的最终检查 | `python -m unittest tests.test_d2_repair_v2_provenance -v` | 4 项通过，exit 0，10.158 秒 |
| 相关回归 | `python -m tmp.d2_repair_v2_20261018.run_verification regression` | testsRun=108；101 success 记录，1 failure，9 error；exit 1 |
| 邻近 checkpoint 与旧 Suite 仅预览补查 | `python -m tmp.d2_repair_v2_20261018.supplemental_checks` | 21 项：18 通过、3 来源门禁 error；exit 1，55.347 秒 |
| 当前来源独立检查 | `python -m tools.verify_d2_repair_v2_source` | 通过，windows_existing，exit 0 |
| 干净父归档＋修复覆盖 | `python -m tmp.d2_repair_v2_20261018.clean_workspace`；子命令见 clean_workspace.json | 最终 source exit 0；27 项 core/历史检查 exit 0，3.717 秒 |
| 历史 E0 前置文件检查 | `Path("experiments/chapter3/e0_core_migration/golden_trace_manifest.json").read_bytes()` | FileNotFoundError，exit 1；未启动黄金轨迹验证，未生成替代文件 |

命令中的 `python` 均为上述 AUV 环境解释器。实际参数、完整测试名称、退出码、版本和失败 traceback 分别保存在 [targeted.json](evidence/targeted.json)、[regression.json](evidence/regression.json)、[additional_runs.json](evidence/additional_runs.json)、[clean_workspace.json](evidence/clean_workspace.json) 和 [e0_preflight.json](evidence/e0_preflight.json)。完整控制台日志同目录保留。runner 文件另保留为 evidence/run_verification.py，方便重跑同一模块集合；其实际运行时位置为 tmp。

早期开发验证也确实执行：初始 14 项快速测试、20 项航点/来源/历史检查、首次真实环境链路、首次 checkpoint 闭环。最终验收以上表最终重跑为准；没有把重复运行相加夸大独立覆盖。

## 定向覆盖

- `test_maddpg_sample_contract`：批长、agent 数、维度、空 batch、weights、flags、有限数及更新前拒绝；独立 snapshot 和加载；step 数量、critic-only 诊断。
- `test_ch3_replay_restore_contract`：容量不匹配拒绝且无 mutation；坏形状、字段长度、指针、frame；同容量环绕、继续采样及优先级。
- `test_execution_replay_gate_contract`：普通 push 旁路、原子转移、环形 gate 对应、未初始化/损坏 gate、动作/终止矛盾。
- `test_maddpg_terminal_target_numerics`：手算 Bellman、终止 NaN 与 live NaN、Huber loss 与导数、非有限 loss/grad 的 optimizer 顺序。
- `test_maddpg_actor_update_mask`：全冻结 Adam 和参数不动、有效行梯度、optimizer 所有权、B3 encoder/context、逐参数软更新。
- `test_waypoint_reward_timing`：真实到达一次、无到达奖励保持、连续步、发现/碰撞同一步、地图 revision 变化、发现后无活动搜索路径。
- `test_d2_repair_v2_provenance`：完整清单、独立验证器、未登记变动/新增生产文件/配置漂移拒绝、正式门禁、只打印计划无产物。
- `test_d2_repair_v2_ablation`：全部关闭与冻结旧参考一致，奖励及超参数一致；冻结单开、保持门控单开；航点单开由时序模块覆盖。
- `test_d2_repair_v2_runtime_contract`：四种模型真实轨迹、事件、物理动作、回放和最多 4 轮更新。
- `test_d2_repair_v2_checkpoint`：原 batch/warmup/频率下真实 Trainer 调度、临时保存/重载、独立评估和来源失败拒绝。

## 真实环境及更新证据

[runtime_chain.json](evidence/runtime_chain.json)：旧 B2/B3 与 v2 B2/B3 各 147 步、7 场景、4 轮完整更新，共 588 个环境步；每场景≤32 步。全部覆盖发现前、首次发现、等待交接、Executor 知情、冻结、成功、碰撞终止及超时。四个模型均使用全新随机初始化。

[scheduler_checkpoint.json](evidence/scheduler_checkpoint.json)：v2 B2/B3 各 256 步、8 回合，按原调度执行 2 轮更新，Actor 计数各为 [2,2,2,2]，twin Critic optimizer 总步数各为 16。各自完成一次 32 步独立场景评估；这些是另外两个临时模型，不与 4 轮探针共享模型预算。

完整原始轨迹未作为正式数据提交。每种对象保留 8 个关键边界的全字段样例：

- [旧 B2](evidence/transition_examples_b2_False.json)、[旧 B3](evidence/transition_examples_b3_False.json)。
- [v2 B2](evidence/transition_examples_b2_True.json)、[v2 B3](evidence/transition_examples_b3_True.json)。
- 完整本地文件路径与 hash：[trace_index.json](evidence/trace_index.json)。

Replay 所存动作为门控后的归一化残差，物理加速度另列。速度差分与控制加速度是不同量，observer 明确区分；没有把位移/dt 当成加速度。真实环境状态转移和团队最终奖励均与该步 Replay 对齐。

## 回归失败完整归类

**SafeSearch source inventory mismatch**（父版本已有同类失败，新增 v2 文件也不符合其旧封闭清单）：

1. `test_ch3_baseline_training...test_real_checkpoints_evaluate_without_optimizer_updates`，B2 和 B3 两条 subTest error。
2. `test_ch3_baseline_training...test_b0_b1_unchanged`，error。
3. `test_ch3_baseline_training...test_training_check_only`，error。
4. `test_ch3_baseline_validation.B0SmokeTests`、`B1SmokeTests` 的 setUpClass，各一条 error，相应类测试未进入运行。
5. `test_ch3_baseline_validation...test_check_only_never_constructs_trainer_or_writes_outputs`，error。
6. `test_ch3_baseline_validation...test_frozen_production_and_baseline_source_gates`，error。
7. `test_ch3_baseline_training...test_evaluation_rejects_hgr_checkpoint`，failure：来源门禁先抛出异常，未到该测试期望的旧 checkpoint 身份拒绝分支。

**旧 repair v1 精确来源拒绝**：

8. `test_d2_implementation_repairs...test_short_training_checkpoint_and_held_out_evaluation`，error：`unreviewed repair source drift: core/algorithms/maddpg.py`。本次 core 明确演进不属于旧 v1 hash，因此其原门禁正确拒绝；不更新 v1 清单来掩盖该拒绝。

总计 1 条 failure、9 条 error 记录；其中 2 条是同一方法的 subTest，2 条来自 setUpClass。testsRun 不等于成功/失败记录简单求和。完整原始 traceback 见 [regression.json](evidence/regression.json) 和 [regression.log](evidence/regression.log)。没有以 skip、捕获后继续训练、删除文件或放宽 hash 的方式处理失败。

core checkpoint、import graph、无 legacy import、场景生成、独立环境、短训练闭环、28D/3D/124D、碰撞/超时、Phase1C/PRRAC Replay、奖励适配、B2/B3 网络和 boundary context、历史 27 记录及精确演进等已运行的其余测试通过。它们不能替代被上述门禁阻断的调用路径。

另对 `test_prrac_checkpoint`、`test_prrac_runtime_checkpoint_compatibility`、`test_prrac_checkpoint_evaluator`、`test_phase1c_checkpoint_evaluator` 补查，共 18 项通过，均只使用测试临时 checkpoint。额外选择的旧 Suite 三个仅预览测试（PipelineTests 的 nine/ten preview、LinuxLaunchTests 的 preflight preview、EvaluationEntryTests 的 preview selection）均被 SafeSearch source inventory mismatch 阻断。没有运行旧 Suite 的实际训练或 HGR。完整 21 项命令、退出码、环境和三个 traceback 见 [supplemental.json](evidence/supplemental.json)、[supplemental.log](evidence/supplemental.log)，其 runner 为 evidence/supplemental_checks.py。此结果独立于 108 项回归，不篡改先前测试记录。

## 来源清单修正与历史 E0

首次新清单使用了错误的统一 LF 假设。干净归档独立验证器返回 exit 1，失败 stdout/stderr 原样留存于 [clean_workspace_initial_failure.json](evidence/clean_workspace_initial_failure.json)。没有因此生成正式 checkpoint。之后从固定父提交实际 git archive 读取原字节，修正**新**清单的归档配置并重新核验；当前和干净目录均通过。checkpoint 闭环在最终清单身份下重新跑过。

[preservation.json](evidence/preservation.json) 核对初始快照中的历史清单、原配置和 302 个保留生产文件。27 条历史记录继续由相关测试精确核验。历史 hash 未改写。

历史 E0 文件确实缺失，文件读取返回 FileNotFoundError，完整命令与 traceback 保留于 [e0_preflight.json](evidence/e0_preflight.json)。E0 状态为 **未完成**，没有用新轨迹补作历史黄金数据。

干净验证是固定父归档＋显式覆盖本次未提交文件；未执行 git commit/push，因此不是修复提交的真正 clean-clone 发布验证。没有运行 Linux/GPU 正式训练、历史 checkpoint 恢复、Replay 扩容、长期稳定性及公平性能评估。

## 安全重查命令

以下 PowerShell 命令只做来源检查和打印训练计划，不启动训练：

```powershell
Set-Location E:\gym\code\WORKSPACE\AUV-Thesis
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& D:\anaconda\anaconda\envs\AUV\python.exe -m tools.verify_d2_repair_v2_source
& D:\anaconda\anaconda\envs\AUV\python.exe -m chapter3_bser.experiments.d2_repair_v2.train --config configs/chapter3/d2_repair_v2/b2_train.json --output outputs/chapter3/d2_repair_v2/collision_terminal/manual_b2
```

第二条应显示 `executed=false`、`admitted=false`，原因是当前验收仍未满足全部来源/回归/E0 条件。不要添加执行标志绕过本次结论；机制执行权限只允许临时、明确限制预算的配置。

最终 receipt：[acceptance.json](evidence/acceptance.json)。其 `formal_training_allowed=false`、`performance_passed=false`；实现修复已交付，正式训练资格和性能结论仍未建立。
