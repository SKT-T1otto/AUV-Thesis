# 非测试源码职责索引（2026-10-18）

本索引覆盖 369 个 Python 文件（68,533 行）：core 44 个、chapter3_bser 251 个，其余为 scripts/tools 和 docs 内的分析脚本。排除 tests、历史 test、输出数据、缓存与 tmp。

覆盖含义：全部文件做 AST/导入/符号与入口扫描；MADDPG 与 D2 B2/B3 主调用链另做源码逻辑审查和独立机制核验。此索引不代表 369 个文件均已逐行语义证明无错。没有运行整仓 test 或正式训练。

`core` 不导入章节方法或 legacy adapters。`chapter4_rcag`、`chapter5_vsgc` 目前没有算法 Python 文件。

## 顶层目录

| 目录 | 实际职责 |
|---|---|
| `core` | 唯一共享执行实现，重点审查对象 |
| `chapter3_bser` | 第三章高层 BSER、章节网络、训练/评估/诊断 |
| `chapter4_rcag` | README 占位；未实现 RCAG |
| `chapter5_vsgc` | README 占位；未实现 VSGC |
| `configs` | 实际已含生效配置，README 的 placeholder 描述过时 |
| `scripts` | 跨平台/Windows/Linux 手动入口 |
| `tools` | 来源、复现、性能和基线管理工具 |
| `docs` | 设计、审查、紧凑证据，另有 31 个分析/复现 Python 脚本 |
| `docs2` | 较早设计与跨平台审计资料 |
| `experiments` | 历史章节实验清单、数据与证据，不是当前训练器代码 |
| `historical_verification` | 历史验收；test 文件不在本次审查范围 |
| `legacy_adapters` | 仅保留说明文件；不能作为运行依赖 |
| `outputs / output / runs` | 实验模型、日志、结果和诊断数据；本次不改写用户留存数据 |
| `3090结果` | 机器导出的实验结果和 manifest；不属于算法实现 |
| `文献` | 参考文献 |
| `tmp` | 本地临时探针、克隆和验证中间产物 |
| `tests` | 本次按要求排除 |
| `.git / .idea / .pytest_cache` | 版本、IDE 和缓存元数据 |

## 易混淆名称

- 本轮 Linux 的 `D2_B2 / D2_B3` 是 experiments/baselines 的两个学习基线。
- PRRAC execution_continuity 中的 `B0/B1/B2/B3` 是执行导航消融编号，与上面无关。
- `direct_mc` 是历史名称；当前实现明确为一步 TD 的 off-policy MADDPG。
- `chapter3_bser/baselines` 是高层分配基线，与 experiments/baselines 的神经网络训练不同。
- D2 零残差运行器仍复用 HGR 目录的通用 MissionRuntime；取消 HGR 算法不等于可以直接删除这个目录。
- long_horizon_maddpg_v1 是上一轮未采纳提案，不能作为已完成奖励修复或已验证算法。

## Python 文件逐项索引

### chapter3_bser

BSER 离线问题定义、候选、探测模型、目标和优化器。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Chapter-3 BSER finite offline allocation package. |
| [candidate_generator.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/candidate_generator.py) | 构造可达搜索和执行待机候选。符号：`_local_peak_indices, _frontier_indices, _ordered_pool, _separated, generate_search_candidates, generate_standby_candidates, generate_candidates`<br>源码说明：Deterministic belief/frontier candidate construction. |
| [config.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/config.py) | BSER 离线目标和候选参数加载/约束。符号：`load_bser_config, load_bser_phase1a1_config`<br>源码说明：Frozen Phase-1A configuration loader. |
| [detection_model.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/detection_model.py) | 高层候选搜索路径对信念格点的探测概率模型。符号：`_point_segment_distance, detection_probability`<br>源码说明：Fixed, interpretable path-coverage detection kernel. |
| [exact_solver.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/exact_solver.py) | 离线穷举最优分配。符号：`partition_groups, feasible_allocations, exact_combination_count, solve_fixed_standby_exact, solve_joint_exact`<br>源码说明：Complete finite enumeration for small BSER instances. |
| [greedy_solver.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/greedy_solver.py) | 分区约束下逐个选取正边际增益候选。符号：`solve_fixed_standby_greedy, solve_joint_greedy`<br>源码说明：Standard marginal-gain greedy solver under a partition matroid. |
| [lazy_greedy_solver.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/lazy_greedy_solver.py) | 缓存边际增益的惰性贪心求解。符号：`solve_fixed_standby_lazy, solve_joint_lazy`<br>源码说明：Deterministic lazy marginal-gain greedy solver. |
| [metrics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/metrics.py) | 高层离线解质量与计算指标。符号：`partition_feasible, validate_small_instance`<br>源码说明：Feasibility and finite-instance mathematical checks. |
| [objective.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/objective.py) | BSER 目标上下文、检测概率、响应时间、目标值和边际增益。符号：`ObjectiveContext, ResponseDiagnostics, build_objective_context, cell_detection_probability, evaluate_objective, marginal_gain, expected_detection_probability, expected_response_time, response_diagnostics, coverage_overlap`<br>源码说明：BSER nonnegative monotone submodular objective and diagnostics. |
| [standby_optimizer.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/standby_optimizer.py) | 对选定搜索组合寻找执行待机位置<br>源码说明：Finite standby enumeration wrappers. |
| [types.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/types.py) | 离线候选、任务、目标上下文与求解结果 dataclass。符号：`SearchCandidate, StandbyCandidate, CandidateGenerationResult, SolverResult`<br>源码说明：Immutable BSER value objects. |

### chapter3_bser/baselines

高层分配基线；不是训练 B2/B3 的目录。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/baselines/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Offline comparator allocators for BSER-E1. |
| [legacy_pse_snapshot.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/baselines/legacy_pse_snapshot.py) | 高层分配基线；不是训练 B2/B3 的目录；具体入口见符号。符号：`evaluate_legacy_pse_snapshot`<br>源码说明：Read-only evaluation of current PSE navigation outputs. |
| [random_allocator.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/baselines/random_allocator.py) | 高层分配基线；不是训练 B2/B3 的目录；具体入口见符号。符号：`RandomAllocatorSummary, evaluate_random_allocator`<br>源码说明：Fixed-seed random feasible allocation baseline. |
| [search_only_allocator.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/baselines/search_only_allocator.py) | 高层分配基线；不是训练 B2/B3 的目录；具体入口见符号。符号：`solve_search_only_greedy`<br>源码说明：Search-only comparator using identical candidates and detection kernels. |

### chapter3_bser/controllers

公共状态采样、路径跟踪；固定动作适配只用于早期 E2。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/controllers/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Experiment-only adapters around high-level BSER assignments. |
| [action_adapter.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/controllers/action_adapter.py) | 早期 E2 固定反馈动作，D2 B2/B3 不使用这套动作生成。符号：`assignment_to_fixed_actions`<br>源码说明：Convert high-level targets to legal fixed feedback actions for E2 only. |
| [path_tracker.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/controllers/path_tracker.py) | 对不可变路径维护局部目标索引和跟踪/剩余路径诊断。符号：`_TrackingState, PathTrackerSnapshot, PathTracker`<br>源码说明：Stateful local-target tracking along an immutable planned path. |
| [state_provider.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/controllers/state_provider.py) | 每步更新位置/速度；周期或交接事件刷新完整规划图。符号：`OnlinePlanningStateProvider`<br>源码说明：Public-interface state sampling for the E2 experiment harness. |

### chapter3_bser/diagnostics

事件语义诊断。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/diagnostics/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Information-bounded diagnostics for BSER experiments. |
| [event_semantics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/diagnostics/event_semantics.py) | 事件语义诊断；具体入口见符号。符号：`mission_phase, classify_executor_invalid, classify_waypoint_stale, AssignmentVersionTracker, diagnostic_event_key, public_target_lock_violation, _InstalledRoute, _route_snapshot, _current_components, _newly_occupied, _relative_cost_change, _distance, _json, Phase1B3ADiagnosticRecorder`<br>源码说明：Read-only Phase 1B.3A event semantics over planner-visible state only. |

### chapter3_bser/events

事件定义、检测及触发条件。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/events/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Observable-state BSER events. |
| [event_detector.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/events/event_detector.py) | 根据公共状态变化检测事件。符号：`EventDetector`<br>源码说明：Deterministic event detection from two information-bounded snapshots. |
| [event_types.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/events/event_types.py) | 事件枚举与诊断结构。符号：`BSEREvent, EventDetection`<br>源码说明：Immutable event types and diagnostics for online BSER. |
| [trigger_policy.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/events/trigger_policy.py) | 重规划触发规则。符号：`TriggerDecision`<br>源码说明：Event-to-replanning trigger decision value object. |

### chapter3_bser/experiments

章节实验入口及共同奖励目标转换。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：BSER-E1 offline experiment entry points. |
| [instance_builder.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/instance_builder.py) | 离线 BSER 实验问题实例构建。符号：`PlanningSnapshotMetadata, SnapshotInstance, SkippedSnapshot, action_trace, _seed_all, _metadata, iter_e1_snapshots`<br>源码说明：Build the E1-v2 snapshot protocol with metadata isolated from algorithms. |
| [plot_e1.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/plot_e1.py) | 离线结果分析、汇总或绘图。符号：`_rows, _save, generate_figures`<br>源码说明：Generate the six bounded BSER-E1 thesis-candidate figures. |
| [plot_e1_v2.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/plot_e1_v2.py) | 离线结果分析、汇总或绘图。符号：`_rows, _save, generate_figures`<br>源码说明：Generate the ten fixed E1-v2 descriptive figures (no cross-scale chart). |
| [reward_objective.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/reward_objective.py) | 执行奖励 adapter 之后唯一的 team_mean 转换和折扣/不折扣回报记账。符号：`objective_identity, RewardAccounting`<br>源码说明：The sole final reward transform, after the existing execution adapter. |
| [run_e1_offline.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/run_e1_offline.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_write_csv, _stable_hash, _round, write_protocol_aggregates, _best_standby, run, main`<br>源码说明：Run BSER-E1 finite mathematical and solver-efficiency validation. |
| [run_e1_v2.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/run_e1_v2.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_write_csv, _write_json, _stable_hash, _finite, _metadata_row, _best_standby, _diagnostic_columns, _method_metrics, _random_metrics, _allocation_jaccard, _aggregate, run, _aggregate_methods, main`<br>源码说明：Run the deduplicated BSER E1-v2 offline validation protocol. |
| [run_s1_sensitivity.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/run_s1_sensitivity.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_write_csv, _jaccard, _displacement, _collect_states, run, main`<br>源码说明：One-factor-at-a-time BSER sensitivity analysis on fixed-hash M20 states. |

### chapter3_bser/experiments/baselines

独立 B2/B3 训练系统。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Independent Chapter 3 comparison methods. |

### chapter3_bser/experiments/baselines/common

B2/B3 共用真实训练、回放、运行器及 checkpoint 身份校验。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/common/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Shared plumbing for the independent direct MADDPG baselines. |
| [checkpoint.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/common/checkpoint.py) | 配置/方法身份、来源/模型/优化器完整性、计数和有限值校验。符号：`digest, write_json, validate_config, validated_output, fresh_source_identity, validate_source_identity, require_source_match, state_digest, _finite_state, load_model, load_checkpoint`<br>源码说明：Independent MADDPG baseline identities and real model/optimizer checkpoints. |
| [model.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/common/model.py) | B2 构建原 core MADDPG；B3 包装 actor 并从联合观测重建当前/后继边界上下文。符号：`boundary_features, BoundaryEncoder, BoundaryConditionedPolicy, BoundaryConditionedMADDPG, _contract_environment, build_model`<br>源码说明：Construct the frozen core MADDPG, with optional boundary conditioning. |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/common/runtime.py) | 真实 B2/B3 仿真、D2 控制器、guidance 更新、转移采集和时钟校验。符号：`seed_innovations, _make_base_env, _public_context, _observations, BaselineMissionRuntime`<br>源码说明：Independent simulator/planner rollout for the direct MADDPG baselines. |
| [train.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/common/train.py) | 真实 B2/B3 更新调度：uniform replay、原团队奖励、每轮更新四个 actor/critic。符号：`BaselineTrainer`<br>源码说明：Plain replay-based MADDPG training shared only by the independent B2/B3 baselines. |

### chapter3_bser/experiments/baselines/direct_boundary

B3 的薄入口；actor 多一个边界编码器，共用 B2 的 TD 学习。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_boundary/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：B3 boundary-conditioned MADDPG baseline. |
| [checkpoint.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_boundary/checkpoint.py) | 该方法 checkpoint 保存、恢复、迁移或兼容性核验。符号：`load_checkpoint`<br>源码说明：Strict B3 checkpoint loading for evaluation. |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_boundary/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`build_model`<br>源码说明：B3 runtime and boundary-conditioned core MADDPG model factory. |
| [train.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_boundary/train.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`DirectBoundaryTrainer`<br>源码说明：B3: BSER plus boundary-conditioned, independent MADDPG training. |

### chapter3_bser/experiments/baselines/direct_mc

B2 的薄入口；当前实现为 off-policy MADDPG 一步 TD，不是 MC 回报训练。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_mc/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：B2 direct MADDPG baseline. |
| [checkpoint.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_mc/checkpoint.py) | 该方法 checkpoint 保存、恢复、迁移或兼容性核验。符号：`load_checkpoint`<br>源码说明：Strict B2 checkpoint loading for evaluation. |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_mc/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`build_model`<br>源码说明：B2 runtime and the ordinary core MADDPG model factory. |
| [train.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/baselines/direct_mc/train.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`DirectMCTrainer`<br>源码说明：B2: BSER plus direct, independent MADDPG training. |

### chapter3_bser/experiments/bser_effect_v1

D0/D1/D2/D3 高层机制消融；D2 先纯搜索分配、后选择执行待机位置。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in D0--D3 BSER contribution evaluation; no training entry point. |
| [allocator.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/allocator.py) | 高层候选/任务分配与部分重分配。符号：`finite, EffectAllocator`<br>源码说明：Chapter-local BSER intervention; unchanged public candidate/detection models. |
| [analysis.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/analysis.py) | 离线结果分析、汇总或绘图。符号：`mean_or_none, paired_interval, penalized, summarize`<br>源码说明：Predeclared paired summaries. Development evidence never establishes efficacy. |
| [controller.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/controller.py) | 事件驱动控制流程和分配提交。符号：`EffectPolicy, ScoredWaypoints, EffectController`<br>源码说明：Common current-state scores and objective-scale-independent event acceptance. |
| [options.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/options.py) | 该方法参数默认值、解析和合法性约束。符号：`parse_arms`<br>源码说明：Frozen intervention definitions for the manual paired experiment. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile, framework_sources, verify_sources`<br>源码说明：Exact, additive D-experiment source evolution; all earlier seals stay frozen. |
| [run_windows.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/run_windows.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`write_json, request, validate_terminal, run_episode, read_completed, run_jobs, parser, main`<br>源码说明：Manual Windows-compatible paired development evaluation (plan by default). |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`EffectRuntime, make_runtime`<br>源码说明：The four opt-in arms share one V4 runtime and unchanged post-Found execution. |
| [solver.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_effect_v1/solver.py) | 该实验的候选组合求解。符号：`fixed_greedy, best_standby, solve`<br>源码说明：Controlled greedy ablations, with fixed unaffected agents in partial solves. |

### chapter3_bser/experiments/bser_final_v1

后续高层分配方案、车辆轨迹预测和候选评分；不是 core MADDPG。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in final BSER assessment; historical D experiments remain unchanged. |
| [allocator.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/allocator.py) | 高层候选/任务分配与部分重分配。符号：`FinalAllocator`<br>源码说明：Same-pool D2 reference, installable proposals, unchanged event lifecycle. |
| [analysis.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/analysis.py) | 离线结果分析、汇总或绘图。符号：`mean_or_none, paired_interval, penalized, summarize`<br>源码说明：Predeclared paired summaries. Development evidence never establishes efficacy. |
| [forecast.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/forecast.py) | 基于车辆物理约束的轨迹预测。符号：`Incomparable, vehicle_constants, observed_disturbance, preview_bridge, Forecast, _unknown_length, predict`<br>源码说明：Read-only public-kinematics prediction, with the actual tracker/bridge preview. |
| [options.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/options.py) | 该方法参数默认值、解析和合法性约束。符号：`parse_arms`<br>源码说明：Frozen F interventions. No evaluation-time hyperparameter search. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile, framework_sources, verify_sources`<br>源码说明：Exact, additive F-experiment source evolution; all earlier seals stay frozen. |
| [run_windows.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/run_windows.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`write_json, request, validate_terminal, run_episode, read_completed, run_jobs, parser, main`<br>源码说明：Manual Windows-compatible paired F evaluation (plan by default). |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`FinalRuntime, make_runtime`<br>源码说明：All F arms preserve V4 physics/events; F0 delegates to untouched D2. |
| [scenarios.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/scenarios.py) | 后续高层分配方案、车辆轨迹预测和候选评分；不是 core MADDPG；具体入口见符号。符号：`content_key, validate_confirmation, prepare, main`<br>源码说明：Prepare and freeze a fresh paired evaluation population, without simulation. |
| [scoring.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/scoring.py) | 候选/路径评分。符号：`route_key, allocation_key, connected_times, PlanScore, Scorer`<br>源码说明：Cached same-state plan scoring; query limits are independent of wall time. |
| [solver.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/bser_final_v1/solver.py) | 该实验的候选组合求解。符号：`guard, rank, choose`<br>源码说明：D2-referenced guards and deterministic bounded candidate selection. |

### chapter3_bser/experiments/d2_performance

D2 语义等价性能工具：轻量安全视图、日志、计算设备与封存增量。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_performance/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Performance-only D2 runtime utilities; no algorithm implementation. |
| [compute.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_performance/compute.py) | 计算设备与 CPU 探索随机流保持。符号：`CpuExplorationNoise, build_model`<br>源码说明：Explicit learner placement while preserving the original CPU exploration RNG. |
| [logging.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_performance/logging.py) | 追加 JSONL 日志和最终物化。符号：`read_records, JsonlJournal`<br>源码说明：Single-owner append-only JSONL journals with atomic final JSON export. |
| [options.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_performance/options.py) | 该方法参数默认值、解析和合法性约束。符号：`performance_options, require_device`<br>源码说明：Explicit compute options, independent of algorithm and experiment budgets. |
| [planning.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_performance/planning.py) | 不建立规划图的实时安全投影缓存。符号：`locked, SafetyOccupancy, SafetyState, D2PlanningViews`<br>源码说明：D2's live safety projection, deliberately without a planning graph. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_performance/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile`<br>源码说明：Exact performance evolution; all historical source seals remain frozen. |
| [round2_provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_performance/round2_provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile`<br>源码说明：Exact second-round performance evolution; historical seals stay frozen. |

### chapter3_bser/experiments/d2_suite_v1

Linux 四臂计划、锁、训练/评估作业、来源验收、回执和配对汇总。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Frozen-scene, four-arm D2 experiment orchestration; execution is opt-in. |
| [cli.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/cli.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`main`<br>源码说明：Unified four-arm preparation, manual launch, and paired summary. |
| [evaluate.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/evaluate.py) | D2、B2、B3、HGR 的同场景完整任务评估；校验策略不被更新。符号：`evaluate`<br>源码说明：One full-task evaluator and read-only search diagnostics for all four arms. |
| [evaluation_cli.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/evaluation_cli.py) | 完整任务/模型评估与相关指标、跟踪或入口。符号：`evaluate, main`<br>源码说明：Separate manual D2 reference and post-training evaluation entrypoints. |
| [evaluation_provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/evaluation_provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile`<br>源码说明：Exact evaluation-entry review; the previous Linux launch seal stays frozen. |
| [linux.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/linux.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`dependencies, check, run_lock, Tee, launch, main`<br>源码说明：Manual Linux launches with explicit compute choices frozen during prepare. |
| [linux_provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/linux_provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile`<br>源码说明：Exact additive Linux launcher review; all historical suite seals stay frozen. |
| [pipeline.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/pipeline.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`require_complete, run, main`<br>源码说明：One manual launch: all training, then all evaluation, then paired summary. |
| [pipeline_provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/pipeline_provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile`<br>源码说明：Exact full-pipeline review; all earlier source seals stay frozen. |
| [plan.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/plan.py) | 构建不可变四臂实验计划和训练/评估输入。符号：`digest, sha, read, write, settings, scenes, validate_pair, prepare, load_plan`<br>源码说明：Prepare immutable experiment inputs without constructing a trainer. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile`<br>源码说明：Exact successor for orchestration only; the frozen D2 seal is unchanged. |
| [run.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/run.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`receipt, save_receipt, execute`<br>源码说明：Manual execution gate, verified completed-job reuse, and immutable receipts. |
| [summarize.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_suite_v1/summarize.py) | 按任务和种子汇总、成对比较；缺失 HGR 记 pending 而非失败。符号：`metrics, paired, summarize`<br>源码说明：Strict paired descriptive comparisons; partial jobs never become failures. |

### chapter3_bser/experiments/d2_v1

将已封存 D2 的控制器和 V4 guidance 安全组件装配到学习/零动作运行器。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_v1/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Frozen D2 planning foundation for independent B2/B3 and HGR policies. |
| [assembly.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_v1/assembly.py) | 复用 D2 搜索优化、V4 失败恢复与 SafetyBridge 组件。符号：`prepare, build_controller, build_bridge, diagnostics`<br>源码说明：Reuse the sealed D2 algorithm without its zero-action experiment runtime. |
| [contract.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_v1/contract.py) | 将已封存 D2 的控制器和 V4 guidance 安全组件装配到学习/零动作运行器；具体入口见符号。符号：`planner_protocol, enabled`<br>源码说明：Explicit planner selection; omitted fields retain historical behavior. |
| [controller.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_v1/controller.py) | 事件驱动控制流程和分配提交。符号：`D2SearchController`<br>源码说明：D2's original search controller with snapshot-safe attribute delegation. |
| [preflight.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_v1/preflight.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`main`<br>源码说明：Read-only D2 source/config preflight. Never constructs a trainer or simulator. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/d2_v1/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile, framework_sources, verify_if_enabled, baseline_sources`<br>源码说明：Exact additive D2 integration review; historical seals remain authoritative. |

### chapter3_bser/experiments/hgr

HGR 训练与 snapshot/restore；零残差 D2 仍借用其中的通用运行器。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Production full-task collectors and complete-cycle HGR training. |
| [build_phase2_source.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/build_phase2_source.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`PolicySourceUnavailable, _sha, _path, load_config, _resolve_config, validate_policy_pair, load_policy_pair, _scenario_audit, build_cases, build_phase2_source, main`<br>源码说明：Prepare a Phase2 frozen source without training or simulator rollouts. |
| [cli.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/cli.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`main`<br>源码说明：Explicit train/resume/fixed-policy evaluation entry points. |
| [evaluation.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/evaluation.py) | 完整任务/模型评估与相关指标、跟踪或入口。符号：`file_sha256, resolved_config, evaluation_manifest, evaluation_identity, episode_row, evaluation_summary, evaluate`<br>源码说明：Fixed-policy HGR-family evaluation, with strict shared mission accounting. |
| [phase1_acceptance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/phase1_acceptance.py) | HGR 训练与 snapshot/restore；零残差 D2 仍借用其中的通用运行器；具体入口见符号。符号：`plan, _spawn_pair, _query, _same_trace, run, main`<br>源码说明：NEW manual-only bounded runtime acceptance with fresh actors, no training. |
| [phase2_gradient_efficiency.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/phase2_gradient_efficiency.py) | HGR 训练与 snapshot/restore；零残差 D2 仍借用其中的通用运行器；具体入口见符号。符号：`_positive_integer, _digest, load_config, resolve_config, FrozenSnapshotSource, _pack, _unpack, _policy_from_payload, save_snapshot_source, load_snapshot_source, VectorMoments, estimator_vectors, RuntimeBackend, _validate_trajectory, Evaluation, _validate_source, run_phase2_gradient_efficiency, main`<br>源码说明：Frozen-policy gradient diagnostics. No training, optimizer, or policy update. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`fresh_source_identity, source_identity, validate_source_identity, require_source_match, checkout_identity`<br>源码说明：Production source identity shared by policy checkpoints and simulator state. |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`seed_innovations, rng_state, restore_rng, DecisionSnapshot, _GraphPickler, _GraphUnpickler, MissionRuntime, collect_trajectory, _collect_trajectory, continue_branch, _continue_branch`<br>源码说明：Real simulator continuation with an explicit, versioned object-graph snapshot. |
| [train.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/hgr/train.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`digest, write_json, immutable_config, phase1_predictor_from_state, validated_output, load_config, validate_config, Trainer, load_checkpoint, run_training`<br>源码说明：Complete-cycle production training; no replay, PPO, critic or reward shaping. |

### chapter3_bser/experiments/long_horizon_maddpg_v1

上一轮新增的未采纳实验提案；本次撤回其作为推荐修复的结论；未接入旧 B2/B3。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/long_horizon_maddpg_v1/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in long-horizon MADDPG repair; historical B2/B3 remain frozen. |
| [cli.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/long_horizon_maddpg_v1/cli.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`read, write, population, manifest, check_disjoint, new_output, state_digest, load_checkpoint, train, evaluate, main`<br>源码说明：Manual train/evaluate entrypoints; defaults only preview, never resume. |
| [config.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/long_horizon_maddpg_v1/config.py) | 该方法参数默认值、解析和合法性约束。符号：`discount_from_retention, validate, discount_report`<br>源码说明：Independent identity and an explicit physical-horizon discount contract. |
| [model.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/long_horizon_maddpg_v1/model.py) | 上一轮新增的未采纳实验提案；本次撤回其作为推荐修复的结论；未接入旧 B2/B3；具体入口见符号。符号：`FieldScale, ScaledMLP, LongHorizonMADDPG, build_model`<br>源码说明：Chapter-local learner repairs with unchanged external tensor dimensions. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/long_horizon_maddpg_v1/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`sha, digest, preflight, require_ready`<br>源码说明：New opt-in gate: preserve every parent byte and reject unreviewed additions. |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/long_horizon_maddpg_v1/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`seed_all, observations, MissionRuntime`<br>源码说明：D2 rollout adapter independent of the frozen B2/B3 training package. |

### chapter3_bser/experiments/map_stp_shadow_v1

MAP_STP 有界快照采集与离线候选路径评分；不安装 shadow 路径、不训练。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/map_stp_shadow_v1/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in offline map diagnostics; never imported by a production runtime. |
| [io.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/map_stp_shadow_v1/io.py) | 来源、快照序列化与独立输出目录。符号：`plain, read, sha, digest, write, new_directory, source_identity, locked, state_from_dict, save_snapshot, load_snapshot`<br>源码说明：JSON-only public snapshot transport and exact experiment source identity. |
| [routes.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/map_stp_shadow_v1/routes.py) | 候选路径生成与车辆轨迹预测。符号：`Route, route_pool, forecast`<br>源码说明：Bounded same-goal candidates on the existing immutable public graph. |
| [run.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/map_stp_shadow_v1/run.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`config, collect, score_snapshot, summarize, score, main`<br>源码说明：Bounded collection and separate offline scoring; no routes are installed. |
| [scoring.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/map_stp_shadow_v1/scoring.py) | 候选/路径评分。符号：`swept_events, first_values, marginal_utility, known_target_boxes, hypotheses, route_scores, team_scores, choose, compare_case`<br>源码说明：Matched-detector scores over common public-model trajectory hypotheses. |

### chapter3_bser/experiments/phase1b1_pilot

历史 Phase 1B.1 在线机制试验。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b1_pilot/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Small paired pilot for the corrected Phase 1B.1 online mechanism. |
| [run_pilot.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b1_pilot/run_pilot.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_seed_all, _write_json, _write_csv, _mean, _dither, MemoizedAllocator, _public_context, _serial_step, _config_for_method, _state_refresh_interval, _periodic_replan_due, _episode, _worker, _seed_worker, _summary, run, main`<br>源码说明：Run the preregistered 80-condition-episode BSER Phase 1B.1 pilot. |

### chapter3_bser/experiments/phase1b2_pilot

历史 Phase 1B.2 部分分配与路径跟踪试验。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b2_pilot/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Execution-consistent BSER Phase 1B.2 pilot. |
| [run_pilot.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b2_pilot/run_pilot.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_worker, _summary, _checkpoint_path, _valid_checkpoint, run, main`<br>源码说明：Run the 80-condition-episode execution-consistent BSER pilot. |

### chapter3_bser/experiments/phase1b3a_diagnosis

历史事件/候选失败诊断与基线复现。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b3a_diagnosis/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Phase 1B.3A bounded diagnostic experiment. |
| [run_diagnosis.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b3a_diagnosis/run_diagnosis.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`_read_csv, _as_bool, _optional_number, _behavior_equal, _reason_summary, _phase_summary, _duplicate_summary, _rejection_summary, _counts, _make_recommendations, _target_summary, _changed_files, _diagnostic_case_worker, run, run_specialized_tests, finalize_tests, main`<br>源码说明：Run the five preregistered Phase 1B.3A diagnosis-only cases. |

### chapter3_bser/experiments/phase1b_online

历史 E2 在线分配实验入口。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b_online/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：BSER-E2 online reallocation experiment. |
| [run_e2.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1b_online/run_e2.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_seed_all, _write_csv, _write_json, _mean, MemoizedAllocator, _episode_worker, _condition_config, _dither, _episode, _aggregate, run, main`<br>源码说明：Run BSER-E2 online reallocation and fixed event ablations without training. |

### chapter3_bser/experiments/phase1c_bser_rmaddpg

历史独立 BSER-RMADDPG 训练器和 checkpoint 评估。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Independent Phase 1C BSER-RMADDPG preflight runtime. |
| [evaluate_phase1c_checkpoints.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg/evaluate_phase1c_checkpoints.py) | 该方法 checkpoint 保存、恢复、迁移或兼容性核验。符号：`_json_safe, _write_json, _write_csv, _load_config, _validate_checkpoint_payload, _build_scenarios, _env_config, _make_env, _load_actor, _public_context, _oracle_context, _apply_residual_mode, _evaluate_episode, _mean_numeric, _aggregate_checkpoint, _checkpoint_info, _resolve_checkpoints, _plot_outputs, run_evaluation, main`<br>源码说明：Read-only deterministic execution diagnostics for Phase 1C checkpoints. |
| [metrics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg/metrics.py) | 历史独立 BSER-RMADDPG 训练器和 checkpoint 评估；具体入口见符号。符号：`_mean, EpisodeMetrics, summarize_preflight`<br>源码说明：Metric accumulation for the action-free-update Phase 1C preflight. |
| [run_phase1c.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg/run_phase1c.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_seed_all, _public_context, _load_config, _episode, _write_json, _write_csv, _write_jsonl, _prepare_output, run_preflight, main`<br>源码说明：Run the independent, no-update Phase 1C BSER-RMADDPG preflight. |
| [train_phase1c.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg/train_phase1c.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`_seed_all, _public_context, _load_config, _config_hash, _policy_snapshot, _collect_episode, _build_learner, _apply_transitions, _checkpoint_metadata, _save_checkpoint, _load_checkpoint, _write_json, _write_csv, _rolling, _plot_curves, run_training, main`<br>源码说明：Independent synchronous-rollout trainer for Phase 1C BSER-RMADDPG. |

### chapter3_bser/experiments/phase1c_bser_rmaddpg_v2

执行阶段奖励 adapter、环境 wrapper、阶段回放与历史 v2 训练器。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Independent Phase 1C-v2 execution-learning implementation. |
| [phase_aware_replay.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/phase_aware_replay.py) | 该版本经验存储、抽样、阶段/辅助标签适配。符号：`_resolve_device, PhaseAwareReplayBuffer`<br>源码说明：Independent phase-aware prioritized replay for Phase 1C-v2. |
| [reward_adapter.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/reward_adapter.py) | 发现奖励恢复、发现后搜索者奖励清零、接触/保持/成功附加奖励和碰撞覆盖。符号：`Phase1CExecutionRewardConfig, RewardAdjustmentResult, Phase1CExecutionRewardAdapter`<br>源码说明：Phase 1C-v2 execution-stage reward adapter. |
| [train_phase1c_v2.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/train_phase1c_v2.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`_seed_all, _json_safe, _write_json, _write_csv, _reward_config, _load_config, _execution_runtime_revision, _config_hash, _public_context, _policy_snapshot, _resolve_training_device, _numpy_copy, _make_base_env, _collect_episode, _build_learner, _apply_transitions, _checkpoint_metadata, _save_checkpoint, _load_checkpoint, _rolling, _plot_curves, run_training, main`<br>源码说明：Independent synchronous-rollout trainer for Phase 1C-v2 execution learning. |
| [training_env.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/training_env.py) | 把真实任务前后状态和接触计数交给执行奖励 adapter，维护转移元数据。符号：`Phase1CV2TrainingEnv`<br>源码说明：Transparent Phase 1C-v2 training wrapper. |

### chapter3_bser/experiments/phase1c_common

发现/交接/接触/保持/成功的转移元数据与执行诊断。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_common/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Shared, non-core helpers for Chapter 3 Phase 1C experiments. |
| [execution_diagnostics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_common/execution_diagnostics.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`_runtime, _number, _integer, _vector, _mean, _sum_reward, _event_names, ExecutionEpisodeDiagnostics`<br>源码说明：Episode-level execution diagnostics for Chapter 3 Phase 1C. |
| [transition_schema.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_common/transition_schema.py) | 该目录的数据结构、阶段或转移契约。符号：`TransitionPhase, classify_transition_phase, Phase1CTransitionMetadata`<br>源码说明：Replay-only transition metadata for Chapter 3 Phase 1C. |

### chapter3_bser/experiments/phase1c_prrac

独立 PRRAC 训练、评估、运行器工厂及执行/搜索诊断；公共 wrapper 被 B2/B3 复用。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Independent Phase 1C PRRAC experiment namespace. |
| [beds.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/beds.py) | 独立 PRRAC 训练、评估、运行器工厂及执行/搜索诊断；公共 wrapper 被 B2/B3 复用；具体入口见符号。符号：`write_diagnostics, resolve_beds, beds_requested, beds_enabled, validate_beds, BEDSEpisodeAdapter`<br>源码说明：Evaluation-only BEDS adapter. No truth target, policy or training changes. |
| [checkpoint_transfer.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/checkpoint_transfer.py) | 该方法 checkpoint 保存、恢复、迁移或兼容性核验。符号：`checkpoint_path, file_sha256, source_training_config, import_actors`<br>源码说明：Explicit actor import and source identity; never migrates replay or optimizers. |
| [collision_terminal_cli.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/collision_terminal_cli.py) | 独立 PRRAC 训练、评估、运行器工厂及执行/搜索诊断；公共 wrapper 被 B2/B3 复用；具体入口见符号。符号：`main`<br>源码说明：Manual collision_terminal_v1 entry points; parsing/preparation never trains. |
| [diagnostics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/diagnostics.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`_mean, _quantile, PRRACDiagnostics`<br>源码说明：PRRAC routing, expert, gate, and stage-critic diagnostics. |
| [evaluate_prrac_checkpoints.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py) | 该方法 checkpoint 保存、恢复、迁移或兼容性核验。符号：`_json_safe, _canonical_json, _hash, _atomic_text, _write_json, _csv_value, _write_csv, _read_csv, _dedupe_activation_rows, _contains_tensor, _load_config, _validate_checkpoint_payload, load_prrac_checkpoint, _checkpoint_info, _resolve_checkpoints, _build_evaluation_manifest, _load_scenario_id_file, _make_env, _public_context, _build_episode_controller, _oracle_context, _install_next_guidance, _apply_residual_mode, _policy_outputs, _controller_mode, _controller_actions, _event_names, _trace_step, _evaluate_episode_job, _combo_key, _validate_resume_search_diagnostics, _failure_funnel, _paired_rows, _plot, _plot_execution_variants, _plot_search_recovery, _execution_variant_summary, _evaluation_summary, _write_outputs, run_evaluation, main`<br>源码说明：Read-only deterministic evaluation for independent PRRAC checkpoints. |
| [evaluation_metrics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/evaluation_metrics.py) | 完整任务/模型评估与相关指标、跟踪或入口。符号：`EvaluationTransitionDiagnostics, safe_rate, wilson_interval, router_class_metrics, failure_stage, _numbers, _mean, _median, _sum_confusions, aggregate_checkpoint, mcnemar_exact_p_value, paired_checkpoint_comparison, recommend_checkpoint`<br>源码说明：Pure aggregation helpers for deterministic PRRAC checkpoint evaluation. |
| [evaluation_provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/evaluation_provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`row_controller_mode, require_single_controller, validate_controller_artifacts, _field, _normal, derive_unique_provenance, _require_equal, validate_evaluation_provenance, validate_summary_provenance, validate_resume_config`<br>源码说明：Strict row-derived provenance and cross-artifact consistency checks. |
| [evaluation_trace.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/evaluation_trace.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`_plain, vector3, failure_trace_row, FailureTraceRecorder`<br>源码说明：Privileged failure tracing isolated from PRRAC policy inputs. |
| [paired_evaluation.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/paired_evaluation.py) | 完整任务/模型评估与相关指标、跟踪或入口。符号：`sha256, read_json, write_new_json, parse_optional_bool, read_episode_csv, outcome_summary, reject_fixture, prepare_pair, validate_checkpoint_for_pair, analyze_pair, main`<br>源码说明：Manual paired evaluation and strict, simulation-free CSV analysis. |
| [replay_adapter.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/replay_adapter.py) | 该版本经验存储、抽样、阶段/辅助标签适配。符号：`PRRACBatch, PRRACReplayAdapter`<br>源码说明：Composition adapter adding PRRAC stage pairs to frozen v2 replay. |
| [residual_branch_diagnostic.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/residual_branch_diagnostic.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`scale_commands, ResidualBranchDiagnostic`<br>源码说明：Explicit diagnostic command intervention; no alternative physics or tracker. |
| [run_residual_branch_sensitivity.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/run_residual_branch_sensitivity.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`alpha_grid, validate_inputs, make_jobs, branch_worker, write_csv, run, main`<br>源码说明：Validated historical-anchor replay; explicit CLI only, no training path. |
| [run_searcher_residual_trace.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/run_searcher_residual_trace.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`sha, read_json, indexed, behavior, validate_sources, trace_worker, collect_jobs, write_results, run, main`<br>源码说明：Explicitly invoked fixed-case trace diagnostic; never generates new scenarios. |
| [runtime_factory.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/runtime_factory.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`PRRACRuntimeContract, runtime_contract, build_prrac_online_controller`<br>源码说明：Single construction boundary for PRRAC online controllers. |
| [search_value_decision.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_decision.py) | 独立 PRRAC 训练、评估、运行器工厂及执行/搜索诊断；公共 wrapper 被 B2/B3 复用；具体入口见符号。符号：`resolve_search_value_decision_config, SearchValueDecision, SearchValueDecisionController`<br>源码说明：Hysteretic decision sidecar for Search Value guided BSER re-evaluation. |
| [search_value_guidance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_guidance.py) | 独立 PRRAC 训练、评估、运行器工厂及执行/搜索诊断；公共 wrapper 被 B2/B3 复用；具体入口见符号。符号：`resolve_search_value_guidance, CandidateScore, SearchValueGuidedCandidateScore, SearchValueGuidedBSERAllocator, aggregate_search_value_guidance`<br>源码说明：S2B inference-only, bounded auxiliary ranking of existing BSER candidates. |
| [searcher_residual_trace.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/searcher_residual_trace.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`array, vector_fields, guidance_signature, SearcherResidualTrace`<br>源码说明：Read-only trace sink; consumes the authoritative SEARCH diagnostic rows. |
| [standby_diagnostics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/standby_diagnostics.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`vector, StandbyDiagnostics`<br>源码说明：Read-only transition diagnostics; never supplies values to control. |
| [task_metrics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/task_metrics.py) | 独立 PRRAC 训练、评估、运行器工厂及执行/搜索诊断；公共 wrapper 被 B2/B3 复用；具体入口见符号。符号：`terminal_planning_snapshot, boolean, strict_outcome, validated_rows, aggregate_task_outcomes`<br>源码说明：Complete mission accounting, separate from partial/aborted evaluation work. |
| [train_phase1c_prrac.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/train_phase1c_prrac.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`_json_safe, _write_json, _write_csv, _config_hash, _config_hash_without_search_value, _config_hash_without_search_value_decision, _config_hash_without_search_modules, _load_config, _parameter_counts, _build_episode_controller, _checkpoint_metadata, _save_checkpoint, _load_checkpoint, _verify_checkpoint_roundtrip, _build_learner, _clone_parameters, _changed, _numpy_copy, _contains_tensor, _combine_episode_diagnostics, _sum_confusion_matrices, _confusion_accuracy, _router_summary, _runtime_metadata, _apply_transitions, _collect_episode, _rolling, _save_line_plot, _plot_outputs, _group_parameters, _current_group_parameters, _dry_run_requirements_met, run_training, main`<br>源码说明：Independent PRRAC trainer for Chapter 3 Phase 1C. |
| [training_env.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/training_env.py) | 公共元数据 wrapper 和最终 team reward 记账；并非调用 PRRAC 网络。符号：`PRRACTrainingEnv`<br>源码说明：Transparent PRRAC metadata wrapper over Phase1CV2TrainingEnv. |
| [transition_protocol.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/transition_protocol.py) | 该目录的数据结构、阶段或转移契约。符号：`PRRACTransitionMetadata`<br>源码说明：Replay-only PRRAC phase-before/phase-after transition protocol. |

### chapter3_bser/experiments/phase1c_prrac/execution_continuity

执行导航连续性变体 B0/B1/B2/B3；这里的编号不是 D2 的学习基线 B2/B3。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in PRRAC execution-continuity ablation runtime. |
| [action_adapter.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/action_adapter.py) | 执行导航连续性变体 B0/B1/B2/B3；这里的编号不是 D2 的学习基线 B2/B3；具体入口见符号。符号：`_norm, ExecutionContinuityActionAdapter`<br>源码说明：Shared residual-action adapter for B3 explicit SAFE_HOLD. |
| [config.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/config.py) | 该方法参数默认值、解析和合法性约束。符号：`parse_execution_variant, overlay_enabled, overlay_config`<br>源码说明：Frozen variant registry and validation for execution continuity. |
| [controller.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/controller.py) | 事件驱动控制流程和分配提交。符号：`ExecutionContinuityController`<br>源码说明：Opt-in v3 controller overlay around the frozen Phase 1B controller. |
| [diagnostics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/diagnostics.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`_rate, _mean, ExecutionContinuityDiagnostics, aggregate_execution_variant, paired_execution_variant_comparisons`<br>源码说明：Episode and aggregate diagnostics for execution-continuity ablations. |
| [event_detector.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/event_detector.py) | 根据公共状态变化检测事件。符号：`_relative_change, ExecutionContinuityEventDetector`<br>源码说明：Endpoint-aware event semantics for the opt-in v3 runtime overlay. |
| [planner.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/planner.py) | 执行导航连续性变体 B0/B1/B2/B3；这里的编号不是 D2 的学习基线 B2/B3；具体入口见符号。符号：`_vector3, _path, _plan_from_query, assign_reachable_public_proxy, _safe_hold, plan_atomic_execution_continuity`<br>源码说明：Deterministic, public-information-only execution-continuity planning. |
| [types.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/execution_continuity/types.py) | 该目录的数据结构、阶段或转移契约。符号：`ExecutionVariant, NavigationMode, ExecutionNavigationPlanV3, ExecutionContinuityDetectionV3, ResidualSuppressionDiagnostics`<br>源码说明：Immutable contracts for the opt-in PRRAC execution-continuity overlay. |

### chapter3_bser/experiments/phase1c_prrac/search_collision_recovery

搜索阶段碰撞恢复、局部连接器和 guidance overlay。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Public evaluation-only Searcher collision recovery API. |
| [aggregation.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/aggregation.py) | 搜索阶段碰撞恢复、局部连接器和 guidance overlay；具体入口见符号。符号：`_numbers, _mean, _median, _rate, _percentile, aggregate_search_collision_recovery, _paired_groups, paired_search_collision_recovery_comparisons, paired_search_collision_recovery_baseline_strata, search_collision_recovery_failure_funnel, validate_s2a_baseline_regression`<br>源码说明：S2-A aggregation, exact pairing, baseline strata, funnel, and preflight. |
| [config.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/config.py) | 该方法参数默认值、解析和合法性约束。符号：`parse_search_recovery_variant, search_collision_recovery_config, _search_collision_recovery_config_v2, search_collision_recovery_config_hash`<br>源码说明：Configuration and hashing for the S2-A evaluation overlay. |
| [controller.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/controller.py) | 事件驱动控制流程和分配提交。符号：`_AgentRecovery, SearchCollisionRecoveryController, build_search_recovery_controller`<br>源码说明：Evaluation-only state machine for pre-found Searcher collision recovery. |
| [controller_v2.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/controller_v2.py) | 第二版事件/恢复控制流程。符号：`_AgentRecoveryV2, SearchCollisionRecoveryControllerV2`<br>源码说明：S2-A.1 forced-refresh and public local-connector recovery controller. |
| [detector.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/detector.py) | 搜索阶段碰撞恢复、局部连接器和 guidance overlay；具体入口见符号。符号：`CollisionEdgeDetector`<br>源码说明：Collision-edge detector with one collision-free SEARCH transition re-arm. |
| [diagnostics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/diagnostics.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`baseline_recovery_summary`<br>源码说明：Plain scalar diagnostics for recovery variants, including strict C0 zeros. |
| [guidance_overlay.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/guidance_overlay.py) | 搜索阶段碰撞恢复、局部连接器和 guidance overlay；具体入口见符号。符号：`apply_search_recovery_guidance`<br>源码说明：Minimal immutable guidance substitution for active Searcher recovery plans. |
| [planner.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/planner.py) | 搜索阶段碰撞恢复、局部连接器和 guidance overlay；具体入口见符号。符号：`_vector3, _plan, plan_route_refresh, _hop_counts, select_egress_route`<br>源码说明：Deterministic recovery planning over public immutable planning views only. |
| [planner_v2.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/planner_v2.py) | 搜索阶段碰撞恢复、局部连接器和 guidance overlay；具体入口见符号。符号：`canonical_path_hash, _bounds, _in_bounds, public_cell_index, audit_public_segment, _distance_to_mask, _start_connector_count, _audit_base, plan_forced_route_refresh, _candidate, _ranking, plan_local_connector`<br>源码说明：S2-A.1 recovery planning using only immutable public planning views. |
| [types.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/types.py) | 该目录的数据结构、阶段或转移契约。符号：`SearchRecoveryVariant, SearchRecoveryMode, RecoveryNavigationPlan, RecoveryStepSnapshot`<br>源码说明：Immutable contracts for the evaluation-only search collision recovery overlay. |
| [types_v2.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/types_v2.py) | 该目录的数据结构、阶段或转移契约。符号：`SearchRecoveryVariantV2, RecoveryModeV2, LastCollisionFreeState, PublicSegmentAudit, LocalConnectorCandidate, LocalConnectorPlan, RecoveryPlanningAudit, ActivationAuditStep`<br>源码说明：Immutable S2-A.1 public recovery and activation contracts. |

### chapter3_bser/experiments/phase1c_prrac/search_continuity

搜索动作连续性变体与诊断汇总。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_continuity/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Shared SEARCH continuity diagnostics and evaluation aggregation. |
| [action_adapter.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_continuity/action_adapter.py) | 搜索动作连续性变体与诊断汇总；具体入口见符号。符号：`apply_residual_mode`<br>源码说明：Copy-only deterministic evaluation action ablations. |
| [aggregation.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_continuity/aggregation.py) | 搜索动作连续性变体与诊断汇总；具体入口见符号。符号：`_values, _mean, _median, _percentile, aggregate_search_continuity, paired_searcher_residual_comparisons, search_failure_funnel`<br>源码说明：Deterministic aggregation for SEARCH continuity episode records. |
| [diagnostics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_continuity/diagnostics.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`_array, _scalar, SearchContinuityDiagnostics`<br>源码说明：One authoritative, information-bounded SEARCH diagnostic implementation. |
| [types.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_continuity/types.py) | 该目录的数据结构、阶段或转移契约。符号：`search_continuity_config, search_continuity_config_hash, nullable_rate`<br>源码说明：Versioned contracts for read-only SEARCH continuity diagnostics. |

### chapter3_bser/experiments/phase1c_prrac/search_value_audit

搜索价值预测、分支敏感性、状态指纹及离线分析。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：D1/D2 diagnostic entry points; never imported by default evaluation/training. |
| [analysis_bundle.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/analysis_bundle.py) | 离线结果分析、汇总或绘图。符号：`bundle, main`<br>源码说明：Package only an explicit small-artifact allowlist from completed D1/D2 runs. |
| [branch_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/branch_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`installed_signature, root_decision, needs_root_probe, replay_equal, locate_work, branch_work, branch_outcome, validate_branches, run, main`<br>源码说明：D2: common-ON-prefix, one-decision A/B/C interventions, then all OFF. |
| [features.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/features.py) | 搜索价值预测、分支敏感性、状态指纹及离线分析；具体入口见符号。符号：`feature_catalog, exact_feature_id, tolerance_groups, candidate_geometry, assignment_geometry, guidance_geometry, compare_feature, pool_audit`<br>源码说明：Public-only feature comparisons and candidate representation diagnostics. |
| [metrics.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/metrics.py) | 搜索价值预测、分支敏感性、状态指纹及离线分析；具体入口见符号。符号：`window_label, binary_metrics, calibration, scenario_key, prediction_summary, paired_outcomes`<br>源码说明：Offline labels and scenario-cluster statistics. No runtime state dependencies. |
| [prediction_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/prediction_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`training_scenarios, freeze_scenarios, run, main`<br>源码说明：D1: frozen-head prediction and same-state candidate representation audit. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`json_value, canonical, digest, file_hash, weights_hash, actor_hash, read_json, atomic_json, atomic_csv, git_identity, training_label_baseline, load_checkpoint, scenarios_from_manifest, scenario_identity, overlap_report, experiment_identity, Progress, fresh_output`<br>源码说明：Read-only checkpoint, scenario identity, and atomic audit output utilities. |
| [runner.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/runner.py) | 搜索价值预测、分支敏感性、状态指纹及离线分析；具体入口见符号。符号：`arguments, key, historical_inputs, make_job, execute_work, run_work, task_signature, no_op_check, record_exception`<br>源码说明：Spawn-safe orchestration of the existing evaluator, not a second simulator. |
| [runtime_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/runtime_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`BoundaryProbeComplete, CUnavailable, InvalidRoot, solution_geometry, alternative_c, AuditAllocator, RuntimeAudit`<br>源码说明：Optional observer and single-decision intervention for the existing evaluator. |
| [state_fingerprint.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/phase1c_prrac/search_value_audit/state_fingerprint.py) | 搜索价值预测、分支敏感性、状态指纹及离线分析；具体入口见符号。符号：`rng_state, normalized, component_fingerprint, runtime_fingerprint`<br>源码说明：Strict read-only state inventory for deterministic replay, not restoration. |

### chapter3_bser/experiments/safe_search_v1

公共几何、失败恢复和路径 guidance 安全层；D2 使用 V4 组件。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in, training-free safe-search experiments; frozen methods stay separate. |
| [allocators.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/allocators.py) | 高层分配器变体。符号：`SearchPoolMixin, SafePriorAllocator, SafeJointAllocator`<br>源码说明：Filter SEARCH routes while retaining original standby queries and solvers. |
| [analyze_development.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/analyze_development.py) | 离线结果分析、汇总或绘图。符号：`validate_rows, collision_role, outcome, quantile, paired_bootstrap, trace_summary, aggregate, choose_variant, analyze_rows, markdown, main`<br>源码说明：Read-only, paired analysis of the frozen 20-scenario development stage. |
| [failure_policy.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/failure_policy.py) | 搜索失败的重试/冷却策略。符号：`SearchFailurePolicy`<br>源码说明：Failed attempts do not consume successful-replan cooldowns. |
| [planning_refresh.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/planning_refresh.py) | 规划快照刷新与端点状态判断。符号：`start_status, topology_key, PlanningViews`<br>源码说明：Public snapshot consistency and bounded continuous-start preflight. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`digest, file_sha256, read_json, _validate_files, inventory, _historical_profiles, _prior_changes, framework_sources, verify_sources, build_reviewed_manifest`<br>源码说明：Exact opt-in SafeSearch inventory; never relaxes the historical entry gate. |
| [public_geometry.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/public_geometry.py) | 仅基于公开占据网格的几何路径检查。符号：`_locked, _vector, PublicGeometry, _component_labels, _graph_hash, filter_planning_state`<br>源码说明：Conservative geometry over public occupancy, without a second path planner. |
| [run_ac_development.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/run_ac_development.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`validate_ac_plan, read_reference, child_command, validate_terminal, validate_trace, validate_child_ac, collect_completed_ac, run_ac_development, parser, main`<br>源码说明：Explicit V5=A+C development extension; never overwrites the historical D2. |
| [run_b1_development.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/run_b1_development.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`validate_plan, reference_metadata, validate_references, child_command, arm_directory, validate_terminal, validate_child, collect_completed_b1, run_b1_development, parser, main`<br>源码说明：B1 transfer of frozen V0/C/A+C on the same 20 development scenarios. |
| [run_development.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/run_development.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`validate_development_inputs, child_command, _run_child, validate_child, collect_completed, run_development, parser, main`<br>源码说明：Explicit, checkpoint-free D2 evaluation of the frozen 20-scene split. |
| [run_paired.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/run_paired.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`digest, file_hash, write_json, read_json, parse_variants, select_indices, validate_request, validate_manifest, output_directory, episode_metrics, execute_episode, summarize, load_dependencies, run_experiment, parser, main`<br>源码说明：Serial, checkpoint-free B0/B1 safe-search ablations on an existing manifest. |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`SearchController, SafetyMixin, SafePriorRuntime, SafeJointRuntime, make_runtime`<br>源码说明：Independent B0/B1 assembly with public-state repair and zero residuals. |
| [safe_guidance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/safe_guidance.py) | 检查/修复规划线段，失败时请求当前位置制动；不屏蔽 RL 残差也不保证制动无碰撞。符号：`_signature, SafetyBridge`<br>源码说明：Search-only path validity overlay through the existing guidance contract. |
| [search_coverage.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/search_coverage.py) | 搜索覆盖诊断。符号：`_flat, CoverageObserver`<br>源码说明：Read-only target-evidence footprint from the planner's actual update stamps. |
| [telemetry.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v1/telemetry.py) | 只读逐步机制记录。符号：`_path_geometry, capture_runtime`<br>源码说明：Read-only SafeSearch capture with explicit guidance/tracker route alignment. |

### chapter3_bser/experiments/safe_search_v2

另一套带运动预测与路径恢复的安全搜索试验；未接入旧 D2 B2/B3。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in pre-Found guidance and recovery; historical V0--V5 stay unchanged. |
| [guidance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/guidance.py) | 导航目标编译和路径覆盖。符号：`SearchTracker, FoundSafetyBridge`<br>源码说明：Pre-Found latched Hold and braking through the existing guidance fields. |
| [motion.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/motion.py) | 车辆运动/制动预测。符号：`vector, Vehicle, forecast_safe`<br>源码说明：Public-state forecasts of the unchanged zero-residual waypoint controller. |
| [options.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/options.py) | 该方法参数默认值、解析和合法性约束。符号：`Settings, parse_arms`<br>源码说明：Fixed development arms. No training or simulator configuration overrides. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`extend_profile, framework_sources, verify_sources`<br>源码说明：Exact additive evolution of the frozen SafeSearch v1 source inventory. |
| [recovery.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/recovery.py) | 路径或连接器恢复。符号：`RecoveryController`<br>源码说明：Bounded, per-searcher recovery using the existing B0/B1 candidate solvers. |
| [run_windows.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/run_windows.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`write_json, request, validate_terminal, summarize, run_episode, read_completed, run_jobs, parser, main`<br>源码说明：Manual Windows-compatible paired development evaluation (plan by default). |
| [runtime.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/experiments/safe_search_v2/runtime.py) | 该方法环境、控制器和策略的装配/步进或运行契约。符号：`FoundMixin, FoundPriorRuntime, FoundJointRuntime, make_runtime`<br>源码说明：Explicit descendants of B0 V5 / B1 V4; R0 returns the original runtime. |

### chapter3_bser/hysteresis

重规划增益门槛与冷却，防止分配频繁切换。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/hysteresis/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Replanning cooldown and hysteresis policy. |
| [cooldown.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/hysteresis/cooldown.py) | 冷却时钟。符号：`ReplanningCooldown`<br>源码说明：Step-based deterministic replanning cooldown. |
| [policy.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/hysteresis/policy.py) | 该目录策略与决策规则。符号：`ReplanningPolicy`<br>源码说明：Replanning hysteresis without parameter adaptation. |

### chapter3_bser/integration

将高层分配转换为导航 guidance，再接入原物理先验；不直接替代动力学。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/integration/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Opt-in Phase 1C interfaces between online BSER and residual RMADDPG. |
| [control_context.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/integration/control_context.py) | 不可变 guidance 数据契约。符号：`AgentAssignmentContextV1, ExecutorAssignmentContextV1, BSERControlContextV1`<br>源码说明：Immutable Phase 1C values passed from BSER to low-level guidance. |
| [guided_env.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/integration/guided_env.py) | 在原环境更新导航时重装外部 guidance，刷新观测并提供初始端点保护。符号：`GuidedEnv`<br>源码说明：Opt-in environment wrapper for Phase 1C high-level navigation guidance. |
| [rmaddpg_bridge.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/integration/rmaddpg_bridge.py) | 将分配和公共状态变成导航目标/保持请求，不产生 RL 动作。符号：`_vector3, _mission_phase, RMADDPGGuidanceBridge, compile_guidance, get_tracking_targets`<br>源码说明：Translate immutable BSER allocations into action-free navigation guidance. |

### chapter3_bser/models

可选搜索价值辅助网络。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Chapter-specific neural models for BSER experiments. |
| [search_value_head.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/search_value_head.py) | 搜索价值辅助网络及公开特征/配置。符号：`resolve_search_value_config, SearchValueHead, SearchStateFeatureExtractor, future_found_labels`<br>源码说明：Auxiliary future-discovery value model for Phase 1C search decisions. |

### chapter3_bser/models/hgr

HGR 随机策略、前后缀估计器及预测器；用户已取消的方法。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/hgr/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Independent, density-defined handoff policy gradient methods. |
| [estimator.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/hgr/estimator.py) | HGR score function 与边界修正估计器。符号：`reward_to_go, BoundaryPredictor, prefix_losses, gradient_vector, update_prefix, update_suffix`<br>源码说明：Score-matched estimators, including the strong direct boundary baseline. |
| [phase1.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/hgr/phase1.py) | HGR Phase 1 方法身份、行为一致性和无更新证明。符号：`canonical, phase1_options, runtime_contract, _plain, BehaviorIdentity, behavior_identity, identical_behavior, NoUpdateProof, named_seed, PairNoise, isolated_global_rng, audit_local_generators`<br>源码说明：Opt-in phase-one contracts, exact behavior identities and addressed noise. |
| [policy.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/hgr/policy.py) | 该目录策略与决策规则。符号：`weights_hash, TanhGaussianPolicy, HandoffPolicy`<br>源码说明：28D local stochastic policies; all gates live inside the Gaussian mean. |
| [stable_predictor.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/hgr/stable_predictor.py) | HGR 稳定预测回归。符号：`StablePredictor`<br>源码说明：Frozen 153D, float64 zero/ridge predictor for the opt-in first phase. |

### chapter3_bser/models/prrac

独立 PRRAC 网络与学习器；不是本轮 Linux B2/B3 使用的 core MLP。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Phase-Routed Residual Actor-Critic (PRRAC) components. |
| [phase_routed_actor.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/phase_routed_actor.py) | PRRAC 编码、路由、专家与门控组合。符号：`PRRACActorOutput, PhaseRoutedResidualActor`<br>源码说明：Phase-routed residual actor; waypoint prior remains environment-owned. |
| [phase_router.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/phase_router.py) | PRRAC 阶段权重路由。符号：`PhaseRouter`<br>源码说明：Deterministic soft phase router. |
| [phase_twin_critic.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/phase_twin_critic.py) | PRRAC 124D 双 critic 与阶段输出头。符号：`gather_stage_values, PhaseCritic, PhaseTwinCritic`<br>源码说明：Independent twin critics with three phase heads and a strict 124D input. |
| [prrac_agent.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/prrac_agent.py) | PRRAC agent 的网络、优化器、探索与状态。符号：`PRRACAgent`<br>源码说明：Independent PRRAC agent state; no mutation of core DDPGAgent fields. |
| [prrac_maddpg.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/prrac_maddpg.py) | PRRAC 阶段 TD 与 actor 损失实现。符号：`PRRACMADDPG`<br>源码说明：Independent four-agent PRRAC learner with phase-aware twin-Q updates. |
| [residual_experts.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/residual_experts.py) | PRRAC 三阶段残差专家。符号：`ResidualExpert, ResidualExperts`<br>源码说明：Independent search, intercept, and hold residual expert heads. |
| [residual_trust_gate.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/residual_trust_gate.py) | 按残差与导航方向对齐程度调节残差门控。符号：`_inverse_softplus, ResidualTrustGate`<br>源码说明：Public-observation residual trust gate with monotone alignment response. |
| [stage_mapping.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/models/prrac/stage_mapping.py) | 转移阶段到 Search/Intercept/Hold 的固定映射。符号：`PRRACStage, transition_phase_to_prrac_stage`<br>源码说明：Fixed mapping from Phase 1C replay phases to PRRAC control stages. |

### chapter3_bser/online

在线 BSER 分配、事件控制、执行目标管理、状态缓存与航点更新。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Online, high-level BSER allocation package. |
| [allocator.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/allocator.py) | 高层候选/任务分配与部分重分配。符号：`_path, BSEROnlineAllocator`<br>源码说明：Online wrapper around the frozen Phase 1A.1 BSER components. |
| [config.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/config.py) | 该方法参数默认值、解析和合法性约束。符号：`execution_runtime_config, load_phase1b_config, load_phase1b1_config, load_phase1b2_config`<br>源码说明：Validated Phase 1B configuration loading. |
| [controller.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/controller.py) | 事件驱动控制流程和分配提交。符号：`OnlineBSERController`<br>源码说明：Online BSER event monitor and high-level allocation controller. |
| [early_discovery.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/early_discovery.py) | 高层早发现候选分配变体。符号：`resolve_early_discovery, estimated_arrival_time, rank_candidates, EarlyDiscoveryAllocator`<br>源码说明：Optional time-discount ranking over unchanged conditional BSER gains. |
| [execution_manager.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/execution_manager.py) | 执行路径和待机/目标分配管理。符号：`_path, ExecutionManager`<br>源码说明：Belief-only executor assignment for standby and target-found phases. |
| [executor_standby.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/executor_standby.py) | 执行者待机候选/分配。符号：`resolve_executor_standby, normalized_weights, compute_standby_target, standby_controller, apply_standby_action`<br>源码说明：Geometric median and legacy P helpers (not used by safe BEDS evaluation). |
| [mission_context.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/mission_context.py) | 根据公共状态构造阶段和已知目标上下文。符号：`OnlineMissionContext`<br>源码说明：Public, immutable mission context for corrected online BSER decisions. |
| [route_impact.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/route_impact.py) | 地图/信念变化对已有路径影响判断。符号：`RouteCheck, RouteImpactResult, RouteImpactEvaluator`<br>源码说明：Evaluate whether newly observed occupancy affects the active routes. |
| [safe_executor_standby.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/safe_executor_standby.py) | 公共几何约束下的执行者待机。符号：`StandbyNavigationParameters, min_searcher_distance, local_searcher_clearance, resolve_safe_route, SafeStandbyNavigation`<br>源码说明：BEDS-only public-map routing; shared planners and dynamics remain untouched. |
| [state_cache.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/state_cache.py) | 在线规划状态缓存。符号：`OnlineStateCache`<br>源码说明：Small deterministic cache of online observable planning states. |
| [types.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/types.py) | 该目录的数据结构、阶段或转移契约。符号：`SearchAssignment, ExecutorAssignment, OnlineAllocation, CurrentAssignment, InitialBSERAllocation, BSERActionAssignment, OnlineStepDiagnostics`<br>源码说明：Immutable high-level online allocation and action-assignment values. |
| [waypoint_manager.py](E:/gym/code/WORKSPACE/AUV-Thesis/chapter3_bser/online/waypoint_manager.py) | 航点分配提交、切换和更新时间。符号：`WaypointManager`<br>源码说明：High-level search waypoint replacement accounting. |

### core

共享执行基础；章节专属算法不放这里。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Stable production interfaces for the CRK thesis codebase. |

### core/algorithms

MADDPG 网络、agent、TD/策略梯度、探索噪声与目标网络更新。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/algorithms/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Shared multi-agent algorithms. |
| [agents.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/algorithms/agents.py) | 每个智能体的 actor、两个 critic、对应目标网络、Adam 和 OU 噪声。符号：`DDPGAgent` |
| [maddpg.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/algorithms/maddpg.py) | 四 agent 工厂、联合 124D 输入、双 Q TD 目标、actor/critic 更新、软更新和序列化。符号：`MADDPG` |
| [misc.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/algorithms/misc.py) | hard/soft update 和未在当前连续路径使用的 one-hot/Gumbel 辅助。符号：`soft_update, hard_update, onehot_from_logits, sample_gumbel, gumbel_softmax_sample, gumbel_softmax` |
| [networks.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/algorithms/networks.py) | LayerNorm + 两层 ReLU MLP；连续 actor 输出小初始化与 tanh。符号：`Identity, MLPNetwork` |
| [noise.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/algorithms/noise.py) | OU 状态递推、重置和设备迁移；没有自动衰减。符号：`OUNoise` |

### core/communication

一次性、固定一步延迟、可靠的目标交接。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/communication/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Communication implementations. |
| [basic_communication.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/communication/basic_communication.py) | 一次性、固定一步延迟、可靠的目标交接；具体入口见符号。符号：`_copy_target, FixedReliableHandoff`<br>源码说明：Deterministic Chapter-3 task handoff service. |

### core/config

环境/物理/奖励/算法默认配置和场景协议组合。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/config/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Canonical Chapter-3 configuration contracts. |
| [ch3_config.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/config/ch3_config.py) | 将候选、物理奖励配置和 M/S 场景设置组合为环境配置。符号：`build_ch3_config, build_mission_config, build_unknown_map_config, profile_diff`<br>源码说明：Unified configuration builder for known- and unknown-map Chapter-3 profiles. |
| [ch3_constants.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/config/ch3_constants.py) | 协议名、场景族和保留的公共常量<br>源码说明：Constants for the unified Chapter-3 environment and scenario profiles. |

### core/env

真实物理步进、目标运动、发现/捕获/终止、奖励、28D 观测和公共接口。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Canonical mission-environment API. |
| [base_env.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/base_env.py) | 历史兼容类入口；底层仍委托 uav_env，不是另一个仿真器。符号：`UAVEnv`<br>源码说明：Compatibility adapter for the merged Chapter-3 environment. |
| [mission_env.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/mission_env.py) | 公共 facade、配置到构造参数过滤、只读公共状态与终止结果访问。符号：`environment_kwargs_from_config, _array, _vectors, MissionCoreEnv`<br>源码说明：Stable public facade over the self-contained Chapter-3 core environment. |
| [observation_contract.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/observation_contract.py) | 28 个字段的索引、归一化与角色语义声明。符号：`ObservationField, validate_contract`<br>源码说明：Code-traced Chapter-3 28-dimensional local-observation contract. |
| [target_motion.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/target_motion.py) | 目标位置速度、反射、预测、扫掠最近距离和截获点求解。符号：`_nonnegative_int, _validate_finite_numbers, _vec3, _expanded_boxes, segment_aabb_first_hit, _boundary_first_hit, TargetState, advance_target_state, _plain_prediction_metadata, _prediction_context, _prediction_step, predict_target_state, simulate_target_trajectory, swept_relative_min_distance, solve_intercept_point`<br>源码说明：Deterministic target state, reflection, detection, and interception. |
| [task_protocol.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/task_protocol.py) | legacy/碰撞终止协议、线段 AABB 首碰检测、权威终止结果。符号：`protocol_identity, validate_task_config, require_protocol_output, closed_segment_aabb_first_hit, EpisodeOutcome, strict_terminal, episode_result`<br>源码说明：Versioned mission outcomes. These diagnostics are outside the 28D input. |
| [task_state.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/task_state.py) | 任务、目标、智能体、地图的只读 dataclass 视图。符号：`MissionTaskState, TargetStateView, AgentStateView, SearchExecutionState, MappingStateView`<br>源码说明：Immutable public state views for :class:`MissionCoreEnv`. |
| [uav_env.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/env/uav_env.py) | 唯一真实步进实现；含历史基类与移动目标/未知地图分支，负责物理、奖励和任务事件。符号：`_BaseUAVEnv, DummySpace, UAVEnv, _mission_init_signature`<br>源码说明：Unified Chapter-3 environment: base mechanics plus mission extensions. |

### core/mapping

占据与目标信念、规划图、A*/Dijkstra、几何和路径成本。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/mapping/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Mapping and path-planning implementations. |
| [map_module.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/mapping/map_module.py) | 基础信息素/概率任务地图及搜索/待机候选基础实现。符号：`PheromoneWaypointPlanner, ProbabilisticTaskMapPlanner` |
| [path_planner.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/mapping/path_planner.py) | 已知/未知障碍地图、传感更新、目标信念及栅格路径/旅行成本实现。符号：`ObstacleAwareTaskMapPlanner, _online_cells, _online_segment_cells, _sampled_online_segment_free, _online_physical_time, OnlineUnknownMapTaskPlanner`<br>源码说明：Deterministic obstacle-aware grid planning for Chapter 3. |
| [planning_graph.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/mapping/planning_graph.py) | 不可变规划图、边、连通分量和端点连接器构造。符号：`_readonly, PlanningEdgeView, PlanningConnectorView, EndpointConnectorSet, PlanningGraphView, _flat, _diagnostic_snapshot, _topology, _component_labels_from_topology, _build_role_adjacency, _build_endpoint, _graph_hash, build_planning_graph`<br>源码说明：Immutable adapter over the authoritative Chapter-3 path planner. |
| [planning_state.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/mapping/planning_state.py) | 将公共视图复制为规划快照并计算内容指纹。符号：`_readonly, GridGeometryView, TargetBeliefView, OccupancyBeliefView, PlanningAgentView, PlanningStateView, planning_state_sha256, _triple, extract_planning_state`<br>源码说明：Immutable, information-bounded planning snapshots for Chapter 3. |
| [travel_cost_service.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/mapping/travel_cost_service.py) | 只读规划快照上的 A*/Dijkstra 和路径成本查询。符号：`_locked, PathQueryResult, SingleSourceTravelResult, TravelCostService`<br>源码说明：Pure planning/physical path queries over an immutable planning graph. |

### core/provenance

源码指纹、版本与历史来源核验。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/provenance/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Source and runtime provenance utilities. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/provenance/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`_hash_relative_files, _collect_required_files, base_algorithm_source_files, base_algorithm_source_fingerprint, mission_algorithm_source_files, mission_algorithm_source_fingerprint, unknown_map_algorithm_source_files, unknown_map_algorithm_source_fingerprint, algorithm_source_files, algorithm_source_fingerprint, _iter_repository_source_files, repository_source_fingerprint, source_fingerprint, capture_base_provenance_snapshot, capture_mission_provenance_snapshot, capture_unknown_map_provenance_snapshot, capture_provenance_snapshot, _assert_snapshot, assert_base_algorithm_source_unchanged, assert_mission_algorithm_source_unchanged, assert_unknown_map_source_unchanged, assert_algorithm_source_unchanged, file_sha256, json_file_sha256, runtime_versions`<br>源码说明：Source and runtime provenance helpers for the merged Chapter-3 project. |

### core/registry

允许的方法身份和历史候选配置登记。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/registry/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Experiment registries. |
| [ch3_efficiency_v3_registry.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/registry/ch3_efficiency_v3_registry.py) | 允许的方法身份和历史候选配置登记；具体入口见符号。符号：`get_ch3_efficiency_v3_candidate, resolve_ch3_efficiency_v3_config, config_diff, validate_v3_candidate_registry`<br>源码说明：Registered single-seed candidates for Chapter-3 efficiency screening v3. |
| [experiment_registry.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/registry/experiment_registry.py) | 允许的方法身份和历史候选配置登记；具体入口见符号。符号：`assert_ch3_method, assert_registered_ch3_method`<br>源码说明：Registry for formal and independently orchestrated Chapter-3 methods. |

### core/replay

四智能体联合索引的环形经验缓冲与抽样。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/replay/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Replay buffers. |
| [ch3_buffer.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/replay/ch3_buffer.py) | 环形联合回放、uniform/PER 采样、奖励归一化可选项及状态导入导出。符号：`_resolve_device, CH3ReplayBuffer`<br>源码说明：Replay storage for pure Chapter-3 training. |

### core/runtime

通用训练/评估入口；D2 B2/B3 不使用这里的更新调度。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/runtime/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Runtime construction and training support. |
| [builder.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/runtime/builder.py) | 为已知/未知地图构建通用环境、MADDPG 和 replay。符号：`_declared_parameter_names, build_runtime`<br>源码说明：Unified runtime builder for known- and unknown-map Chapter-3 profiles. |
| [engine.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/runtime/engine.py) | 历史配置来源和通用 episode 更新调度；含 policy_delay/PER 逻辑。符号：`_json_safe, _config_hash, _algorithm_config, _algorithm_config_hash, _evaluation_config, _evaluation_config_hash, _run_config_hash, _resolve_device, _ch3_method, _build_efficiency_v2_method_configs, get_ch3_method_config, _environment_class_for_protocol, _build_ch3_env_kwargs, _build_train_env, CH3Runtime, set_ch3_determinism, build_ch3_runtime, build_ch3_runtime_from_resolved_config, _as_env_actions, add_planner_mechanism_metrics, _run_episode, _write_csv, _finite_values, summarize_evaluation_rows, load_scenario_manifest, _checkpoint_metadata, _read_csv_rows, _rng_state_dict, _restore_rng_state, _resume_identity, save_resume_state, load_resume_state, train_and_evaluate_resolved_config, train_and_evaluate_method, train, main`<br>源码说明：Training and paired evaluation entry point for the Chapter-3 project. |
| [metrics.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/runtime/metrics.py) | 通用任务指标汇总。符号：`_vector_text, _finite_or_none, augment_episode_metrics`<br>源码说明：Unified episode metrics with finite/blank serialization semantics. |
| [training.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/runtime/training.py) | 通用训练/评估数据隔离、恢复、checkpoint 与训练汇总流程。符号：`TrainingProtocolSpec, _protocol_spec, _write_csv, _read_csv, run_episode, _scenario_content_hash, _present_values, validate_dataset_isolation, _load_manifest, _identity, _save_resume, _load_resume, _checkpoint_metadata, _summary, train_and_evaluate`<br>源码说明：Unified Chapter-3 training, evaluation, checkpoint, and resume workflow. |

### core/scenarios

场景 schema、生成、manifest 与场景身份校验。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/scenarios/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Canonical Chapter-3 scenario manifests. |
| [ch3_generator_impl.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/scenarios/ch3_generator_impl.py) | 已知/未知地图具体场景采样与合法性检查。符号：`_canonical_sha, _inside, _sample_point, _sample_agents, _paired_obstacles, _trajectory_record, _planner, validate_scenario_reachability, _build_mission_profile_manifests, _boxes_overlap, _sample_unknown_agents, _sample_unknown_obstacles, _unknown_trajectory_record, _validate_unknown_truth_reachability, _unknown_profile_obstacles, _build_unknown_profile_manifests, _write_manifest, _resolve_profiles, _default_generator_seed, build_scenario_manifests, write_scenario_manifests, main`<br>源码说明：Generate strictly paired Chapter-3 manifests and bounded smoke manifests. |
| [generator.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/scenarios/generator.py) | 统一场景生成接口。符号：`build_e0_manifests, generate_e0_manifests`<br>源码说明：Deterministic E0 manifest generation from the self-contained core builder. |
| [manifest.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/scenarios/manifest.py) | 场景 manifest 加载和内容身份。符号：`canonical_json_bytes, sha256_json, write_manifest`<br>源码说明：Canonical JSON and hashing helpers for scenario manifests. |
| [registry.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/scenarios/registry.py) | 允许的算法、场景或论文基线身份<br>源码说明：Supported scenario-profile registry derived from CH3 configuration. |
| [schema.py](E:/gym/code/WORKSPACE/AUV-Thesis/core/scenarios/schema.py) | 场景数据契约。符号：`ScenarioProfile`<br>源码说明：Scenario schema declarations; numerical records remain CH3-authored dicts. |

### docs/chapter3/d2_suite_v1/performance_round2

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [revalidate_windows.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/d2_suite_v1/performance_round2/revalidate_windows.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`write, sha, topology, PowerStatus, power, plan, affinity, CounterValue, Counters, load, sentinel, child, run`<br>源码说明：External Windows measurement driver; does not modify either source checkout. |

### docs/chapter3/hgr_phase2/analysis/debug_10scene_20260926

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [recompute.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/hgr_phase2/analysis/debug_10scene_20260926/recompute.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`sha, close, summary, main`<br>源码说明：Read-only numerical audit of saved Phase2 JSON/NPZ; no production imports. |

### docs/chapter3/search_diagnostics/bser_effect_v1

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [analyze_completed.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/bser_effect_v1/analyze_completed.py) | 离线结果分析、汇总或绘图。符号：`read, mean, exact_p, main`<br>源码说明：Read-only audit of retained D results. Run from the repository root with -m. |

### docs/chapter3/search_diagnostics/bser_final_v1

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [analyze_completed.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/bser_final_v1/analyze_completed.py) | 离线结果分析、汇总或绘图。符号：`read, mean, digest, signatures, main`<br>源码说明：Read-only audit of the user-run F development evaluation; no simulator calls. |
| [seal_reviewed_sources.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/bser_final_v1/seal_reviewed_sources.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`seal`<br>源码说明：Explicit local release operation, never invoked by an experiment or test. |

### docs/chapter3/search_diagnostics/safe_search_v1

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [ac_figures.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/ac_figures.py) | 离线结果分析、汇总或绘图。符号：`integer, sha256, check_fraction, prepare, render, validate_output, save, self_check, main`<br>源码说明：Static figures from the completed A+C analysis, without simulator access. |
| [ac_source_comparability.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/ac_source_comparability.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`run, main`<br>源码说明：Full-episode unchanged-arm replay audit for the additive A+C experiment. |
| [analyze_ac_development.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/analyze_ac_development.py) | 离线结果分析、汇总或绘图。符号：`validate_new_rows, episode_metrics, paired_metrics, validate_reference_analysis, read_run_traces, watch_run, validate_plan, gate_result, verify_comparability, analyze, markdown, output_directory, main`<br>源码说明：Read-only V5=A+C development analysis against separately verified V0--V4. |
| [analyze_b1_development.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/analyze_b1_development.py) | 离线结果分析、汇总或绘图。符号：`comparison, diagnostics, analyze`<br>源码说明：Offline paired B1 transfer analysis. Requires all 60 real B1 terminals. |
| [analyze_b1_remaining.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/analyze_b1_remaining.py) | 离线结果分析、汇总或绘图。符号：`independent_counts, analyze`<br>源码说明：Offline completion and six-version pairing; requires all 60 new terminals. |
| [b1_case_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/b1_case_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`runs, inspect, audit_run`<br>源码说明：Read-only mechanism evidence from completed B1 traces; no source changes. |
| [b1_figures.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/b1_figures.py) | 离线结果分析、汇总或绘图。符号：`render`<br>源码说明：Static thesis-development figures from completed B1 transfer analysis. |
| [b1_hold_drift_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/b1_hold_drift_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`audit`<br>源码说明：Offline one-step dynamics reconciliation, not a new simulator episode. |
| [b1_numeric_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/b1_numeric_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`read, audit`<br>源码说明：Independent recomputation from terminal JSON, without analysis helpers. |
| [b1_post_found_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/b1_post_found_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`audit`<br>源码说明：Reconcile post-Found executor failure from complete B1 V5 traces. |
| [b1_remaining_gain_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/b1_remaining_gain_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`audit`<br>源码说明：Exploratory positive-case evidence, separately from frozen paired effects. |
| [development_baseline_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/development_baseline_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`Inputs, key_outcome, compare_outcomes, terminal_arm, arm_key, checked_signature, historical_comparison, previous_comparison, telemetry_comparison, case_comparison, mechanism_trace, baseline_mechanisms, audit, output_path, main`<br>源码说明：Offline baseline and replay audit; never starts a simulator or changes inputs. |
| [development_compact_evidence.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/development_compact_evidence.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`count, validate_terminal, validate_coverage, validate_windows, compact_coverage, reject_raw_payload, validate_pairing, build, mapped_run_file, export, main`<br>源码说明：Export small, Git-visible evidence after strict full-stage reconciliation. |
| [development_figures.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/development_figures.py) | 离线结果分析、汇总或绘图。符号：`digest, nonnegative_number, prepare, render, validate_output, save, synthetic_example, main`<br>源码说明：Export one two-panel static research diagnostic from completed D2 analysis. |
| [development_public_map_replay.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/development_public_map_replay.py) | 该版本经验存储、抽样、阶段/辅助标签适配。符号：`sha, segment_touches_box, touched_cells, _copied, add_public_capture, wrapped_loader, run_replay, self_check, main`<br>源码说明：Independent read-only replay for D2 scene 7 / V4; never a D2 arm. |
| [development_telemetry_equivalence.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/development_telemetry_equivalence.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`read_trace, common_public_row, compare_old_prefix, check_terminal, run, main`<br>源码说明：Full-horizon observer noninterference check for the failed D2 scene 10/V3. |
| [development_trace_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/development_trace_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`windows, trailing_zero_steps, audit_trace, aggregate, summarize_audits, reject_constant, trace_rows, check_output, analyze, main`<br>源码说明：Read-only window audit after all 100 fixed-development arms have completed. |
| [run_b1_remaining.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/run_b1_remaining.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`arm_directory, child_command, validate_terminal, validate_child, collect, run`<br>源码说明：Frozen B1 V1/V2/V4 supplement; invokes the existing reviewed CLI unchanged. |
| [write_b1_remaining_report.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v1/write_b1_remaining_report.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`pct, paired, write`<br>源码说明：Build reproducible Markdown tables from the verified six-version analysis. |

### docs/chapter3/search_diagnostics/safe_search_v2

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [analyze_completed.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v2/analyze_completed.py) | 离线结果分析、汇总或绘图。符号：`read, main`<br>源码说明：Read-only audit of completed R0-R4 development outputs; no simulator imports. |
| [audit_bser.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v2/audit_bser.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`read, sha, mean, metric, offline, initial_snapshots, main`<br>源码说明：BSER decision audit. Existing evidence stays read-only. |
| [reproduce_score_retention.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3/search_diagnostics/safe_search_v2/reproduce_score_retention.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`main`<br>源码说明：Synthetic, zero-simulator reproduction; not an experiment result. |

### docs/chapter3_map_stp

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [analyze_shadow.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3_map_stp/analyze_shadow.py) | 离线结果分析、汇总或绘图。符号：`mean, interval, analyze, main`<br>源码说明：Read-only recomputation of compact shadow results; creates a new analysis file. |
| [bottleneck_audit.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3_map_stp/bottleneck_audit.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`sha, read, digest, write, trace, audit, longest, center, summarize, candidate_support, trajectory_diagnostics, belief_diagnostics, main`<br>源码说明：Offline audit of retained CH3 experiments; never creates or steps an environment. |
| [check_sampling_stability.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/chapter3_map_stp/check_sampling_stability.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`analyze, main`<br>源码说明：Post-hoc stability recomputation; does not retune the frozen decision gate. |

### docs/hgr/verification/final_02

历史诊断/离线分析与复现脚本；未被生产训练链导入。

| 文件 | 职责与主要符号 |
|---|---|
| [continue_verification.py](E:/gym/code/WORKSPACE/AUV-Thesis/docs/hgr/verification/final_02/continue_verification.py) | 历史诊断/离线分析与复现脚本；未被生产训练链导入；具体入口见符号。符号：`main`<br>源码说明：Recover only complete method logs after the user interrupted the runner. |

### historical_verification

历史验收包；本次不审查 test 文件。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/historical_verification/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Explicit historical evidence suite, outside recursive current discovery. |

### scripts

Windows/跨平台人工入口和离线诊断工具，不是另一套底层算法。

| 文件 | 职责与主要符号 |
|---|---|
| [analyze_found_executor_distance.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/analyze_found_executor_distance.py) | 离线结果分析、汇总或绘图。符号：`number, integer, boolean, read_csv, sha256, geometry, timing, prepare_population, describe, correlation, ranks, auc, logistic, interval, odds, model_report, quartiles, outcome_statistics, contact_equality, git_identity, audit_inputs, write_csv, write_json, analyze, main`<br>源码说明：Offline, scenario-level Found geometry analysis (NumPy only; no runtime imports). |
| [analyze_found_timing_failure_attribution.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/analyze_found_timing_failure_attribution.py) | 离线结果分析、汇总或绘图。符号：`build_episodes, population_counts, timing_bin, timing_bins, model_inputs, estimate, dominant_factor, contact_budget, audit, analyze, main`<br>源码说明：Offline canonical episode analysis of discovery timing and Found geometry. |
| [analyze_search_failures.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/analyze_search_failures.py) | 离线结果分析、汇总或绘图。符号：`file_hash, digest, read_json, integer, distribution, select_population, normalize_episode, summarize_group, verify_summary, analyze_run, paired_evaluations, analyze, markdown, write_report, main`<br>源码说明：Read-only diagnosis of exported CH3 runs; stdlib only, no model loading. |
| [analyze_searcher_residual_branch_sensitivity.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/analyze_searcher_residual_branch_sensitivity.py) | 离线结果分析、汇总或绘图。符号：`nonempty_output, select_anchor, rows_index, control_reproduction, branch_equal, classify, summarize_scenario, analyze, main`<br>源码说明：Offline categorical branch sensitivity analysis; no simulator/model imports. |
| [analyze_searcher_residual_effect.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/analyze_searcher_residual_effect.py) | 离线结果分析、汇总或绘图。符号：`present, number, integer, boolean, read_csv, unique_rows, file_hash, load_input, merged_row, metric, difference, paired_stats, descriptive_stats, exact_mcnemar, paired_binary_stats, trajectory_rows, analyze, main`<br>源码说明：Read-only paired residual-ablation analysis; Python standard library only. |
| [analyze_searcher_residual_trace.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/analyze_searcher_residual_trace.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`decode, read_csv, file_sha, write_csv, numeric, distance, changed, action_index, mean, before_window, followups, analyze_scenario, analyze, main`<br>源码说明：Strict same-key, no-interpolation offline trace analysis (stdlib only). |
| [check_hgr_phase2_config.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/check_hgr_phase2_config.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`Phase2ConfigError, validate_phase2_config, main`<br>源码说明：Read-only Phase2 preflight; never collect, train, create outputs, or resume. |
| [export_hgr_phase2_policy_pair.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/export_hgr_phase2_policy_pair.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`sha, read_checkpoint, verify_sources, reconstruct_pair, export_pair, main`<br>源码说明：One-time, audited parameter import from the migrated 3090 HGR cycle 4. |
| [hgr_decision_experiment.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/hgr_decision_experiment.py) | Windows/跨平台人工入口和离线诊断工具，不是另一套底层算法；具体入口见符号。符号：`_sha, _path, _int, _finite_positive, load_config, resolve_config, budget_plan, _prepare, preflight, DecisionBackend, decide, Experiment, run_experiment, main`<br>源码说明：Bounded HGR research screen; default CLI is a read-only preflight. |
| [hgr_decision_statistics.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/hgr_decision_statistics.py) | 离线结果分析、汇总或绘图。符号：`_vectors, _costs, _ratio, _moments, summarize_pair`<br>源码说明：Small-budget diagnostics for ONE frozen HGR policy pair, without rollouts. |
| [hgr_suffix_diagnostics.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/hgr_suffix_diagnostics.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`_flat, _gradient_array, _norm, _finite, _validate_trajectories, _group_gradients, _behavior_change, diagnose_suffix_update`<br>源码说明：Observe one unchanged HGR suffix SGD update without constructing a runtime. |
| [run_d2_suite.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_d2_suite.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明<br>源码说明：Cross-platform launcher: python -m scripts.run_d2_suite --help (repository root). |
| [run_search_diagnostic_probe.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_search_diagnostic_probe.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`jsonable, physical_signature, run_probe, main`<br>源码说明：Bounded B0/B1 current-source diagnostic probe; no checkpoints or training. |
| [run_searcher_residual_branch_sensitivity.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_searcher_residual_branch_sensitivity.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明<br>源码说明：Root-relative diagnostic entry point; no sys.path injection. |
| [run_searcher_residual_trace_diagnostic.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_searcher_residual_trace_diagnostic.py) | 该目录机制诊断、轨迹记录、状态/分支分析<br>源码说明：Root-relative launcher, without sys.path injection or legacy imports. |
| [search_diagnostic_observer.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/search_diagnostic_observer.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`vector, norm, subtract, distance, segment_distance, angle, plain, endpoint_present, QueryTap, capture_runtime, TransitionObserver`<br>源码说明：Opt-in read-only telemetry for existing CH3 runtimes. |
| [select_phase1c_prrac_s2a1_activation_scenarios.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/select_phase1c_prrac_s2a1_activation_scenarios.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`select_scenarios, main`<br>源码说明：Select deterministic diagnostic-only S2-A.1 scenarios from a v1 C0 episode CSV. |
| [summarize_search_trace.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/summarize_search_trace.py) | 该目录机制诊断、轨迹记录、状态/分支分析。符号：`public_cell, segment_hits_box, occupied_voxel_intersections, summarize, main`<br>源码说明：Summarize bounded diagnostic traces without inferring unobserved causes. |
| [validate_phase1c_prrac_s2a1_activation.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/validate_phase1c_prrac_s2a1_activation.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_read_csv, _integer, _boolean, _result, validate_activation, validate_output_dir, main`<br>源码说明：Validate S2-A.1 activation summaries or a complete evaluation artifact set. |
| [validate_phase1c_prrac_s2a_baseline.py](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/validate_phase1c_prrac_s2a_baseline.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`main`<br>源码说明：Validate the frozen C0 Ep100 Val100 gate without modifying evaluation results. |

### tools

维护、性能、来源、复现与有界验证工具；不属于训练损失。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Repository command-line tools, invoked with ``python -m tools.<name>``. |
| [benchmark_d2_performance.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/benchmark_d2_performance.py) | 维护、性能、来源、复现与有界验证工具；不属于训练损失；具体入口见符号。符号：`QueryDiagnostics, benchmark_model_builder, Timings, difference, record, cache_diagnostics, update_signature, run, environment_profile, compute_profile, logging_profile, main`<br>源码说明：Opt-in bounded D2 timing and semantic signatures; never a formal experiment. |
| [build_bser_phase1a1_delivery.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/build_bser_phase1a1_delivery.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`sha256, dump_json, load_json, rows, rel_path, git_blob, normalize_before, current_record, freeze_pair, floats, method_map, mean_delta, git_lines, changed_files, markdown, build_docs, collect_metrics, remote_result, main`<br>源码说明：Build auditable BSER Phase 1A.1 freeze manifests and delivery reports. |
| [compare_d2_performance.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/compare_d2_performance.py) | 维护、性能、来源、复现与有界验证工具；不属于训练损失；具体入口见符号。符号：`semantic_summary, compare, main`<br>源码说明：Fail closed on any semantic difference in two bounded D2 benchmarks. |
| [establish_working_baseline.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/establish_working_baseline.py) | 维护、性能、来源、复现与有界验证工具；不属于训练损失；具体入口见符号。符号：`sha256, main`<br>源码说明：Create the user-accepted Phase 0B working baseline and archive waiver. |
| [run_bser_planner_equivalence.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/run_bser_planner_equivalence.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_seed_all, _write_csv, _fixed_targets, _flatten_cells, _compare_query, run, main`<br>源码说明：Run EQ1 differential checks between PlanningGraphView and the frozen planner. |
| [run_core_golden_e0.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/run_core_golden_e0.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`_utc_now, _sha256_file, _write_json, _canonical_json, _seed_all, _to_numpy, _optional_array, _target_payload, _capture_state, _normal, _state_hash, _compare, _actions, _run_profile, _write_csv, _source_manifest, run, main`<br>源码说明：Run 60 core-only E0 trajectories against the frozen Phase 0B-2 golden hashes. |
| [run_core_training_smoke.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/run_core_training_smoke.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`seed_all, sha, run_smoke, main`<br>源码说明：Run a bounded 2x10-step CPU training-closure smoke for the self-contained core. |
| [run_phase1b1_seed_parallel.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/run_phase1b1_seed_parallel.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`run_seed, main`<br>源码说明：Run one Phase 1B.1 pilot seed as independent condition episodes. |
| [verify_bser_core_freeze.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/verify_bser_core_freeze.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`build_after, main`<br>源码说明：Verify the Phase-1A pre-existing core freeze and emit a reproducible report. |
| [verify_collision_terminal.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/verify_collision_terminal.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`utc, sha, dump, git, tracked_and_new, source_files, tree_identity, flatten, RecordingResult, run_methods, module_job, current, run_modules, hgr_entry, hgr_entry_metadata, historical, export_reference, legacy, golden_profile, golden, main`<br>源码说明：Fixed-tree local current/historical/legacy/golden verification. |
| [verify_external_archive.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/verify_external_archive.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`normalized, sha256_file, is_relative_to, is_reparse, scan_archive, read_manifest, expected_rows, ArchiveIndex, verify_one, write_csv, verify_curated, build_parser, main`<br>源码说明：Read-only, resumable verifier for CRK legacy external archives. |

### tools/ch3_baselines

论文 B0—B3 身份、任务条件、零残差与已训练模型评估入口。

| 文件 | 职责与主要符号 |
|---|---|
| [__init__.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/__init__.py) | 包标记或公共导出；不独立启动实验<br>源码说明：Independent, training-free Chapter 3 task baselines. |
| [basic_search_prior.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/basic_search_prior.py) | 论文 B0—B3 身份、任务条件、零残差与已训练模型评估入口；具体入口见符号。符号：`baseline_environment_kwargs, SearchPriorAllocator, SearchPriorController, BasicSearchPriorRuntime`<br>源码说明：Search-only allocation injected before the first guidance installation. |
| [bser_prior.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/bser_prior.py) | 论文 B0—B3 身份、任务条件、零残差与已训练模型评估入口；具体入口见符号。符号：`ZeroResidualSource, BSERPriorRuntime`<br>源码说明：B1: the original MissionRuntime and BSER planner, with zero residuals. |
| [evaluate.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/evaluate.py) | 完整任务/模型评估与相关指标、跟踪或入口。符号：`resolve_config, select_manifest, episode_row, summary, evaluate, main`<br>源码说明：CPU serial complete-task evaluator. No checkpoint, training, resume or scene generation. |
| [framework_provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/framework_provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`baseline_protected_identity, framework_sources, verify_framework_sources`<br>源码说明：CH3-final production and baseline gates, independent of Windows launchers. |
| [learned_evaluation.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/learned_evaluation.py) | 完整任务/模型评估与相关指标、跟踪或入口。符号：`evaluate`<br>源码说明：Frozen independent MADDPG evaluation; no replay, updates or HGR policy. |
| [provenance.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/provenance.py) | 该版本源码/配置来源清单、哈希和变更验收。符号：`file_sha256, digest, baseline_source_identity, baseline_evolution, production_sources, capture_sources, require_unchanged, write_json`<br>源码说明：Separate baseline provenance; historical HGR manifests are never rewritten. |
| [registry.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/registry.py) | 允许的算法、场景或论文基线身份。符号：`load_registry, method_spec, task_conditions, load_reference, validate_method_config, baseline_config, training_config`<br>源码说明：Paper identities mapped to immutable production method identities. |
| [run_baseline.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/run_baseline.py) | 人工运行、配置预检、作业编排或产物构建入口；详见下列符号和模块说明。符号：`normalize_row, unified_summary, evaluation_plan, evaluate, main`<br>源码说明：Unified B0--B3 evaluation. Training is never dispatched from this entry. |
| [run_training.py](E:/gym/code/WORKSPACE/AUV-Thesis/tools/ch3_baselines/run_training.py) | 该版本训练调度/更新入口；仅在显式调用时执行。符号：`training_plan, train, main`<br>源码说明：Independent B2/B3 MADDPG training; --check-only never constructs a model. |

## 配置与 Shell/PowerShell/BAT 入口

下列文件只按配置结构或入口目标核查，不把历史结果 JSON 当作算法源码。

| 文件 | 职责/调用目标 |
|---|---|
| [baseline_registry.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/baselines/baseline_registry.json)<br>`configs/chapter3/baselines/baseline_registry.json` | 配置结构：B0_search_prior, B1_bser_prior, B2_direct_mc, B3_direct_boundary |
| [basic_search_prior_v1.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/baselines/basic_search_prior_v1.json)<br>`configs/chapter3/baselines/basic_search_prior_v1.json` | 配置/场景输入：{"schema": "ch3.basic_search_prior.spec.v1", "method": "ch3_basic_search_prior_v1"} |
| [bser_prior_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/baselines/bser_prior_eval.json)<br>`configs/chapter3/baselines/bser_prior_eval.json` | 配置/场景输入：{"method": "ch3_baseline_bser_prior", "algorithm": null} |
| [direct_boundary_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/baselines/direct_boundary_train.json)<br>`configs/chapter3/baselines/direct_boundary_train.json` | 配置/场景输入：{"schema": "ch3.baseline.training.v1", "method": "ch3_baseline_direct_boundary", "algorithm": "direct_boundary_maddpg", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [direct_mc_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/baselines/direct_mc_train.json)<br>`configs/chapter3/baselines/direct_mc_train.json` | 配置/场景输入：{"schema": "ch3.baseline.training.v1", "method": "ch3_baseline_direct_mc", "algorithm": "maddpg", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [search_prior_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/baselines/search_prior_eval.json)<br>`configs/chapter3/baselines/search_prior_eval.json` | 配置/场景输入：{"method": "ch3_baseline_search_prior", "algorithm": null} |
| [bser_phase1a.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1a.json)<br>`configs/chapter3/bser_phase1a.json` | 配置/场景输入：{"schema": "bser.phase1a.config.v1"} |
| [bser_phase1a1.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1a1.json)<br>`configs/chapter3/bser_phase1a1.json` | 配置/场景输入：{"schema": "bser.phase1a1.config.v1"} |
| [bser_phase1b.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1b.json)<br>`configs/chapter3/bser_phase1b.json` | 配置/场景输入：{"schema": "bser.phase1b.config.v1"} |
| [bser_phase1b1.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1b1.json)<br>`configs/chapter3/bser_phase1b1.json` | 配置/场景输入：{"schema": "bser.phase1b1.config.v1"} |
| [bser_phase1c.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c.json)<br>`configs/chapter3/bser_phase1c.json` | 配置/场景输入：{"schema": "bser.phase1c.integration.v1"} |
| [bser_phase1c_beds.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_beds.json)<br>`configs/chapter3/bser_phase1c_beds.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation_report.v2", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_diagnostic_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_diagnostic_eval.json)<br>`configs/chapter3/bser_phase1c_diagnostic_eval.json` | 配置/场景输入：{"schema": "bser.phase1c.diagnostic_eval.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_early_discovery.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_early_discovery.json)<br>`configs/chapter3/bser_phase1c_early_discovery.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation_report.v2", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_collision_terminal_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_collision_terminal_eval.json)<br>`configs/chapter3/bser_phase1c_prrac_collision_terminal_eval.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_collision_terminal_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_collision_terminal_train.json)<br>`configs/chapter3/bser_phase1c_prrac_collision_terminal_train.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.training.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_eval.json)<br>`configs/chapter3/bser_phase1c_prrac_eval.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_execution_ablation.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_execution_ablation.json)<br>`configs/chapter3/bser_phase1c_prrac_execution_ablation.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.execution_ablation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_individual_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_individual_eval.json)<br>`configs/chapter3/bser_phase1c_prrac_individual_eval.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400, "reward_objective": "individual_v1"} |
| [bser_phase1c_prrac_individual_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_individual_train.json)<br>`configs/chapter3/bser_phase1c_prrac_individual_train.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.training.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400, "reward_objective": "individual_v1"} |
| [bser_phase1c_prrac_s1_search_diag_legacy.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_s1_search_diag_legacy.json)<br>`configs/chapter3/bser_phase1c_prrac_s1_search_diag_legacy.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.execution_ablation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_s1_search_diag_native.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_s1_search_diag_native.json)<br>`configs/chapter3/bser_phase1c_prrac_s1_search_diag_native.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.execution_ablation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_s1_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_s1_train.json)<br>`configs/chapter3/bser_phase1c_prrac_s1_train.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.training.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_s2a1_local_connector_ablation.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json)<br>`configs/chapter3/bser_phase1c_prrac_s2a1_local_connector_ablation.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation_report.v2", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_s2a_collision_ablation.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_s2a_collision_ablation.json)<br>`configs/chapter3/bser_phase1c_prrac_s2a_collision_ablation.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation_report.v2", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_prrac_team_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_team_eval.json)<br>`configs/chapter3/bser_phase1c_prrac_team_eval.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [bser_phase1c_prrac_team_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_team_train.json)<br>`configs/chapter3/bser_phase1c_prrac_team_train.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.training.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [bser_phase1c_prrac_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_prrac_train.json)<br>`configs/chapter3/bser_phase1c_prrac_train.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.training.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_search_value.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_search_value.json)<br>`configs/chapter3/bser_phase1c_search_value.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.training.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_search_value_guided.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_search_value_guided.json)<br>`configs/chapter3/bser_phase1c_search_value_guided.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation_report.v2", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_train.json)<br>`configs/chapter3/bser_phase1c_train.json` | 配置/场景输入：{"schema": "bser.phase1c.training.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_v2_1_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_v2_1_train.json)<br>`configs/chapter3/bser_phase1c_v2_1_train.json` | 配置/场景输入：{"schema": "bser.phase1c.training.v2", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [bser_phase1c_v2_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/bser_phase1c_v2_train.json)<br>`configs/chapter3/bser_phase1c_v2_train.json` | 配置/场景输入：{"schema": "bser.phase1c.training.v2", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [experiment.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/d2_suite_v1/experiment.json)<br>`configs/chapter3/d2_suite_v1/experiment.json` | 配置/场景输入：{"schema": "ch3.d2_suite.v1"} |
| [baseline_registry.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/d2_v1/baseline_registry.json)<br>`configs/chapter3/d2_v1/baseline_registry.json` | 配置结构：B0_search_prior, B1_bser_prior, B2_direct_mc, B3_direct_boundary |
| [direct_boundary_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/d2_v1/direct_boundary_train.json)<br>`configs/chapter3/d2_v1/direct_boundary_train.json` | 配置/场景输入：{"schema": "ch3.baseline.training.v1", "method": "ch3_baseline_direct_boundary", "algorithm": "direct_boundary_maddpg", "planner_protocol": "d2_v1", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [direct_mc_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/d2_v1/direct_mc_train.json)<br>`configs/chapter3/d2_v1/direct_mc_train.json` | 配置/场景输入：{"schema": "ch3.baseline.training.v1", "method": "ch3_baseline_direct_mc", "algorithm": "maddpg", "planner_protocol": "d2_v1", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [hgr_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/d2_v1/hgr_train.json)<br>`configs/chapter3/d2_v1/hgr_train.json` | 配置/场景输入：{"schema": "hgr.training.v1", "method": "ch3_hgr", "algorithm": "hgr", "planner_protocol": "d2_v1", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [hgr_decision_experiment.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/hgr_decision_experiment.json)<br>`configs/chapter3/hgr_decision_experiment.json` | 配置/场景输入：{"schema": "hgr.decision_experiment.v1"} |
| [hgr_phase1_ridge_diagnostic.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/hgr_phase1_ridge_diagnostic.json)<br>`configs/chapter3/hgr_phase1_ridge_diagnostic.json` | 配置/场景输入：{"schema": "hgr.training.v1", "method": "ch3_hgr", "algorithm": "hgr", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [hgr_phase1_zero.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/hgr_phase1_zero.json)<br>`configs/chapter3/hgr_phase1_zero.json` | 配置/场景输入：{"schema": "hgr.training.v1", "method": "ch3_hgr", "algorithm": "hgr", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [hgr_phase2_debug.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/hgr_phase2_debug.json)<br>`configs/chapter3/hgr_phase2_debug.json` | 配置/场景输入：{"schema": "hgr.phase2.gradient_efficiency.v1"} |
| [hgr_phase2_gradient_efficiency.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/hgr_phase2_gradient_efficiency.json)<br>`configs/chapter3/hgr_phase2_gradient_efficiency.json` | 配置/场景输入：{"schema": "hgr.phase2.gradient_efficiency.v1"} |
| [hgr_phase2_source_builder.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/hgr_phase2_source_builder.json)<br>`configs/chapter3/hgr_phase2_source_builder.json` | 配置/场景输入：{"schema": "hgr.phase2.source_builder.v1"} |
| [hgr_train.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/hgr_train.json)<br>`configs/chapter3/hgr_train.json` | 配置/场景输入：{"schema": "hgr.training.v1", "method": "ch3_hgr", "algorithm": "hgr", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [long_horizon_maddpg_v1.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/long_horizon_maddpg_v1.json)<br>`configs/chapter3/long_horizon_maddpg_v1.json` | 配置/场景输入：{"schema": "ch3.long_horizon_maddpg.v1", "method": "ch3_long_horizon_maddpg_v1", "algorithm": "long_horizon_maddpg", "planner_protocol": "d2_v1", "max_steps": 400, "reward_objective": "team_mean_v1"} |
| [map_stp_shadow_v1.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/map_stp_shadow_v1.json)<br>`configs/chapter3/map_stp_shadow_v1.json` | 配置/场景输入：{"schema": "ch3.map_stp.shadow.v1"} |
| [prior_only_eval.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/chapter3/prior_only_eval.json)<br>`configs/chapter3/prior_only_eval.json` | 配置/场景输入：{"schema": "bser.phase1c.prrac.evaluation.v1", "method": "ch3_bser_rmaddpg_phase1c", "max_steps": 400} |
| [conda_environment_full.yml](E:/gym/code/WORKSPACE/AUV-Thesis/configs/environment_lock/conda_environment_full.yml)<br>`configs/environment_lock/conda_environment_full.yml` | 环境锁定/安装清单 |
| [conda_environment_history.yml](E:/gym/code/WORKSPACE/AUV-Thesis/configs/environment_lock/conda_environment_history.yml)<br>`configs/environment_lock/conda_environment_history.yml` | 环境锁定/安装清单 |
| [environment_linux_cuda.yml](E:/gym/code/WORKSPACE/AUV-Thesis/configs/environment_lock/environment_linux_cuda.yml)<br>`configs/environment_lock/environment_linux_cuda.yml` | 环境锁定/安装清单 |
| [torch_runtime.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/environment_lock/torch_runtime.json)<br>`configs/environment_lock/torch_runtime.json` | 配置结构：torch_version, torch_file, cuda_compiled_version, cuda_available, cudnn_version, mps_available, default_dtype, num_threads, num_interop_threads, gpu_devices, environment |
| [WORKING_BASELINE_V2.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/legacy_baseline/WORKING_BASELINE_V2.json)<br>`configs/legacy_baseline/WORKING_BASELINE_V2.json` | 配置/场景输入：{"schema": "crk.working_legacy_baseline.v2"} |
| [M00_MOVING_CLEAR.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/scenarios/e0_equivalence/M00_MOVING_CLEAR.json)<br>`configs/scenarios/e0_equivalence/M00_MOVING_CLEAR.json` | 配置/场景输入：{"max_steps": 400} |
| [M10_MOVING_UNKNOWN_SINGLE.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/scenarios/e0_equivalence/M10_MOVING_UNKNOWN_SINGLE.json)<br>`configs/scenarios/e0_equivalence/M10_MOVING_UNKNOWN_SINGLE.json` | 配置/场景输入：{"max_steps": 400} |
| [M20_MOVING_UNKNOWN_MULTI.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json)<br>`configs/scenarios/e0_equivalence/M20_MOVING_UNKNOWN_MULTI.json` | 配置/场景输入：{"max_steps": 400} |
| [M90_MOVING_KNOWN_ORACLE.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/scenarios/e0_equivalence/M90_MOVING_KNOWN_ORACLE.json)<br>`configs/scenarios/e0_equivalence/M90_MOVING_KNOWN_ORACLE.json` | 配置/场景输入：{"max_steps": 400} |
| [scenario_manifest_index.json](E:/gym/code/WORKSPACE/AUV-Thesis/configs/scenarios/e0_equivalence/scenario_manifest_index.json)<br>`configs/scenarios/e0_equivalence/scenario_manifest_index.json` | 配置结构：experiment, generator_seed, profiles, scenario_seeds |
| [_search_value_audit_common.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/_search_value_audit_common.sh)<br>`scripts/linux/_search_value_audit_common.sh` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [bundle_search_value_audits.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/bundle_search_value_audits.sh)<br>`scripts/linux/bundle_search_value_audits.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.search_value_audit.analysis_bundle |
| [env_preflight.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/env_preflight.sh)<br>`scripts/linux/env_preflight.sh` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [evaluate_d2_reference.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/evaluate_d2_reference.sh)<br>`scripts/linux/evaluate_d2_reference.sh` | 启动 Python 模块：chapter3_bser.experiments.d2_suite_v1.evaluation_cli |
| [evaluate_d2_suite.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/evaluate_d2_suite.sh)<br>`scripts/linux/evaluate_d2_suite.sh` | 启动 Python 模块：chapter3_bser.experiments.d2_suite_v1.linux |
| [evaluate_d2_trained.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/evaluate_d2_trained.sh)<br>`scripts/linux/evaluate_d2_trained.sh` | 启动 Python 模块：chapter3_bser.experiments.d2_suite_v1.evaluation_cli |
| [run_ch3_baseline_eval.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_ch3_baseline_eval.sh)<br>`scripts/linux/run_ch3_baseline_eval.sh` | 启动 Python 模块：tools.ch3_baselines.run_baseline |
| [run_ch3_basic_prior_eval.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_ch3_basic_prior_eval.sh)<br>`scripts/linux/run_ch3_basic_prior_eval.sh` | 启动 Python 模块：tools.ch3_baselines.evaluate |
| [run_ch3_learning.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_ch3_learning.sh)<br>`scripts/linux/run_ch3_learning.sh` | 启动 Python 模块：chapter3_bser.experiments.hgr.cli |
| [run_collision_terminal.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_collision_terminal.sh)<br>`scripts/linux/run_collision_terminal.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.collision_terminal_cli |
| [run_d2_full_suite.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_d2_full_suite.sh)<br>`scripts/linux/run_d2_full_suite.sh` | 启动 Python 模块：chapter3_bser.experiments.d2_suite_v1.pipeline |
| [run_d2_suite.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_d2_suite.sh)<br>`scripts/linux/run_d2_suite.sh` | 启动 Python 模块：scripts.run_d2_suite |
| [run_hgr_phase1_acceptance.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_hgr_phase1_acceptance.sh)<br>`scripts/linux/run_hgr_phase1_acceptance.sh` | 启动 Python 模块：chapter3_bser.experiments.hgr.phase1_acceptance |
| [run_phase1c_prrac_eval.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_prrac_eval.sh)<br>`scripts/linux/run_phase1c_prrac_eval.sh` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_execution_ablation.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_prrac_execution_ablation.sh)<br>`scripts/linux/run_phase1c_prrac_execution_ablation.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints |
| [run_phase1c_prrac_s1_search_diag.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_prrac_s1_search_diag.sh)<br>`scripts/linux/run_phase1c_prrac_s1_search_diag.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints |
| [run_phase1c_prrac_s1_train.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_prrac_s1_train.sh)<br>`scripts/linux/run_phase1c_prrac_s1_train.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac |
| [run_phase1c_prrac_s2a1_local_connector_ablation.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_prrac_s2a1_local_connector_ablation.sh)<br>`scripts/linux/run_phase1c_prrac_s2a1_local_connector_ablation.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints |
| [run_phase1c_prrac_s2a_collision_ablation.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_prrac_s2a_collision_ablation.sh)<br>`scripts/linux/run_phase1c_prrac_s2a_collision_ablation.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints |
| [run_phase1c_prrac_train.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_prrac_train.sh)<br>`scripts/linux/run_phase1c_prrac_train.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.train_phase1c_prrac |
| [run_phase1c_v2_1_train.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_v2_1_train.sh)<br>`scripts/linux/run_phase1c_v2_1_train.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 |
| [run_phase1c_v2_diagnostic_eval.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_v2_diagnostic_eval.sh)<br>`scripts/linux/run_phase1c_v2_diagnostic_eval.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_bser_rmaddpg.evaluate_phase1c_checkpoints |
| [run_phase1c_v2_train.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_phase1c_v2_train.sh)<br>`scripts/linux/run_phase1c_v2_train.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_bser_rmaddpg_v2.train_phase1c_v2 |
| [run_search_value_d1_audit.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_search_value_d1_audit.sh)<br>`scripts/linux/run_search_value_d1_audit.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.search_value_audit.prediction_audit |
| [run_search_value_d2_audit.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/run_search_value_d2_audit.sh)<br>`scripts/linux/run_search_value_d2_audit.sh` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.search_value_audit.branch_audit |
| [summarize_d2_suite.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/summarize_d2_suite.sh)<br>`scripts/linux/summarize_d2_suite.sh` | 启动 Python 模块：scripts.run_d2_suite |
| [train_ch3_direct_boundary.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/train_ch3_direct_boundary.sh)<br>`scripts/linux/train_ch3_direct_boundary.sh` | 启动 Python 模块：tools.ch3_baselines.run_training |
| [train_ch3_direct_mc.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/train_ch3_direct_mc.sh)<br>`scripts/linux/train_ch3_direct_mc.sh` | 启动 Python 模块：tools.ch3_baselines.run_training |
| [train_d2_suite.sh](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/linux/train_d2_suite.sh)<br>`scripts/linux/train_d2_suite.sh` | 启动 Python 模块：chapter3_bser.experiments.d2_suite_v1.linux |
| [run_ch3_baseline_eval.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_ch3_baseline_eval.bat)<br>`scripts/run_ch3_baseline_eval.bat` | 启动 Python 模块：tools.ch3_baselines.run_baseline |
| [run_ch3_basic_prior_eval.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_ch3_basic_prior_eval.bat)<br>`scripts/run_ch3_basic_prior_eval.bat` | 启动 Python 模块：tools.ch3_baselines.evaluate |
| [run_ch3_learning.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_ch3_learning.bat)<br>`scripts/run_ch3_learning.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_ch3_learning.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_ch3_learning.ps1)<br>`scripts/run_ch3_learning.ps1` | 启动 Python 模块：chapter3_bser.experiments.hgr.cli |
| [run_collision_terminal.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_collision_terminal.bat)<br>`scripts/run_collision_terminal.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_collision_terminal.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_collision_terminal.ps1)<br>`scripts/run_collision_terminal.ps1` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.collision_terminal_cli |
| [run_hgr_decision_experiment.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_hgr_decision_experiment.bat)<br>`scripts/run_hgr_decision_experiment.bat` | 启动 Python 模块：scripts.hgr_decision_experiment |
| [run_hgr_phase1_acceptance.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_hgr_phase1_acceptance.bat)<br>`scripts/run_hgr_phase1_acceptance.bat` | 启动 Python 模块：chapter3_bser.experiments.hgr.phase1_acceptance |
| [run_hgr_phase2_gradient_efficiency.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_hgr_phase2_gradient_efficiency.bat)<br>`scripts/run_hgr_phase2_gradient_efficiency.bat` | 启动 Python 模块：scripts.check_hgr_phase2_config, chapter3_bser.experiments.hgr.phase2_gradient_efficiency |
| [run_hgr_phase2_source_builder.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_hgr_phase2_source_builder.bat)<br>`scripts/run_hgr_phase2_source_builder.bat` | 启动 Python 模块：chapter3_bser.experiments.hgr.build_phase2_source |
| [run_phase1c_diagnostic_eval.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_diagnostic_eval.bat)<br>`scripts/run_phase1c_diagnostic_eval.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_diagnostic_eval.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_diagnostic_eval.ps1)<br>`scripts/run_phase1c_diagnostic_eval.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_eval.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_eval.bat)<br>`scripts/run_phase1c_prrac_eval.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_eval.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_eval.ps1)<br>`scripts/run_phase1c_prrac_eval.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_execution_ablation.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_execution_ablation.bat)<br>`scripts/run_phase1c_prrac_execution_ablation.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_execution_ablation.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_execution_ablation.ps1)<br>`scripts/run_phase1c_prrac_execution_ablation.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_s1_search_diag.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s1_search_diag.bat)<br>`scripts/run_phase1c_prrac_s1_search_diag.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_s1_search_diag.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s1_search_diag.ps1)<br>`scripts/run_phase1c_prrac_s1_search_diag.ps1` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints |
| [run_phase1c_prrac_s1_train.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s1_train.bat)<br>`scripts/run_phase1c_prrac_s1_train.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_s1_train.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s1_train.ps1)<br>`scripts/run_phase1c_prrac_s1_train.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_s2a1_local_connector_ablation.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s2a1_local_connector_ablation.bat)<br>`scripts/run_phase1c_prrac_s2a1_local_connector_ablation.bat` | 启动 Python 模块：chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints |
| [run_phase1c_prrac_s2a1_local_connector_ablation.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s2a1_local_connector_ablation.ps1)<br>`scripts/run_phase1c_prrac_s2a1_local_connector_ablation.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_s2a_collision_ablation.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s2a_collision_ablation.bat)<br>`scripts/run_phase1c_prrac_s2a_collision_ablation.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_s2a_collision_ablation.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_s2a_collision_ablation.ps1)<br>`scripts/run_phase1c_prrac_s2a_collision_ablation.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_train.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_train.bat)<br>`scripts/run_phase1c_prrac_train.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_prrac_train.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_prrac_train.ps1)<br>`scripts/run_phase1c_prrac_train.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_train.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_train.bat)<br>`scripts/run_phase1c_train.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_train.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_train.ps1)<br>`scripts/run_phase1c_train.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_v2_1_train.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_v2_1_train.bat)<br>`scripts/run_phase1c_v2_1_train.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_v2_1_train.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_v2_1_train.ps1)<br>`scripts/run_phase1c_v2_1_train.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_v2_train.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_v2_train.bat)<br>`scripts/run_phase1c_v2_train.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_phase1c_v2_train.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_phase1c_v2_train.ps1)<br>`scripts/run_phase1c_v2_train.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_prior_only_eval.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_prior_only_eval.bat)<br>`scripts/run_prior_only_eval.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_prior_only_pair_1.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_prior_only_pair_1.bat)<br>`scripts/run_prior_only_pair_1.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_prior_only_pair_10.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_prior_only_pair_10.bat)<br>`scripts/run_prior_only_pair_10.bat` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [run_prrac_controller_pair.ps1](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/run_prrac_controller_pair.ps1)<br>`scripts/run_prrac_controller_pair.ps1` | 人工启动/预检/批处理包装；读取后交给对应 Python 实现 |
| [train_ch3_direct_boundary.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/train_ch3_direct_boundary.bat)<br>`scripts/train_ch3_direct_boundary.bat` | 启动 Python 模块：tools.ch3_baselines.run_training |
| [train_ch3_direct_mc.bat](E:/gym/code/WORKSPACE/AUV-Thesis/scripts/train_ch3_direct_mc.bat)<br>`scripts/train_ch3_direct_mc.bat` | 启动 Python 模块：tools.ch3_baselines.run_training |
