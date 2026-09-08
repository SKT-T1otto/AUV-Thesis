# BEDS-PRRAC：可开关的早期发现排序与 Executor 预定位

本次仅实现代码结构与 evaluation 接口。没有执行 10-episode smoke、正式 100×400 实验或用户 checkpoint 的训练/恢复，尚未建立性能改进结论。

## 开关与范围

新增 `configs/chapter3/bser_phase1c_beds.json`，默认两个 enabled 均为 false；`early_discovery.lambda=0.01`、`nominal_speed=1.0`、`executor_standby.gain=0.5`。代码在未提供 early-discovery 配置时的 lambda 默认值是 0.0。另提供 `bser_phase1c_early_discovery.json`。

| CLI variant | early_discovery.enabled | executor_standby.enabled |
| --- | --- | --- |
| baseline | false | false |
| early_only | true | false |
| standby_only | false | true |
| full | true | true |

启用 BEDS 时要求 OFF + native B1 + C2 + full_prrac；不与 SearchValue head 的辅助排序叠加。`BEDS-PRRAC` 是新增 evaluation 扩展标识，保存在 episode 的 `beds` 字段及 resolved config 中；没有变更冻结的 registered method、checkpoint schema 或模型结构。

两个开关关闭时，不构造 episode adapter，不修改 guidance/action，不新增 worker payload 字段，也不产生 BEDS sidecar。已有配置没有 BEDS 字段时不会被注入新默认字段，原配置和 resume 语义保留。显式带 BEDS 字段的新 config 自身的配置 hash 自然不同；bit-level 等价指相同初始状态、参数和种子下的算法与物理执行，不指两个不同配置文件字节相同。

## Early discovery 插入位置

`chapter3_bser/online/early_discovery.py` 实现 `EarlyDiscoveryAllocator`，复用 `BSEROnlineAllocator` 的候选生成、ObjectiveContext 和原始 solver。PRRAC controller factory 仅在 early 开关开启时注入该 allocator。

在 `_solve_candidates()` 中，先计算原 BSER 解，保留原 Executor standby 选择。对于尚未 Found 的 Search 候选，仍用原 `marginal_gain(selected, candidate, standby, context)` 得到条件增益，只在每一轮 greedy 候选排序时应用：

`final_score = original_bser_score * exp(-lambda * estimated_arrival_time)`。

到达时间优先使用候选已有 A* `path_length / nominal_speed`。只有缺少该字段时才使用候选 waypoint 与当前 Searcher 位置的欧氏距离。距离单位为米、速度为米/秒，因此 arrival_time 为秒、lambda 为秒的倒数；没有把它误记成完成步数，也没有新增路径规划调用。现有 BSER gain 本身已使用公开 belief，本扩展不新增 belief 估计器。

只对原始正边际增益候选排序，保留每个 Searcher 最多一个分配的约束和原 candidate.key 的确定性 tie-break。最终仍以原 `evaluate_objective()` 计算并报告 objective，原 controller 的接受/滞回规则也保留。因此一个重排建议仍可能被已有规则拒绝。本次没有改 BSER objective、candidate 数据对象或冻结 solver。

early disabled 或 lambda=0 时直接返回原 solver 结果，避免无效的到达时间计算或 score 浮点运算。Found 后恢复原 solver 路径。

## Executor standby 插入位置

`chapter3_bser/online/executor_standby.py` 提供 `compute_standby_target(executor_position, searcher_positions, optional_weights)`，以 modified Weiszfeld 迭代近似求解加权几何中位点 `min sum(w_i * ||p-p_i||)`。几何中位点通常不是算术重心；等边三角形对称输入才相同。实现处理重复点、共线点、单点权重、无有效权重及大幅度有限坐标。

当前 OnlineAllocation 没有导出可靠的逐 Searcher 当前搜索价值，因此接入层明确使用 1/3 等权，并记录 `weights_source=uniform_no_audited_searcher_values`。接口接受调用方明确提供的有效非负权重；无效权重回退为等权，不把 marginal gain、距离或残差伪称为 belief 权重。

`phase1c_prrac/beds.py` 的 `BEDSEpisodeAdapter` 只在 evaluation worker 的 Search 阶段启用：

1. 在既有 guidance 编译完成后、安装前，只替换 Executor 的 standby/tracking 目标；Searcher assignment 对象保持原样。
2. 在原 actor 推理、mode 和 continuity action adapter 完成后、`env.step` 前，只将第 3 号 Executor 的动作替换为 `clip(gain * (target-current), -1, 1)`，保持 tensor dtype/device 和 4×3 contract。
3. 输入仍是项目原有归一化残差动作接口，原动作到加速度尺度、导航 prior、阻力、碰撞和边界动力学全部保留。新的 standby 导航目标让 prior 也指向预定位位置；比例动作不是对物理位置或速度的直接写入。
4. 首次 Found 的物理步本来仍从 Search 状态开始。该步后，adapter 在下次 guidance 安装时退出；包括 HANDOFF_PENDING 在内的所有 Found 状态直接返回原 guidance/action 对象，恢复原 Executor 逻辑。

实际 baseline 已有 BSER standby guidance，并非所有情况下 Executor 都静止。本扩展是在开启开关时替换该 Search standby 行为。目标是三个 Searcher 的几何中位点，不是新增障碍物规划器；它不保证目标或直线路径位于无障碍区域。原碰撞处理保留，复杂地形下的可达性及耗能需要后续获准的 smoke 观察，本次不声称已验证。

BEDS 接入只读已有公开 PlanningStateView 的 agent 位置。Searcher observation 没有增加 Executor 信息，仍为 28D，action 为 3D，critic 为 124D；communication/control context schema 未改。未修改 Actor、Critic、reward、PathTracker、C2、训练脚本或 shared core。

## Diagnostics 与恢复

启用任一 BEDS 开关后，原 evaluator 的父进程额外写出：

- `early_discovery_diagnostics.csv`：scenario_id、episode、decision step、开关、candidate_count、mean_time_discount、top_candidate_changed、action_applied。正常 pre-action 记录为 true；终止步已计算但未执行动作的排序仍被汇总，标为 false。
- `early_discovery_candidates.csv`：每次 greedy 排序的 candidate_id、agent_id、original_bser_score、estimated_arrival_time、time_discount、final_score，另含 step、decision_index 和 ranking_round。
- `executor_standby_diagnostics.csv`：episode、pre-action step、物理 Executor position、standby target、距离、到三个 Searcher 的平均距离、归一化动作范数及权重来源。

所有 sidecar 都带 checkpoint、checkpoint episode、mode、execution variant、recovery variant。候选级记录的单位是一次排序评估；同一候选在不同 greedy round 的条件 gain 不同，不能把这些行当独立 episode。step 聚合的 candidate_count 是该步新发生的候选打分次数，没有进行新排序时为 0，mean_time_discount 为 NA。仅 standby 启用或 lambda=0 时不会伪造 candidate 打分记录。

这些记录是 episode 内的逐决策/逐步诊断，正式统计仍以 scenario/episode 为采样单位。父进程原子写入稳定表头，恢复时按 checkpoint/episode/step/decision/candidate 去重；配置开关或参数变化、已有 BEDS sidecar 缺失都会拒绝 resume。不同 variant 应使用不同输出目录。原 `episode_evaluation.csv` 额外携带明确的 BEDS 标识，避免把启用扩展的结果误认为 canonical OFF baseline。

## Linux smoke 命令（未执行）

使用原 native B1 episode-100 checkpoint 的实际路径替换占位符。新增配置默认 evaluation_episodes=10、max_steps=400；以下命令显式选择 full。没有改训练或 checkpoint 文件。

```bash
python -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints --config configs/chapter3/bser_phase1c_beds.json --checkpoint "<原native-B1-checkpoint绝对路径>" --beds-variant full --episodes 10 --workers 2 --device cpu --output-dir outputs/chapter3/phase1c_prrac/beds_full_smoke10_v1
```

其他三个组合将 `--beds-variant` 替换成 `baseline`、`early_only` 或 `standby_only`，并分别使用新的输出目录。不传该选项时，新配置保持两个开关关闭。没有生成任何正式实验结果或 performance_passed 声明。

## 文件范围

修改：`chapter3_bser/experiments/phase1c_prrac/runtime_factory.py`、`chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py`。

新增：`chapter3_bser/online/early_discovery.py`、`chapter3_bser/online/executor_standby.py`、`chapter3_bser/experiments/phase1c_prrac/beds.py`、两个上述 JSON configs、`tests/test_early_discovery_search.py`、`tests/test_executor_standby.py`、`tests/test_beds_evaluation.py` 和本文档。

前两轮离线分析的脚本、测试、报告和证据未修改。历史 outputs、27 条 source-provenance 记录及冻结 baseline hashes 未修改。

## 验证范围

单元测试覆盖 disabled score identity、lambda=0、时间折扣顺序、A* 长度优先、稳定 tie-break、候选不变、原 objective 保持，以及几何中位点、重复点、共线点、有效/无效权重、有限输出、动作 dtype/device 和关闭/Found identity。

集成测试使用临时合成权重及四步 worker，覆盖三个 active 组合、公开 guidance 角色边界、diagnostic 序列化/去重、CLI 分派，以及与修改前 HEAD evaluator 的 OFF payload 逐位一致。该 fixture 验证不等于正式 100 场性能或跨平台逐位一致性验证。

新增测试共 21 项已通过（14 项排序/几何动作单元测试，7 项 evaluation 接口与诊断测试，分批执行）。OFF 对照基准是修改前 commit `4f38028689b4594db7ff9cdab4c741de71df4ffd`。`python -m compileall chapter3_bser scripts tests` 已通过；`git diff --check` 及新增文件空白检查通过。canonical 输入 bundle 与两轮历史离线输出的 SHA-256 复核均保持不变。

110 个相关回归模块共执行 202 项测试，耗时 2871.219 秒：200 项通过、1 项失败、1 项错误。该集合包含上述 14 项新单元测试，不应重复相加。27 条 provenance、28D contract 以及其余选定 BSER / B1 / C2 / PRRAC 检查通过；整轮回归不能标为全绿。

两项未通过均已核对为修改前 HEAD 中存在的问题：

- `tests.test_bser_event_detection` 期待 `len(BSEREvent)==8`，HEAD 中枚举实际已有 9 项；测试和枚举文件均与 HEAD 内容一致。
- `tests.test_bser_v1_artifacts_frozen` 从 HEAD 读取冻结清单中的 `experiments/chapter3/bser_e1_offline/aggregate_by_profile.csv` 时失败。HEAD 的原冻结清单中有 19 个路径不在 HEAD Git tree 中；测试和清单文件均与 HEAD 内容一致。

没有为通过测试而改枚举、历史清单、哈希或测试断言。这两项历史问题仍未解决。上述均为本地验证，不是 CI；合成测试不替代 10 场 smoke 或正式实验。
