# Searcher Residual–Waypoint Branch Sensitivity

这是显式调用的局部分支诊断，不是策略、训练或完整 episode 性能评测。实际修改仓库为 `AUV-Thesis`；Linux 使用者在自己的 `AUV-Thesis-D1D2` 根目录运行下列命令。历史目录、checkpoint 和配置均只读；不会生成新场景或自动启动下一项实验。

## 实现与控制路径

仅在 `evaluate_prrac_checkpoints._evaluate_episode_job` 增加默认 `None` 的 `branch_diagnostic` 参数。普通 evaluator 不传参数时不调用新 hook，原动作和终止路径保持不变。

独立 helper `residual_branch_diagnostic.py` 在原 Actor forward、residual mode、continuity action adapter 之后，`env.step(actions)` 之前复制并缩放指定 Searcher 的 residual **command**。未选 Searcher 和 Executor 的 command 保持位级不变。alpha=1 直接返回原 tensor；anchor 前也返回原 tensor。未修改 runtime 的 `_residual_scale`，也未修改 Actor、Critic、PathTracker、BSER、C2、reward 或训练模块。

原环境继续执行 command clipping、物理单位转换、既有 residual scale、prior 合成、加速度 clipping、碰撞与动力学。诊断 runner 不计算或替换 final action。raw residual 仍取原 forward 输出；applied residual 为本步实际送入原环境的 command。`full_applied_residual_*` 额外保留缩放前 command。`prior_action_*`、`final_action_*` 读取原环境本次转移缓存，单位是物理加速度，不能与 dimensionless command 直接相加。

“Executor 不变”指从不对其 command 做诊断缩放，也不改其正式控制逻辑；不同 alpha 后续改变公共状态时，原控制流程自然产生的 Executor 输出不保证跨分支相同。单元测试另核验同一 anchor 首次干预的 Executor 物理 action 不变。

alignment 和 PRE_FOUND waypoint_switch_event 来自原 SearchContinuityDiagnostics；零范数/无有效定义填 NA。`path_cursor_advance_event` 另记转移前后 cursor 的差异，不替换原 switch 公式。POST_FOUND 若无原 alignment，则填 NA。

## 同状态与来源核验

1. 必须提供成功完成的正式 15-scenario trace，不能用 smoke trace 当正式来源。
2. 恢复两份原 canonical 100-scenario manifests，复用前一诊断的严格配置/场景/配对 provenance 核验。来源 CSV 的 discordant 集合必须正好为 5 help、10 hurt，且与 paired divergence CSV 及 trace manifest 选定集合完全一致。
3. 当前 checkpoint 路径与 SHA256、OFF config SHA256、历史文件及 trace 文件 hash 必须与 trace manifest 一致。继续 CPU、native B1、C2、max_steps=400、explore=false、training_update=false。
4. 每个场景重新从原 action trace 计算第一次 nav/cursor 差异，核对 paired_first_divergence.csv；两个值皆 NA 时拒绝。最早时刻全部 tie agents 都保留。anchor 为最早离散分支 step 减一、下限为零；不参考 0.5m 位置分叉。
5. 全部 15 场验证后才能 smoke 切片，保留原场景内容、scenario seed 和 canonical episode index。不生成、重新编号或采样场景。
6. 每个 alpha 从 step 0 以完整 full_prrac 重放，干预前所有动作相同；不 deepcopy/restore 未验证的运行时对象。

`anchor_state_hash` 在当前 Actor 输出已产生、任何 alpha 缩放之前取得。复用现有严格只读 `runtime_fingerprint`，包括环境动力学/地图状态、wrapper、provider 缓存、controller 当前 allocation、allocator、bridge/PathTracker、C2、scorer 公共状态、context、观测、installed guidance、action adapter、Python/NumPy/Torch 与可枚举环境 RNG。额外 hash 显式 scenario/step、四 agent 位置、导航目标、三个 cursor、路径 endpoint/hash、C2、阶段、公共 observation、raw residual、full applied residual，以及原纯函数 `_compute_waypoint_prior_acc` 的当前 preview 和缓存 prior。Preview 不更新缓存、不重新实现 prior 公式。

manifest 保存每个 alpha 的分组件 hash、inventory 与显式 exclusions；墙钟时间和已有 reporting-only 状态沿用原 fingerprint 的排除清单。未知不可序列化类型直接失败，不能默默忽略。该 hash 是对已列举状态的验证，不声称证明未知/排除状态也完全相同。真实目标若在环境 inventory 中，只进入 audit hash，不馈入 policy/BSER。

## 十步与 alpha=1 control

干预转移为 `[anchor, anchor+10)`，严格十次；在原 controller/PathTracker 完成第十次正常转移后终止。环境 max_steps 参数仍为 400，不改场景或 dynamics；这不是 75 个完整 400-step episode。

step trace 每个 scenario×alpha×Searcher 包含十条 `sample_kind=transition`，再加一条 `sample_kind=endpoint`（relative_step=10）。endpoint 仅记录状态，动作、alignment、collision/found transition event 等不存在字段填 NA，不执行第十一次动作。

“immediate”指 `anchor+1`：第一条干预动作可以影响的下一次正常 PathTracker guidance 状态。`position_separation_at_horizon` 是 endpoint 三个 Searcher 中最大的同 agent 欧氏距离；step trace 保留所有 agent 的距离。

alpha=1 必须复现原 R0 在 anchor 到 anchor+10 的所有可要求状态（位置、导航目标、cursor、C2）；十条转移的 collision 与 mission found event 也核对。位置/导航采用欧氏绝对容差 1e-6；cursor、布尔值及 C2 tier 精确一致。历史 PRE_FOUND action trace 若不足以覆盖所需 endpoint，记录缺失并失败，不插值或补造 POST_FOUND 状态。其它 alpha 提前终止/窗口不足也失败。

每场所有 alpha 的 anchor hash 必须相同；否则或 control 失败，该场 `interpretation_allowed=false`、分类 unresolved、解释指标 NA，manifest 状态 failed。离线分析拒绝 failed manifest。alpha=0 只是同一 R0 anchor 上指定 agents 的局部十步关闭，不是从 episode 开始关闭三个 Searcher 的历史 R1。

## 分类与解释

比较所有三个 Searcher 的 `(waypoint_cursor, navigation_target)` 联合类别；导航等价容差为 1e-8，与原首次分叉分析一致。它是可观察导航分支定义，不凭它推断内部某个 predicate 已被证明触发。

- stable：完整 H 窗口内网格上所有 alpha 的联合分支相同。
- immediate_threshold：最早差异出现在 anchor+1，且观测支持两类别单次切换。
- delayed_threshold：稍后才出现上述切换。
- non_monotonic：任意观测 step 的 alpha 类别序列离开某类别后又返回，如 A/B/A；不报告 threshold 区间。
- unresolved：缺失/失败或出现超过两个无自然顺序的类别，不能强加单调数值顺序。

`alpha_switch_interval` 仅报告首次分叉 step 的相邻采样 alpha 区间，且整个观测窗不存在已见非单调/多类别歧义；不是连续空间精确阈值。分类中的 non_monotonic 优先于 immediate/delayed。smoke 分类只适用于其记录的 alpha 网格。

help/hurt 输出场景数、分类计数/描述性比例、切换区间、首次离散分叉、各 alpha endpoint 距离，以及 `smallest_sampled_alpha_reduction_with_discrete_divergence`（已采样的最小 `1-alpha`，不外推）。分组均值等权对待场景，缺失与有效数同时输出。15 个事后选择案例不是独立统计验证集；不输出显著性或 causes failure 判断，负对齐也不是有害标签。

## 输出与写入安全

runner 输出 `branch_sensitivity_manifest.json`、`selected_anchors.csv`、`branch_sensitivity_step_trace.csv`、`branch_sensitivity_scenario.csv`，另有父进程更新的 `progress.json`。worker 只返回内存结果；复用 spawn collector，父进程统一原子写 CSV，最终按 scenario_id/alpha/step/agent_id 排序。输入开始与结束核对 SHA；非空输出拒绝，不能写入历史/trace 输入目录或其祖先。

分析器为纯标准库，无 torch、环境或 evaluator 导入；验证输出 hash、完整键集合、同状态标记，并从逐步行重新核验分类。显式生成 `analysis/branch_sensitivity_help.csv`、`branch_sensitivity_hurt.csv`、`branch_sensitivity_summary.json`。runner 不自动运行分析。

## Linux 命令（全部单行）

替换 checkpoint 为原 trace manifest 中的绝对路径；若原输出不在下列路径，替换所有对应路径。smoke 不改变原 15 场输入核验，只减少实际 local branch runs。

Smoke：1 help＋1 hurt，alpha=0,0.5,1，共六个局部分支，由用户执行：

```bash
python scripts/run_searcher_residual_branch_sensitivity.py --source-root "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2" --trace-root "searcher_residual_trace15_v1" --paired-first-divergence "searcher_residual_trace15_v1/analysis/paired_first_divergence.csv" --checkpoint "<原checkpoint绝对路径>" --config "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/searcher_residual_off/resolved_evaluation_config.json" --workers 4 --device cpu --max-per-transition 1 --alphas 0 0.5 1 --output-dir "searcher_residual_branch_sensitivity15_smoke_v1"
```

正式：5 help＋10 hurt，固定五个 alpha，共 75 个局部分支，由用户执行：

```bash
python scripts/run_searcher_residual_branch_sensitivity.py --source-root "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2" --trace-root "searcher_residual_trace15_v1" --paired-first-divergence "searcher_residual_trace15_v1/analysis/paired_first_divergence.csv" --checkpoint "<原checkpoint绝对路径>" --config "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/searcher_residual_off/resolved_evaluation_config.json" --workers 4 --device cpu --output-dir "searcher_residual_branch_sensitivity15_v1"
```

离线分析（仅接受 completed manifest）：

```bash
python scripts/analyze_searcher_residual_branch_sensitivity.py --input-dir "searcher_residual_branch_sensitivity15_v1" --output-dir "searcher_residual_branch_sensitivity15_v1/analysis"
```

验证命令：

```bash
python -m compileall scripts tests chapter3_bser
```

```bash
python -m unittest tests.test_searcher_residual_branch_sensitivity -v
```

以上实验命令只是交付给用户，未由 Codex 执行。单元测试使用合成输入，不能视作真实 checkpoint 的 alpha=1 reproduction 已通过。

## 最新记录的本地验证（非 CI）

新增 23 项测试分批通过：最后一轮完整模块当时包含 22 项（325.477 秒），随后覆盖新增 authoritative-norm 断言的 22 项非 native 测试也通过（10.136 秒）。两批合计覆盖当前 23 项不同测试。已有 residual/off、continuity、paired evaluation、checkpoint evaluator 的 17 项相关回归通过；额外双模式单步 trace no-op 回归通过（291.443 秒），共 18 项既有测试。

Native 夹具用一条三步合成 R0 和三条两步合成重放验证相同 anchor hash、alpha=1 复现、anchor 前 action、局部 alpha command、物理 residual/clip 和 Executor 首次 action。H=10 输出、分类及正式 75 份 job 的组装/写入使用 mock 合成数据测试，没有执行真实 checkpoint 的六分支 smoke 或 75 分支实验。

`python -m compileall scripts tests chapter3_bser`、`git diff --check` 及新增文件 whitespace 检查通过。Actor/Critic、core、PathTracker、BSER online 与 C2 目录无 diff。未执行训练、100 场实验、正式 smoke/75-run、commit 或 push。
