# BEDS 第二阶段：Safe Executor Standby / Handoff-ready Standby

基准 commit：`4f11e65e41147e5ced953be0c70c5b0eab13dde4`。本阶段仅实现安全导航接入和诊断，未运行 simulator smoke、训练、用户 checkpoint 或正式实验。本地单元测试不能证明 Linux 两场一定无碰撞，也不建立成功率提升结论。

## 根因与导航审计

旧 `beds.py` 将 `planned_path=()`、`tracking_waypoint=raw target` 强行写入 guidance，再在 `before_action` 用 `clip(gain*(target-current))` 覆盖 Executor residual。这样既跳过了原路径追踪，也在 waypoint prior 之外增加了没有速度阻尼的推进项。该动作是归一化残差加速度，不是物理速度；距离近并不意味着速度或接管状态合适。

已核对以下现有链路，均未修改：

| 链路 | 本阶段复用与限制 |
| --- | --- |
| `core/mapping/path_planner.py` / `OnlineUnknownMapTaskPlanner` | 只使用在线已知 occupancy 的 endpoint/segment 语义；每周期用其原 segment API 核验保留路径与当前位置到 tracking target 的连接。该实现按 occupied cell 判定，原方法忽略传入的 clearance 数值，本阶段不新增或承诺连续空间裕量。 |
| `core/mapping/planning_state.py` / `TravelCostService` | 需要重规划时单独提取公开快照，复用原 Dijkstra/A*、有效 mask、endpoint connectors 和路径成本。没有另写 A*。 |
| `controllers/path_tracker.py` | 为 BEDS 创建独立实例，消费完整 planned path，保持原桥接器 tracker 状态不变。 |
| `integration/rmaddpg_bridge.py` / `GuidedEnv` | 仅覆盖 Executor assignment；保持 Searcher assignment 对象、28D/3D/124D、communication schema 不变。GuidedEnv 安装 tracking/hold target。 |
| `UAVEnv._compute_waypoint_prior_acc` | 既有 prior 按 XY/Z slow radius 产生期望速度，再用当前速度反馈产生加速度；无需新 P 控制器。 |
| `execution_continuity/action_adapter.py` | 原 B1/B3 和 Found 后 action 路径保持不变；BEDS 只在未 Found 的 Search 阶段作用。 |
| C2 local connector | C2 的 Searcher controller、碰撞触发与原 segment audit 未改。复核了公开图重连能力；Standby 起点无合法 connector 时采用 hold，没有强行调用 Searcher recovery 控制器。 |
| PSE standby | 复用现有 PSE update interval 的配置来源。PSE repair 内部调用 `is_inside_obstacle`，且包含不同 objective 的 gain/hysteresis，不能直接复用其控制逻辑。 |
| Found intercept | 原 native B1 controller → bridge → guidance/prior/action 路径保留。Standby 不接入该阶段。 |

## raw target → safe target → local target

`compute_standby_target()` 的几何中位点算法不变。没有可靠逐 Searcher 价值时仍使用明确等权，称为 `raw_standby_target`。

新增 `safe_executor_standby.py` 的薄封装先请求原 TravelCostService 到 raw target 的路径。raw target 不满足 endpoint connector/有效性或 Searcher 间距时，在公开图内按以下固定顺序修复：距 raw target 最近、原 planning cost 最低、cell index 最小。只保留原 public valid mask、single-source reachable mask 中的点，再以原 A* 验证路径，同时排除与当前 Searcher 的距离小于环境 `safe_dist` 的终点。

现有 `assign_reachable_public_proxy` 要求严格向 intercept 目标取得进展，也不接受动态 Searcher 排除条件，所以没有直接调用；新封装只扩展终点筛选，底层图查询完全复用。无法得到路径时设置 current-position hold，不将 raw target 装成直达目标。

仅在 update interval 已到且 raw target 相对上次接受/尝试的 raw target 移动达到 threshold 时更新正常目标。首次调用、到期后的无路径重试例外；已知地图使保留路径失效时立即 hold，等 interval 到期再规划。小位移保持原路径与 tracker index。不会逐步执行 A*。

重规划快照独立于 BSER 的 OnlinePlanningStateProvider，不改变 Searcher/early 的 provider 刷新节奏。每周期的路径安全检查直接调用 live unknown-map planner 原 segment API，避免依赖 BSER 缓存 occupancy 的刷新周期。

## 动作边界与减速

正常控制链是 `safe target → planned_path → PathTracker local target → guidance → GuidedEnv → 原 waypoint prior + 原 residual → 原 dynamics`。远离局部目标时不替换动作；near-target residual 仅按原 XY/Z slow radius 缩小，保持方向与 dtype/device。减速与速度阻尼本身由原 waypoint prior 完成。

进入环境 `executor_hold_radius` 且 tracker 已到最后一段，或当前至 local target 的线段违反 `safe_dist` / public-map 安全检查时，guidance 进入 current-position hold，并仅将 Executor residual 归零。这是必要的最小 action safety gate：原 `GuidedEnv` 的 hold flag 只安装位置目标，并不会屏蔽 learned residual。若不屏蔽，prior 已经减速后仍可能受到非零 residual 推进。不是给最终 raw target 生成新的方向动作，也没有改 Actor 或 Searcher residual。

动态检查仅使用当前 Searcher 位置到当前 Executor→local-target 线段的距离，复用原 PathTracker point-segment distance 与环境 `safe_dist`；不预测未来轨迹，不移动 Searcher 路径。已有过近状态也会 hold，等待 Searcher 离开。hold 通过 prior 对当前速度反馈制动，不写 `_agent_vel`。该机制可能保守地等待；未探索障碍、惯性和 residual 导致的实际风险仍需用户 smoke 检验。

每次物理步结束后先锁存 diagnostic，再在 `target_found` 首次为 true 时清空 raw/safe target、path、tracker index、residual factor 等全部 Standby 软件状态。包括 HANDOFF_PENDING，下一周期的 guidance/action 返回原对象；物理位置、速度连续，不 teleport、不 reset velocity。

## 配置与身份

仍使用 `bser_phase1c_beds.json`，两个机制默认 false。新增 `update_interval=10`（现有 Chapter 3 PSE interval）和 `target_shift_threshold=0.75`（现有 public target update distance）。旧 `{enabled,gain}` 配置可加载；开启时自动解析上述缺省值，写入 resolved evaluation config。`gain` 保留用于旧配置兼容和记录，新安全链不再用它生成 P 动作。

physical 参数从实际 runtime 只读绑定：`safe_dist`、`prior_slow_radius_xy/z`、`executor_hold_radius`；tracking threshold 继承正式 Phase 1B.2 的配置。其实际值随每条 Standby 诊断记录，避免重复定义物理阈值。只有 active standby 才绑定 unknown-map planner，并要求原 Executor prior 已启用；不满足时在运行前报错。

active standby 的 resolved config 和 evaluation manifest 记录 `safe_public_path_v2` control revision 与解析后参数。scenario manifest identity 仍用于同场景配对，不将控制参数混入 scenario ID。不同 revision/参数不得恢复到旧输出；新增 found-state sidecar 纳入存在性、resume 和原子写入检查。

`standby_diagnostics_enabled=true` 是新模板的只读观测开关，使 baseline/early-only 也输出 Found 速度和 per-agent collision 作为比较参照。关闭它不改变控制；旧配置没有该字段时，standby 开启默认记录，关闭则不新增 Standby 记录。baseline 不被标成 BEDS-PRRAC。

## 诊断定义

`executor_standby_diagnostics.csv` 为每个物理动作一行，`step` 为动作前时刻、`transition_step` 为对应动作后时刻。记录 raw/safe target、更新事件/原因/位移、路由可用性/长度/点数、local target、Executor 位置/速度/速率、目标距离、动作和动作范数、减速/hold、最近 Searcher 距离、C2 是否有 active Searcher、Found 前后状态、实际参数等。

`target_update_event` 表示该周期尝试生成新目标/路径，失败原因可为 `NO_SAFE_PUBLIC_ROUTE`；不表示尝试一定成功。动作字段是**送入 env.step 的归一化 residual**，并非 prior 加 residual 后的物理总加速度。hold 时 residual 为 0 仍可能存在自然速度与 prior 制动加速度。post-Found 的 standby 路径等字段为空，`standby_active=false`。

collision 来自 `get_agent_state().collision_flags`，不改环境判定：

- `searcher_collision_flags` 与三人的累计 collision-bearing transition 数。
- `executor_pre_found_collision_count`，以动作前 `target_found=false` 归类，因此首次触发 Found 的物理步计入 pre-Found。
- `executor_post_found_collision_count`，以动作前 `target_found=true` 归类。
- `executor_collision_event` 是该步真实 flag，不是 rising edge；连续 true 的多步逐步计数。agent-agent 间距违规与环境障碍 collision flag 不是同一指标。

缺失的字段输出 NA。若中途缺失 collision flags，受影响的累计计数从此保持 NA，不把不完整历史当成零。

新增 `executor_standby_found_state.csv`，每个 Found episode 只锁存一次：

- 原正式 `executor_distance_to_target_at_found` 的只读副本，不在 Standby 控制中读取真实 target。
- 物理动作后 `executor_velocity_at_found`、`executor_speed_at_found`。
- 触发 Found 的那次动作的 `executor_action_norm_at_found`。
- 相对动作前 safe standby target 的 `standby_distance_at_found`。
- `min_executor_searcher_distance_at_found`、`standby_active_before_found`。

未 Found episode 不伪造 Found 行。所有 sidecar 保留 checkpoint/mode/B1/C2 身份字段。Early diagnostics 的五个原字段与数学语义不变。正式比较仍以 episode 为采样单位，不能把逐步行当独立 episode。

## 本地验证

针对性测试共 71 项通过（分批执行），包含 28 项新安全 Standby 测试、14 项已有 early/geometric-median 测试，以及 PathTracker、C2 public segment/local connector/reconnect、evaluation 指标与信息边界、BEDS 接入和 27 条 provenance 检查。Early 源文件对照指定 commit 未变；Early-only adapter 的 guidance/action 对象及排序诊断对照旧版一致。原 planner 返回无路由或查询抛出 RuntimeError/ValueError 时均进入 fail-safe hold。

`python -m compileall chapter3_bser scripts tests` 和 `git diff --check` 通过；历史 canonical 输入与两轮离线输出共 68 条 SHA-256 校验通过。上述为最新本地验证，不是 CI。

未运行两项旧的四步 simulator worker 测试，也未运行任何 2/10/100 场 smoke；本次使用纯公开图、tensor prior 函数及 mock 接入测试。disabled 的 action/guidance identity 已验证，未宣称同两场物理 rollout 的 bit-level 等价已重新验证。Linux 配对 smoke 仍须按用户验收标准验证 baseline、early-only、pre-Found Executor collision 和 Found 速度。

## Linux 同两场四组命令（仅提供，未执行）

在 Linux AUV-Thesis 仓库根目录执行下列一行。先核对指定 checkpoint SHA-256 与配置中的 scenario_seed=1729，再运行 episodes=2；每次使用时间戳/PID 新目录，任一组失败即退出。scenario 生成代码未修改，使用同一 seed 的原前两场。若旧 smoke 使用过另一个自定义 scenario 子集，则须改用其原 scenario-id-file，而不能仅依赖 seed。

```bash
bash -lc 'set -euo pipefail; ckpt="/home/legion/AUV-Thesis/baselines/AUV-Thesis-SearchValue/outputs/chapter3/phase1c_prrac/s2b_search_value_pilot_seed1_ep100_v1/checkpoints/phase1c_prrac_episode_0100.pt"; printf "%s  %s\n" "abc7e1c885cc57455f47683d0cb177bacd22706ab65706c64f26e6425e023a0b" "$ckpt" | sha256sum -c -; python -c "import json; c=json.load(open(\"configs/chapter3/bser_phase1c_beds.json\")); assert c[\"scenario_seed\"] == 1729"; out="outputs/chapter3/phase1c_prrac/beds_safe_pair2_v2_$(date -u +%Y%m%dT%H%M%S)_$$"; test ! -e "$out"; for variant in baseline early_only standby_only full; do python -m chapter3_bser.experiments.phase1c_prrac.evaluate_prrac_checkpoints --config configs/chapter3/bser_phase1c_beds.json --checkpoint "$ckpt" --beds-variant "$variant" --episodes 2 --workers 2 --device cpu --output-dir "$out/$variant"; done'
```

验收时按 scenario_id 对齐四组，分别统计每组 Executor pre/post-Found 和三名 Searcher collision；同时查看 Found speed/action norm/standby_active_before_found。2 场不要求 Success 提高，也不足以作性能结论。

## 文件范围

修改：`online/executor_standby.py`（仅兼容配置/说明，几何中位点未改）、`phase1c_prrac/beds.py`、`phase1c_prrac/evaluate_prrac_checkpoints.py`、`configs/chapter3/bser_phase1c_beds.json`、`tests/test_beds_evaluation.py`。

新增：`online/safe_executor_standby.py`、`phase1c_prrac/standby_diagnostics.py`、`tests/test_beds_safe_standby.py`、本文档。旧 BEDS 文档作为第一阶段记录保留。本阶段未改 Early Discovery、Actor/Critic、reward、Searcher residual、C2、Search Value、检测/Found/Handoff、Found 后 Executor、通用 planner/PathTracker、checkpoint 或历史 outputs；未 commit/push。
