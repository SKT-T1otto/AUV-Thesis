# BSER-RMADDPG 接口设计

## 1. 当前 BSER 输出的真实形态

BSER 是高层分配器，不输出速度或加速度。核心类型位于 `chapter3_bser/online/types.py`：

- `OnlineAllocation.search_assignments
  - `agent_id
  - `candidate_id
  - `waypoin
  - `path` / `path_cell_indices
  - travel time、planning cost、failure reason
- `OnlineAllocation.executor_assignmen
  - `executor_id
  - `target_region
  - `path` / `path_cell_indices
  - ETA、source、reachable、planning cost、failure reason
- `OnlineAllocation
  - objective、detection probability、response time
  - trigger reason、status、search frozen、allocation hash
- `BSERActionAssignmen
  - 本步事件、是否重规划、当前 allocation、waypoint updates、decision reason、diagnostics

用户提出的五类接口语义与当前类型的对应关系如下：

| 期望语义 | 当前来源 | 结论 |
|---|---|---|
| agent assignment | search/executor assignment | 已显式提供 |
| waypoint | search `waypoint`；executor `target_region`；实际跟踪点可由 `path` + `PathTracker` 得出 | 已显式提供 |
| target priority | allocation 只有全局 `detection_probability`，没有逐 agent 显式 priority | 若方案 A 需要，只能由适配器基于公开 belief 在已分配位置取值并归一化；不得改 candidate generator |
| standby state | executor `source`、`reachable`、target region 和公开任务阶段可推导 | 建议由接口适配器显式编码，不回写 BSER 内部类型 |
| execution request | 没有同名字段；可由公开 mission context、事件和当前 executor assignment 推导 | 建议由接口适配器生成只读布尔值 |

建议新建不可变、版本化的边界对象，例如 `BSERControlContextV1`，统一携带 allocation hash/version、任务阶段、每 agent assignment、跟踪航点、reachable/hold 状态和 execution request。该对象属于 Phase 1C 适配层，不改变 BSER 的冻结数据结构。

## 2. 调用频率结论

对 `phase1b1_pilot/run_pilot.py` 和 `phase1b_online/run_e2.py` 的调用链核对结果：

- episode 开始：`OnlinePlanningStateProvider.initialize()` 一次，`OnlineBSERController.initialize()` 一次，初始 allocation 一次。
- 每个环境 step 之后：`provider.snapshot(...)`，随后 `controller.step(...)`。
- `controller.step` 每步运行事件检测，但不会每步运行完整 BSER 优化。
- 只有出现 actionable event、通过相关事件过滤/cooldown 后才会请求对应的 full/partial allocation；该候选仍可能被 atomic/hysteresis 拒绝，拒绝时不提交新 allocation。
- episode 级别只负责初始化，不是 episode 结束后才规划。
- planning state 的“完整地图刷新频率”和“BSER 优化频率”是两个概念。状态提供器可在 transition、forced refresh 或配置间隔时完整刷新，其他步只更新公开运动/任务状态。

Phase 1C 应保持相同时间尺度：

1. 事件检测每环境步一次。
2. BSER 仅在初始化和有效事件路径上计算新 proposal。
3. 当前 allocation 在两次事件之间缓存并持续生效。
4. RMADDPG actor 每环境步输出一次低层残差动作。
5. 仅当 allocation hash/version 改变时提交新任务分配；PathTracker 可每步更新“当前路径跟踪点”，但不得修改其机制。

## 3. 方案 A：BSER 作为额外 observation

结构：

```tex
local observation (28D)
  + BSERAssignmentEncoder(context_i)
  -> augmented observation
  -> RMADDPG actor
  -> residual action


一个可审计的候选编码是每 agent 追加 14 维 `BSER feature v1`：

| 字段 | 维度 | 来源 |
|---|---:|---|
| assignment kind one-hot：search/executor/hold-or-unassigned | 3 | allocation + mission phase |
| 归一化 waypoint delta | 3 | tracking waypoint - current position |
| 归一化 waypoint distance | 1 | 同上 |
| assigned belief priority | 1 | 公开 belief 在已分配位置的取值；适配器派生 |
| standby flag | 1 | 适配器派生 |
| execution request flag | 1 | 公开 mission context + allocation 派生 |
| reachable flag | 1 | assignment/path result |
| allocation changed this step | 1 | allocation hash/version |
| normalized allocation age | 1 | 适配器状态 |
| normalized planning/path cost | 1 | assignment planning cost |

维度影响：

- actor 输入：`28 + 14 = 42
- actor 输出：仍为 `3
- centralized critic 输入：`4 × 42 + 4 × 3 = 180
- replay observation：每 agent 从 `28` 变为 `42
- 旧 checkpoint 不可直接加载；必须新建模型、replay 与 checkpoint namespace

实现边界：必须使用 observation wrapper/decorator，不应修改冻结的 28D 基础契约或直接把 BSER 字段塞入 `UAVEnv._get_obs`。编码器只能读取公开状态与已产生的 allocation，不能改变 candidate、event 或 allocator 结果。

主要风险：

- policy 可以忽略或对抗 BSER assignment，BSER 的高层约束退化为“提示”。
- assignment 更新是事件驱动的，而 observation 每步写入 replay；若不记录 allocation version/age，训练数据难以解释。
- 网络、replay、checkpoint、两条 runtime builder 和所有 shape 测试都需要同步修改。
- 逐 agent target priority 当前不是原生输出，派生规则需要单独锁定，否则会形成新的未声明算法机制。

## 4. 方案 B：BSER 作为 high-level controller

推荐结构：

```tex
公开 belief / map / task / agent state
  -> OnlinePlanningStateProvider
  -> OnlineBSERController
     -> 初始或事件触发的 OnlineAllocation
  -> Phase1C BSER guidance bridge
     -> agent assignment + PathTracker tracking waypoint/hold
  -> 环境 navigation target / waypoint prior
  -> 现有 28D observation 中的 nav delta/direction/distance
  -> RMADDPG actor 每步输出 3D residual action
  -> UAVEnv: final acceleration = waypoint prior + learned residual


方案 B 不把 `assignment_to_fixed_actions` 接到 RMADDPG 前面。该函数是 Phase 1B 固定反馈控制实验的执行器，会直接生成完整归一化动作；Phase 1C 应复用其“allocation/path 到跟踪目标”的语义，而不是复用完整动作输出。

### 建议边界 API

Phase 1C 适配层至少需要：

```tex
compile(allocation, planning_state, mission_context)
  -> BSERControlContextV1

tracking_targets(context, public_agent_state)
  -> {agent_id: target_or_hold_position}

install_guidance(targets, allocation_version)
  -> 无动力学推进、无奖励、无任务状态变化

observe_after_guidance()
  -> 与当前状态和新导航目标一致的 28D observations


`install_guidance` 必须是显式公共接口；不能让训练循环直接写 `env._nav_targets`。接口应是通用“外部高层导航指导”，BSER 依赖留在 `chapter3_bser/integration/`，避免把 Chapter 3 特定类型引入 core 环境。

### 时序约束

当前 `env.step` 在返回 `next_observation` 前就已构造 observation，而事件检测发生在 step 之后。因此首版实现必须处理以下顺序：

```tex
env.step(residual_action)
  -> public state snapsho
  -> controller.step
  -> 若 allocation/跟踪点变化，install_guidance
  -> 重新读取 observation（只重算 observation，不推进环境）
  -> 写入 replay 的 next_observation


reset 后也应先初始化 BSER、安装初始 guidance，再取得训练使用的首个 observation。否则 replay 中会出现“observation 导航目标”和“实际下一步 prior 导航目标”不一致。

### 维度影响

首版方案 B 保持：

- observation：每 agent `28
- actor：`28 -> 3
- centralized critic：`124 -> 1
- replay shape：不变

原因是 28D 观测已经包含 navigation target delta、direction 和 distance；当 BSER guidance 成为当前导航目标后，低层 actor 已能看到执行所需的局部目标几何信息。assignment priority 等高层信息继续由 BSER 决策，不需要重复交给低层 actor。

## 5. 两方案比较

| 维度 | 方案 A：额外 observation | 方案 B：high-level controller |
|---|---|---|
| 修改代码量 | 中到高；网络输入、replay、builder、checkpoint、shape tests 全部变化 | 中；新增 bridge/runner，并提供受控 guidance API；算法网络与 replay 可不变 |
| 理论一致性 | 中；BSER 只是可被忽略的策略输入 | 高；BSER 明确负责离散任务分配，RMADDPG 明确负责连续低层控制 |
| 实验可解释性 | 中低；失败难区分是 BSER 信息无效还是 policy 未使用 | 高；可分别统计高层 allocation、事件重规划、prior 与 residual 贡献 |
| 与第三章创新匹配 | 中；更像 feature augmentation | 高；直接体现 BSER 搜索-执行优化与 RMADDPG 控制的分层协同 |
| 观测/网络维度 | 28→42，critic 124→180（按 14D v1） | 28、3、124 全部不变 |
| 旧模型兼容性 | 网络结构不兼容 | 结构兼容，但 Phase 1C 仍应使用独立 checkpoint/实验命名 |
| 破坏 Phase 1B 风险 | 较高，若误改基础 observation | 低，前提是新功能默认关闭且 runner 独立 |

## 6. 推荐结论

推荐方案 B 作为 Phase 1C 主方案。

它与现有“waypoint prior + RMADDPG residual”动作语义天然对齐，也能保持 28D observation、3D action、124D centralized critic、replay schema 和冻结 BSER 组件不变。方案 A 可保留为后续消融项，但不应作为首个集成版本；在方案 B 的 smoke test 与训练管线稳定前，不应引入 14D 扩维。

## 7. Phase 1B 隔离要求

- 不修改 `chapter3_bser/experiments/phase1b*` 下的正式实验运行器与配置。
- 不修改 `objective.py`、`greedy_solver.py`、`candidate_generator.py`、`exact_solver.py`。
- 不修改 EventDetector 规则、ReplanningPolicy/cooldown、allocator 决策、PathTracker 或 WaypointManager。
- 新功能通过独立 Phase 1C method/config 开启，默认关闭；旧方法构建出的环境与 runtime 必须保持原路径。
- Phase 1B 固定控制器 `assignment_to_fixed_actions` 不应被替换或改写。
