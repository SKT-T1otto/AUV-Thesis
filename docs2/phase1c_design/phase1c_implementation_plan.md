# Chapter 3 Phase 1C 实施计划

## 1. 目标与冻结边界

目标是把已冻结的事件触发 BSER 作为高层任务规划器，把现有 RMADDPG 作为每步低层残差控制器。首版采用方案 B，不改变 BSER 优化目标、候选规则、事件规则、cooldown、奖励、动作或任务成功语义。

当前基线结果仅作为冻结验收参考，本设计阶段未复跑实验：

- Success rate：`0.20`
- Found rate：`0.55`
- Executor invalid：`1336`
- Waypoint stale：`740`

## 2. Step 1：接口添加

### 2.1 新增 Phase 1C 层

建议新增：

- `chapter3_bser/integration/control_context.py`
  - 定义不可变、版本化的 `BSERControlContextV1` 和 per-agent guidance 数据。
- `chapter3_bser/integration/rmaddpg_bridge.py`
  - 读取公开 planning/mission state 与 `OnlineAllocation`。
  - 将 search waypoint、executor target/path、reachable/hold 状态转换为通用 tracking targets。
  - 只调用现有 `PathTracker`，不修改 PathTracker。
  - 用 allocation hash/version 保证重复提交幂等。
- `chapter3_bser/integration/guided_env.py`（如采用 wrapper）
  - 管理 reset/step 后的 BSER 时序和 observation 刷新。
  - 禁止直接从训练 runner 写环境私有字段。
- `configs/chapter3/bser_phase1c.json`
  - 独立 method/config 名、机制版本、输出和 checkpoint namespace。

### 2.2 环境最小公共能力

若 wrapper 无法完全通过现有公共 API 完成，需要在以下文件增加通用、默认关闭的接口：

- `core/env/mission_env.py`
  - 转发 `install_navigation_guidance(...)` 与只读 `observe()`。
- `core/env/uav_env.py`
  - 实现外部 guidance 的受控安装和无状态推进的 observation 重算。
  - 仅 Phase 1C 配置启用；旧配置行为必须逐位/逐指标保持。

接口验收条件：

- 安装 guidance 不推进 `step_count`，不计算 reward，不移动 target/agent，不更新 belief/event/cooldown。
- `observe()` 只反映当前物理状态与已安装导航目标。
- 同一 allocation version 重复安装无副作用。
- 不可达 executor 进入 hold，不允许直线穿障碍 fallback。
- `next_observation` 中的导航字段与下一步实际 waypoint prior 使用同一个 tracking target。

### 2.3 注册与配置

建议新方法名：`ch3_bser_rmaddpg_phase1c`。未来可能修改：

- `core/registry/experiment_registry.py`：只新增方法注册，不改变旧方法。
- `core/runtime/training.py` / `core/runtime/engine.py`：仅在需要通用 high-level hook 时增加可选 hook，默认 `None`。
- `core/runtime/builder.py`：按新方法注入 wrapper/bridge；方案 B 下保持 replay `(28,)*4` 和 action `(3,)*4`。

若 chapter-specific runner 能复用现有 `_run_episode` 而无需侵入 core，应优先把编排放在 `chapter3_bser/experiments/phase1c_bser_rmaddpg/`，减少对共享 runtime 的影响。

## 3. Step 2：single episode smoke test

先测试，不训练：

- 固定一个正式 profile 的单 seed、单 episode、较短 max_steps。
- 用未训练/指定 checkpoint 的 actor 做 rollout；smoke 目的只验证接口与时序，不解释性能。
- 验证 controller 初始化一次、event detection 每 step 一次、allocator 只在初始化/有效事件路径调用。
- 验证 allocation 未变化时 guidance version 不变。
- 验证 allocation 变化后，写入 replay 的 `next_observation[6:12,15]` 与新 tracking target 一致。
- 验证 actor 输出 shape `(4,3)`、值域 `[-1,1]`，critic 输入维度仍为 `124`。
- 验证 replay push/sample 完成且 observation shape 仍为四组 28。
- 验证 target found、executor public handoff、unreachable hold、episode done 四条边界路径。
- 输出每步 `allocation_version`、`replanned`、`decision_reason`、tracking target、prior/residual norm，便于诊断但不参与决策。

必要测试文件建议：

- `tests/test_phase1c_bridge_contract.py`
- `tests/test_phase1c_guidance_timing.py`
- `tests/test_phase1c_observation_shape.py`
- `tests/test_phase1c_phase1b_isolation.py`
- `tests/test_phase1c_single_episode_smoke.py`

## 4. Step 3：training pipeline

Smoke 全部通过后再接训练：

1. 新建独立的 `chapter3_bser/experiments/phase1c_bser_rmaddpg/` runner。
2. 使用新的 output、resume 与 checkpoint 目录；不覆盖 Phase 1B 或现有 RMADDPG 结果。
3. 每条 replay transition 必须满足：obs 所表达的 guidance 与生成 action 时一致，next_obs 与下一步 guidance 一致。
4. 记录 high-level 与 low-level 两套指标：
   - BSER event/optimizer/proposal/accept/reject 次数；
   - allocation version、任务阶段、waypoint switches；
   - prior norm、residual norm、residual contribution ratio；
   - found/success、碰撞、路径距离、能耗与响应时间。
5. checkpoint metadata 增加 integration schema/version、BSER mechanism version、observation schema 和配置 hash。
6. 训练启动前加入 dry-run/preflight；本设计阶段不启动任何训练。

方案 B 网络结构虽与旧 RMADDPG 一致，正式实验仍建议从明确记录的初始化策略开始，并使用 Phase 1C 独立 checkpoint。是否允许 warm-start 应作为单独实验因素，不能与主结果混用。

## 5. Step 4：baseline comparison

最低比较组建议：

| 组别 | 高层规划 | 低层控制 | 用途 |
|---|---|---|---|
| 当前 RMADDPG/PSE-RMADDPG baseline | 现有导航机制 | RMADDPG residual | 学习基线 |
| Phase 1B.2_corrected | Event-BSER | 固定反馈 action adapter | 冻结规划/执行基线 |
| Phase 1C 主方法 | Event-BSER | RMADDPG residual | 验证分层集成收益 |
| 可选消融：BSER prior only | Event-BSER | residual=0 | 分离高层先验贡献 |
| 可选后续方案 A | BSER observation augmentation | RMADDPG | 只在主方案稳定后比较接口形式 |

所有比较必须锁定 profile、scenario manifests、seeds、episodes、max_steps、环境语义和统计口径。Phase 1B 的四个冻结指标应先通过隔离回归验证；若发生变化，停止训练并定位隔离失败。

## 6. 风险与控制

| 风险 | 后果 | 控制 |
|---|---|---|
| 把 `controller.step` 误当成每步 optimizer | 计算频率和算法语义改变 | 分别计数 detector calls、optimizer invocations、accepted replans |
| env.step 返回的 obs 早于 BSER guidance 更新 | transition 语义错位 | guidance commit 后无推进地重算 next observation |
| 直接修改 `_nav_targets` | 私有耦合、进度/奖励状态可能不一致 | 新增受控公共 API 或严格 wrapper |
| 不可达路径退回直线目标 | 穿障碍且破坏 Phase 1B 语义 | 保留 reachable/hold 语义，复用 PathTracker 结果 |
| 方案 A 扩维但漏改 builder/replay | 运行时 shape 错误 | 首版采用 B；A 独立 schema 与 shape tests |
| 新功能默认开启 | 破坏 Phase 1B/旧 RMADDPG | 新 method/config 显式 opt-in，默认 hook 为 `None` |
| checkpoint 名称或 metadata 混用 | 无法复现或错误 resume | Phase 1C 独立目录和 integration schema hash |
| target priority 派生形成新机制 | 暗中改变算法 | B 首版不把 priority 输入 actor；A 若实施先冻结派生规范 |

## 7. 未来允许与禁止修改清单

### 可新增

- `chapter3_bser/integration/*`
- `chapter3_bser/experiments/phase1c_bser_rmaddpg/*`
- `configs/chapter3/bser_phase1c.json`
- `tests/test_phase1c_*.py`
- `docs2/phase1c_design/*`

### 可在严格 feature gate 下最小修改

- `core/env/mission_env.py`
- `core/env/uav_env.py`
- `core/runtime/builder.py`
- `core/runtime/training.py`
- `core/runtime/engine.py`
- `core/registry/experiment_registry.py`

### 方案 B 首版不需要修改

- `core/algorithms/agents.py`
- `core/algorithms/networks.py`
- `core/algorithms/maddpg.py`
- `core/replay/ch3_buffer.py`
- `core/env/observation_contract.py`

### 禁止修改

- `chapter3_bser/objective.py`
- `chapter3_bser/greedy_solver.py`
- `chapter3_bser/candidate_generator.py`
- `chapter3_bser/exact_solver.py`
- EventDetector 逻辑、cooldown/ReplanningPolicy、allocator 决策
- `chapter3_bser/controllers/path_tracker.py`
- `chapter3_bser/online/waypoint_manager.py`
- 所有 Phase 1B 正式实验代码和历史结果/checkpoint

## 8. 阶段退出条件

进入正式 Phase 1C 训练前必须同时满足：

1. Phase 1B.2_corrected 隔离回归仍为 `0.20 / 0.55 / 1336 / 740`。
2. 冻结 BSER 文件哈希无变化。
3. 方案 B 的 observation/action/critic/replay 维度分别保持 `28 / 3 / 124 / 28`。
4. 单 episode smoke test 通过且没有 observation-guidance 时序错位。
5. 新方法默认关闭，旧方法输出和调用链不变。
6. 所有 Phase 1C 产物进入独立目录，可单独清理或复现实验。
