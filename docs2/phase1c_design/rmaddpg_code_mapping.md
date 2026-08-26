# Chapter 3 RMADDPG 代码映射

## 1. 审查边界

- 仓库快照：`cd83bd3ef6339de4319786dd60028f50f131adbf`
- 审查对象：当前工作树（含用户已有的未提交 Phase 1B/1B.3A 文件）
- 本阶段只做静态代码追踪与接口设计；未启动训练，未修改算法、环境或 Phase 1B 实验代码。
- `chapter3_bser/objective.py`、`greedy_solver.py`、`candidate_generator.py`、`exact_solver.py` 均视为冻结组件。

## 2. RMADDPG 组件位置

| 职责 | 文件 | 类/函数 | 当前作用 |
|---|---|---|---|
| 单智能体定义 | `core/algorithms/agents.py` | `DDPGAgent` | 持有 actor、target actor、双 critic、target critics、优化器与 OU 噪声 |
| Actor 网络 | `core/algorithms/networks.py` | `MLPNetwork` | `DDPGAgent.policy`/`target_policy` 使用的 MLP；连续输出约束到 `[-1, 1]` |
| Critic 网络 | `core/algorithms/networks.py` | `MLPNetwork` | `critic1`/`critic2` 及对应 target 网络；输出标量 Q 值 |
| 多智能体协调 | `core/algorithms/maddpg.py` | `MADDPG` | 动作推理、集中式 critic 输入拼接、双 critic 更新、延迟 actor 更新、target 软更新 |
| Replay buffer | `core/replay/ch3_buffer.py` | `CH3ReplayBuffer` | 保存 `obs/actions/rewards/next_obs/dones/success_flags`，支持优先级采样 |
| 统一训练入口 | `core/runtime/training.py` | `train_and_evaluate` | 读取训练/验证场景，构建 runtime，管理 checkpoint、resume 与结果输出 |
| 单回合入口 | `core/runtime/training.py` | `run_episode` | 委托 `core.runtime.engine._run_episode` 并追加 episode metrics |
| 核心训练循环 | `core/runtime/engine.py` | `_run_episode` | actor 推理、环境推进、replay 写入、采样更新、target 更新 |
| 运行时构建 | `core/runtime/builder.py` | `build_runtime` | 创建 `UAVEnv`、`MADDPG` 和 `CH3ReplayBuffer` |
| 兼容/注册运行路径 | `core/runtime/engine.py` | `build_ch3_runtime*`、`train_and_evaluate_resolved_config`、`train_and_evaluate_method` | 另一条配置注册兼容路径；其 replay 维度从环境空间动态读取 |
| 环境稳定外观 | `core/env/mission_env.py` | `MissionCoreEnv` | 对 `UAVEnv` 的 reset/step 与任务、智能体、地图公共视图进行转发 |
| 实际环境实现 | `core/env/uav_env.py` | `UAVEnv` | 动力学、奖励、目标/地图状态、导航目标、观测与动作执行 |
| 观测契约 | `core/env/observation_contract.py` | `FIELDS`、`OBSERVATION_DIM` | 明确定义每智能体 28 维本地观测 |
| 方法注册表 | `core/registry/experiment_registry.py` | `LEARNING_METHODS` 等 | 当前包含 `ch3_pheromone_rmaddpg`、`ch3_pse_rmaddpg`；尚无 BSER-RMADDPG 方法 |

说明：actor 与 critic 没有各自独立的文件；它们共享 `MLPNetwork` 网络类，在 `DDPGAgent.__init__` 中分别实例化。

## 3. 当前观测结构

`core/env/uav_env.py::_get_obs` 为四个智能体分别生成 28 维向量：

| 索引 | 字段 | 维度 |
|---|---|---:|
| `0:3` | 位置 | 3 |
| `3:6` | 速度 | 3 |
| `6:9` | 当前导航目标相对位移 | 3 |
| `9:12` | 当前导航目标单位方向 | 3 |
| `12:15` | 已知任务目标相对位移；未知时为零 | 3 |
| `15` | 归一化导航距离 | 1 |
| `16` | 归一化速度 | 1 |
| `17` | 朝导航目标的 closing speed | 1 |
| `18` | 最近障碍距离 | 1 |
| `19` | 航点进度 | 1 |
| `20` | agent finished | 1 |
| `21` | hold progress | 1 |
| `22:26` | 角色 one-hot | 4 |
| `26:28` | 目标知识阶段 | 2 |

当前空间与网络维度：

- 每智能体 observation：`28`
- 每智能体 actor 输入/输出：`28 -> 3`
- 四智能体联合 observation：`4 × 28 = 112`
- 四智能体联合 action：`4 × 3 = 12`
- 每个集中式 critic 输入：`112 + 12 = 124`
- Replay buffer：四组 `obs_dim=28`、四组 `action_dim=3`

`core/runtime/builder.py::build_runtime` 的 replay 维度目前硬编码为 `(28,) * 4` 和 `(3,) * 4`；`core/runtime/engine.py::build_ch3_runtime_from_resolved_config` 则从环境空间动态读取。若未来改变 observation shape，这两条构建路径必须同时审计。

## 4. AUV observation 完整调用链

### Reset 路径

```text
core.runtime.training.train_and_evaluate
  -> core.runtime.builder.build_runtime
     -> core.env.uav_env.UAVEnv(...)
     -> core.algorithms.maddpg.MADDPG.init_from_env(env)
        -> 读取 env.observation_space / env.action_space
     -> core.replay.ch3_buffer.CH3ReplayBuffer(...)

core.runtime.training.run_episode
  -> core.runtime.engine._run_episode
     -> observations = env.reset(scenario)
        -> UAVEnv.reset
        -> UAVEnv._get_obs
     -> core.runtime.engine._as_env_actions(runtime, observations)
        -> MADDPG.step(observations)
           -> MADDPG._ensure_obs_tensor
           -> DDPGAgent.step(obs_i)
              -> DDPGAgent.policy(obs_i)
                 -> MLPNetwork.forward
```

若外层使用 `MissionCoreEnv`，则在 `UAVEnv.reset` 前多一层 `MissionCoreEnv.reset` 透明转发，不改变观测语义。

### 每步 next observation 路径

```text
UAVEnv.step(actions)
  -> 更新动力学、目标、地图、任务状态和导航目标
  -> UAVEnv._get_obs
  -> next_observations 返回 _run_episode
  -> replay_buffer.push(observations, actions, rewards,
                        next_observations, dones, success_flags)
  -> observations = next_observations
  -> 下一步 MADDPG.step
```

集中式训练时，`MADDPG._prepare_sample` 将四个 observation 和四个 action 拼接成 critic 输入；执行阶段每个 actor 只读取自身的本地 observation。

## 5. Action 完整调用链

```text
core.runtime.engine._run_episode
  -> _as_env_actions(runtime, observations, explore)
     -> MADDPG.step
        -> DDPGAgent.step
           -> policy(obs)
           -> 可选 OU exploration noise
           -> clamp[-1, 1]
     -> stack 为 (4, 3)
  -> UAVEnv.step(actions)
     -> UAVEnv._apply_agent_dynamics(actions)
        -> clamp[-1, 1]
        -> _actions_to_residual_acc
        -> residual = residual_scale * raw_residual
        -> prior = prior_strength * _compute_waypoint_prior_acc()
        -> final_acc = clamp(prior + residual)
        -> 更新速度与位置，并处理碰撞/边界
```

因此，RMADDPG 输出并不是完整的高层任务动作，而是三轴归一化残差加速度。环境已依据 `_nav_targets` 计算航点先验，再把学习残差叠加到先验上。这一现状是 Phase 1C 采用高低层方案的主要结构依据。

## 6. 训练更新链

```text
_run_episode(train_updates=True)
  -> CH3ReplayBuffer.push(...)
  -> 达到 warmup_steps 且满足 update_frequency
  -> CH3ReplayBuffer.sample(batch_size)
  -> 对每个 agent：
       MADDPG.update_critic_only(...) 或 MADDPG.update(...)
       -> 双 centralized critic
       -> policy_delay 到期时更新 actor
  -> CH3ReplayBuffer.update_priorities(...)
  -> MADDPG.update_all_targets(...)
```

Checkpoint 的 `init_dict`/resume identity 包含网络输入维度。任何 observation 扩维都会使旧 checkpoint 与新模型结构不兼容，必须使用新的 Phase 1C checkpoint 命名空间，不能覆盖或冒充 Phase 1B/现有 RMADDPG checkpoint。
