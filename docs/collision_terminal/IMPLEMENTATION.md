# 首次实体障碍碰撞即团队失败

> 2026-09-13 维护索引：新协议现已提交为 `06167b5807a05b29ee0f42e21556015c17cfd5c1`。下文“本轮/未提交/验证”保留原交付时点含义；本次修复与实际验收见 [maintenance/SUMMARY.md](maintenance/SUMMARY.md)。三份过期调试日志的精确 Git blob 恢复索引见 [cleanup_manifest.json](maintenance/cleanup_manifest.json)，其余历史结果未改写。

本轮实现新增 `collision_terminal_v1`，接入真实 PRRAC 训练、评价、配对评价、checkpoint 和 replay 路径。历史配置缺少字段时仍解释为 `legacy_nonterminal_v1`。

仓库核对：2026-09-12 通过 GitHub 连接器读取远程默认分支 `main`，提交为 `93a9c8fb53857051390265e3035061bf05402e25`；本地 root 为 `E:/gym/code/WORKSPACE/AUV-Thesis`，origin 为 `https://github.com/SKT-T1otto/AUV-Thesis.git`，分支/HEAD 相同，开始时工作区干净。直接 Git HTTPS 连接失败，连接器核对成功。本机未发现 Python 训练进程；没有操作 Linux 训练目录。没有启动正式训练或性能评价，没有 commit/push，没有改写旧 outputs。

## 行为与版本

| 项目 | legacy_nonterminal_v1 | collision_terminal_v1 |
|---|---|---|
| 检测版本 | endpoint_rollback_v1 | segment_closed_aabb_v1 |
| 奖励版本 | legacy_shaping_v1 | team_failure_override_v1 |
| 实体障碍 | 步末点检测，回退和反弹，继续回合 | 实际运动段闭集 AABB 检测，任一 agent 命中全队失败 |
| 冻结 | 保留历史实现 | 生成实际运动段之前生效，静止点仍接受几何检测 |
| 事件优先级 | 保留历史行为 | collision > success > timeout |
| 碰撞步奖励 | 保留原 shaping/adapter | 最终四个 reward 均为 collision_terminal_reward，默认 -2 |
| 期限 | 保留历史行为 | max_steps 为任务截止期限，截止步无碰撞成功有效 |
| bootstrap | 保留原 done 语义 | success/collision/timeout 都为零；外部截断不冒充任务结束 |
| 再次 step | 保留历史行为 | 抛出 RuntimeError，要求 reset |

点艇模型不新增艇体半径。物理闭集容差为 `1e-9` 世界单位，包含端点、切线、零长度点接触，不使用 planner clearance。核对发现 `target_motion.segment_aabb_first_hit` 为目标反射排除了内部/静止起点，因此新建专门的 `closed_segment_aabb_first_hit`，原目标反射函数保持不变。

碰撞 agent 保存于其首次命中点，速度置零；其他 agent 完成本离散步的候选运动。记录 agent/obstacle ID、运动前位置、未经碰撞回退的候选终点、命中点和比例。采用步级优先级，没有引入连续时间的全队同步碰撞引擎。无碰撞的积分、控制、接近及保持逻辑保留。

初始位置位于真实障碍闭集内时，reset 明确抛出 ValueError，并设置 `invalid_initial_state`。评价异常写入失败文件并传播退出码，不重采样、不跳过 manifest 场景。近障碍、separation、正常目标 Contact 和世界边界不属于本次新增终止事件。

## 调用链与数据

`CLI → resolved config → worker job → environment_kwargs_from_config → MissionCoreEnv/UAVEnv → GuidedEnv → Phase1CV2TrainingEnv → PRRACTrainingEnv → collector → PRRACReplayAdapter → PRRACMADDPG`。

`EpisodeOutcome` 是终止结果权威；公开 `get_episode_result()` 返回其副本。碰撞在动力学后立即锁定失败，跳过新增 Found、Contact、Hold、Success、地图更新及后续重规划，保留进入本步前已经发生的发现和交接。正常 success/timeout 在既有任务更新完成后确定。

严格终端 collector 保存环境返回的最终 next_obs，然后结束回合，不运行 controller、recovery、guidance 安装或 observation refresh。只读末态诊断使用复制后的公共运动学快照。终止原因与 Search/Intercept/Hold 三头标签分开，不扩展网络维数。

碰撞时 core 输出一次最终失败奖励；adapter 只确保这个值原样输出，不重复叠加。即使 adapter disabled，奖励仍为四个相同负值。日志包含原始分项、覆写前/后和 `final_reward_by_agent`。Searchers freeze、discovery-only 及 success bonus 都不能覆盖碰撞信号。

Replay 保存协议身份，拒绝混装或跨协议恢复，碰撞样本不能设置 success 或进入 success-tail。critic 只对可 bootstrap 行计算 target actor/Q，终端行直接取 reward，避免 NaN×0。28D 局部观测、3D 残差动作、124D critic 输入、PRRAC 网络、BSER、A* 与 PathTracker 参数未变。剩余任务时间仅写入 metadata；不宣称 28D 是严格完整 Markov 状态，也不宣称个体 shaping 等价于最大化团队成功概率。

## Checkpoint 的四条路径

| 路径 | 读取旧数据 | 新状态 |
|---|---|---|
| evaluate + allow-protocol-transfer | 只读源 Actor 和必要辅助预测头；核对真实源配置哈希 | 新严格评价环境，无训练、无 replay 恢复 |
| train | 无 | 全新 Actor/critic/optimizer/replay/计数 |
| warmstart | 完整四个 Actor 参数和 buffers，strict 键/形状检查 | target_actor 同步；critic/optimizer/replay/计数/RNG 不继承 |
| resume | 同协议、检测、奖励语义、config hash、架构、runtime 一致的完整状态 | 恢复 learner/optimizer/replay/计数 |

普通 resume 不能跨协议。跨协议评价白名单仅允许 legacy → strict，并要求显式授权；B0/B1、原 architecture、28D/3D 和原 schema 检查继续执行。结果明确包含 `checkpoint_task_protocol`、`evaluation_task_protocol`、`protocol_transfer_evaluation`。

旧 checkpoint 若没有内嵌 `resolved_training_config`，跨协议评价和 warmstart 需要源 run 下的原始 `resolved_training_config.json`，其 canonical hash 必须匹配 checkpoint。不得编辑源 metadata 来通过校验。新 checkpoint 内嵌完整 resolved config，并以临时文件写完后原子替换正式 `.pt`；拒绝临时路径。

Warmstart 默认 `critic_only_warmup_updates=256`，单位为满足 replay 样本量和普通 warmup 条件后的一次采样更新，每次对四个 agent 调用 `update_critic_only`。Actor 在 256 次更新之前不更新，之后恢复原 policy_delay。实际执行数写入 checkpoint/summary。它是工程初值，未经过性能调优。`-2.0` 同样不构成最优性或安全性保证。负奖励必须有限且严格小于零，clip 不得静默改变它。

源 checkpoint SHA256、训练协议、完成 episode、可取得的训练步数/更新数/成本记录在 initialization；未知成本为 null。旧 replay 不迁移、不删除。本轮只使用独立合成权重验证接口，没有验证真实旧 1000ep 权重或做其性能评价。

训练 `learner_update_rounds` 与 checkpoint `update_step` 都按采样更新轮数计；既有 `optimizer_update_count` 按每个 agent 的 learner 更新调用计。`wall_seconds` 沿用本次调用的采样、更新和收尾计时区间；`collector_wall_seconds` 是各回合采样耗时之和，并行 worker 的这个和不能当作整机实际墙钟耗时。恢复前的成本与本次调用成本不混写成一次连续训练成本。

## 手动运行

从本次代码的仓库根目录执行。Linux 应使用独立开发/实验副本，不在正在运行的旧训练目录中 pull、切换或覆盖代码。所有严格输出必须包含独立的 `collision_terminal` 目录。以下命令提示输入真实 checkpoint，不假设 ep1000 存在、不自动选择 latest/best。

本轮文件尚未 commit/push。Linux 执行前需要将本轮修改和新增文件一并同步到独立副本，文件清单见 [CHANGES.md](CHANGES.md)。

Windows Conda 可通过 `CRK_CONDA_EXE` 指定可执行文件，通过 `-CondaEnv` 指定环境；Linux 可使用 `CRK_CONDA_EXE` 与 `CRK_CONDA_ENV`。PowerShell 使用 conda run 初始化环境，SH 使用 conda.sh + activate 并用 exec 传播退出码。BAT 与 PowerShell 参数相同。

旧 checkpoint 新协议 Prior-only / PRRAC 配对评价，100 场/控制器：

```powershell
$SourceCheckpoint=Read-Host '旧checkpoint完整路径'; .\scripts\run_collision_terminal.ps1 -CondaEnv AUV evaluate --checkpoint "$SourceCheckpoint" --allow-protocol-transfer --episodes 100 --workers 4 --device cpu --output-dir outputs/chapter3/phase1c_prrac/collision_terminal/paired
```

```bash
read -r -p '旧checkpoint完整路径: ' SOURCE_CHECKPOINT; CRK_CONDA_ENV=AUV bash scripts/linux/run_collision_terminal.sh evaluate --checkpoint "$SOURCE_CHECKPOINT" --allow-protocol-transfer --episodes 100 --workers 4 --device cpu --output-dir outputs/chapter3/phase1c_prrac/collision_terminal/paired
```

从头训练，1000 episodes、400 max_steps、每 100 episodes checkpoint：

```powershell
.\scripts\run_collision_terminal.ps1 -CondaEnv AUV train --seed 2729 --episodes 1000 --workers 4 --device cpu --output-dir outputs/chapter3/phase1c_prrac/collision_terminal/scratch_seed2729
```

```bash
CRK_CONDA_ENV=AUV bash scripts/linux/run_collision_terminal.sh train --seed 2729 --episodes 1000 --workers 4 --device cpu --output-dir outputs/chapter3/phase1c_prrac/collision_terminal/scratch_seed2729
```

Actor 热启动新训练：

```powershell
$SourceCheckpoint=Read-Host '旧checkpoint完整路径'; .\scripts\run_collision_terminal.ps1 -CondaEnv AUV warmstart --checkpoint "$SourceCheckpoint" --critic-warmup-updates 256 --seed 2729 --episodes 1000 --workers 4 --device cpu --output-dir outputs/chapter3/phase1c_prrac/collision_terminal/actor_warmstart_seed2729
```

```bash
read -r -p '旧checkpoint完整路径: ' SOURCE_CHECKPOINT; CRK_CONDA_ENV=AUV bash scripts/linux/run_collision_terminal.sh warmstart --checkpoint "$SOURCE_CHECKPOINT" --critic-warmup-updates 256 --seed 2729 --episodes 1000 --workers 4 --device cpu --output-dir outputs/chapter3/phase1c_prrac/collision_terminal/actor_warmstart_seed2729
```

恢复同协议的新 checkpoint，其他参数须与该 run 原配置一致。若选择的 checkpoint 后面已经有更晚权重，必须指定新的 `collision_terminal` 输出子目录；程序拒绝覆盖已有 checkpoint。

```powershell
$NewCheckpoint=Read-Host '新协议checkpoint完整路径'; $RunDirectory=Read-Host '该checkpoint所属的collision_terminal run目录'; .\scripts\run_collision_terminal.ps1 -CondaEnv AUV resume --checkpoint "$NewCheckpoint" --seed 2729 --episodes 1000 --workers 4 --device cpu --output-dir "$RunDirectory"
```

```bash
read -r -p '新协议checkpoint完整路径: ' NEW_CHECKPOINT; read -r -p '该checkpoint所属的collision_terminal run目录: ' RUN_DIRECTORY; CRK_CONDA_ENV=AUV bash scripts/linux/run_collision_terminal.sh resume --checkpoint "$NEW_CHECKPOINT" --seed 2729 --episodes 1000 --workers 4 --device cpu --output-dir "$RUN_DIRECTORY"
```

评价加 `--prepare-only` 仅生成计划，不加载 checkpoint 或执行回合。配对目录自动附加时间戳和 UUID。每个 train/warmstart run 目录必须独立且不存在；resume 仅恢复同配置 run。直接 Python 入口的 `--help` 列出全部参数。

## 结果和验收

训练：`resolved_training_config.json`、`metrics/episode_metrics.csv`、`metrics/execution_diagnostics.csv`、`metrics/training_summary.json`、`checkpoints/phase1c_prrac_episode_NNNN.pt`；进度按已完成 batch 输出真实 episode、环境步、更新数、近期安全成功/碰撞失败、checkpoint 和耗时。

评价：每个控制器独立输出 `episode_evaluation.csv`、`checkpoint_summary.csv`、`evaluation_summary.json`、`evaluation_progress.json`、`failure_funnel.csv`、`failure_trace.jsonl`、`failure_trace_index.csv`、`collision_terminal_outcomes.png`。配对根目录保存 `pair_plan.json`、共享配置、`paired_analysis.json`。

严格汇总满足 `n_success + n_collision_failure + n_timeout = n_valid_episodes`。异常/未完成单独计数，完整率不可用时为 null；无 Found 的条件成功率为 null，首次碰撞无事件为 null。多 agent 同步碰撞只计一个团队失败；角色统计为出现该角色首次碰撞的 episode 数。碰撞记录不受 only_found_failures 过滤，但仍受已有数量上限控制。碰撞次数随提前终止减少，不解释为避障能力提升。

最新本地验证的命令与结果见 [VERIFICATION.md](VERIFICATION.md)。这些是自动回归，不是 CI、性能提升证据、1000ep 完成证明或真实旧模型评估结果。历史 golden 文件缺失会如实记录；历史基准与 27 条历史哈希保持原样，新增逐文件 evolution 记录单独维护。
