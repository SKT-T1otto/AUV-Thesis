# 第三章团队目标与交接梯度重估

本实现依据用户的《第三章研究方案.md》（读取位置：`E:/gym/code/WORKSPACE/CRK-Thesis-v2/docs/research/20260911_chapter3_refinement/第三章研究方案.md`）及 2026-09-13 的完整实现要求。工作开始时，本地 HEAD 与远程默认分支 `main` 均为 `0a5bd3a4efdb68f2cca8721ffe0e8dbf2d54e265`，工作区干净。

HGR 是工程方法名。实现与有界机制验收不证明论文创新性、性能优势或安全成功概率最大化。正式实验由用户手动启动。

本轮最终状态、600 项固定树回归、真实更新证据及历史输入缺口见[验收报告](verification/RESULTS.md)与[机器可读状态](verification/acceptance_summary.json)。

## 奖励与任务边界

`chapter3_bser/experiments/reward_objective.py::RewardAccounting.apply` 是唯一最终变换位置。调用顺序为物理环境 → GuidedEnv → 原 Phase1CV2 execution reward adapter → PRRACTrainingEnv 的最终奖励记账。原适配器输出的四维向量就是 `source_reward_by_agent`；先求固定四人均值，再按 `team_mean_v1` 广播为四个相同 learner reward。广播后没有 clipping、tanh、Searcher zeroing 或额外奖金。

`individual_v1` 保留原向量，历史缺失字段按此解释，且保留其旧输出结构。新增 `bser_phase1c_prrac_individual_train/eval.json` 显式声明个体目标，输出相同的团队诊断指标；与对应 team 配置只差目标和输出路径。`task_protocol` 仍独立标识物理与终止规则。碰撞 source 为 `[-2,-2,-2,-2]` 时最终仍为同一向量。发现、Searcher 冻结和可靠交接均不设置 episode done，只有成功、首次实体碰撞或原任务截止终止 bootstrap。

包装器、PRRAC collector/replay、HGR 主轨迹与分支使用同一步最终值。报告分别保存团队未折扣回报、团队折扣回报和诊断用原个体回报；兼容四维向量不会在主团队指标中再次求和。原 reward adapter 的 Searcher zeroing 计数是 source 诊断。PRRAC 的 router/gate 正则仍是算法损失，不计入团队环境回报，也不声称其更新是精确 MC score 梯度。

默认 `gamma=0.95` 直接继承已核对的现有配置；所有方法和旧新分支一致，数学测试另覆盖 `gamma=1`。团队折扣回报和安全成功率分别报告。

## 策略、时间与恢复

`models/hgr/policy.py` 使用五个完全独立的随机策略模块：三个 Searcher 与 Executor standby 属于 `theta_minus`，另一个 Executor execution 属于 `phi`。所有参数、LayerNorm、router/gate 及优化器均分开。每个 Actor 只接收原 28D 本地观测，仍输出 3D 残差命令。

随机分布为 `u ~ Normal(mu, std)`、`a=tanh(u)`；复用的 PRRAC router/gate 在 `mu=atanh(clamp(PRRAC_action))` 内部。最终 tanh 与参数无关，密度对三维求和，使用稳定 Jacobian，并保存采样时的 pre-tanh latent。score 重算时 detach 动作样本和观测，不混入 pathwise 导数。std 为新增训练状态，初始 0.12，log-std 的固定范围为 [-10,2]。没有 OU 行为密度、旧 replay、PPO、多 epoch actor 重用、GAE、critic bootstrap、熵目标或回报标准化。

`experiments/hgr/runtime.py::MissionRuntime` 沿用真实 BSER/controller/provider/PathTracker 与物理任务。`tau` 是已完成的转移数：Executor 已合法获得目标信息且完成下一决策观测准备，即将第一次使用 execution 策略时捕获快照。前缀是 `a_0..a_(tau-1)` 和对应奖励；第一条 suffix 动作是 `a_tau`。Found 与交接事件、交接决策时间分别记录。终端出现时不再捕获交接。

快照保存完整对象状态图，根组件包括物理环境、Mission facade、全部奖励/引导包装器、runtime、provider、controller 和 bridge。其包含动力学/目标运动/冻结/消息/地图/belief/接触保持/奖励记忆/任务时钟、已安装 guidance、PathTracker 和 RNG。字段清单、版本和字节摘要一起保存，跨 spawn 恢复保持对象引用关系。导航闭包只重新绑定一次；不 reset 环境、不重复安装 guidance、不额外推进 step。快照不包含 policy、learner、optimizer 或 replay。

同策略恢复检查继续原随机流；正式标签使用新的独立子流，只改变未来策略采样及随机创新，不重抽已实现的物理隐藏状态。现有目标反射过程与流场是确定状态过程，原状态完整延续。重复抽中同一主索引保留重数并使用新的分支流。旧、新策略从同一快照分别运行到真正终止或原始截止，不使用 critic 补尾。

预测器输入为 153D 合法特征：四个本地 28D 观测、公开位置/速度/导航目标和剩余时间、公开地图/belief 摘要。真实目标与未知障碍仅存在于仿真恢复载荷，既不送入 Actor，也不新增给 planner。模型执行端只需随机策略与本地观测。

## 公式到生产函数

| 研究方案 | 实现位置 | 实际用途 |
|---|---|---|
| 3-2 固定团队均值 | `reward_objective.py::RewardAccounting.apply` | 最后一次奖励变换与回报记账 |
| 3-11 联合前缀 score | `policy.py::HandoffPolicy.score_log_prob` | 仅累加实际随机决策者；冻结动作无 score |
| 3-13 完整旧新续值差 | `runtime.py::continue_branch`, `train.py::Trainer.paired_label` | 从同一边界得到可正可负 MC 标签 |
| 3-15/3-16 预测加残差校正 | `estimator.py::prefix_losses` | `-C*f/N` 与 `-C*(label-f)/(N*K*q)` |
| 完整旧梯度 | `estimator.py::prefix_losses` 的 `old` | `-sum(gamma**t * log_prob_t * G0_t)/N` |
| 3-21 旧梯度加增量 | `estimator.py::update_prefix` | 三项梯度合并后仅执行一次 SGD |
| 3.6 分块外循环 | `train.py::Trainer.run_cycle` | 独立 suffix 训练、主轨迹、先导拟合、正式校正与发布 |
| 3.7 强直接边界对照 | `prefix_losses(method='direct_boundary_corrected')` | 前缀奖励梯度 + 当前边界值预测 + 校正 |

`N` 始终包括无交接轨迹。无交接时 `U=0`，抽中该索引贡献零且不重抽，旧梯度仍保留整条真实回报。生产默认均匀有放回 `q=1/N`，一般公式的 `1/q` 位置由人工非均匀例子验证。

`C=gamma**tau * sum(log_prob)` 只乘一次边界折扣，不按随机长度取时间均值。回报、预测、抽样概率和标签均作为常量。预测器及其 normalization 在正式校正前冻结；正式标签不参与本轮拟合或选择。前阶段采用单次 SGD 上升，固定学习率、无 momentum/weight decay/梯度裁剪，并核对参数变化与完整原始估计量一致。更新后丢弃主前缀。

执行策略训练使用独立完整任务轨迹的 `sum(gamma**t * score_phi(t) * G_t)/N_suffix`；无交接轨迹保留零贡献与原始分母。主数据不参与选择 `phi1`。先导、主轨迹、后阶段训练和每次正式分支均有分离的身份与随机流编号。

## 对照、成本与恢复

可运行方法为 PRRAC-individual、PRRAC-team、`stochastic_direct_mc`、`direct_boundary_corrected`、`hgr`。两个 PRRAC 配置除目标/输出路径外保持相同常规参数；三个随机方法共用策略结构、环境、信息、分块时序、优化器及 gamma。

直接 MC 的主轨迹使用新执行策略。直接边界校正保留前缀奖励梯度，预测当前 `V_plus(phi1)`，使用相同恢复接口校正当前续值；其主采集可停在边界，并明确标记为未完成前缀，固定模型评价仍运行完整任务。HGR 使用完整旧主轨迹与新旧差值。`predictor_zero` 是合法零控制变量；`remove_correction`、`remove_prediction` 明确标记为有偏消融，不是默认 HGR。

每轮分别记录主前缀、旧参考后续、执行策略训练、先导前缀/旧/新后续、正式旧/新后续的环境步，以及恢复次数、预测器/前后 Actor 更新数、墙钟时间。直接 MC 另列主当前策略后续步数。总步数等于这些互不重叠的实际转移之和；旧梯度和先导不是免费资源。

默认初值为主 batch 16、后阶段训练 4、先导 8、正式抽样 8，尚未调优。`total_main_trajectories=1000` 只统计主轨迹，不含额外成本。外循环 batch 在 100、200 等保存边界缩短，模型绝不把 112 标成 100。每个完整 cycle 原子保存，可同方法恢复；不保留有效旧前缀。预算策略为“仅在剩余预算为正时开始新 cycle，已开始 cycle 完整运行后披露超额并停止”，不截断 MC 标签。

同方法恢复校验目标、物理协议、schema、architecture 和不可变配置，加载前后策略、预测器、优化器、计数及采样状态。跨目标不能完整恢复 PRRAC critic/replay/optimizer。旧 PRRAC 可显式 Actor-only warmstart、只读跨目标评价，或显式映射为 HGR 随机均值。均值映射 epsilon=1e-6，边界误差上限随初始化记录；复制 standby 与 suffix 后不共享存储。不导入旧 critic/replay/optimizer 或伪造旧行为概率。

## 手动命令

模型另外保存完整生产源码摘要及策略参数摘要；同方法续训要求源码身份匹配。快照也校验生产源码身份，并核对恢复后的原始时钟、实际时限和对象引用。首版 HGR 使用 CPU 逐条采集真实环境，不包含并行吞吐优化。预测器每轮以新先导标签重新拟合，Adam 状态保存在该轮 checkpoint 中；下一轮创建新的拟合优化器，正式校正标签始终不用于拟合。

从仓库根目录运行；正式输出目录必须是全新路径，且包含 `collision_terminal` 目录组件。以下 checkpoint 占位路径须由用户明确替换。默认 HGR train 执行完整梯度更新。

恢复随机基线时传入相同的 `--algorithm`，或通过 `--config` 指向其原配置；修改了 batch、网络大小或先导预算的实验也必须用原配置恢复。`prrac-train --resume USER_CHECKPOINT.pt` 是 PRRAC 同方法恢复入口，按 checkpoint 原目标选择配置。

Windows PowerShell（BAT 可将 `.ps1` 换为 `.bat`）：

```powershell
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV train --output-dir outputs/chapter3/hgr/collision_terminal/run_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV train --algorithm stochastic_direct_mc --output-dir outputs/chapter3/direct_mc/collision_terminal/run_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV train --algorithm direct_boundary_corrected --output-dir outputs/chapter3/direct_boundary/collision_terminal/run_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV prrac-train --episodes 1000 --output-dir outputs/chapter3/team_reward/collision_terminal/run_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV prrac-train --config configs/chapter3/bser_phase1c_prrac_individual_train.json --episodes 1000 --output-dir outputs/chapter3/individual_reward/collision_terminal/run_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV resume --checkpoint "USER_CHECKPOINT.pt" --output-dir outputs/chapter3/hgr/collision_terminal/resume_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV train --mean-initialization "USER_PRRAC_CHECKPOINT.pt" --output-dir outputs/chapter3/hgr/collision_terminal/mean_init_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV prrac-train --init-actors-from "USER_PRRAC_CHECKPOINT.pt" --critic-warmup-updates 256 --output-dir outputs/chapter3/team_reward/collision_terminal/warmstart_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV evaluate --checkpoint "USER_HGR_CHECKPOINT.pt" --episodes 100 --policy-mode stochastic --output-dir outputs/chapter3/hgr/collision_terminal/eval_001
.\scripts\run_ch3_learning.ps1 -CondaEnv AUV prrac-evaluate --checkpoint "USER_PRRAC_CHECKPOINT.pt" --allow-objective-transfer --allow-protocol-transfer --episodes 100 --output-dir outputs/chapter3/team_reward/collision_terminal/eval_001
```

Linux 对应完整入口：

```bash
CRK_CONDA_ENV=AUV bash scripts/linux/run_ch3_learning.sh train --output-dir outputs/chapter3/hgr/collision_terminal/run_001
CRK_CONDA_ENV=AUV bash scripts/linux/run_ch3_learning.sh train --algorithm stochastic_direct_mc --output-dir outputs/chapter3/direct_mc/collision_terminal/run_001
CRK_CONDA_ENV=AUV bash scripts/linux/run_ch3_learning.sh train --algorithm direct_boundary_corrected --output-dir outputs/chapter3/direct_boundary/collision_terminal/run_001
CRK_CONDA_ENV=AUV bash scripts/linux/run_ch3_learning.sh prrac-train --episodes 1000 --output-dir outputs/chapter3/team_reward/collision_terminal/run_001
CRK_CONDA_ENV=AUV bash scripts/linux/run_ch3_learning.sh resume --checkpoint USER_CHECKPOINT.pt --output-dir outputs/chapter3/hgr/collision_terminal/resume_001
CRK_CONDA_ENV=AUV bash scripts/linux/run_ch3_learning.sh evaluate --checkpoint USER_HGR_CHECKPOINT.pt --episodes 100 --policy-mode stochastic --output-dir outputs/chapter3/hgr/collision_terminal/eval_001
```

`--policy-mode deterministic_mean` 单独评价确定性均值，不作为随机策略 J 的无偏样本。按总仿真步预算可为 train/resume 增加 `--max-total-environment-steps 1000000`。恢复时也可提高 `--total-main-trajectories`，不可混淆额外轨迹计数。Linux/Windows 入口转发实际 Python 参数并保留退出码，`--help` 不启动环境。

有界验收入口（不是正式性能实验）：

```bash
python -B -m unittest tests.test_hgr_mechanism tests.test_team_reward tests.test_hgr_integration -v
python -B -m chapter3_bser.experiments.hgr.cli train --config tests/fixtures/hgr/integration_config.json --output-dir outputs/chapter3/hgr/collision_terminal/manual_bounded_check
python -B -m tools.verify_collision_terminal --suite all --workers 4 --output-dir docs/hgr/verification/manual_fresh_run
```

最终本地验收记录置于 `docs/hgr/verification/`。历史 E0/golden 缺失输入单独记为 blocked，不生成替代历史产物。模型、checkpoint、原始训练输出继续忽略；本轮不 commit、push，也不运行正式 1000 主轨迹实验。

## 本轮真实有界集成证据

`verification/final_02/current/tests.test_hgr_integration.log` 中四项集成检查全部通过：真实 M20 生成场景、无交接/预算收尾、生产入口两个完整外循环与续训/对照/评价、原流及 spawn 快照续跑。固定合法小场景的任务截止为 8 步，真实可靠交接决策为 `tau=2`；各分支沿用原截止，不补尾或重置时钟。

| 外循环 | g0 范数 | 预测项范数 | 校正项范数 | 增量范数 | g1 范数 | 实际前阶段参数变化范数 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.316336 | 1.037221 | 1.037979 | 0.000757706 | 3.317059 | 0.003317084 |
| 2 | 3.253211 | 1.703721 | 1.703796 | 0.000075027 | 3.253140 | 0.003253160 |

数据来自 [cycles.json](verification/final_02/integration/cycles.json)，并由实际 SGD 参数差核验。第二轮正式标签包含 `-0.00011279397566439035`，没有截为零；这些真实任务结果不提供伪造的梯度 MSE。

该两轮 HGR 运行共 108 个环境步、12 次快照恢复、6 次预测器更新，前后 Actor 各更新 2 次。原参考后续、先导和正式旧新后续均计入成本；续训和其他基线另有自己的成本记录，不把它们混称为整个验收仅用 108 步。[完整分项成本](verification/final_02/integration/summary.json)、[分支记录](verification/final_02/integration/branches.json)、[续训记录](verification/final_02/integration/resumed/summary.json) 均可复核。

从第一轮 checkpoint 恢复并进入第二轮后，前后策略哈希及累计成本与连续运行一致。直接 MC、直接边界校正都通过生产入口完成实际参数更新；固定模型评价单独标记为随机策略且没有训练更新。评价复用了合法小场景作为接口夹具，不能用于正式性能结论。

两份测试模型保存在 Git 忽略的 `outputs/chapter3/hgr/collision_terminal/verification_final_02/checkpoints/`。原始集成 summary 中的临时 checkpoint 路径已随测试清理；持久路径见 [retained_checkpoint.json](verification/final_02/integration/retained_checkpoint.json)，复制后的模型已[再次读回核验](verification/final_02/integration/retained_checkpoint_readback.json)。这些是有界机制测试模型，不是正式实验模型。
