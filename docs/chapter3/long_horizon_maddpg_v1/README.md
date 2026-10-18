# 独立长视野 MADDPG 修正版

本版本用于核查并修复旧 B2/B3 的训练设置风险，不是已经验证有效的新算法。
旧 B2/B3、D2 规划、core、奖励和历史来源记录均不修改。
方法标识 `ch3_long_horizon_maddpg_v1`，独立配置、入口、checkpoint、输出。

## 后续发现奖励衰减

配置明确声明：400 步跨度后事件权重至少保留 0.5。
gamma = exp(log(0.5)/400) ≈ 0.9982686326。
150 步后的权重约 0.7711，旧 gamma=0.95 时约 0.00045555。
这改变了训练优化目标，不是保持旧目标的等价加速，也不是奖励重分配。
保留原有早发现奖励，因此不意味着所有发现时刻同等好。
多步目标不能消除原 gamma 的数学折扣；本版本不通过加入未经校正的多步离策略回报伪装解决问题。
单步 Bellman 的信用传播速度、部分可观测性仍是需要实验验证的问题。

## 网络和学习设置

- 保留每个 actor 的 28→128→128→3，critic 的 124→256→256→1；不增加隐藏层或维度。
- 默认使用语义字段的固定单位缩放；位置、速度、目标相对量分别缩放。
  二进制标志、角色 one-hot、单位方向、归一化距离和动作不与世界坐标一起做 LayerNorm。
  `legacy_layernorm` 开关用于单因素对照。旧 LayerNorm 不是已证实的实现错误。
- actor 输出层仍使用小权重和 tanh，初始残差接近零；目标网络深拷贝、禁止梯度更新。
- actor 更新只使用该 actor 尚能影响物理动作的样本；全冻结 minibatch 不调用 Adam.step，避免历史动量继续漂移。
- critic 的当前/目标输入将已冻结智能体的动作规范为零；发现后不把团队 done 改成 true。
- actor 的队友动作默认来自 replay，减少联合动作外推；这与旧版不同，是可关闭的对照项。
- critic 先预热 256 轮，actor 每两轮 critic 更新一次；actor lr=0.0003。
- OU sigma 按总环境步数衰减至初始角色值的 25%；评估不采样探索噪声。
  固定噪声本身并不是实现 bug，调度效果需要实测。
- 移除固定 [-10,10] TD target clip，保留 Huber loss 和梯度裁剪，记录 target/Q/TD 分布。
  增大 gamma 后只改 gamma、保留任意价值截断是不充分的修复。
- 移除仅按 id(sample) 的缓存依赖，按本轮数据重建目标；四智能体更新完后才软更新目标网络。

延迟更新借鉴标准 TD3 的稳定化做法；本版本不实现 target-policy smoothing，不能称为完整 MATD3。
参考：https://spinningup.openai.com/en/latest/algorithms/td3.html

## 不混入此次修正的奖励问题

旧环境的搜索航点 bonus 时序缺陷、持续分散奖励与任务指标的偏差仍需独立奖励协议审查。
本版本保留每步原奖励，用相同零残差轨迹的逐步对照证明没有暗改环境。
不能把这里的长视野修正表述成所有奖励设计问题都已解决。

## 来源与运行

新 gate 要求 `d2_performance_v2_evolution.json` 的一个完整父档案逐字节匹配，
并验证本版本新增文件的独立封存记录。旧来源检查不放宽、不改写历史哈希。
任何不在父档案和本版本增量中的生产源码/配置新增，都阻止显式执行。
当前工作区若仍含 `map_stp_shadow_v1` 等其他未封存新增，check 会明确返回 ready=false。
此时不能把结果解释成当前源码已通过来源验收；应在保留旧实验目录的前提下准备经过审查的独立源码树。

在仓库根目录执行，只检查，不创建网络或启动仿真：

```bash
python -m chapter3_bser.experiments.long_horizon_maddpg_v1.cli check
```

手动训练预览；预算必须明确指定，以下 8000 步仅是短诊断预算，不是正式性能实验。
默认预热需要 4096 步，过短的预算会没有 actor 更新，不能当作学习实验：

```bash
python -m chapter3_bser.experiments.long_horizon_maddpg_v1.cli train \
  --manifest /absolute/path/to/train.json \
  --output outputs/chapter3/long_horizon_maddpg_v1/collision_terminal/diagnostic_01 \
  --environment-steps 8000
```

只有来源检查通过后，人工添加 `--execute` 才执行；完成已开始的 400 步任务回合后停止，报告实际预算超量。
不会自动恢复旧权重。新格式不接受 B2/B3/HGR 的 checkpoint。

评估同样默认只预览，显式 `--execute` 才加载这个新方法的 checkpoint：

```bash
python -m chapter3_bser.experiments.long_horizon_maddpg_v1.cli evaluate \
  --checkpoint /absolute/path/to/diagnostic_01/final.pt \
  --manifest /absolute/path/to/evaluation.json \
  --output outputs/chapter3/long_horizon_maddpg_v1/collision_terminal/evaluation_01
```

评估跑完整任务，禁止训练与评估场景按 ID、seed、内容重叠。
同一组评估场景可与旧方法做任务指标配对；新旧 gamma 不同，训练折扣回报不能当作同一个目标直接排名。

## 验证与实验边界

`tests/test_long_horizon_maddpg_v1.py` 覆盖衰减、完整400步反向目标、终止、梯度方向、
冻结阶段 Adam 动量、Polyak 公式、输入缩放、噪声调度、序列化、零动作 D2 等价、来源拒绝与预览不执行。
没有自动运行正式训练，未建立 performance_passed。
方法效果需要分别关闭网络缩放、阶段掩码、队友动作选择等单因素对照，不能将整包变动的收益归功于一个组件。

2026-10-17 最新记录的本地验证：60 项测试、23 项子测试通过，涵盖冻结观测、路径跟踪、
core checkpoint、团队奖励、碰撞协议和历史来源元数据；不是 CI 或整仓测试全通过声明。
在仅包含完整父来源和本增量的临时源码副本中，另完成一个 400 步机制训练回合，
100 轮 critic 更新、保存和重新加载新 checkpoint，以及一个禁止更新的评估回合。
此机制测试使用 hidden_dim=16 和关闭预热，目的是验证工程链路，不代表默认配置的收敛或任务性能。
主工作区仍有 6 个 MAP_STP 源码/配置新增未纳入该来源封存，因此正式执行 gate 保持关闭。
紧凑记录见 [verification_20261017.json](verification_20261017.json)。
