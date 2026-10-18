# D2 实现修复 v1（2026-10-18）

本版本落实用户确认的边界：保留原复合奖励及各项权重，保留 γ=0.95、28D 观测、3D 归一化残差动作、124D 集中 critic。它是独立的 `D2_B2_repair_v1` / `D2_B3_repair_v1`，不是旧 Linux 实验的续跑，也不是已验证的算法性能提升。没有启动长训练。

## 修改及适用范围

1. `core/env/uav_env.py` 增加显式 `arrival_before_reward_v1` 时序。物理移动及发现判定之后、奖励计算之前，先读取合法的内部搜索航点到达事件；地图访问登记、内部航点切换和导航刷新仍在奖励之后。保留进展奖励使用的旧目标，避免简单交换两行导致覆盖量或目标距离一起变化。发现阶段优先级、碰撞终止覆盖奖励保持原语义。缺省 `legacy_v1` 保留历史行为。
2. 修复版执行时屏蔽冻结、保持和不可达情况下的 RL 残差。回放保存实际应用的归一化残差，并在八项样本之外保存当前及下一状态的四个执行门控位；网络输入、回放观测和动作维度没有增加。actor 更新、其他 actor 动作、target actor 动作均复用门控。
3. actor 损失只对该 agent 可作用的样本求平均；整批无效时不调用 Adam.step，包括已有动量时。critic 仍学习这些状态的团队回报，不把搜索者冻结误当作任务终止。实际 actor 更新次数分别记录，不再一律按每轮四次计数。
4. 原网络、LayerNorm、隐藏层、学习率、OU 参数、twin-min TD、目标裁剪 ±10、梯度裁剪、软更新和 γ 均保留。新增目标值均值、绝对最大值和达到裁剪边界的比例，便于后续检查饱和现象；这个比例不能区分恰好等于边界与实际被裁剪。
5. 共享 MADDPG 修复未被当前 B2/B3 工厂选用的本地 DDPG 分支，使 target、critic 和 actor 三处均使用本地 31D 输入。修复终止样本的 `0 × NaN`，对非终止非有限目标、非有限损失和梯度报错，在对应优化器写入前检查。双 critic 的梯度都检查后才执行任一个 critic 优化器。无效 actor 梯度仍会终止当轮训练；这不承诺整个 actor+critic 更新的事务回滚。
6. 移除仅依赖 `id(sample)` 的批次缓存，避免可变样本及对象 ID 重用产生陈旧 target 输入。

## 奖励和观测诊断

不增加新的奖励项，不修改持续分散、覆盖、发现、接触、保持、成功等奖励的权重。保存原有各项 **tanh 前** 分量、发现前后团队回报及原有完整任务回报；这些分量经过 tanh、执行奖励适配和团队平均后，不具有简单相加得到最终回报的性质。

`diagnostics.paired_ordering` 按场景 ID、场景种子、完整场景内容指纹、环境创新种子、γ、任务协议、奖励来源版本、航点奖励时序和时限配对。分别统计“较高折扣回报却具有较低发现率／更多发现前碰撞／更高失败惩罚发现步数”的排序冲突。不构造新的加权指标，不以零冲突证明奖励等价，不把不完整任务当作失败，不把不同训练种子间的比较当作独立统计重复。真实排序需要后续同场景评估数据；当前合成反例只验证诊断器能识别冲突。

保留并明确记录风险：critic 看不到完整地图、剩余时间、上一时刻加速度和执行门控，不能宣称满足 Markov 性。旧奖励中的航点 bonus 仍针对 **环境内部搜索航点**，没有悄悄改成 D2 的语义候选点或路径子目标。引导航点与内部航点不一致时，奖励目标关联性仍需研究。

γ=0.95 仍会衰减远期发现事件；本次遵循用户决定，不通过整体提高 γ、重设权重或奖励重构去消除这个问题。

## 动作安全边界

保持或制动请求下，RL 残差变为零，原物理先验和动力学继续运行。惯性、流场、先验制动及未知障碍仍可能导致碰撞。本修复没有实现停距证明、完整最终加速度投影或未知地图下的无碰撞保证。安全统计是抑制残差的次数，不能解释为“避免碰撞次数”。

## 版本与来源

旧 B2/B3 配置和历史清单不改写。共享源文件的前后精确哈希记录在 `docs/provenance/d2_repair_v1_evolution.json`，27 条迁移记录继续按历史基线加明确演进检查。修复版自己的源文件和配置也在独立清单内逐项校验。检查点记录完整当前生产源码指纹，加载时要求完全匹配，并检查优化器状态的有限性及实际更新次数；旧权重、其他修复开关的权重不允许改名加载。本版本仅支持从头训练，不支持恢复 replay 或隐式续跑。

这不使当前工作目录自动满足旧 D2/HGR/SafeSearch 的完整源码封存要求；那些旧入口仍会拒绝未经其历史清单认可的新代码。修复版本的精确改动检查与检查点全源码指纹，也不等于对工作目录中其他既有实验分支的重新审计或正式发布验证。

## 手动入口

默认只打印计划，不训练、不创建训练输出：

```bash
python -m chapter3_bser.experiments.d2_repair_v1.train \
  --config configs/chapter3/d2_repair_v1/b2_train.json \
  --output outputs/chapter3/d2_repair_v1/collision_terminal/b2_new_run
```

将配置换成 `b3_train.json` 使用 B3。配置里的 `repairs.waypoint_arrival`、`repairs.inactive_actor`、`repairs.hold_residual` 是明确的消融开关；改变开关时必须使用新的输出目录，删除自动生成的 `waypoint_reward_timing` 字段或使其与开关一致。冻结和保持条件可能重叠，因此关闭单个开关不保证所有相关门控都消失。

用户日后手动决定执行训练时，才添加 `--execute`。现有配置仍是 1000 回合计划；本次没有执行这个计划。不能把旧 Linux checkpoint 交给这个入口继续训练。

手动评估修复版自身产生的检查点，要求显式 eval/smoke_eval 场景角色、正确 profile/时限、场景 ID、场景种子和去身份字段后的初始任务配置均不得与实际训练重合：

```bash
python -m chapter3_bser.experiments.d2_repair_v1.evaluate evaluate \
  --checkpoint /path/to/new/checkpoint_final.pt \
  --manifest /path/to/held_out_manifest.json \
  --output outputs/chapter3/d2_repair_v1/collision_terminal/b2_new_eval \
  --execute
```

对同场景、同环境创新种子的多个修复版评估结果进行排序诊断：

```bash
python -m chapter3_bser.experiments.d2_repair_v1.evaluate ordering \
  --episodes /path/to/eval_a/episodes.jsonl /path/to/eval_b/episodes.jsonl \
  --output /path/to/new_ordering_report.json
```

新配置及修复版源文件统一使用 LF 换行，避免 Git 在 Linux 上转换换行后破坏精确来源校验。

没有配对数据时，只报告零个配对组，不得推断奖励已对齐。排序报告和所有实验输出均拒绝覆盖已有结果。

最新本地验证结果另见 `VERIFICATION.md`。这些检查不是 CI，也不是论文正式训练与比较。
