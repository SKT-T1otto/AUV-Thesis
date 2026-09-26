# 交给 GPT：HGR Phase2 debug 结果及需要继续判断的问题

这是一份真实已完成实验的结果分析材料，不是新实验授权。请先阅读事实与统计限制，再回答末尾问题。
请区分“实现与日志检查通过”“本次样本的数值表现”“总体梯度效率结论”，不要把任何一种等同于另外两种。
本次没有提出或实施 HGR 算法、estimator、Phase1 runtime、reward、environment、planner 或策略架构修改。

## 1. 当前可支持的结论

**debug 流程已完成，真实 handoff correction 已执行，但本次没有观察到足以抵偿额外采样成本的梯度修正。**

本批 HGR 与 Direct New MC 的平均单样本梯度 MSE 基本相同：475.8867 对 475.3126。
HGR 消耗 23,290 环境步，Direct New MC 消耗 3,313 步，前者为后者的 7.03 倍。
日志定义的平均 cost-normalized error 为 132,356.00 对 24,389.89，比例为 5.43。

进一步拆开同一条 HGR 样本的 g0 和 g_delta 后，发现修正几乎没有改变梯度，
因此不能把 HGR 与独立 Old MC 样本间的差别解释为校正的效果。
当前只运行 10 场景 × 每方法每场景 2 次重复，reference 每场景也只有 2 条。
这不足以证明 HGR 一般有效或无效，也不足以给出稳定的优劣排序或论文级显著性结论。

## 2. 数据与实验定义

唯一发现的 Phase2 结果来自：

`E:\gym\code\WORKSPACE\AUV-Thesis\outputs\chapter3\hgr_phase2\collision_terminal\debug_gradient_efficiency_10scene\phase2_results.json`

- 结果文件最后修改时间：2026-09-26 22:08:31（本机 Asia/Shanghai）。这不是起止时间或墙钟耗时。
- 文件 SHA256：`02f932372fe272c3c7ebb0050efdcae9e49ca44fd0671c14cc49d2fb578213fb`。
- 结果 schema：`hgr.phase2.gradient_efficiency.v1`。
- 运行源码 SHA256：`100f369e2c1958a065905e379bd13ca63cdc7f76000352f2ee8fa267ff9aabca`。
- Git HEAD：`16ae663fdc3d9d702f43e0f5de4bf0dfafaca7c3`；工作区有未提交的配置/文档等修改。
- frozen source SHA256：`4a60f4228ef8f8982f9f9ff0959a06574384fe043141e4dba6100e97d893278d`。
- policy pair SHA256：`fee275f981417e05cbc9ad4b98c677b9f27390d9052413f17b028f4402b49542`。

| 项目 | 实际配置 |
|---|---|
| scope | fresh_main：固定初始场景，每次重新采完整主轨迹 |
| 场景 | scene_00–scene_09，10 个预声明场景，无运行时子采样 |
| profile / horizon | M20_MOVING_UNKNOWN_MULTI / H=400 |
| task protocol / reward objective | collision_terminal_v1 / team_mean_v1 |
| 求导对象 | theta_minus，321,316 个参数；phi0、phi1 固定 |
| 比较样本 | 每方法 20 个，共 60 个；每场景每方法 2 个 |
| reference | 总计 20 条 Direct New MC，每场景 2 条 |
| HGR | 每样本 N=1，q=1，固定 K=8，zero predictor，lambda=0 |
| 配对 | policy_crn；仅续分支 old/new 共享策略噪声，环境流独立 |
| gamma | 0.95 |
| 参数更新 | 0 |

Direct New MC 是使用当前 HandoffPolicy 的 (theta,phi1) 直接计算前级梯度，
不是历史 B2 MADDPG 网络；Old MC 使用 (theta,phi0)，其目标本来与新目标不同。

策略来自已审计的真实训练周期：共享 theta 与 phi0 取自 ep48/cycle3，phi1 取自 ep64/cycle4。
cycle4 日志的 theta_before、phi0、phi1 哈希与张量对应；不是把两个最终 checkpoint 的不同 theta 混用。
来源选择在本次 Phase2 之前完成，未根据本次成功率、handoff 或 MSE 选择。

来源审计已经提示差异很小：后级参数差 L2=1.05659e-6；固定合成输入下 Gaussian 均值最大差=3.36440e-7。
这个合成检查不是 on-policy 行为距离，不过此次真实续分支回报差也很小，见第 5 节。

## 3. 完整性与核验边界

运行记录为 status=PASS、frozen_policy_check=PASS、handoff_correction_status=EXERCISED。
performance_claims_supported=false、formal_experiment=false；实际预算也确认这是 debug。

本次后处理从原始 NPZ 重算，没有调用实验或训练代码，核验了：

- 64 个被结果引用的 NPZ 文件全部存在，文件 SHA、数组 key/shape/dtype 和有限性匹配。
- 60 条比较样本与 20 条 reference 元数据完整，无重复 (scenario,method,repeat)。
- 所有比较样本的梯度范数、MSE、case_reference_mse、MSE×步数均与向量/日志一致。
- 三方法均值、逐维样本方差、方差 trace、场景内方差和成本汇总一致。
- 20 条 HGR 样本均满足逐元素 g_full=g0+g_delta。
- 40 对实际续分支的快照、pair_id、共享 policy seed、独立 environment seed 与终止时钟一致。
- 运行记录中的全部源码文件哈希与当前源码一致；原始结果与所有 NPZ 在分析前后未变。

reference 的原始逐轨迹梯度向量没有单独保存：可由已保存的场景均值重算 g_ref，
也可由两条样本的梯度范数与场景均值范数独立复核参考方差 trace；
无法仅靠当前输出独立重建参考梯度的完整逐维方差。这一限制不能写成“参考真值已验证”。
这些都是保存结果的一致性检查，不是重新执行环境，也不是无偏性的数学证明。

## 4. 三方法结果与指标口径

| 指标 | Direct New MC | Old MC | HGR |
|---|---:|---:|---:|
| 样本数 | 20 | 20 | 20 |
| mean_gradient_mse | 475.312568 | 379.816700 | 475.886673 |
| pooled variance trace（ddof=1） | 455.833551 | 361.228902 | 454.943814 |
| 方法总环境步数 | 3,313 | 4,365 | 23,290 |
| 平均每样本环境步数 | 165.65 | 218.25 | 1,164.50 |
| mean_cost_normalized_error | 24,389.886635 | 29,341.857812 | 132,355.997211 |
| mean_case_reference_mse | 827.311359 | 671.519035 | 696.677834 |
| 平均梯度向量的范数 | 3.489436 | 3.020198 | 4.476705 |
| 平均梯度相对 g_ref 的平方误差 | 42.270695 | 36.649243 | 43.690049 |
| 分层方法均值估计的方差 trace | 24.860523 | 20.291070 | 22.566531 |
| 自然 handoff 数 | 6/20 | 6/20 | 5/20 |

HGR / Direct New MC 比值：MSE 1.001208，pooled variance trace 0.998048，环境步数 7.029882，
平均 cost-normalized error 5.426675。它们是本批描述性比值，不带显著性结论。
Old MC 的低 MSE 也不能作为旧目标对新目标无偏或优于新策略的证据。

必须保持下列口径：

1. g_ref 是 10 个固定场景 reference 均值的等权平均。
2. gradient_mse=||g_sample-g_ref||²；表内平均 MSE 是 20 个单样本平方误差的均值。
3. ||mean(g_sample)-g_ref||² 是另一个指标，已另列，不可与单样本 MSE 混淆。
4. cost-normalized error 逐样本计算 MSE_i×steps_i，再平均；不是 mean(MSE)×mean(steps)，也不是乘总步数。
5. pooled variance 混合了场景间差异和场景内随机性。case_reference_mse 使用每场景仅 2 条 reference，仍很嘈杂。
6. 所有方法使用全部 10 场景，未筛除无交接/提前终止轨迹。不能把最有利的几个场景抽出作为总体结果。
7. 三方法的完整主轨迹流相互独立；相同 comparison pair_id 不表示使用了相同主轨迹。
8. 当前配置固定重复次数，三方法实际成本不同；这不是一次直接实施的等环境预算实验。

## 5. HGR 的额外成本是否产生了相应修正

HGR 的 20 条主轨迹只有 5 条发生自然交接。对应 40 对 old/new 续分支，即 80 条续分支。
其他 15 条没有交接，g_delta=0；它们的 K_actual=8 记录的是 8 个零标签抽样项，
不是执行了 8 对续分支。校正成本为零，不能把 K_actual 总和误算为实际 rollout 数。

| 成本组成 | 环境步数 |
|---|---:|
| HGR 主轨迹 | 4,046 |
| 旧策略校正续分支 | 9,622 |
| 新策略校正续分支 | 9,622 |
| HGR 合计 | 23,290 |

校正续分支共 19,244 步，占 HGR 成本 **82.6277%**。
40 对的 old/new 步数全部相同，轨迹 trace hash 全部不同，因此不能说两条分支完全重复。
40 对的回报差没有精确为零的项，但绝对回报差中位数仅 3.58586e-10，最大为 1.58800e-8。

| HGR 样本 | tau | gamma^tau | ||g0|| | ||g_delta|| | ||g_delta||/||g0|| | 双侧续分支总步数 |
|---|---:|---:|---:|---:|---:|---:|
| scene_00 repeat0 | 45 | 0.0994403 | 11.407989 | 3.93271e-9 | 3.44733e-10 | 2,988 |
| scene_00 repeat1 | 45 | 0.0994403 | 9.528576 | 1.92621e-8 | 2.02151e-9 | 2,740 |
| scene_03 repeat1 | 251 | 2.56227e-6 | 8.689766 | 9.63908e-13 | 1.10925e-13 | 2,384 |
| scene_04 repeat0 | 41 | 0.122087 | 10.720368 | 1.58165e-9 | 1.47537e-10 | 5,744 |
| scene_04 repeat1 | 42 | 0.115982 | 9.985920 | 4.02995e-9 | 4.03564e-10 | 5,388 |

**同一批 HGR 主轨迹内部的修正效果**：

- 只用已保存 g0 的平均 MSE：475.8866726391858。
- 使用 g_full 后：475.8866726237490。
- 稳定公式 2(g0-g_ref)·g_delta+||g_delta||² 给出的平均 MSE 改变量：**-1.54368e-8**。
- 这相当于在本批数据中增加 19,244 步，MSE 变化约为 1e-8 量级。

上述 g0 诊断是从同一批 HGR 向量推导出来的，不是额外跑了一个实验，也不是独立 Old MC 的结果。
原实现的严格旁路要求 phi 行为身份完全相同；当前两者不同，因此没有旁路，符合既有设计。
这不是请求把“小差异”偷偷当成“零差异”，也没有修改阈值或 budget。

续分支终止记录为 obstacle_collision 24 条、success 12 条、timeout 44 条，均为这 5 个交接快照的重复续分支。
不能把这些数除以 80 后当成总体任务成功率，更不能与历史 ep100 成功率比较。
主轨迹最终任务结果没有完整逐项保存在此日志中，不能由长度反推出成功/碰撞原因。

## 6. 不确定性、场景差异与成本总账

reference 梯度范数为 **4.677192**；均值估计的方差 trace 为 **18.313045**，
其平方根（向量均方意义的标准误尺度）为 **4.279374**，约为范数的 **91.49%**。
这不是 95% 置信区间，也不是已知的真实参考误差，但显示 reference 预算不足以给出精确方向判断。
各方法的平均梯度平方误差约 36.65–43.69，与各自均值噪声 trace 加 reference 噪声 trace 的规模接近。
不能据此可靠区分系统偏差。JSON 中均值梯度对 reference 的 cosine 为负，也不足以单独诊断梯度符号错误。

每场景每方法只有 2 次重复，场景内样本方差只有 1 个自由度。
Direct New MC 和 HGR 最大的 3 个单样本 MSE 分别占总 MSE 的 58.6585% 和 58.6589%。
均值受到少数高误差样本明显影响，不适合根据约 0.12% 的总体 MSE 差异排算法名次。

| 场景 | Direct 平均 MSE | HGR 平均 MSE | Direct 步数 | HGR 步数 | HGR handoff |
|---|---:|---:|---:|---:|---:|
| scene_00 | 150.127 | 124.522 | 415 | 6,146 | 2/2 |
| scene_01 | 82.098 | 97.015 | 415 | 800 | 0/2 |
| scene_02 | 570.588 | 870.935 | 93 | 93 | 0/2 |
| scene_03 | 104.443 | 84.817 | 455 | 3,184 | 1/2 |
| scene_04 | 96.665 | 108.054 | 800 | 11,932 | 2/2 |
| scene_05 | 336.709 | 467.793 | 76 | 76 | 0/2 |
| scene_06 | 82.922 | 67.355 | 150 | 149 | 0/2 |
| scene_07 | 2,210.785 | 1,962.649 | 42 | 42 | 0/2 |
| scene_08 | 41.739 | 57.580 | 800 | 800 | 0/2 |
| scene_09 | 1,077.050 | 918.145 | 67 | 68 | 0/2 |

scene_07 的主轨迹都仅有 21 步且无交接，误差很大；后级 handoff correction 对这种样本本来就为零。
实际额外预算主要花在 scene_00/03/04，不能指望它减少其他无交接样本的前级采样波动。
这是当前数据和校正作用范围的解释，不是错误实现的证明。

| 全部成本 | 环境步数 |
|---|---:|
| 历史策略来源准备（截至 ep64） | 44,610 |
| 本次 reference | 4,090 |
| 本次 Direct New MC | 3,313 |
| 本次 Old MC | 4,365 |
| 本次 HGR（含两侧校正） | 23,290 |
| 本次新增环境步数 | **35,058** |
| 日志 total_environment_steps（含历史准备） | **79,668** |

不要把 79,668 全部说成本次实验新消耗，也不要重复计入 HGR 主轨迹与续分支。
逐方法 MSE×成本不摊入共享 reference 或历史准备成本；当前文件没有足够的墙钟分解来比较 GPU/CPU 时间效率。

## 7. 对照代码后的解释与尚未证明的事

代码位置（均保持未改）：

- `chapter3_bser/experiments/hgr/phase2_gradient_efficiency.py:216`：调用现有 prefix_losses/gradient_vector；保存 g0、g_delta、g_full。
- 同文件 `:326`：每个 HGR 样本新采一条旧策略主轨迹，发生交接时再采 K 对续分支。
- 同文件 `:491`：reference 方差按场景分层合成。
- 同文件 `:507`、`:530`：样本误差及 cost-normalized error 的实际口径。
- `chapter3_bser/models/hgr/estimator.py:79`、`:89`：交接校正乘 gamma^tau，与固定 K 均值相乘。

当前 zero-predictor、N=1、q=1 的公式为：

    g_full = g0 + g_delta
    g_delta = 1_handoff * gamma^tau * sum(t<tau, grad_theta log pi_theta(a_t|h_t))
              * mean(k=1..K, G_plus_new,k - G_plus_old,k)

据此，已观察到的现象与以下解释一致：

1. 真实 phi 差异很小，40 对实际续分支回报差也极小；correction 对 g0 的相对影响约 1e-13–2e-9。
2. 晚交接进一步被既有 gamma^tau 衰减，例如 tau=251 的样本。没有理由据此自动修改 gamma 或 reward。
3. 固定 K=8 仍支付续分支成本；精确身份不同使严格零更新旁路不适用。
4. 本实验每个 replicate 重新采 g0，并完整计入其成本。它衡量的是**单次完整主轨迹加校正**的梯度估计，
   没有跨训练周期复用或摊销历史 g0 的成本，因此不能直接推断完整训练流程中的跨周期复用收益。
5. 方法之间的独立主轨迹采样造成不同样本值与终止长度；独立 Old MC 的均值差不能替代 HGR 内部 g0→g_full 的比较。

本次没有发现保存的向量、误差公式、账目、配对或截止时钟的明显不一致。
但是后处理检查不能排除所有 simulator bug，也没有证明估计器在总体上的无偏性、方差界或训练收敛性。

## 8. 请 GPT 继续回答

请基于以上真实结果和公式，按“事实 / 解释 / 无法判断 / 最小下一步”组织回答：

1. 这次应归类为“运行流程通过、校正信号极弱”，还是已经足以判断 HGR 数学机制无效？逐条说明证据范围。
2. 在本组几乎相同的 phi0/phi1 下，放大 repeat/reference 主要解决什么问题，不能解决什么问题？
   请把估计精度不足与目标策略变化过小分开，不要笼统要求多训练。
3. 本框架的 fresh_main、N=1、zero predictor、g0 全额计费，是否对应论文想论证的梯度复用优势？
   如不对应，指出需要补充的实验定义，而不是直接改 estimator 或偷偷免去成本。
4. 当前 mean(MSE_i×steps_i) 与等预算平均梯度 MSE 的关系和区别是什么？正式实验应怎样避免混淆？
5. 在不挑选成功案例、不改变 reward/environment/planner、不随机扰动 phi 的前提下，
   给出最小的后续验证顺序；哪些只需重分析，哪些需要另行授权运行或准备新来源？
6. 如果怀疑实现问题，请明确可证伪的检查与预期现象；不能仅凭 HGR 未占优或 cosine 为负就认定代码错。

请勿把 debug 的描述性比值写成显著改善/显著退化，不根据此处数据宣称训练成功率变化，
不要自动提出 adaptive budget、跨版本 predictor 或算法重设计作为已经获准的修改。

附件 `evidence.json` 包含完整精度的三方法汇总、30 个场景×方法条目、60 条比较样本摘要、
20 条 HGR 分量诊断、40 对续分支摘要与 64 个向量文件校验和。`recompute.py` 是本次数值复核代码，
只读取原始结果，写入新的分析摘要，不执行 rollout。无需把 321,316 维全部向量粘贴到对话中。
