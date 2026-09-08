# Found-State Executor Geometry 离线诊断

现有 canonical 100 场结果足够计算全部 Found 场景的真实三维距离。当前样本的关联方向为负，但置信区间包含零效应，**尚不支持“稳定负关联”**。这是条件于 Found 的观察性分析，不是因果结论，也不代表跨种子、checkpoint 或分布的稳定性。

## 数据与覆盖率

本机输入是 `E:/gym/code/WORKSPACE/3090结果/907/analysis_bundle/full_prrac`。其 `resolved_evaluation_config.json` 中的原始输出目录为 `/home/legion/AUV-Thesis/AUV-Thesis-D1D2/outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/full_prrac`。不是依据文件夹名称或预期计数认定身份：脚本核对 evaluation manifest、resolved config、逐场 scenario ID/seed、checkpoint、manifest identity、正式模式以及 OFF + B1 + C2 + native 配置和 28D/3D/124D 契约。

| 计数 | 实际值 |
| --- | ---: |
| 原始 episode 数 | 100 |
| 唯一 scenario 数 | 100 |
| Found | 55 |
| Found 中 Success | 32 |
| Found-but-failed | 23 |
| Found 中 Contact | 32 |
| Found 中缺少主距离 | 0 |
| Found 中缺少 found_step | 0 |

所有 55 个 Found episode 均纳入主分析；没有混入 `searcher_residual_off`、15 个 post-hoc discordant trace 或逐步重复样本。

## 原始字段与事件时序

主变量直接读取 `episode_evaluation.csv.executor_distance_to_target_at_found`。

`chapter3_bser/experiments/phase1c_common/execution_diagnostics.py` 的 `_distances()` 读取物理 `_agent_pos[3]` 和 `target_state.position`，转换为 float64 三维向量，计算欧氏范数。`observe_step()` 在首次 `task_after.target_found` 为真且尚未记录 `found_step` 时保存该距离和 `task_after.step`，后续步不会覆盖它。

`phase1c_bser_rmaddpg_v2/training_env.py` 先执行底层 `env.step(actions)`，再读取 `task_after` 并调用诊断。PRRAC wrapper 沿用该路径。`core/env/uav_env.py` 在 agent 动力学更新及 target advance 后做 swept detection；`_publish_detection()` 使用该步结束的状态发布 Found。因此 Found step 是从 1 开始的已完成物理步编号，距离属于 **post-env.step 的发布状态**。swept 检测中的步内最近点参数 tau 没有被用于 Found 距离，也没有在本分析中插值。

已有 `scripts/analyze_searcher_residual_effect.py` 的 `found_state_comparison.csv.executor_target_distance` 通过别名映射读取 `executor_target_distance_at_found` 或 `executor_distance_to_target_at_found`。其 `found_execution_transition_comparison.csv.executor_distance_at_found` 也直接读取这些明确的 Found 字段。本数据提供的是后者，并非从 final distance 计算。在本机检查的结果目录中未找到已有 `found_state_comparison.csv` 成品，因此本次以原始 episode CSV 和生产该字段的源码为依据，不声称核验了未提供的旧分析成品。

原始完整 bundle 已逐文件列入 manifest 的 SHA-256 清单。其他 CSV 包含 episode 聚合、规划/恢复 activation 或有限 failure trace index，没有为所有 Found 场景保存两端 Found 坐标。本次不使用有限 trace 补齐位置。

55/55 场均缺少可用的 Found executor/target xyz、XY 距离、绝对 dz 和速度；输出对应字段全部为 `NA`，JSON 为 `null`。三维标量不能唯一确定这些量。速度扩展模型不拟合。未来最小补充是：在同一个首次 Found 的 post-step 诊断钩子中锁存并序列化 `_agent_pos[3]`、`target_state.position`、`target_state.velocity`、完成步号和 scenario ID；无需把诊断加入 actor 观测。此次没有添加 runtime instrumentation，也没有重跑实验。

## 主统计结果

bootstrap 固定 seed=1729，重复 5,000 次；每次按 55 个完整 scenario 有放回抽样，所有统计量及模型使用同一组抽样索引。区间为 percentile 95% CI，标准差使用样本标准差，分位数使用线性插值。

| d_found，米 | Success | Found-but-failed |
| --- | ---: | ---: |
| n | 32 | 23 |
| mean | 6.5852 | 7.4624 |
| std | 4.0151 | 2.8852 |
| median | 5.7351 | 7.0485 |
| Q1 | 3.6434 | 5.7443 |
| Q3 | 8.2037 | 9.8005 |
| min | 0.6668 | 2.6891 |
| max | 18.5989 | 13.6418 |

| 统计量 | 估计 | bootstrap 95% CI |
| --- | ---: | --- |
| mean difference，failed − success，米 | 0.8772 | [−0.9717, 2.6657] |
| median difference，failed − success，米 | 1.3134 | 未请求区间 |
| point-biserial r | −0.1219 | [−0.3941, 0.1265] |
| Spearman r，平均秩处理 ties | −0.1834 | [−0.4290, 0.0794] |
| ROC AUC，score = −distance | 0.6073 | [0.4519, 0.7535] |

没有选择控制阈值；这些数据不足以建立物理安全边界。

## Logistic 与 Found 时间调整

模型为无惩罚 Bernoulli MLE，计算前标准化连续变量，报告时换算回原始单位；使用 NumPy Newton 求解和回溯步长。脚本检测单一结局、常数/共线预测量、分离、病态信息矩阵及不收敛，并记录 bootstrap 拟合状态。没有新增 requirements。

| 模型与项 | 原始单位 beta | OR | bootstrap 95% CI of OR |
| --- | ---: | ---: | --- |
| Success ~ distance：distance 每 1m | −0.06975 | 0.9326 | [0.7402, 1.0746] |
| 同上：distance 每 5m | — | 0.7056 | [0.2223, 1.4328] |
| Success ~ distance + found_step：distance 每 1m | −0.11848 | 0.8883 | [0.6674, 1.0598] |
| 同上：distance 每 5m | — | 0.5530 | [0.1324, 1.3368] |
| 同上：found_step 每 10 steps | −0.008711 / step | 0.9166 | [0.8460, 0.9581] |

调整后 distance beta 的 bootstrap CI 为 [−0.40439, 0.05806]。distance 与 found_step 的相关系数是 −0.09761。控制 Found 时间后，distance 的点估计仍为负且绝对值增大，但区间仍包含零，因此不能确认稳定的距离关联，也不能断言它只是“更晚发现”的替代变量。Found 时间与成功的负关联在此样本中更明确。

两个原始模型均收敛；两组各 5,000 次 bootstrap 均有效，无分离或数值失败。JSON 另提供 Wald 区间、标准化系数、标准误、信息矩阵条件数和每类失败计数。若其他数据出现失败重抽样，区间仅基于有效重复，缺失重复数会明确输出；有效率低于 90% 时标注不稳定。

## 距离分箱与辅助变量

采用实际距离的四分位数边界；相等距离留在同一箱，重复边界和空箱显式合并。

| 箱 | 实际距离范围，米 | n | Success | Success rate |
| --- | --- | ---: | ---: | ---: |
| Q1 | 0.6668–4.4662 | 14 | 11 | 78.57% |
| Q2 | 4.8088–6.5071 | 14 | 7 | 50.00% |
| Q3 | 6.7876–9.1731 | 13 | 8 | 61.54% |
| Q4 | 9.3016–18.5989 | 14 | 6 | 42.86% |

分箱成功率并不单调。这支持谨慎解释，不应仅用首尾箱宣称稳定趋势。

全部 55 场 `Contact == Success`，因此未重复拟合同一模型。

Handoff 名称存在必须保留的语义差异：

- runtime 的 `handoff_step` 是发布步，`_publish_detection()` 将它与 `found_step` 同时赋值。因此可以从已记录的 found_step 无歧义导出；输出附带 `handoff_step_source=derived_runtime_publish_equals_found`，不伪称它是 CSV 已保存字段。
- CSV 的 `executor_target_received_step` 是 runtime 的 `executor_received_target_step`，表示下一步动力学开始前的 delivery step 标签。
- 输出 `handoff_delay_steps` 是发布步减 Found 步；55 场均为 0，std=0。
- 输出 `executor_receive_delay` 是接收步标签减 Found 步；55 场均为 1，std=0。
- CSV 旧字段 `handoff_delay` 实际是上述接收延迟。保留为 `source_handoff_delay` 用于核对，不能将其解释成发布延迟。
- CSV 的 `executor_distance_at_handoff` 是 `executor_distance_to_target_at_received` 的别名；诊断是在首次看到 receipt 的该步结束采样，不能当成下一步动力学开始前瞬间的 delivery 几何状态。

上述两个 delay 都是常数：`not explanatory because no episode-level variance`。

## 可复现运行与文件

在 Linux `AUV-Thesis-D1D2` 仓库根目录执行以下单行命令。输出目录必须不存在；脚本拒绝覆盖现有分析目录，也拒绝写入历史输入目录。

```bash
python scripts/analyze_found_executor_distance.py --source-dir outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/full_prrac --output-dir found_executor_distance_analysis_v1 --bootstrap-seed 1729 --bootstrap-repetitions 5000
```

生成四个文件：

- `found_executor_distance_episode.csv`：完整 55 场 Found 表，含 NA 字段、距离来源和 handoff 导出来源。
- `found_executor_distance_bins.csv`：距离分箱。
- `found_executor_distance_summary.json`：计数、逐字段缺失数量及完整 scenario IDs、统计、bootstrap、模型、辅助分析、限制。
- `found_executor_distance_manifest.json`：Git commit/status、原始输入路径及 SHA-256、完整输入 bundle 清单、源码 hashes、时序语义、seed、重复次数、版本及其他三个输出文件 hashes。manifest 不对自身做循环 hash。

本次实际产物保存在本仓库 `docs/found_executor_distance/`，可供 Git 审阅。数据输入的内部 scenario-manifest identity 与 manifest 文件本身的 SHA-256 是两个概念，均分别保存。输入 resolved config 没有记录可独立核验的历史 evaluation Git commit；本次记录当前分析 Git commit 和所审计源码 hashes，并明确保留这一来源限制。

## 最新记录的本地验证

新增 `tests/test_found_executor_distance_analysis.py`，25 项测试通过，覆盖用户要求的 Found 过滤、首次事件、物理 3D/XY/dz、post-step 时序、分组、缺失门禁、点二列相关、bootstrap 重现性、Logistic 方向和时间调整、AUC 方向、分箱、Contact 一致性、handoff 语义、manifest hashes 和历史输入不变。事件测试执行现有诊断源码的事件锁存/距离方法，并检查 wrapper 调用顺序，不加载 simulator 或 checkpoint。

相关历史离线分析 31 项测试通过；repository metadata 4 项测试通过，27 项历史 provenance 记录及允许演化约束保持有效。以上合计 60 项，为最新记录的本地验证，不是 CI。`git diff --check` 通过。

仅新增离线 analyzer、测试、说明和紧凑分析证据。没有修改 runtime、历史 inputs、Actor、Critic、BSER、C2、PathTracker、reward 或 checkpoint；没有训练、重跑 100 场、commit 或 push。
