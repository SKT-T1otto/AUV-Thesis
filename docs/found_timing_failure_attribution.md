# Found Timing and Geometry Failure Attribution Analysis

当前 canonical 100 场数据的自动描述性结论为 **Evidence favors early discovery limitation**。在完整 Found 子样本中，发现时间与最终成功的关联证据强于 Found 时 Executor 距离。该结论不把失败逐场判定为时间不足，也不意味着增加时限必然能解决失败。

本次完全离线，只新增 analyzer、测试、说明和紧凑分析文件。未修改 runtime、算法、历史 outputs 或 checkpoint，未训练、重跑实验、commit 或 push；不需要新增 runtime instrumentation。

## 输入审计

本机使用 `E:/gym/code/WORKSPACE/3090结果/907/analysis_bundle/full_prrac` 中的：

- `episode_evaluation.csv`
- `evaluation_manifest.json`
- `resolved_evaluation_config.json`
- `checkpoint_metadata.json`

Resolved evaluation config 记录的原始 Linux 输出路径是 `/home/legion/AUV-Thesis/AUV-Thesis-D1D2/outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/full_prrac`。身份判定依据文件内容，而非目录名称。已核验 100 个唯一 scenario 的完整集合、逐场 seed、generator seed=1729、checkpoint 路径/episode/config hash/runtime revision、full_prrac mode、OFF、B1_ATOMIC_LAST_VALID、S2A1_C2_LOCAL_CONNECTOR、native runtime，以及 28D/3D/124D 契约。

Checkpoint metadata 与 episode/config 中记录的身份一致；本次没有打开权重文件，因此不声称验证了 checkpoint 权重字节。输入 resolved config 没有独立记录历史 evaluation Git commit；analysis manifest 记录当前分析 commit、Git status 和审计源码 hashes，并保留这一来源限制。

Manifest 还保存整个原始 bundle 的文件路径及 SHA-256，分析前后逐文件比对，确保历史输入没有改变。

## Population 与失败构成

| 类别 | 定义 | count | 占全部 episode |
| --- | --- | ---: | ---: |
| A：Not Found | found=false | 45 | 45% |
| B：Found but Failed | found=true, success=false | 23 | 23% |
| C：Found and Success | found=true, success=true | 32 | 32% |

总失败数为 68；Not Found 占全部失败的 45/68=66.18%，Found 后失败占 23/68=33.82%。Found 条件失败率为 23/55=41.82%。JSON 的 `not_found_ratio` 和 `found_failure_ratio` 均以全部 100 场为分母，其他条件比例另行命名并注明分母。

A/B/C 是互斥的观察结果。Timing 与 geometry 是 B/C 中可能同时存在的关联因素，不能与 A 拼成三个互斥的失败机制。

全部 100 场进入 timing episode 表；45 场 Not Found 的 found_step 和 remaining_steps 均为 NA。所有模型、Found 分箱及 geometry 表使用全部 55 个 Found episode。不存在逐步重复样本，也没有补入 post-hoc trace。

## Found timing

Found step 与上一轮审计一致：首次 `task_after.target_found` 为真的 post-env.step 完成步号。距离也使用同一 post-step 的物理状态。Remaining steps 精确等于 `400 - found_step`，因此它是发现时间的等价表达，不能作为独立解释变量再次加入模型。

| Found step 统计 | Success | Found-but-failed |
| --- | ---: | ---: |
| n | 32 | 23 |
| mean | 79.1563 | 196.5652 |
| std，样本标准差 | 95.2359 | 136.1105 |
| median | 44.5 | 219.0 |
| Q1 | 22.0 | 58.5 |
| Q3 | 84.5 | 309.0 |
| min | 1 | 1 |
| max | 338 | 398 |

按 scenario 有放回抽样 5,000 次，seed=1729；所有统计和模型共用每次抽样索引。差值方向为 failed − success：

- Mean difference：117.409 steps，bootstrap 95% CI [50.528, 181.574]。
- Median difference：174.5 steps，bootstrap 95% CI [42.488, 243.5]。

固定 400-step horizon 的分箱结果如下。各箱概率均为 `P(Success | Found, timing bin)`。

| Found step bin | Found count | Success count | Success rate |
| --- | ---: | ---: | ---: |
| 0–100 | 32 | 25 | 78.125% |
| 101–200 | 5 | 3 | 60.000% |
| 201–300 | 10 | 2 | 20.000% |
| 301–400 | 8 | 2 | 25.000% |

较晚分箱成功率较低，但并非逐箱严格单调，而且后续箱样本量较小。

## Geometry 与联合模型

Geometry 直接读取原始 `executor_distance_to_target_at_found`，55/55 场均可用，无缺失样本。定义是物理 `_agent_pos[3]` 与真实 `target_state.position` 的三维欧氏距离；不用 final distance、初始位置或 trace 回推。

成功组平均距离 6.5852m，Found 后失败组 7.4624m，差值 0.8772m，bootstrap 95% CI [−0.9717, 2.6657]。中位数差 1.3134m，bootstrap 95% CI [−0.8223, 3.9138]。

以下三个模型均使用全部 55 个 Found episode；连续变量在拟合内部标准化，输出 beta 换回原始单位。原始 beta 单位分别为每 step 和每 metre，Found 的 OR 则按每 10 steps 表示。使用既有离线 NumPy MLE 实现，不新增 requirements。

| 模型 | 项 | beta，原始单位 | OR | Wald 95% CI of OR | bootstrap 95% CI of OR |
| --- | --- | ---: | ---: | --- | --- |
| Success ~ found_step | Found 每晚 10 steps | −0.008186 / step | 0.9214 | [0.8757, 0.9695] | [0.8560, 0.9654] |
| Success ~ distance | distance 每增加 1m | −0.069751 / m | 0.9326 | [0.8005, 1.0865] | [0.7402, 1.0746] |
| Success ~ found_step + distance | Found 每晚 10 steps | −0.008711 / step | 0.9166 | [0.8696, 0.9660] | [0.8460, 0.9581] |
| 同上 | distance 每增加 1m | −0.118475 / m | 0.8883 | [0.7477, 1.0553] | [0.6674, 1.0598] |

三个原始模型均收敛，分别 5,000 次 bootstrap 全部有效。JSON 保存 beta、标准误、标准化参数、beta 的 Wald/bootstrap CI、OR 的两类 CI、拟合状态和失败重复次数。联合模型的 timing、distance 结果与上一轮 geometry 分析一致。

`success_probability_summary.csv` 同时提供 timing bins 和 geometry quartiles 的经验概率。Geometry quartiles 的成功率依次为 78.57%、50.00%、61.54%、42.86%，分母仍然是对应距离箱内的 Found episode。它们不是包含 Not Found 的总体成功概率，也不是额外独立样本。

## 剩余 Contact 时间

Contact 与 Success 在全部 55 场 Found episode 上完全一致。32 个观察到 Contact 的 episode，Found→首次 Contact 耗时均值为 117.031 steps，中位数 92.5，范围 1–295。

成功组 Found 时平均剩余 320.844 steps；失败组平均剩余 203.435 steps，范围 2–399。23 个无 Contact 的 Found episode 全部运行到 400-step horizon，因此其 Contact 时间属于右删失：我们观察到了截至 horizon 仍未 Contact，但没有观察到“最终需要多少步才能 Contact”。

不能将成功组平均 Contact 耗时作为每个失败场景的必要时间阈值，也不能把这 23 场全部标记为“剩余时间不足”。失败组中甚至存在第 1 步就 Found、仍在 400 步未 Contact 的场景。Timing 的负关联较明确，但不足以独立解释所有 Found 后失败。

既有 runtime 的 publish 步等于 Found 步，receive 步等于 Found+1；这个延迟缺少 episode 间方差，本次没有将其作为解释变量。

## 自动描述性结论规则

规则在运行正式分析前写入脚本，不根据结果调阈值。某因素只有同时满足以下条件才标为受到支持：单变量模型和联合模型的 beta bootstrap/Wald 95% CI 都严格位于零以下；模型收敛；三种模型的有效 bootstrap 比例均不少于 90%；正式重复次数不少于 5,000。任何必需变量缺失、拟合不可用或重复不足都会阻止强行选择。

只允许用户指定的四种结论。当前输出为：

```json
{
  "interpretation_type": "descriptive",
  "conclusion": "Evidence favors early discovery limitation",
  "associated_factor": "discovery_timing",
  "population": "Found episodes only",
  "timing_supported": true,
  "geometry_supported": false
}
```

“Geometry 未获支持”不代表不存在距离关联。这里比较的是关联证据，而不是直接比较不同单位 beta 的绝对大小。结论也不外推到其他 seeds、checkpoints 或任务分布。

## 运行与文件

Linux 仓库根目录执行以下单行命令。`scripts/analyze_found_executor_distance.py` 是上一轮已新增的离线统计依赖，需要随本脚本一起保留；无需新依赖库。输出目录必须不存在，脚本拒绝覆盖或写入历史输入目录。

```bash
python scripts/analyze_found_timing_failure_attribution.py --source-dir outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/full_prrac --output-dir found_timing_failure_attribution_v1 --bootstrap-seed 1729 --bootstrap-repetitions 5000
```

输出六个文件：

- `analysis_manifest.json`：输入路径及 hashes、完整 input bundle、身份验证、Git commit/status、源码 hashes、时序语义、bootstrap 设置和其他五个输出 hashes。
- `found_timing_episode.csv`：完整 100 场，Not Found 时间为 NA。
- `found_timing_bins.csv`：全部 55 场 Found 的固定时间分箱。
- `found_geometry_episode.csv`：全部 55 场 Found 的原始三维距离。
- `success_probability_summary.csv`：时间分箱和距离四分位箱的条件成功概率。
- `failure_attribution_summary.json`：失败构成、时间/距离统计、三个模型、自动描述性结论、Contact 时间与删失说明。

本机正式离线分析产物位于本仓库 `docs/found_timing_failure_attribution_v1/`，用于 Git 可见的紧凑证据留存。Manifest 排除自身的循环 hash。若主距离缺失，geometry 模型/分箱停止，缺失数量与 scenario IDs 明确记录，geometry episode 表仍保留全部 Found 行；timing 数据完整时继续 timing 分析，命令返回 2 表示部分数据不足。

## 最新记录的本地验证

新增 `tests/test_found_timing_failure_attribution.py` 的 19 项测试通过，覆盖用户要求的十类行为，并增加结论规则、Contact 删失及直接 CLI 测试。联合运行本次测试、既有 geometry 25 项、既有 residual analyzer 31 项和 repository metadata 4 项，合计 **79 项通过**。27 条历史 provenance 记录及允许演化约束保持有效。

另以原始 CSV 独立复核计数、分箱、均值差及三个 Logistic score equations；与上一轮 geometry 输出核对联合模型和距离区间。全部输入与生成输出 hashes 复核通过，`git diff --check` 和新增文件空白检查通过。上述是最新记录的本地验证，不是 CI。
