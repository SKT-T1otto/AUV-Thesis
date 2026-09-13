# HGR 评价入口收尾（2026-09-13）

本轮仅修改入口、评价适配与身份校验。工作基线为本地及远程默认分支 `main` 的 `0fed8e1650f2747bb58f17bc0b375b69870ea0b5`；通过远程接口实际核对，未假设已知 SHA 必然仍是最新。开始时工作区干净。没有 commit、push、正式长训练或 100 场性能评价；不修改保留的旧 checkpoint、配置或输出。

## 实现与兼容

| 生产文件 | 职责 |
|---|---|
| `chapter3_bser/experiments/hgr/train.py` | `validated_output` 先 resolve，再复用 `require_protocol_output`，拒绝非空目录；覆盖 train/resume/mean-initialization 的 API 与 CLI。checkpoint 额外验证源码清单聚合及不可变配置摘要。 |
| `chapter3_bser/experiments/hgr/provenance.py` | 按既有清单哈希规范校验来源、报告精确差异文件；增加不走缓存的实际源码扫描与 HEAD/dirty 记录。 |
| `chapter3_bser/experiments/hgr/evaluation.py` | 三个随机方法唯一固定模型评价器；完整配置/场景/流身份、共享严格任务统计、逐回合原子保存及异常记录、前后输入/模型/源码只读验证。 |
| `chapter3_bser/experiments/hgr/cli.py` | 保留 evaluate 的函数入口与命令参数，转交同一评价模块；异常和中断退出非零。 |

既有 `episodes.json` 行与 `summary.json` 对象继续存在。完整评价保留原 `episodes`、`method`、`policy_mode`、`safe_success_rate` 及两种回报均值等字段；增加协议、完整性、计数、事件分解与身份字段。旧缺失字段不会被解释为 False。HGR `actual_length` 必须等于真实 collector 转移数，显式适配 `episode_length`；contact/hold 来自权威累计计数，缺失保持 null。任务结局由 PRRAC-team 同样使用的 `strict_outcome`、`validated_rows`、`aggregate_task_outcomes` 判定。

输出路径必须在归一化后包含独立的 `collision_terminal` 组件；已有空目录允许使用，非空目录及文件拒绝。不会自动清理或另选输出。检查在建输出目录、建环境和采集前完成，checkpoint 元数据只读检查可先进行。旧 legacy 保护函数的行为没有改变。

## 身份口径

`resolved_evaluation_config.json` 保存实际 episodes、seed、mode、gamma、时域、方法、策略与环境配置、Phase 1B 在线控制参数及执行 runtime。环境参数复用公共 `environment_kwargs_from_config` 过滤，候选配置中未被环境构造器使用的学习率、replay、训练回合数等不进入实际执行配置；Phase 1B 的历史离线实验清单也不冒充当前任务配置，其完整参考配置另以规范化 SHA256 标识。训练路径、训练计数和训练优化参数不冒充评价配置。完整 checkpoint 训练配置和实际训练成本保存在 `evaluation_identity.json/checkpoint_training`。评价 `actual_training_updates=0` 仅指本次评价。

`evaluation_manifest.json` 保留所选完整场景及原始顺序。外部输入另记绝对原路径、原始字节 SHA256、原 schema；生成输入记录生成器、源码 SHA、generator_seed、profile/split/数量。选中列表的内容 SHA 为排序键、无额外空白、UTF-8、禁止 NaN 的规范 JSON SHA256。只选择前 `episodes` 项并核对数量、唯一 scenario_id、整数 scenario_seed、profile、时域和 validation 标签。不改标签或补生成替代输入；标签校验不构成对所有训练数据无重叠的证明。测试里重用的场景明确为接口夹具。

`evaluation_identity.sha256` 对 canonical 对象计算同样的规范 SHA，覆盖 checkpoint 文件 SHA、实际 policy state_dict 摘要、完整生产源码内容摘要、实际评价配置、选中场景内容和随机流。绝对输出路径、时间、Git dirty、外部 JSON 缩进不参与 canonical；原始文件 SHA 与内容 SHA 单独命名。checkpoint 训练出处仍完整保存，未覆盖原 metadata。`pairing_sha256` 排除 method/algorithm/策略结构/checkpoint，保留环境、gamma、协议、场景、模式和流，可用于不同随机方法的同条件配对；不能将其当作性能胜负判断。

采样仍为零基 `policy_sampling_seed=seed+evaluation_episode_index`，与场景 generator_seed 和 scenario_seed 分别记录。临时网络构造保护 Torch RNG；评价不调用优化器。`deterministic_mean` 与 `stochastic` 分开记录，前者不声称估计随机策略的期望回报。HGR 仍为 CPU 串行采集，未增加 workers/device 参数。

源码文件清单聚合沿用既有 `json.dumps(files, sort_keys=True)` 的哈希规范。评价默认要求 checkpoint 清单及聚合有效，并与当前全部 `core`/`chapter3_bser` Python 源文件精确一致。修补前 checkpoint 即使网络兼容，也默认要求回到其原固定代码版本评价。本轮没有迁移白名单或忽略校验开关，没有改写旧 checkpoint 来源。resume 原检查保留。

运行前后重新读 checkpoint 字节、policy 全部参数及 buffer、实际磁盘源码、有效 runtime 配置和外部 manifest；结束检查不使用 source_identity 缓存。HEAD 与 dirty 只作出处，内容摘要才决定一致性。已保存的配置/manifest/identity 在结束时还会核对是否被外部修改。

## 中断语义

开始采集前即写入完整身份、空 episode 列表、`evaluation_complete=false` 的 summary/progress。每个合法 episode 后原子替换三个进度文件；全过程未全部完成前，整体率值和回报均值均为 null。异常及可捕获 KeyboardInterrupt 保存失败索引、完整 scenario、异常类型、消息、traceback 和已有合法前缀，然后重新抛出使 CLI 退出非零。不重采样、不跳过，不将程序错误计为碰撞或超时。

只有全数合法回合、输入与模型终检均通过后才发布 complete，progress 最后写入。消费者应同时核对 summary/progress 的完整标志、身份和有效数。文件替换是逐文件原子的，不是多文件事务；强杀不能保证清理，最后一次已保存的 incomplete 进度仍可辨识未完成 run。重新执行要求新目录，不实现断点续评。

## 验证记录

完整入口回归位于 `final_02/`：100 项实际执行通过，源码前后保持一致。最后一次仅修正配置元数据的投影后，重新固定源码运行相关专项，记录位于 `final_metadata/`；该次包含真实生产评价、固定旧提交行为对照、输出与身份、异常、HGR 数学、团队奖励、碰撞协议、27 条来源记录及 28D 契约。两次运行分别记录发现/执行名单、失败、错误、跳过、源码摘要、平台与依赖、开始结束时间和退出码，不合并为一次执行计数。

独立测试模块复用现有 runner 并行执行，每个 HGR 方法的采集仍为 CPU 串行。最终源码的真实评价与行为对照在 `final_metadata/hgr_entry_metadata/bounded_evaluation/`；完整入口回归的训练/恢复/两种基线及其评价紧凑证据在 `final_02/hgr_entry/production_entries/`。后续元数据投影未改变这些训练、恢复、采集、奖励或更新函数。测试模型由临时目录真实完整有界 cycle 产生，未复制或提交模型。

第一次固定树尝试 `final/` 因复核发现合法的可选配置默认值处理缺陷而主动停止；它的部分日志不作为最终回归通过依据。直接复现为：训练配置校验接受省略 `collision_terminal_reward` 的配置，评价解析却抛出 KeyError。修复后从实际环境基类构造参数读取默认值，显式解析缺省协议版本和候选配置，并补充等价回归；奖励值仍为原默认 `-2.0`。额外确保异常的非有限 episode 数据在进入合法前缀前被拒绝。之后重新固定源码并完整重跑，不合并旧日志凑数。

参考行为固定为 `0fed8e1650f2747bb58f17bc0b375b69870ea0b5`，不使用浮动 HEAD。两个源码副本在相同 Windows/Python/Torch/NumPy 下，使用同一 checkpoint、8 步合法小场景、seed 839，分别运行真实 CLI 与 collector。只读观察器逐项比对全部记录数组原始字节、浮点十六进制、任务结果与 Found/交接事件。明确新增字段名单为 `evaluation_episode_index`、`policy_sampling_seed`、`contact_episode`、`hold_episode`、`episode_length`、`wall_seconds`、`optimizer_update_count`、`failure_stage`；原有 episode 字段及 summary 公共字段全部逐项相等。修补前程序加载测试模型仅用于此隔离行为对照，不能绕过修补后生产评价的来源门禁。

旧算法验收记录保持原样。本轮历史输入检查另存 `historical/` 和 `golden/`，缺失时明确 blocked，不生成替代黄金数据。当前本地 Windows 结果不代表 Linux 主机通过。正式实验与性能结论状态始终为 false。

## 最终本地验收（2026-09-14）

| 独立执行 | 发现 | 实际执行 / 通过 | 失败 / 错误 / 跳过 | 模块 | 退出码 | 用时 |
|---|---:|---:|---|---:|---:|---:|
| `final_metadata`：最终源码相关专项 | 61 | 61 / 61 | 0 / 0 / 0 | 7 | 0 | 677.681 秒 |
| `final_02`：元数据投影修正前的完整相关回归 | 100 | 100 / 100 | 0 / 0 / 0 | 17 | 0 | 1860.220 秒 |

两次均严格匹配发现与执行名单，源码在各自测试期间不变。最终源码专项包含新评价模块的 11 项测试、HGR 数学 8 项、团队奖励 5 项、共享评价指标 6 项、碰撞协议 24 项、仓库来源 6 项及观测契约 1 项。此前完整回归另含真实 HGR 训练、恢复、两种直接基线及其固定策略评价等检查；这 100 项发生在最终配置元数据修正之前，不能描述为最终树上的 100 项通过，也不与 61 项相加。

最终验收源码树 SHA256 为 `7e66931d838817395ec1a7d62a16420cd1cdbd1db6a24fca0e403d5c91b26753`；此前完整回归为 `463078eb225373c6bcf1cac62e02dfd57c27a7ba5e306aec67bc242e3816ba81`。源码树口径包含测试、配置与启动脚本，排除验收产物。最终 checkpoint/评价所用生产 Python 源码摘要为 `7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491`，与保存的全部逐文件 SHA 及最终磁盘源码再次核对一致。

实际环境为 Windows 10 build 26200、Python 3.10.20、Torch 2.11.0+cu126、NumPy 2.2.6；HGR 评价实际使用 CPU 串行。最终专项于 UTC 2026-09-13 16:27:14 开始、16:38:32 结束（北京时间 2026-09-14 00:27–00:38）。这是本地验证，不是 CI 或 Linux 验证。

真实固定模型对照在 stochastic 与 deterministic_mean 下各执行 1 回合、8 步：Found 为第 1 步，交接决策为第 2 步，最终均为 timeout。两种模式各自与固定旧版本的动作、奖励、事件及原输出字段精确一致，二者采样轨迹不同。模型与 checkpoint 均未改变，评价训练更新为 0。这些复用训练场景的接口夹具不构成独立性能样本。最终六类输出、32 个必需行字段、规范化身份、完整场景、实际配置、计数和时钟的再读回均通过，见 [artifact_readback.json](artifact_readback.json)。此前完整回归的三个随机方法还验证了同场景/任务/流的配对 hash 相同。

历史检查发现 4 项，实际运行 1 项并通过，其余 3 项因冻结的离线 BSER 文件、E0 结果或旧 overlay manifest 缺失而阻塞；独立 golden 检查还缺少 `golden_trace_manifest.json`，实际可验轨迹为 0。这两类历史检查均退出 1，保持 `blocked_missing_historical_input`，没有伪造、跳过后宣称通过或削弱 27 条来源检查。

| 交付状态 | 值 |
|---|---|
| `evaluation_hardening_complete` | `true` |
| `current_regression_passed` | `true`（最终专项；完整回归版本另列） |
| `bounded_evaluation_passed` | `true` |
| `historical_verification_status` | `blocked_missing_historical_input` |
| `formal_experiments_completed` | `false` |
| `performance_claims_supported` | `false` |

机器可读状态与完整执行命令见 [acceptance_summary.json](acceptance_summary.json)，准确的一行式 Windows/Linux 训练、恢复及评价命令见 [HGR README](../../README.md#手动命令)。Git 的 diff 统计、空白检查、未跟踪文件清单与产物忽略检查见 [git_review.txt](git_review.txt) 和 [repository_artifacts_check.json](repository_artifacts_check.json)。修改保留在工作区，未 commit、push；旧输出及模型未修改。后续提交、推送和正式 Linux 实验由用户手动执行。
