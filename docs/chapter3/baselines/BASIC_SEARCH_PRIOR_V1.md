# 仅搜索收益规划与先验控制基线

新增方法 `ch3_basic_search_prior_v1`，英文 Search-only Planning + Prior-only Control，简称 Basic Search-Prior / B0。这里的 B0 是本基线简称，不是生产运行时枚举 `B0_LEGACY_V2_1` 的方法身份。本方法无需模型文件或训练。

## 方法合同

Searcher 目标为 `F_search(A) = sum_c b(c) [1 - product_(a in A)(1 - p_a(c))]`。直接复用现有候选生成、检测概率、可行路径服务及 `solve_search_only_greedy`。检测模型原本的时间、距离因素保留；不生成联合待命候选，不使用 Executor 响应权重，不使用学习价值头或 BEDS 排序。

完整分配将 solver 的 `selected` 直接转换为 SearchAssignment，独立建立 anchor ExecutorAssignment，所以 `standby=None` 不再丢弃搜索任务。零边际收益时保留原 greedy 停止条件；没有候选、路径不可达或没有正收益选择均计入诊断，不重新抽场景。

局部重规划固定未受影响 Searcher 的原对象和路径。对受影响候选使用 `b(c) * [1-P_fixed(c)]` 计算条件边际发现收益，调用同一 search-only greedy；合并后再用原 `evaluate_objective(search_only=True)` 求整个分配的收益。缺失受影响搜索路线或 greedy 选择时原子拒绝。保留原事件、路径影响检测、冷却和 waypoint 稳定机制。原控制器只比较 allocator 提供的收益；外层 controller 在比较前、稳定化后对实际保留的搜索分配重新求 search-only 值，避免缓存分值与实际路径不一致。

每个 episode 完成原 reset 和公共状态初始化后，只记录一次 Executor 初始位置为 anchor。交接前，完整分配和 Executor-invalid fallback 都查询到该 anchor 的路径。anchor 不可达时，仍保留 anchor 作为任务终点，复用 bridge 的不可达当前位置保持；记录原因和漂移，不改位置、速度、冻结标志或物理扰动。Found 不等于 handoff。原 corrected controller 在 Executor 合法获得消息后，切换到公共目标路径，并继续原动态公开目标更新、路径失效重规划、接近/接触/保持/完成流程。

每一步直接创建 CPU `zeros((4,3))` 传给原环境包装链；检查 shape、有限性和严格零，随后检查物理 `_last_residual_acc` 仍严格为零。保留 `use_residual_prior=True`、原先验强度、残差缩放、限幅和动力学；物理先验通常非零。没有 policy 实例、Actor forward、探索采样、gate/专家推理、critic、replay、predictor、HGR 分支或 optimizer。诊断中的三个学习调用计数为无学习组件架构的零计数；物理残差由每步断言核查。

依然复用第三章 belief、候选、路径和先验控制基础设施，不声称是“没有第三章组件”，也不冒充已有标准算法。在相同候选、belief、检测模型下，纯 Executor 响应权重不影响评分和选择；共享感知地图、合法通信、运动及障碍约束仍可能形成真实物理耦合。

## 参考配置与来源

入口要求显式提供 `--reference-training-config`。优先使用实际 HGR run 的 `config.json`。复用生产 HGR 配置校验及评价配置解析，并按 `_make_base_env` 相同的 `build_ch3_config` / `environment_kwargs_from_config` 流程构造原 MissionCoreEnv，再复用 GuidedEnv、Phase1CV2TrainingEnv、PRRACTrainingEnv、OnlinePlanningStateProvider、RMADDPGGuidanceBridge。包装器名称中有 Training 不代表启动训练；它们提供原奖励和事件记账。

额外显式关闭构造参数 `pse_use_standby`，防止原环境中独立的后台 PSE 待命优化（包括构造/reset 期间）仍然运行。该开关只作用于 standby planner，不关闭 belief、地图感知或路径跟踪。它作为待命干预记录在 SPEC 中，从公共任务条件中分离；完整 reference 原值和实际 `effective_environment_config` 分别保留。解析器检查实际构造 kwargs 与 HGR 相比只能有这一项差异，没有全局 monkeypatch 或生产源码修改。

v1 要求 HGR、M20_MOVING_UNKNOWN_MULTI、400 步、28D/3D、三 Searcher 后一 Executor、collision_terminal_v1、segment_closed_aabb_v1、team_failure_override_v1、碰撞奖励 -2、team_mean_v1、gamma 0.95、legacy dynamic_public_intercept_v2_1。环境 kwargs、执行运行时参数、奖励 adapter 参数来自实际 reference 解析，完整 reference 另行保存。环境构造与 HGR 相同：只通过原 factory 支持的配置字段传递参数，不把 reference 任意字段猜作新增环境开关。

checked-in 配置路径标记为 `default_config_reference`；`outputs` 下的显式配置标为 `run_config_reference`，这个标签只是配置来源分类，不证明它与某个 checkpoint 绑定。Windows 未找到用户指定的 HGR 实际 run 配置，未核查 Linux 文件。参考方法 `ch3_hgr` 与实际运行方法分别记录。

`frozen_production_source.json` 新增记录本次开始时 `core` 和 `chapter3_bser` 的完整 Python 清单及字节 SHA256，启动时严格核对。未修改生产目录，也未改写历史 provenance。每次评价还记录外层工具全部 Python、Chapter 3 JSON 配置和两个启动脚本的独立清单/哈希，验证开始、逐场及结束来源，并核对 reference 和 manifest 文件。额外的 `delivery_manifest.json` 是交付时文件清单，生成后不自包含。不能用新的 baseline 清单替代 HGR 原来的 checkpoint 来源校验。

## 评价输入与输出

仅实现 CPU 串行，无 workers/device/resume 参数。输出必须是新目录或空目录，解析后的路径必须含独立 `collision_terminal` 组件；非空目录直接拒绝。没有 checkpoint 参数，无模型加载或新训练权重。

`--manifest` 必须是用户已冻结的现有文件。验证数量、全部 scenario_id 唯一、非负合法 int64 scenario_seed、validation 标签、profile 和至少 400 步 horizon。保留所选前 N 场的顺序与内容，记录原始文件哈希及选中内容哈希；不生成、覆盖或修正 manifest，不按失败重抽场景。标签检查不证明训练/验证集全局无重叠。

随机流沿用 HGR runtime 的 `seed_innovations(seed + evaluation_episode_index)`，在构造前和 reset 后设置；场景自身 scenario/planner/target motion seed 保留。没有 policy RNG 抽样，不关闭环境随机性，不消耗网络初始化随机数。同 seed 不保证不同方法的整条轨迹逐项随机事件一致，也没有复制 HGR pairing_sha256。

输出包括：

- `resolved_evaluation_config.json`：完整 reference 与解析后公共任务条件、方法干预。
- `baseline_spec.json`、`evaluation_manifest.json`、`evaluation_identity.json`：方法、场景、来源、随机流和 checkout 起止状态。
- `evaluation_progress.json`、`episodes.json`、`summary.json`：进度、原任务逐场结果和汇总。
- `controller_diagnostics.json`：零残差、非零先验、分配/重规划、候选/路径失败、anchor 漂移、Found/handoff、接触/保持、首次碰撞。
- `run_console.log`：每完成一场 flush 一行。异常另写 `evaluation_failure.json`，保存失败场景及异常，不伪装成超时。

使用原 finalize_episode、strict_outcome、validated_rows、aggregate_task_outcomes。不可用的 contact/hold 计数为 null。原候选生成接口对不可达路径只暴露数量，无法获取细粒度原因时标明 `SEARCH_PATH_UNREACHABLE_UNSPECIFIED_BY_GENERATOR`，不编造原因。未完成全部场景并通过最终来源核查前，正式率值及平均回报为 null；异常中已消耗的物理步也保留。

保留 HGR 现有映射：`handoff_event_step` 取任务接口的 `handoff_step`。生产 `_publish_detection` 将这个字段设为发布步，所以可与 Found 同号，不能将它误解为接收步。实际合法接收另见原 `executor_target_received_step`，首次合法接收后的控制决策见 `handoff_decision_step`。测试分别核查这些语义。

`evaluation_identity.sha256` 标识开始时身份（计算时不含 sha256 自身）；结束后追加 sources_after、checkout_after、end_state_sha256，不修改开始时哈希。不要将该字段解释为整个可追加 JSON 文件的字节哈希。

## 手动运行命令

使用已激活的现有 AUV Python 环境，不安装依赖。脚本从自身位置定位仓库，参数完整透传，保留 Python 退出码。下面全部为单行命令，本轮不自动执行正式评价。

Windows PowerShell（需先将冻结 validation manifest 放在示例路径；此处明确使用 checked-in 默认参考）：

```powershell
& 'E:\gym\code\WORKSPACE\AUV-Thesis\scripts\run_ch3_basic_prior_eval.bat' --reference-training-config 'E:\gym\code\WORKSPACE\AUV-Thesis\configs\chapter3\hgr_train.json' --manifest 'E:\gym\code\WORKSPACE\AUV-Thesis\outputs\chapter3\final_manifests\ch3_validation_M20_seed12729_n100.json' --episodes 100 --seed 12729 --output-dir 'E:\gym\code\WORKSPACE\AUV-Thesis\outputs\chapter3\baselines\collision_terminal\basic_search_prior_eval100_seed12729_v1'
```

Linux（用户自行部署新增文件到目标 checkout 后运行；本轮没有 SSH、修改或运行 Linux 目录）：

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/run_ch3_basic_prior_eval.sh --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --manifest /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/final_manifests/ch3_validation_M20_seed12729_n100.json --episodes 100 --seed 12729 --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/collision_terminal/basic_search_prior_eval100_seed12729_v1
```

Linux 动态查看进度，不依赖另一 shell 的变量：

```bash
tail -F /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/collision_terminal/basic_search_prior_eval100_seed12729_v1/run_console.log
```

本地测试：

```powershell
python -B -m unittest tests.test_ch3_basic_search_prior -v
```

## 比较边界

它是完整搜索—交接—执行任务的基础能力参照，相比完整 HGR 同时去掉联合规划、优化待命、学习残差。因此差异不能单独归因于 HGR 梯度校正；后续仍需要 BSER+Prior-only、Direct-MC 等桥接对照，本轮没有实现或训练这些额外对照。结果可能高于或低于 HGR，必须原样保留，无成功率高低验收条件。

合成/故障注入测试仅验证接口。真实数值复现限于相同输入、源码、依赖与平台；跨平台浮点与图搜索细节可能产生差异，不能声称 Linux 正式评价已经通过。正式 100 场由用户随后手动运行。
