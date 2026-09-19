# CH3 baseline framework：B0–B3

本框架为完整搜索、发现、合法交接、接近、接触保持和任务完成提供分层对照。实现位于 `tools/ch3_baselines/`，生产 `core/`、`chapter3_bser/`、原 HGR 算法、checkpoint 和来源校验均保持冻结。本次交付不启动训练或正式评价，不以成功率为验收条件。

## 方法与贡献链

| 注册键 | 论文 method | 规划 | 学习与残差来源 | 是否需要独立训练 |
|---|---|---|---|---|
| B0_search_prior | ch3_baseline_search_prior | Search-only | 无学习；严格零残差 | 否 |
| B1_bser_prior | ch3_baseline_bser_prior | 原生 BSER joint | 无学习；严格零残差 | 否 |
| B2_direct_mc | ch3_baseline_direct_mc | 原生 BSER joint | 随机 residual policy；完整 MC return | 是 |
| B3_direct_boundary | ch3_baseline_direct_boundary | 原生 BSER joint | 随机 residual policy；当前 boundary value + correction | 是 |
| HGR，既有参考方法 | ch3_hgr | 原生 BSER joint | 随机 residual policy；handoff gradient reconstruction | 本框架不管理 |

B0 衡量先验控制下搜索规划的完整任务基础能力；B0→B1 衡量联合响应规划和优化待命的增益。B1 是评估 RL residual 贡献所需的无学习对照，B1→B2 衡量普通 MC residual RL 的增益。B2→B3 研究 boundary information 与校正估计的贡献；B3→HGR 研究 old–new suffix difference 与 handoff gradient reconstruction 的贡献。

这是待检验的贡献链，不是性能结论。B0→B1 包含明确的 standby 干预，B2→B3 还改变估计器使用的信息和采样成本，不能将任一差值解释成无条件的单因素因果证明。B0 本身不能替代 HGR 的直接对照；B2/B3 是从各自原生算法入口独立训练的方法，不是对 HGR 已训练权重的 evaluation ablation。

## 实现边界

### B0：沿用既有实现

完整复用 `basic_search_prior.py` 中的 `BasicSearchPriorRuntime`，不改动已有 B0 Python、配置或旧脚本。其目标保持 `F_search(A) = sum_c b(c) P_A(c)`；不使用 Executor 响应权重，交接前保持初始 anchor，并保留原 belief、候选生成、路径、碰撞、合法交接与完成流程。`pse_use_standby=False` 是既有 B0 的显式方法干预。统一入口只增加论文身份和输出适配，原 B0 入口仍输出 `ch3_basic_search_prior_v1`。

### B1：原生 MissionRuntime + 零动作源

`BSERPriorRuntime` 继承原 `MissionRuntime`，直接调用其构造和 `advance`。唯一替换的是传入 `advance` 的动作源：`ZeroResidualSource.actions` 返回 CPU float32 `zeros((4,3))`，不消耗 action generator，不创建网络。

原 controller factory、`BSEROnlineAllocator`、联合目标 `F_joint(A,y) = sum_c b(c) P_A(c) w_y(c)`、待命规划、事件更新和 bridge 均沿用生产代码。构造时检查 allocator 必须为原生确切类型；B1 不引用 `SearchOnlyAllocator`。每一步核查命令残差和物理 `_last_residual_acc` 均为严格零，同时保留原 `_last_prior_acc`、动力学、限幅和残差先验开关。因此实际执行仍是原 `a_prior`。

没有 Actor forward、Gaussian sampling、BoundaryPredictor 或 optimizer。零学习调用计数来自无学习组件的结构；测试再用禁止调用的补丁验证，物理零残差由运行时逐步断言。补丁只存在于测试中。

### B2/B3：仅配置和路由

现有 `chapter3_bser.models.hgr.METHODS` 已包含 `hgr`、`stochastic_direct_mc`、`direct_boundary_corrected`，`chapter3_bser.experiments.hgr.train.Trainer` 已有对应实现，无需重写或复制算法。

- B2 使用原 `stochastic_direct_mc`：完整主轨迹的 MC return；不使用交接梯度重构。
- B3 使用原 `direct_boundary_corrected`：prefix reward gradient、当前 boundary value 与 correction；原 `paired_label` 只请求 current/new suffix，不计算 old–new difference，`old` 和 `delta_hat` 保持空值。
- HGR 原分支才请求 old/new 两个 suffix 并形成差值。此路径完全未修改。

原 Trainer 的共同初始化、suffix 训练、预算策略、优化器、网络和 checkpoint 文件命名保留；B2 仍可能构造原 Trainer 的 predictor 对象，但不使用它拟合 MC 主轨迹标签。这里不做可能改变训练随机流的“优化清理”。评价不构造 Trainer。

论文身份与原生 checkpoint 身份分别记录，严格映射如下：

| 论文 method | runtime_method / 原生 config.method | algorithm |
|---|---|---|
| ch3_baseline_search_prior | ch3_basic_search_prior_v1 | null |
| ch3_baseline_bser_prior | ch3_baseline_bser_prior | null |
| ch3_baseline_direct_mc | ch3_stochastic_direct_mc | stochastic_direct_mc |
| ch3_baseline_direct_boundary | ch3_direct_boundary_corrected | direct_boundary_corrected |

不将论文别名写入生产 registry 或原生 checkpoint。B2/B3 的原生 checkpoint 仍符合 `hgr.complete_cycle.v1`，文件名仍由原 Trainer 产生，例如 `hgr_main_..._cycle_....pt`；文件名前缀不能判定训练算法。统一 evaluator 调用原 `load_checkpoint` 校验 schema、完整 cycle、config/hash、网络和来源，再核对本方法 identity；拒绝 HGR 或另一个 baseline 的 checkpoint，绝不重标记权重。

## 公平配置与输入

四个方法统一要求 M20_MOVING_UNKNOWN_MULTI、max_steps=400、collision_terminal_v1、segment_closed_aabb_v1、team_failure_override_v1、碰撞 terminal reward=-2、team_mean_v1、gamma=0.95、dynamic_public_intercept_v2_1、28D observation 和 3D action。历史 124D critic 不变。默认关闭所有 ablation 和 early_discovery 实验干预。

优先传入实际 HGR run 的 `config.json` 作为 `--reference-training-config`。B2/B3 训练配置从它完整复制，仅更改 `method`、`algorithm` 和 `output_dir`，包括 seed、预算、更新参数、网络、reward 和 planner 在内的其余字段均保留。不传 reference 时，使用 `configs/chapter3/hgr_train.json`；仓库内两个 baseline train JSON 也严格只允许上述三个字段不同，不进行调参。

评价时校验独立 checkpoint 与 reference 的公共任务条件、原生网络 schema 和 policy 参数相同。训练超参数和已完成数量仍由 checkpoint 完整 config 暴露，是否已达到计划训练预算必须另行审计；框架不把一个中途 checkpoint 自动视为正式训练结果。`run_config_reference` 标签只表示路径含 `outputs`，不证明它与用户的某个 HGR checkpoint 绑定。

`--manifest` 必须是已经冻结的 validation manifest，所有方法显式传同一文件。检查全部场景身份、seed、profile、数量、顺序和 horizon，包括未选择的尾部；只使用前 N 场，不新建、不修改、不重抽场景。记录原始文件 SHA256、选中场景内容 SHA256。环境创新 seed 沿用原运行时的 `seed + zero_based_episode_index`；B0/B1 不采样策略，B2/B3 沿用原独立 action generator 的随机策略评价。

`comparable_inputs_sha256` 只覆盖共同任务条件、选中场景、评价 seed 和 N，不包含方法差异。B0 的 standby 干预单独完整记录，不进入共同任务条件。此哈希相同不证明跨方法轨迹随机事件逐项相同，也不是生产 HGR `pairing_sha256`。训练同为 1000 条 main trajectory 不代表总环境成本相同；原训练 summary 的 costs、actual_total_environment_steps、wall_seconds、预算超支应与性能一起报告。Native learned evaluator 的完整来源文件保留以便检查。

## 入口与产物

统一评价：`python -B -m tools.ch3_baselines.run_baseline --baseline ... --manifest ... --episodes 100 --seed 12729 --output-dir ...`。B2/B3 额外必须给独立 `--checkpoint`；B0/B1 禁止 checkpoint。可选 reference 如上。CPU 串行，B2/B3 固定使用原 stochastic 评价模式，没有训练、resume 或 worker 分支。

默认目录按 `outputs/chapter3/baselines/<B0–B3 注册键>/collision_terminal/<run_id>/` 隔离。保留独立 `collision_terminal` 路径组件以满足生产输出验证。CLI 的输出必须为新目录或空目录，非空目录直接拒绝。目录和结果在显式运行时才创建；本次交付不在保留的 outputs 下预放空结果或伪造 summary。

每次评价根目录统一写入：

| 文件 | 内容 |
|---|---|
| resolved_config.json | 论文与原生方法、planner/learning/residual、完整 reference/runtime config、任务条件、checkpoint 身份、随机流与比较范围 |
| evaluation_manifest.json | 选中原场景、原始文件及内容哈希 |
| summary.json | method、planner_mode、learning_mode、n_success、safe_success_rate、found_rate、success_if_found_rate、collision_failure_rate、timeout_failure_rate、mean_team_discounted_return、actual_environment_steps、wall_seconds 等 |
| episodes.json | 原 finalize_episode / validated_rows 所定义的逐场结果，追加两层方法身份和 optimizer_update_count=0 |
| identity.json | 开始/结束 source 与 checkout、reference/manifest/checkpoint 身份关联 |

另有 `evaluation_progress.json`、`controller_diagnostics.json`、`run_console.log`；异常写 `evaluation_failure.json`。B0/B1 每完成一场打印并 flush 一行。B2/B3 完整委托原 evaluator，保留其 `native_evaluation/` 下原生 config、manifest、identity、episodes、summary 和 progress；运行过程中查看该子目录的 `evaluation_progress.json`，根目录汇总在委托返回后更新，不虚构逐步学习调用计数。

统计只使用原 strict outcome、validated rows 和聚合器。未完成全部场景或最终来源检查失败时，正式率值和平均回报保持 null。程序错误不计为 timeout。B0/B1 中断时可记录已执行的物理步数；原 learned collector 不暴露中断场景已执行步数，因此 learned 失败结果的 `actual_environment_steps=null`，`completed_episode_environment_steps` 仅为有效完整场景的成本下界。正常完成时才有精确总步数。

`handoff_event_step` 沿用生产接口的发布步，不能解释为合法接收步。合法接收使用 `executor_target_received_step`，接收后决策使用 `handoff_decision_step`。不可得的接触/保持等数据保留 null。

手动训练入口为 `python -B -m tools.ch3_baselines.run_training --baseline B2_direct_mc|B3_direct_boundary ...`。`--check-only` 只校验/打印计划，不创建模型、Trainer 或结果目录。显式训练时调用原 Trainer，另写 `baseline_training_identity.json`，不改写原 config、summary、checkpoint。若预算导致实际 main 数量不足，sidecar 标为 `stopped_before_requested_main_count`，不宣称完成。

## Provenance 与本轮验证边界

原 `frozen_production_source.json` 保持不变：196 个生产 Python 文件，聚合 SHA256 为 `7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491`。原 27 条历史 provenance 记录及所有历史 hash 不变。

新增 `framework_protected_b0.json` 钉住本轮开始 commit `c7be3fdbf9e7305308ed5aac3ea3a896dfc86b26` 下七个既有 B0 实现/配置/脚本文件。新增 `framework_provenance.py` 复用原来源机制，再记录六个新入口脚本；基线工具全部 Python 和 CH3 JSON 配置继续由原清单收集。评价前、逐场（prior）及结束核对来源和输入；原 learned evaluator 自行逐场核对生产来源和 policy 权重，外层核对新增框架及 checkpoint 文件。训练前后同样验证框架和 reference。

`identity.start_identity_sha256` 是开始身份内容的摘要，不包含自身及后续追加字段；它不是整个最终 JSON 的文件 SHA。`end_state_sha256` 对结束来源内容取摘要。新增交付清单另行提供文件字节哈希，不更新旧 B0 交付清单或历史验证结果。

最新本地验证和范围见 `framework_verification_results.json` 与 `FRAMEWORK_IMPLEMENTATION_REPORT.md`。合成场景的物理接口测试不构成正式性能数据；learned 路由测试使用明确的临时 mock，并未加载真实 B2/B3 权重。用户提供 HGR 已完成 checkpoint 的状态，本任务不重新认证该训练。本轮不执行 Linux 任务，也不宣布正式论文比较完成。

后续手动 Linux 命令见 `linux_commands.md`。
