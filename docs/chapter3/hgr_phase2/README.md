# HGR Phase2 冻结策略梯度效率实验协议

日期：2026-09-25。实施位置：`E:\gym\code\WORKSPACE\AUV-Thesis`，当前分支 `main`。
本阶段交付评估框架及单元测试；真实环境实验为 **NOT_RUN**。
本文不作为 Phase1 真实 acceptance、任务性能或梯度效率收益的证据。

2026-09-26 配置更新：已连接 policy-pair 输入、frozen-source 输出和评估输入路径。
配置更新时为 **PARTIAL READY / SOURCE_NOT_AVAILABLE**，当时没有生成真实来源或执行实验。
同日后续已审计用户迁入的 3090 checkpoint，并导出真实策略对：**POLICY_PAIR_READY**。
随后已使用原 builder 生成 frozen source，10 个场景读取复核及 19 项 source builder 测试通过。
当前进入配置预检查阶段，真实 Phase2 仍为 NOT_RUN。策略核验和差异限制见第 8 节，启动检查见第 9 节。

## 1. 实验对象与固定项

比较同一 θ 下的三个前级梯度估计器：Direct New MC、Old MC、HGR。
φ0、φ1 是调用方明确提供的两个冻结后级策略。求导对象只有 `theta_minus`；
“完整梯度”在这里指重新采样完整主轨迹后的前级梯度，包含随机前缀与无交接事件，
不表示同时求 θ 和 φ 的联合梯度。

直接调用已有 `prefix_losses` 和 `gradient_vector` 计算梯度；不调用 optimizer、训练周期、
后级更新或 predictor 拟合。旧、新策略的 θ 必须具有完全相同的权重、buffer、前向属性和模块模式。
运行使用策略副本，持续核对完整行为身份，输入策略及其 .grad 不变。

沿用 `hgr_phase1_zero.json` 的环境、奖励、规划器、终止、交接与网络合同：
M20_MOVING_UNKNOWN_MULTI、H=400、γ=.95、collision_terminal_v1、team_mean_v1、28D/3D。
Phase2 v1 只支持已有 zero predictor / λ=0 路径，不引入 adaptive budget 或跨版本 predictor。
导入 train 模块仅复用配置校验、输出路径检查和 JSON 写入，不构造 Trainer。

## 2. 默认采样单位

用户已选择 `scope=fresh_main`：

| 项目 | 默认协议 |
|---|---|
| snapshot_count | 10 个预先固定的初始场景，保持顺序 |
| repeat_count | 每个场景、每种方法重采 20 条完整主轨迹 |
| 比较样本数 | 每种方法 200 个；三种方法合计 600 个 |
| reference_rollouts | **总计 1000 条** Direct New MC，均分为每场景 100 条 |
| HGR 固定 K | 沿用运行配置，默认每个梯度样本 8 次校正抽样 |
| 参数更新数 | 0 |

reference_budget 的单位是总 rollout 数，必须能被 snapshot_count 整除，每场景至少 2 条。
API 的 num_snapshots / num_repeats / reference_budget 可显式覆盖配置；不会自动截取输入场景。
输入必须恰好有指定数量的场景，至少 2 次 repeat 才报告样本方差。

这里的 snapshot_id 是预声明场景 ID；每条完整轨迹的自然交接时刻 τ 和交接快照 hash 随样本记录。
不同 repeat 可以有不同 τ，也可以没有交接。交接快照由现有 collect_trajectory 捕获，并立即用于 HGR 续分支。
日志保存快照摘要与续分支审计字段；不把交接快照对象图复制进 JSON。
无交接、碰撞和提前终止的主轨迹保留在分母内，不重采、不挑选成功案例、不注入 Found。

可选 `fixed_boundary` 仅供条件诊断：输入必须同时包含完整已记录前缀和真实 DecisionSnapshot；
裸快照缺少 score 信息，会被拒绝。它估计给定前缀下的 score 估计器统计量，
不能解释为初始场景分布上的完整梯度效率。默认配置不使用此模式。

## 3. 三种估计器

令 s_t = ∇θ log πθ(a_t|h_t)，R_t 是既有 γ 折扣 reward-to-go。

1. **direct_new_mc**：使用 (θ,φ1) 从场景初始状态采到真实终止，
   g = Σ(t<τ) γ^t s_t R_t；无交接时累加全部前级动作。
2. **old_mc**：使用 (θ,φ0) 独立采样完整轨迹，使用同一 score 公式。
   它估计旧目标梯度，作为旧策略差异的对照；不能假定对新目标无偏。
3. **hgr**：独立采一条 (θ,φ0) 完整轨迹得到 g0。在该轨迹自然产生的交接快照上，
   每次抽样执行一对独立于主轨迹后缀的旧、新续分支：

       C = 1_handoff * γ^τ * Σ(t<τ) s_t
       g_delta = C * mean_k(G_plus_new,k - G_plus_old,k)
       g_full = g0 + g_delta

这是现有固定 K 估计器取 N=1、q=1、预测值为 0 的情况，不是将原训练批大小 16 偷换成 200。
每个 replicate 是单条主轨迹的梯度估计，200 是统计样本数。
旧校正分支重新采样，不能直接复用主轨迹旧后缀来抵消 g0。
不改变 K、不根据标签大小调预算、不复用跨 repeat 标签。
完整 φ 行为完全一致且原 Phase1 开关允许时，使用原 NoUpdateProof 旁路：K_actual=0、g_delta=0。
无交接时仍保留 K 个零标签抽样项，实际续分支步数为 0。

## 4. 参考梯度、误差与方差

每个场景 s 使用独立参考流得到均值 g_ref,s；整体参考：

    g_ref = (1 / 10) * Σ_s g_ref,s
    gradient_mse = ||g_hat - g_ref||²
    case_reference_mse = ||g_hat - g_ref,s||²
    cost_normalized_error = gradient_mse * 本样本实际 environment_steps

gradient_mse 按请求采用平方欧氏范数，不再除以参数维度。目标是 10 个固定场景的均匀混合。
每行是某一场景的梯度，因此与整体 g_ref 的误差含场景间差异；case_reference_mse 供场景内比较。
不能把固定的这 10 个场景推广为所有场景，也不能把 200 个分层样本声称为独立同分布场景抽样。

每种方法输出 float64 Welford 均值和逐维样本方差（ddof=1）：
per_dimension_variance 为合并描述方差，包含场景差异；
within_case_variance 和 case_means 分别保留每个场景的方差与均值，顺序对应 snapshot_manifest。
mean_gradient_mse 是逐样本平方误差的平均，**不是**平均梯度的平方误差。
mean_cost_normalized_error 是逐样本乘积的平均；另外报告每种方法 total_environment_steps。

g_ref 是有限 MC 估计，不是真值。保存 reference_gradient_norm，
并报告 reference_mean_variance = Σ_s(var_s / reference_rollouts_per_case) / snapshot_count²
及其 trace，提示参考误差水平。参考流与比较流隔离，比较误差包含参考估计的不确定性。
本框架不据此自动给出“有效”“提升”等结论。

## 5. 随机流、成本与审计

保留 `hgr.phase1.named_streams.v1`。run seed 只作为具名派生的根：
按 scope、场景索引、method、repeat、draw 寻址，不在 Phase2 手工覆盖全局随机 seed。
真实采集和续分支调用原 Phase1 API，使用其 RNG 隔离与随机创新规则。

每个 (场景,repeat) 有唯一 comparison pair_id，三种方法共享该组标识但主轨迹随机流相互独立。
reference pair_id 使用独立命名域。每次 HGR draw 再有唯一 pair_id，
old/new 两条续分支共享这一标识；policy_crn 只共享策略噪声，环境创新保持独立。
改变 reference budget 不会移动已有比较样本的随机流。
fixed_boundary 的 random_stream_id 记录实际 PairNoise trace_id。

记录实际步数，不用 400 或剩余时域代替实际终止长度：

| 成本项 | 含义 |
|---|---|
| source_preparation_steps | 调用方声明的来源准备成本；固定前缀模式至少覆盖全部前缀长度 |
| reference_steps | 所有参考完整轨迹；条件模式为实际续分支 |
| direct_new_mc_steps / old_mc_steps | 各方法实际主轨迹成本 |
| hgr_main_steps | HGR 旧主轨迹成本 |
| hgr_correction_old_steps / hgr_correction_new_steps | 所有实际校正续分支成本 |

总成本包含上述全部项目；逐样本和逐方法误差乘成本不摊入共享参考与来源准备成本。
不包含训练、优化器、CPU 求导时间，也不宣称等墙钟时间公平性。
默认理论环境步数上界为 1,920,000（不含来源准备），实际按终止计账；本次没有执行。

每条比较样本包含 snapshot_id、tau、phi0_hash、phi1_hash、snapshot_hash、method、repeat_id、
pair_id、environment_steps、gradient_norm、gradient_mse、random_stream_id、random_source_revision。
无交接时 tau / snapshot_hash 为 null。另存 K_requested / K_actual、旁路原因、续分支标签和审计结果。

## 6. 输入、文件与手动入口

入口：

```python
run_phase2_gradient_efficiency(
    config,
    snapshot_source=None,
    num_snapshots=None,
    num_repeats=None,
    reference_budget=None,
)
```

snapshot_source 接受 FrozenSnapshotSource 对象或显式的
`hgr.phase2.frozen_source.v1` 文件路径。文件使用 weights_only=True 读取，
只含两份冻结策略状态/行为身份、预声明场景或前缀、来源身份和准备成本。
不接受 Trainer checkpoint；不从 ep100 恢复；不自动构造有意义的 φ0/φ1 差异。

默认配置已按下表连接，路径相对于仓库根目录；此处仅声明预期位置，不创建目录或占位模型：

| 配置文件 / 字段 | 路径 |
|---|---|
| hgr_phase2_source_builder.json / policy_source_path | outputs/chapter3/hgr_phase2/policy_pair_source.pt |
| hgr_phase2_source_builder.json / source_output_path | outputs/chapter3/hgr_phase2/frozen_phase2_source.pt |
| hgr_phase2_gradient_efficiency.json / snapshot_source | outputs/chapter3/hgr_phase2/frozen_phase2_source.pt |
| hgr_phase2_gradient_efficiency.json / output_dir | outputs/chapter3/hgr_phase2/collision_terminal/gradient_efficiency_01 |

人工准备好真实 policy-only 导出后，可放到第一行路径，也可只将 policy_source_path 改为其实际路径。
不要把 Trainer checkpoint 改名放入该位置。如果缺少输入文件，source builder 返回
SOURCE_NOT_AVAILABLE（退出码 2）；评估入口缺少 frozen source 时也会在采样和输出创建前失败。

policy-only 导出必须采用 `hgr.phase2.policy_pair.v1`，字段结构见
`chapter3_bser/experiments/hgr/build_phase2_source.py` 的模块说明：

- old_policy / new_policy：完整 HandoffPolicy 的 state、training_modes、behavior_sha256；
  两份 state 都包括 theta_minus.* 和 phi.*。θ 完全一致，φ 的行为身份不同。
- source_identity、runtime_contract_sha256：真实来源对应的完整源码身份和运行合同摘要。
- origin：kind=real_policy_export，以及 description / theta_id / phi0_id / phi1_id 的真实来源说明。
- preparation_environment_steps：来源准备实际消耗的环境步数，不能把已经发生的成本填成 0。

当前运行合同保持 M20_MOVING_UNKNOWN_MULTI、H=400、Actor hidden_dim=128、expert_hidden_dim=128。
此前发现的 H=8、hidden_dim=8 历史测试模型不兼容；本次迁入的真实 HGR 模型为 H=400、hidden_dim=128。
准备器不会替换网络结构、补权重或修补 θ。
只提供来源说明不能证明模型的训练历史；策略来源仍须人工核实，禁止用新随机初始化及扰动替代。

准备器按固定 seed 202609260–202609269 调用既有生成器，构建 scene_00–scene_09，
记录初始位置、目标轨迹投影、障碍物布局 hash 和 profile；不筛选交接结果。
人工提供 FrozenSnapshotSource 时，场景也应使用完整既有定义，至少包含
scenario_profile、max_steps、scenario_id、scenario_seed，保持相同几何、目标和任务设置。
source_identity 必须描述来源实际生成时的当前源码；不可给旧版本来源重填当前身份来绕过检查。
新增章内 Python 会自然进入现有源码清单；旧来源不匹配时拒绝，不修改历史 provenance 或 source gate。

以下是后续人工运行顺序，本次没有执行。先激活 AUV 环境并单独运行来源准备：

```powershell
conda activate AUV
python -m chapter3_bser.experiments.hgr.build_phase2_source --config configs/chapter3/hgr_phase2_source_builder.json
```

只有来源准备返回 READY、确认输出目录是新或空目录，并由用户明确发起真实实验后，再单独运行：

```powershell
python -m chapter3_bser.experiments.hgr.phase2_gradient_efficiency --config configs/chapter3/hgr_phase2_gradient_efficiency.json
```

Windows 对应两个独立入口：
`scripts\run_hgr_phase2_source_builder.bat [准备配置路径]` 和
`scripts\run_hgr_phase2_gradient_efficiency.bat [实验配置路径]`。
可用 PYTHON 指定解释器；包装本身不激活环境、不训练、不恢复 checkpoint；
准备脚本保留来源缺失的退出码 2，成功后也不会自动启动实验。

运行结果：

- `phase2_results.json`：配置、源码/策略身份、参数展平顺序、参考摘要、逐样本日志、方差与成本统计。
- `vectors/*.npz`：默认存梯度向量、参考梯度、逐维方差；JSON 用 path / key / shape / dtype / sha256 定位。
  HGR 每个样本分别保存 g0、g_delta、g_full，gradient_vector 指向 g_full。
- 小型诊断可设 vector_storage=inline，将向量以 values 数组直接写入 JSON。

不覆盖非空输出目录；增量落盘。失败保留已有结果、完整 traceback、进行中的调用和已知成本，
无法从失败调用恢复的步数明确标为 unknown，不自动重试或修复。
PASS 只表示本轮框架执行及冻结检查通过；handoff_correction_status 独立给出 EXERCISED / NOT_EXERCISED。
所有样本无自然交接时保留全部样本，校正状态为 NOT_EXERCISED。

## 7. 单元验证与后续

`tests/test_hgr_phase2_gradient_efficiency.py` 覆盖：

- 合成解析梯度、g0 + g_delta、MSE、实际主轨迹及双续分支成本；
- 同一来源重复运行、named streams、唯一配对身份、全局 RNG 隔离；
- JSON/NPZ schema、校验和、逐维 ddof=1 方差；
- 精确零更新旁路、无交接保留、条件模式界限；
- 冻结策略变更/非有限结果诊断、来源/配置/输出拒绝；
- 专用来源文件往返、错误 checkpoint schema 拒绝，以及真实 API 的模拟委派。

测试用替身阻止真实 MissionRuntime 构造/恢复和 optimizer.step；小型结果只写 TemporaryDirectory。
2026-09-25 本地验证记录：9 项 Phase2 测试、8 项既有梯度机制测试、8 项元数据/来源测试，
合计 **25 项通过（14.658 秒）**；这是本地记录，不是 CI。
在已有 AUV 环境执行的命令为：

```powershell
python -m unittest tests.test_hgr_phase2_gradient_efficiency tests.test_hgr_mechanism tests.test_repository_metadata tests.test_core_source_provenance -v
```

2026-09-26 配置连接的最新本地验证：以下两组单元测试 **28 项通过（20.29 秒）**。
覆盖准备输出与评估输入路径一致，以及预期策略文件尚不存在时返回 SOURCE_NOT_AVAILABLE、退出码 2，且不创建输出目录。

```powershell
python -m pytest tests/test_hgr_phase2_gradient_efficiency.py tests/test_hgr_phase2_source_builder.py -q
```

本次配置更新核对的 221 个既有生产源码和 provenance 文件哈希不变，没有生成真实来源文件或实验输出目录。

修改前后核对的 65 个既有文件 SHA256 全部一致，覆盖既有 HGR 源码、Phase1 配置、core Python 和 provenance 文件。
本次仅运行单元测试，不执行上述真实入口，不读取用户 checkpoint，不写历史 outputs，
不修改既有算法文件、core、reward、environment、planner 或历史 provenance。
实际运行前仍需合格的 frozen source，并核对适用于当前运行源码的 Phase1 真实 runtime acceptance。

## 8. 3090 真实 checkpoint 核验与策略导出（2026-09-26）

详细记录见 [3090 权重审计](checkpoint_3090_audit.md)。HGR 的 ep16–ep100 文件包含真实训练权重；
B2/B3 文件也是真实权重，但属于 MADDPG baseline，不能作为当前 HandoffPolicy 的 φ0/φ1。
本 Phase2 的 Direct New MC 是对同一个 (θ,φ1) 的直接梯度估计，不是导入历史 B2 的网络。

已生成 `outputs/chapter3/hgr_phase2/policy_pair_source.pt`，采用既有 `hgr.phase2.policy_pair.v1`：

- θ 和 φ0 取自 ep48/cycle3；φ1 取自 ep64/cycle4。
- cycle4 记录的 θ_before、φ0、φ1 哈希与这些实际张量完全对应，恢复的是前级更新前的真实周期策略对。
- ep64 中的 θ_after 不用于导出。两个 checkpoint 的最终 θ 不相同，不能直接当作冻结策略对。
- ep96→ep100 仅有约 4.63e-14 的权重变化，固定 float32 输入检查无可见分布差异；cycle5/6 完全零变化。
  cycle4 是保存的相邻周期中最后一个在同一固定输入检查下有非零分布差异的周期。没有运行或按 Phase2 结果选策略。
- cycle4 的 φ 差异也很小：参数差 L2 约 1.06e-6，固定输入下均值最大差约 3.36e-7。
  这不证明真实场景中的差异强度或实验统计功效，不应预期一定能观察到梯度效率收益。

独立入口为 `python -m scripts.export_hgr_phase2_policy_pair`，仅用于本次已审计的历史输入和当前源码。
它固定输入文件 SHA256、历史 Git 源码与当前生产源码摘要；源码变化时拒绝，并非通用跨版本加载器。
原始 checkpoint 的完整来源清单与 Git `da4a8a64cba26b2adf7d248fd8486e517422628c` 完全一致。
导出是将学习到的张量明确导入当前策略对象，生成新的 policy-only 文件；不把原 checkpoint 改名或改写为当前来源，
不声称历史运行等同于 Phase1。原来源清单保存在 `policy_pair_source.audit.json`，其 hash 绑定于新文件 origin。
当前 source gate、历史 provenance、算法与运行源码均未修改。

导出后通过现有 `load_policy_pair` 严格读取验证。历史准备成本为 44,610 环境步，导出新增环境步和参数更新均为 0。
没有创建 Trainer、载入优化器、恢复环境、训练、生成场景、生成 frozen source 或运行梯度效率实验。
已有输出一律拒绝覆盖，因此不要为重复执行而删除已生成文件。

最新本地验证（不是 CI）：**43 passed in 25.79s**。

```powershell
python -B -m pytest tests/test_hgr_phase2_policy_export.py tests/test_hgr_phase2_source_builder.py tests/test_hgr_phase2_gradient_efficiency.py tests/test_hgr_phase1_legacy_fixture.py -q -p no:cacheprovider
```

该导出阶段后已单独运行原 source builder，完成 10 个预声明场景的 frozen source 生成；
真实 Phase2 仍按第 6 节由用户单独发起。

## 9. 正式入口的输出协议与运行前检查

本次报错来自 debug 配置路径缺少独立的 `collision_terminal` 目录分量，已修正。
正式配置原本合规，保持不变。两份配置均使用已有 `frozen_phase2_source.pt`，不生成新来源：

| 配置 | 场景数 | 每场景重复 | 参考 rollout 总数 | output_dir |
|---|---:|---:|---:|---|
| hgr_phase2_debug.json | 10 | 2 | 20 | outputs/chapter3/hgr_phase2/collision_terminal/debug_gradient_efficiency_10scene |
| hgr_phase2_gradient_efficiency.json | 10 | 20 | 1000 | outputs/chapter3/hgr_phase2/collision_terminal/gradient_efficiency_01 |

debug 仍保留全部 10 个场景，只降低重复和参考采样预算。不能把 snapshot_count 改为 2 后从来源中截取场景。
20 条参考轨迹表示每场景 2 条；debug 用于检查运行流程，不能替代正式统计实验。

现有正式 Python 入口已经在第一次 rollout 和输出创建前依次执行：

1. `resolve_config` 校验配置及 runtime_config。
2. `load_snapshot_source` 读取指定文件，严格校验 frozen schema、source identity、运行合同及策略载荷。
3. `_validate_source` 要求来源场景数与 snapshot_count 完全一致，并验证场景或前缀。
4. `validated_output` 要求 collision_terminal 协议及新的或空的输出目录。
5. `Evaluation` 构造阶段验证共享 θ 及冻结策略身份。

这些检查保持原样。生产 Python 文件属于 frozen source 的完整源码清单；
修改它们会使现有来源身份不再匹配，因此本次不为重复同一检查而改动生产文件，
也不修改 source gate、重写来源或把 mismatch 豁免掉。

新增 `scripts/check_hgr_phase2_config.py` 复用上述检查，提供**独立只读预检查**：

```powershell
conda activate AUV
python -m scripts.check_hgr_phase2_config --config configs/chapter3/hgr_phase2_debug.json
python -m scripts.check_hgr_phase2_config --config configs/chapter3/hgr_phase2_gradient_efficiency.json
```

通过返回 `PREFLIGHT_PASS`、退出码 0；失败返回 `PREFLIGHT_FAIL`、退出码 2，
并指出 snapshot_source、snapshot_count、output_dir 或 runtime_config 的错误。
检查不会创建目录、写文件、恢复环境、更新参数或进入 rollout。
输出目录不允许恢复或追加已有实验；当前框架只支持新目录或空目录，保留非空目录中的历史结果。

Windows 入口 `scripts/run_hgr_phase2_gradient_efficiency.bat [配置路径]` 已接入该检查，
只有退出码 0 才会继续调用原正式 Python 入口。该 bat 检查通过后会实际开始实验，
如果只想检查，使用上面的 `scripts.check_hgr_phase2_config` 命令。

后续由用户明确启动时，正式 Python 命令保持不变（本次没有执行）：

```powershell
python -m chapter3_bser.experiments.hgr.phase2_gradient_efficiency --config configs/chapter3/hgr_phase2_debug.json
python -m chapter3_bser.experiments.hgr.phase2_gradient_efficiency --config configs/chapter3/hgr_phase2_gradient_efficiency.json
```

`tests/test_hgr_phase2_config_validation.py` 覆盖非法输出、合法目录、10→2 场景数不匹配、
来源缺失、错误 schema、非空目录保留、错误 runtime、预检查 CLI 和 Windows 入口顺序。
它还对本机存在的真实 frozen source 和两份配置只读检查，禁止 MissionRuntime、Trainer、
optimizer.step 和 rollout。真实来源未随仓库提供时，仅该可选本机来源检查跳过。

本次最新本地验证（不是 CI），仅运行用户指定的两个测试文件：

```powershell
python -B -m pytest tests/test_hgr_phase2_config_validation.py -q -p no:cacheprovider
# 13 passed in 25.30s（包含现有真实来源的两份配置检查，无跳过）
python -B -m pytest tests/test_hgr_phase2_gradient_efficiency.py -q -p no:cacheprovider
# 9 passed in 26.82s（合成梯度测试）
```

没有执行真实 debug/正式实验、训练或生成新策略来源。本次配置修复不改变 HGR、estimator、
Phase1 runtime、reward、环境、planner、网络或 Phase2 梯度计算。
