# HGR Phase2 冻结策略梯度效率实验协议

日期：2026-09-25。实施位置：`E:\gym\code\WORKSPACE\AUV-Thesis`，当前分支 `main`。
本阶段交付评估框架及单元测试；真实环境实验为 **NOT_RUN**。
本文不作为 Phase1 真实 acceptance、任务性能或梯度效率收益的证据。

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

配置中的 snapshot_source 默认 null，因此示例入口在未准备来源时明确报错，并且不创建实验输出。
以下是未来明确准备来源后的导出方式，old_policy / new_policy / ten_scenarios 均须由调用方提供；
当前实现工作没有生成真实来源文件：

```python
from chapter3_bser.experiments.hgr.phase2_gradient_efficiency import (
    FrozenSnapshotSource, load_config, resolve_config, save_snapshot_source,
)
from chapter3_bser.experiments.hgr.provenance import fresh_source_identity

config = resolve_config(load_config("configs/chapter3/hgr_phase2_gradient_efficiency.json"))
source = FrozenSnapshotSource(
    old_policy=old_policy, new_policy=new_policy,
    cases=[{"snapshot_id": f"scene_{i:02d}", "scenario": scenario}
           for i, scenario in enumerate(ten_scenarios)],
    scope="fresh_main",
    source_identity=fresh_source_identity(),
    preparation_environment_steps=0,
)
save_snapshot_source("frozen_phase2_source.pt", source, config["runtime_config"])
```

场景应使用完整既有场景定义，至少校验 scenario_profile、max_steps、scenario_id、scenario_seed，
保持相同几何、目标和任务设置。示例的准备成本 0 只适用于没有为准备来源执行环境步进的情况。
source_identity 必须描述来源实际生成时的当前源码；不可给旧版本来源重填当前身份来绕过检查。
新增章内 Python 会自然进入现有源码清单；旧来源不匹配时拒绝，不修改历史 provenance 或 source gate。

准备来源后，把配置 snapshot_source 指向该文件，并指定新或空的输出目录。相对 CLI 路径从仓库根解析。
后续由用户明确发起实验时，在 AUV 环境执行：

```powershell
conda activate AUV
python -m chapter3_bser.experiments.hgr.phase2_gradient_efficiency --config configs/chapter3/hgr_phase2_gradient_efficiency.json
```

Windows 包装入口：`scripts\run_hgr_phase2_gradient_efficiency.bat [配置路径]`。
可用 PYTHON 指定解释器；包装本身不激活环境、不训练、不恢复 checkpoint。

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
2026-09-25 最新本地验证：9 项 Phase2 测试、8 项既有梯度机制测试、8 项元数据/来源测试，
合计 **25 项通过（14.658 秒）**；这是本地记录，不是 CI。
在已有 AUV 环境执行的命令为：

```powershell
python -m unittest tests.test_hgr_phase2_gradient_efficiency tests.test_hgr_mechanism tests.test_repository_metadata tests.test_core_source_provenance -v
```

修改前后核对的 65 个既有文件 SHA256 全部一致，覆盖既有 HGR 源码、Phase1 配置、core Python 和 provenance 文件。
本次仅运行单元测试，不执行上述真实入口，不读取用户 checkpoint，不写历史 outputs，
不修改既有算法文件、core、reward、environment、planner 或历史 provenance。
实际运行前仍需独立提供合格冻结来源，并完成应有的 Phase1 真实 runtime acceptance。
