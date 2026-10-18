# 第三章 BA-STPR 方案：源码审计与科学可行性评审

日期：2026-10-17（Asia/Shanghai）。状态：方案评审与只读源码审计；未实现 BA-STPR，未运行训练、checkpoint 加载、完整回合评估或正式实验。用户提供的方案在 4.5 的团队收益公式处截断，本文不假定后续内容。

## 1. 结论

方向科学上成立，但当前文本还不是可以直接实施并声称创新完成的算法规格。当前系统确实存在“静态空间覆盖代理与移动目标首次发现目标不一致”的建模缺口；该缺口的实际重要程度和新方法的净收益尚未验证。

单独采用“目标信念预测 + 路径探测收益 + 行程成本 + 贪心去重”，创新性偏弱，已有最优搜索、信息路径规划和移动目标搜索研究覆盖这些思想。硕士论文可以将其发展为有明确适用条件和验证证据的方法贡献，但不能以更名、组合现成算法或优于单一最短路对照作为充分创新证据。

建议收敛为：**在公开信息边界内，对固定上层分配目标的有界候选路线，估计保留时间相关性的团队首次发现收益，并验证该收益在现有执行链中的兑现程度。** 位置—速度信念不足、未知障碍反射以及路线执行时间误差，是实施前必须解决或明确承认的模型限制。

## 2. 版本与审计边界

- 工作目录：`E:/gym/code/WORKSPACE/AUV-Thesis`；分支：`main`。
- HEAD：`c468663acf05fa11b41eb068169ae44cc79a2376`。
- 审计开始时工作树不干净：6 个已跟踪文件存在修改，另有 D2 round2 性能验证、相关测试/工具以及文献目录未跟踪。
- 已跟踪改动包括 `core/mapping/path_planner.py`、`chapter3_bser/experiments/safe_search_v1/provenance.py`、`tests/d2_performance_review.py`、`tests/test_ch3_baseline_provenance.py`、`tests/test_safe_search_provenance.py`、`tools/benchmark_d2_performance.py`。
- 因此本文针对 **HEAD 加当前工作树**，不能仅用 HEAD 复现全部审计对象。`path_planner.py` 的当前 diff 是 CPU 上小张量读取/计算的性能改动；未据此认定其完整行为等价或完成 provenance 验证。
- 本轮仅新增本文档；未修改现有生产源码、历史证据、provenance manifest、outputs 或 checkpoint，未 commit/push。没有重跑完整回归或 27 条 provenance 校验，不能把本文表述为仓库发布验收。

关键文件当前原始字节 SHA-256（用于本次审计身份，不更新历史清单）：

| 文件 | SHA-256 |
|---|---|
| `chapter3_bser/online/allocator.py` | `6f77ec62083446d1fa62bec8406ebfdd5988e6708614497a2fc6ba108f730ff2` |
| `chapter3_bser/candidate_generator.py` | `d80edd01d8f0568350dafc589ca818182b5fc0543e751bd7e9e6d6ff353e8187` |
| `chapter3_bser/detection_model.py` | `2f3ca09e49a4d04f0b62951bc711f091bbc4914a8f854a5e9ecd957ae3b0dcfa` |
| `chapter3_bser/objective.py` | `09df76bf604f51fe335d30f852d9b4fe4efe26aa673a8889344040dc16f83ac9` |
| `core/mapping/path_planner.py` | `032cc1f5f1488f668ac7c1b4e8f7bb0a01003f9546f04312008dd464ffcb269c` |
| `core/mapping/planning_state.py` | `0c0a72bad93d11bd3d42cb132e8449958ddefb7acb9fe2156250808dc2a5b6a5` |
| `core/mapping/travel_cost_service.py` | `185cfc49deab62e33374dc423fb582b0e54ead7b8a6c1675827022a316f69c2e` |
| `core/env/target_motion.py` | `e35ae02447723dfdc6be23d31fa4ed73a992ddcf12998947bd2af0b6084844af` |
| `core/env/uav_env.py` | `2626d3f957a8e448db34868442d28cc20988cbb46292dd48700eab0335ad7eaf` |
| `chapter3_bser/experiments/bser_final_v1/forecast.py` | `392a082c52f5b54febab9d35c2752a3fe1a79707a55fe9876063fbbdc17a846b` |
| `chapter3_bser/experiments/bser_final_v1/scoring.py` | `f4242b675e189d2a3737ca382f7665dabda9ed32743a5099efa075726042b1ed` |

## 3. 真实调用链与已有功能

### 3.1 上层输出包含完整路线

所有用户列举的五个入口均存在。实际主要链路为：

```text
公开环境接口/在线地图
  -> extract_planning_state -> 不可变 PlanningStateView
  -> OnlineBSERController.initialize / step
  -> BSEROnlineAllocator.allocate / allocate_partial
  -> generate_candidates / generate_search_candidates
  -> TravelCostService.query -> A* -> PathQueryResult
  -> detection_probability + build_objective_context
  -> solve_joint_greedy / 独立实验求解器
  -> OnlineAllocation[SearchAssignment(waypoint, path, costs, cells)]
  -> WaypointManager / 独立实验稳定和重评分
  -> RMADDPGGuidanceBridge / SafetyBridge
  -> PathTracker.tracking_target
  -> GuidedEnv.install_guidance / 导航 hook
  -> 环境原有 waypoint prior 与学习残差融合
  -> 原有动力学、碰撞和发现判据
```

分配器并非只输出目标点。`SearchAssignment` 已含终点、完整路径、栅格索引、physical travel time、planning cost；导航桥将完整路径转成逐段 tracking waypoint，不生成动作。

`TravelCostService` 在不可变公开图上运行 A*；`single_source` 使用 Dijkstra。对同一起终点，它尝试起终点连接器组合后只返回最小 planning cost 的一个结果。内部连接器有多个不等于对上层暴露多条候选路线。默认每搜索者生成 4 个**不同终点**候选；这也不等于固定同一终点的 K 条路线。

### 3.2 必须区分三套评分

**原 BSER 路线覆盖与响应目标**，见 `detection_model.py`、`objective.py`：

\[
p_\tau(c)=\operatorname{clip}\left[p_{\max,i}
\exp\left(-\frac{d(c,\tau)^2}{2\sigma_i^2}\right)(1-o(c)),0,1\right],
\quad \sigma_i=\kappa r_i.
\]

`d(c,τ)` 是栅格中心到整条折线所有线段的最小距离。默认 `p_max` 为 fast/balanced/precise 的 0.72/0.78/0.84，`κ=1`，见 `configs/chapter3/bser_phase1a1.json`。高斯核在探测半径外仍非零，不是环境确定性发现概率的精确表达；`1-o(c)` 表示该格为空闲的代理权重，不是新增声学遮挡模型。

\[
Q_S(c)=1-\prod_{\tau\in S}(1-p_\tau(c)),\qquad
F(S,y)=\sum_c b_t(c)Q_S(c)\exp[-T_E(y,c)/18].
\]

不可达格的响应权重为零。纯搜索目标为 `Σ b_t Q_S`。联合贪心枚举待命点，对每个待命点按边际增益选取每搜索者至多一个候选，再取最优组合。**团队重复覆盖已经受到抑制**，不应再次把去重列作从零新增贡献。这里的乘积是固定格条件下的覆盖模型，也不自动代表移动目标的真实团队发现概率。

`maximum_physical_travel_time=120` 是候选过滤预算；路线长度被记录，但原 BSER 上述目标没有直接的 `-λ_L L` 项。行程成本会影响 A* 路线、候选可达性和执行者响应权重，不能说“现有分配只按最短路打分”。

**环境内原始候选目标点评分**，见 `core/mapping/map_module.py: _candidate_points` 和 `path_planner.py: score_search_candidates`：基础值包含 pheromone、`exp(-coverage_weight*coverage)/(1+claim_weight*claims)`、frontier、距离偏好、高度偏好和保留点分离；然后加入当前信念权重，扣除归一化搜索/执行行程成本。在线未知图再加入占据熵收益并扣除带半衰期的目标观测 recency。该路径与独立 BSER 候选池不是同一个评分入口；不能把这些项都当作 BSER 路线目标已经使用的项。

**已有独立时序实验**：

- `online/early_discovery.py` 用 `ΔF × exp(-λ L/v_nominal)` 重排既有候选。它只有终点级到达时间折扣，不预测目标、不计算沿途首次发现；默认关闭。
- `experiments/d2_v1/assembly.py` 接入 `bser_effect_v1` 的 D2：先优化纯静态搜索，再顺序优化执行者待命；保留 V4 安全与失效处理。
- `experiments/bser_final_v1/forecast.py` 已复制公开 guidance/tracker 状态，以公开运动学预测 20 步，考虑观测扰动估计和小扰动变体；没有 simulator rollout、目标真值或随机抽样。
- 其时刻 k 覆盖是 `p_i,k(c)=max_{j<=k} kernel(c, predicted_segment_i,j)`，团队覆盖仍是 `1-Π_i(1-p_i,k)`。`near=mean_k Σ b_t(c)[Q_k(c)-Q_0(c)]`；动态响应项为 `Σ_k,c b_t(c)[Q_k-Q_(k-1)]_+ exp(-T_E(predicted_executor_k,c)/18)`。
- **F 系列已经显式考虑搜索器的路径执行时间顺序，但冻结当前目标信念和未来地图**。因此“现有代码完全没有路径时序评分”不成立；“已有精确移动目标首次探测评分”也不成立。

### 3.3 行程成本不是纯长度，也不是实际到达时间

公开图对应在线规划边代价：

\[
c_{plan}(e)=t_e+0.35h_2(\bar o_e)+1.25\bar o_e t_e,
\quad \bar o_e=(o_u+o_v)/2.
\]

系数为标准未知图配置值；端点连接器也有其既有成本规则。`physical_travel_time` 单独累计无占据惩罚的名义边时间。普通搜索器使用 `d_xy/1.00 + 0.6|d_z|/0.55`；执行者使用 `d_xy/1.15 + 0.6|d_z|/0.60`。三个异构搜索器在该行程估计中共用角色速度，而真实最大速度、加速度、阻尼和探测半径不同。因此其“physical”名称不能作为经过校准的真实执行 ETA 证明。

新评分若再同时减路径长度、行驶时间、占据风险，可能把已有代价重复惩罚。第一版更适合固定相对基线路线的绕行预算和剩余物理步预算，再比较首次发现收益；必要时只保留一个经归一化、定义清楚的额外成本项。

## 4. 信念、运动与信息边界

### 4.1 现有信念不足以直接恢复轨迹联合分布

`TargetBeliefView` 只有归一化的位置栅格概率、熵、峰值和 revision；没有目标 ID 列表、速度后验、粒子或轨迹假设。`PlanningStateView` 暴露的是**智能体**位置和速度，不能把该 velocity 误认为目标速度。标准 M20 是单移动目标、多 AABB 障碍，`MULTI` 不表示多目标；首版不需要扩展多目标任务或环境。

`core/config/ch3_config.py` 的 M20 继承占据约束扩散、`alpha=0.12`，不是构造函数的默认静态模式。在线扩散用中心加六邻域的七点核卷积，混合旧信念；占据约束模式再乘空闲权重并归一化。边界零填充、mask 和归一化没有携带反射后的速度方向。它是现有估计近似，**并不等价于匀速反射动力学的准确传播**。

负观测用当前搜索器位置/半径内高斯置信度构造非零下限 likelihood，再归一化，并记录时间戳和置信度；不是复用了发现判据的步内相对运动核。预约衰减、访问与重访信息已经存在，但不都直接暴露于 BSER 的公开快照。不能将近似 belief 的采样结果无条件叫作真实后验样本。

### 4.2 运动函数可以复用，但输入来源必须合法

`core/env/target_motion.py` 提供 `TargetState`、`advance_target_state`、`predict_target_state`、`simulate_target_trajectory`，复用同一套边界/AABB 反射语义；预测所需的状态包括位置、速度、sample_step、motion_mode。因此无需另写反射模拟器。

但当前搜索公开视图没有目标速度后验，也没有完整运动规格（dt、clearance、反射上限等）。`target_motion_known=True` 说明运动规律公开，不意味着 scenario 中的该回合真实 `target_initial_velocity` 或 motion seed 可以被规划器读取。生成器的速度区间 0.25–0.55、方向分布等可以作为**候选公开先验规格**讨论，实施前必须明确其协议地位，不能从当前隐藏样本反推先验。

带持久速度和反射的目标通常需要位置—速度联合状态 z=(x,v) 才具备所需 Markov 性。只有 b(x) 不能唯一确定未来轨迹；给所有位置格临时随机配速度，也不保证得到与历史负观测条件一致的 b(x,v)。

真实环境用 ground-truth 障碍推进目标，而搜索规划只能知道在线占据地图。即便复用完全相同的反射函数，传入已知障碍也只能得到**给定已知地图的条件预测**，不能保证与隐藏障碍导致的真实未来反射一致。声称“未知障碍下精确复现真实目标未来”会违反信息边界。需明确采用地图条件近似还是对未知地图不确定性建模；后者显著扩大首版范围。

### 4.3 真实发现判据

环境默认 dt=0.2。三个 Searcher 的半径分别为 2.00、2.35、2.75，阈值另加 `detect_eps_bias=0.10`。`_maybe_detect_swept` 调用 `swept_relative_min_distance`：

\[
r_0=a_0-x_0,\ \delta=(a_1-a_0)-(x_1-x_0),\quad
\theta^*=\operatorname{clip}\left(-r_0^T\delta/\|\delta\|^2,0,1\right),
\quad D_i=\mathbf1[\|r_0+\theta^*\delta\|\le r_i+0.10].
\]

零相对位移按既有 EPS 分支处理。这里用一步起终点弦段，即使目标一步内发生反射；新评分应匹配当前实现，不能擅自改为另一套更精细折线检测或增加遮挡判据。环境在多智能体同一步满足发现时按其既有距离/ID规则决定 finder，不能另改发现和交接协议。

评估器可读取真实目标/障碍并计算预测误差；这些值、真实轨迹、完整 simulator snapshot 不能成为搜索规划器输入。公开标量和公开运动规格宜通过受限 chapter adapter 提供，不能把整个 env 或 scenario 真值字典交给新评分器。

## 5. 安全与实际安装路线

- 图连通和线段有效只针对当前地图。在线 core 的 segment predicate 采样检查已知占据格，允许未知空间，且该实现忽略 clearance 参数；不能由此推出真实未知环境无碰撞保证。
- D2 的公开安全视图、`PublicGeometry` 与 `SafetyBridge` 提供额外过滤、逐步检查剩余路径、必要时替换路线或 Hold；部分分配有原子拒绝/保留机制。必须复用与对照同样的安全链。
- `PathTracker` 默认阈值 0.75，按顺序更新局部点；它没有保证真实轨迹严格落在折线上。惯性、流场、动作残差和 Hold 都影响时间参数化；已有 snapshot、cross-track、remaining-path 诊断可复用。
- `WaypointManager.stabilize` 会因相同 candidate ID、近似相同终点、未受影响或 cooldown 保留旧 assignment。这意味着固定终点换路线可能被直接保留逻辑吞掉。新路线应在明确的稳定后路线细化位置安装，并独立处理路线版本与 tracker 状态；不能只改路径评分却不确认实际执行。
- SafetyBridge 的覆盖签名已包含路线，能识别同终点换路线，但也可能将新路线替换为其安全修复路线。必须分别记录 generated、selected、stabilized、safety-effective、tracking 路线。

已知历史开发报告 `docs/chapter3/search_diagnostics/bser_final_v1/results_20260930/report.md` 记录：20 个配对开发场景，F0/D2 Found 12/20、发现前碰撞 3/20；F1 为 11/20、4/20，F2 的主结果与 F1 相同。该报告还记录 Hold 且速度非零的碰撞和安全连接器失败。这里只引用既有报告，**本轮未重新校验其原始轨迹**。结果不证明新 BA-STPR 无效，但警示“代理分数提高”不能替代真实闭环发现收益，且应先隔离安全失败的影响。

## 6. 首次发现概率的数学修正

### 6.1 A 方案有条件成立

对来自合法联合轨迹分布的固定假设 m，用共享 swept 判据得到首次发现步 T_i,m，则

\[
J_i=\sum_m w_m\gamma_D^{T_{i,m}}\mathbf1[T_{i,m}\le H]
\]

是该模型下的加权样本估计，不能仅因调用真实运动函数就声称是精确环境概率。有限 M 具有采样误差；预测搜索器轨迹具有控制误差；当前 belief 和未知地图还有模型误差。不同路线应共用同一组目标假设及权重，避免采样噪声主导排序。

团队使用 T_team,m=min_i T_i,m，仅计一次发现。固定、相互独立执行的候选路线条件下，这成为加权 max/覆盖目标，具有可证明的单调次模结构；该结构本身是已有数学性质，不构成原创。加入负行程成本后不再自动单调；每智能体至多选一条是 partition matroid，不能直接套用普通 cardinality greedy 的 1-1/e 保证。路线间避碰、观测触发重规划等耦合也需重新说明假设。

### 6.2 B 方案不必止步于边际独立乘积

若 q_k 定义为 `P(D_k=1 | D_1=...=D_(k-1)=0, 当前信息, τ)`，链式法则给出 `P(T=k)=q_k Π_(j<k)(1-q_j)`，**不需要另加时间独立假设**。困难在于从未条件化的边际 b_(t+k) 算出的探测率不是这个 hazard。

相同的两个边际探测率 (0.5,0.5)，实际至少发现一次的概率可以是 0.5（完全相关），也可以是 1（事件互斥）；独立公式给 0.75。仅靠边际无法识别二者。

若拥有合法的 Markov 转移模型，可使用未发现概率质量递推，而非直接将边际相乘。令 s_0(z)=b_t(z)，对确定的团队路线：

\[
a_k=\sum_{z,z'}s_{k-1}(z)P(z'|z)D_{S,k}(z,z'),
\quad
s_k(z')=\sum_zs_{k-1}(z)P(z'|z)[1-D_{S,k}(z,z')],
\quad J=\sum_{k=1}^H\gamma_D^k a_k.
\]

这里 D 依赖转移两端，以匹配 swept 判据；团队 D 是任一 Searcher 满足发现。s 不重新归一化，质量守恒为 `Σ_k a_k+Σ_z s_H(z)=1`。该递推保留所选模型的时间相关性，是可用的标准生存概率计算；对匀速反射仍需 z 包含必要的速度等状态。若只用现有扩散，则只能宣称“扩散模型下”的首次发现收益，不能声称已实现匀速反射预测。

### 6.3 截止发现概率与提前发现偏好要分开

gamma=1 最大化 H 内发现概率；gamma<1 兼顾提前发现但可能牺牲期限内发现率。这是设计取舍，不能把二者称为同一指标。建议同时报告 `P(T<=H)`、首次发现分布、全样本惩罚发现时间；不能只报告已发现回合的条件均值。

路线到达目标点后继续 Hold、继续搜索还是截断，也必须在固定 H 的模型中明确。否则不同路线的收益可能比较了不同曝光时间。H 不得超过剩余任务步数；未知目标速度和到达时间误差需单独做敏感性检验。

## 7. 相关论文及创新边界

以下为截至 2026-10-17 检索到的相关原始来源，属于定向查新而非穷尽查新；出版论文与 arXiv 作者稿分开标示。未将论文自己的实验数字转述为本项目效果。

| 原始来源 | 已有内容与本方案的关系 | 本项目可能保留的差异 |
|---|---|---|
| [Morin, Abi-Zeid, Quimper, 2023, EJOR: Ant colony optimization for path planning in search and rescue operations](https://www.sciencedirect.com/science/article/pii/S0377221722004945) | 有限时域/资源下，对 Markov 移动搜索对象最大化发现概率；说明目标本身属于成熟最优搜索问题。 | 不使用其 visibility 模型改写当前环境；需比较相同公开模型下的标准搜索规划。 |
| [Wei 等, 2024, 系统仿真学报: Platform Path Optimization Method Based on Cumulative Detection Probability of Sonar Search](https://www.china-simulation.com/EN/10.16182/j.issn1004731x.joss.23-0953) | 移动目标轨迹情景、时间相关声纳 CDP、Bayesian 情景修正及累计发现/发现效率多目标优化。与本方案非常接近。 | 当前系统的确定性 swept 发现、未知图条件预测和异构团队执行，而非重新发明 CDP。 |
| [Xiao 等, 2024, Journal of Field Robotics: MPC-based cooperative multiagent search for multiple targets using a Bayesian framework](https://onlinelibrary.wiley.com/doi/10.1002/rob.22382) | Bayesian 概率图、基于探测概率/熵的 MPC、多智能体观测点与安全路径，包含三 UAV 实验。 | 固定上层目标后的路线细化、时间相关首次发现和严格冻结底层接口。仅核到摘要/书目信息，不据此认定其完整时序处理方式。 |
| [Jeong 等, 2025, arXiv:2502.01041](https://arxiv.org/html/2502.01041v1) | 时间变化的加权信念、轨迹预测、time-configuration space 和异构多智能体团队效用；假定已知环境。 | 不增加 LSTM，不依赖完整已知图；需说明合法先验与未知反射处理。 |
| [Sudha, Popović, Coates, 2025, arXiv:2508.14636](https://arxiv.org/html/2508.14636v1) | 海上 ASV 移动目标的时空分布预测、候选轨迹评分与探索/重探测效用，含实地验证。 | 首次发现终止型目标与模型语义一致性；不是其持续跟踪/熵目标的直接复制。 |
| [Yuan 等, 2025, arXiv:2510.10421](https://arxiv.org/html/2510.10421v1) | 在演化 belief 区域生成覆盖路径，估计发现成功率和所需时间，再做分层规划。 | 异构多 Searcher、未知障碍、固定终点路线选择；不能声称首次把发现概率与路径时间结合。 |
| [Zhang 等, 2026-07-15, JMSE: Synergy Strategy Bezier Curve Optimization for UAV-Based Moving Target Search in Maritime Search and Rescue](https://www.mdpi.com/2077-1312/14/14/1302) | 出版商页面索引显示：时间变化目标 belief、可达运动向量路线、CDP 优化及有限时域首次发现建模。 | 有界在线路线细化、团队联合首次发现与实际执行误差。该站正文请求受限，本轮依据出版商检索片段，未审完全文公式。 |

本地 `文献/第三章文献/papers/manifest.json` 还列有 2023 intent-based MAIPP、2025 多目标主动搜索和 2026 高低层解耦搜索论文。已核查相关在线原始来源；本轮未重新抽取本地 PDF 全文。高低层解耦、belief 更新和合作去重亦不能单独作为本章新颖性。

研究贡献更有希望落在：**明确刻画静态代理何时排序失真；在合法先验与已知地图条件下保持轨迹相关性；把团队去重和异构执行时间纳入可计算路线收益；用校准和闭环对照验证收益。** 这些仍是待验证的贡献候选，不能据本轮定向检索保证可发表或达到答辩标准。

## 8. 已运行的有界机制诊断

目的仅为证明静态评分与移动首次发现排序可以不一致，不测试真实场景性能。没有环境实例、训练、checkpoint、随机抽样或 RNG 状态读取；随机种子不适用（零随机调用）。固定假设是测试输入，不来自任何待评估回合真值。

- 空间边界 (20,20,8)，无障碍，起点 (0,5,2)，终点 (8,5,2)。
- 上路线经过 (0,7,2)、(8,7,2)；下路线经过 (0,3,2)、(8,3,2)。两者长度/名义时间均为 12，搜索器按固定 1 单位/秒沿折线执行。
- 目标假设初始 (4,4,2)，速度 (0,0.5,0)，dt=0.2，H=60。半径 2.0 加偏置 0.10。
- 复用 `TargetState`、`simulate_target_trajectory`、`swept_relative_min_distance` 和静态核的 `_point_segment_distance`。

| 路线 | 静态核分数（p_max=0.72, sigma=2, q_free=1） | 首次发现步 | 最小 swept 距离 |
|---|---:|---:|---:|
| 上 | 0.2337497764980118 | 21 | 约 0 |
| 下 | 0.6353977698609087 | 无 | 3.5777087639996563 |

这是单连续位置假设上的核诊断，不是完整栅格 belief、A* 候选生成或动力学闭环实验。所用目标速度 0.5 在标准生成器 0.25–0.55 范围内，但固定直线方向和给定初始位置并不代表从标准场景分布抽样；后续必须在实际候选通道和真实公开 belief 上验证差异频率，不能用此数值宣传标准场景收益。早先还检查过速度 1、初始 (3,2,2) 的存在性例子，上路线第 18 步发现；该速度超出标准范围，不作为本节最终复现案例。

复现代码（从仓库根目录用项目 AUV Python 执行；不写文件）：

```python
import numpy as np
from core.env.target_motion import (
    TargetState, simulate_target_trajectory, swept_relative_min_distance,
)
from chapter3_bser.detection_model import _point_segment_distance

dt, H = 0.2, 60
start, goal = np.array([0., 5., 2.]), np.array([8., 5., 2.])
hypothesis = TargetState([4., 4., 2.], [0., .5, 0.], 0,
                         'constant_velocity_reflect_v1')
target = simulate_target_trajectory(hypothesis, H, dt, [20., 20., 8.], ())
for name, y in [('upper', 7.), ('lower', 3.)]:
    path = np.array([start, [0., y, 2.], [8., y, 2.], goal])
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    times = np.r_[0., np.cumsum(lengths)]
    positions = np.array([[np.interp(k*dt, times, path[:, j])
                           for j in range(3)] for k in range(H+1)])
    d = min(_point_segment_distance(hypothesis.position[None, :], a, b)[0]
            for a, b in zip(path[:-1], path[1:]))
    static = .72*np.exp(-d*d/(2*2.**2))
    first = None
    for k in range(1, H+1):
        distance, _ = swept_relative_min_distance(
            positions[k-1], positions[k], target[k-1].position, target[k].position)
        if first is None and distance <= 2.10:
            first = k
    print(name, float(lengths.sum()), float(static), first)
```

## 9. 最小改造位置、复杂度和实施前门槛

生产算法尚未修改。建议新的 chapter 命名空间放路线候选、公开规格、轨迹假设/生存质量评分、路线细化缓存及日志。复用 `TravelCostService` 的公开图和 shared target motion 函数，不把本章评分放进 core。

首版 K=3 或 4：保留原路线，以少量合法栅格中间点或受限边惩罚获取替代通道；中间点应是现有图可连接节点，不能假定任意偏移点已有 connector。按路线/通道而非只按终点去重，过滤循环和超预算路径，失败回退原路线。此生成方式没有全空间完备性/最优性保证。

在上层 assignment 稳定后、guidance 安装前细化路线；保持目标点、待命策略、事件和安全机制相同。须处理原稳定器的同终点保留规则，并验证 SafetyBridge 最后实际安装路线。目标点和路线可以在未来合法事件触发时更新，“目标点不变”应定义为每次同快照比较，而非永久冻结整回合目标。

若图为 V/E、目标轨迹假设 M、时域 H、每搜索者 K：有限中间点/有限次 A* 的生成成本约 O(K(E+V log V))（连接器组合等实际常数另计）；共享目标传播 O(MH C_motion)，所有搜索路线的 swept 评分 O(3KMH)。预缓存每假设首次发现时间后，三 Searcher 的 K^3 个组合最多 64（K=4），团队组合评分 O(K^3 M)。无需因三个智能体就无条件引入复杂大规模求解器。稀疏 Markov 生存递推另依赖增广状态图边数；速度维度和未知地图假设可能主导成本。

实施前应先完成以下诊断，当前均未运行：

1. 冻结公开运动先验：仅位置 belief 如何连接合法速度假设；是否维护 chapter 私有联合后验；未知障碍反射条件假设。若不允许这些附加估计状态，应降级为明确的扩散/边际近似，不宣传准确反射预测。
2. 同一真实公开快照、同终点生成候选，统计候选不足率、几何差异、沿途概率差异和排序翻转率。若几乎只有一条可行路线或 belief 近均匀，固定终点路线细化的增益上限很低。
3. 先用确定性运动先验/零残差控制隔离地图贡献；在标准速度和已冻结场景里比较预测 ETA、实际 ETA、跟踪偏差与真实发现事件。不得为此自动恢复 Actor/checkpoint。
4. 同池对照：原路线；同候选池静态核；同池静态核加到达时间折扣；同池时变边际近似；同池相关性保留的首次发现；再独立去掉团队去重。已有 F2/F 系列作为相关机制参考，保持其目标点差异等解释边界。
5. 正式配对评估中冻结安全层、地图更新、目标点/待命机制、下层策略、算力预算和种子；收益差异不能归于更大的候选池或额外真值权限。报告 Found、全样本惩罚发现步数、发现前碰撞、候选失败、安装率、回退率、规划 P50/P95 及模型校准。碰撞为竞争终止事件，不能简单当作独立删失。
6. 固定终点只是条件路线优化，非完整 POMDP 全局最优；长绕行要受剩余时间/相对基线预算约束。gamma 和预算须在开发集确定，确认集不得继续调参；既有 20 开发场景已反复使用，不宜充作独立确认集。

## 10. 最新记录的本地任务范围验证

项目解释器：`D:/anaconda/anaconda/envs/AUV/python.exe`。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -B -m unittest `
  tests.test_bser_detection_probability `
  tests.test_bser_objective_properties `
  tests.test_bser_planning_state_readonly `
  tests.test_early_discovery_search `
  tests.test_bser_final -q
```

结果：25 tests，0.187s，OK（unittest 计时，不含进程和依赖加载）。这是现有组件的任务范围本地验证，不是 CI、全回归或 BA-STPR 性能通过。首次使用系统 base Python 时出现 OpenMP 重复运行库错误并退出；改用上述已有 AUV 环境后通过，没有启用 `KMP_DUPLICATE_LIB_OK` 绕过错误。

当前不能直接实现/宣称的内容：从位置边际唯一恢复真实速度轨迹；在未知障碍图上精确预测真实反射；仅用不带时间戳的折线确保执行 ETA；已知图可达推出真实闭环无碰撞；从定向文献检索确认全球新颖性；从合成反例或历史报告证明 BA-STPR 有性能增益。上述限制不是代码缺陷的统一判断，而是需要单独解决的研究假设和验证边界。
