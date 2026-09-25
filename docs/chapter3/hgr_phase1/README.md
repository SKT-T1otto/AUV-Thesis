# HGR 第一阶段实现与验收报告

日期：2026-09-24。本文记录**最新本地限定验收**，不是 CI、真实任务性能或论文结论。

代码已在独立研究工作树实现，34 项限定单元/静态测试通过。
真实 runtime 验收为 **NOT_RUN**；效率收益为 **NOT_RUN**。
下一步是用户手动执行本文的有限真实验收，然后依据结果决定后续实验。

## 1. 版本与范围

- 实际核对的基线 HEAD：da4a8a64cba26b2adf7d248fd8486e517422628c。
- 研究分支：codex/hgr-phase1；尚未 commit/push。
- 原工作树：E:\gym\code\WORKSPACE\AUV-Thesis，交付检查仍为 clean。
- 研究工作树：C:\Users\lenovo\.codex\visualizations\2026\09\24\01a0d1f6-73d0-7d43-83b3-856fc1c8a6ef\hgr-phase1。
- 原工作树、历史输出、core、公共 planner、安全层、奖励和历史 provenance 文件均未修改。
- 冻结 M20_MOVING_UNKNOWN_MULTI、H=400、gamma=.95、team_mean_v1、collision_terminal_v1；
  28D/3D Actor、153D 辅助输入和历史 124D critic 合同保持原样。
- 不包含跨版本预测、Bernoulli 纳入、自适应预算、critic、replay 或新强化学习算法。

static_evidence.json 给出研究版完整生产源码身份、相对 HEAD 的修改/新增文件及哈希。
研究版源码身份与历史实验不同；**不宣称旧 checkpoint 可在研究版恢复**。
原 B0123 正式入口的 production_sources() 实际检查为 **FAIL：源码不匹配**。
这是正式实验入口的未解除阻塞，未修改旧 pin、历史摘要或白名单去放行。
诊断还涉及 Windows/LF 工作树版本比较；以 Git 修改清单判断本次真正改动。

## 2. 修改文件及理由

| 文件 | 改动 |
|---|---|
| chapter3_bser/models/hgr/phase1.py（新增） | 显式开关/修订、完整行为身份、受验证旁路、具名种子、配对噪声和 RNG 清点 |
| chapter3_bser/models/hgr/stable_predictor.py（新增） | 每周期清空、zero/ridge、float64 solve、固定 lambda、整批数值回退 |
| chapter3_bser/models/hgr/policy.py | 可选标准高斯噪声；未提供时保持旧 randn/latent/log_prob 路径 |
| chapter3_bser/models/hgr/estimator.py | 仅显式授权且重新核验身份的 HGR 可 K=0；原固定 K 公式、SGD 和一次消费不变 |
| chapter3_bser/experiments/hgr/runtime.py | 分支噪声表、串行 RNG 隔离、合法交接/原时钟断言、实际终止与轨迹摘要 |
| chapter3_bser/experiments/hgr/train.py | 新模式编排、zero 跳过 pilot、旁路、冻结预测、成本/失败诊断、新状态修订 |
| chapter3_bser/experiments/hgr/phase1_acceptance.py（新增） | 用户手动运行的有限真实验收，不调用训练/恢复入口 |
| configs/chapter3/hgr_phase1_zero.json（新增） | 默认 zero、lambda=0、policy_crn、精确零更新旁路 |
| configs/chapter3/hgr_phase1_ridge_diagnostic.json（新增） | ridge 诊断：alpha=1、kappa=4、数值参考尺度=1、lambda=.5 |
| tests/test_hgr_phase1*.py（3 个新增文件） | T1–T8、合成 runtime/周期、固定基线源码 fixture 和内存状态往返 |
| 两个平台的 run_hgr_phase1_acceptance 包装（新增） | 手动验收；失败返回非零；Linux tee 管道失败传播 |

provenance.py 未修改。新增章内 Python 自然进入完整源码清单。
旧配置缺少 phase1 或 enabled=false 时保留旧计算和流计数顺序；
旧 predictor_zero 仍采 pilot。新 zero 才跳过拟合专用 pilot。

新状态 schema 为 hgr.complete_cycle.phase1.v1，另存 phase1_revision、
predictor_revision、random_source_revision、named_counts。
恢复构造依赖明确 predictor 修订并 strict 加载。自动测试仅进行新对象内存往返；
未执行训练 checkpoint 保存/恢复验收，未读取用户 ep100。

## 3. 数学假设与源码对应

| 条件 | 源码和检查 | 证据边界 |
|---|---|---|
| 交接前分布不依赖 phi | HandoffPolicy.actions 前级全部使用 theta，含 Executor 待命；参数存储隔离 | 源码和合成测试通过 |
| tau 是动作前停止时刻 | collect_trajectory 在 suffix 首次为真后、advance 前保存边界；续分支核对恢复的 153D 合法特征摘要 | 真实边界恢复 NOT_RUN |
| Found 不等于可靠交接 | suffix 读 Executor 的 target_known_by_agent[3] | 既有合法知识接口未改 |
| 交接后不回到 theta | 后级只允许 Executor 活跃；已交接后知识回退则报错，不改状态 | 原知识标记仅 reset 清零；真实动态检查 NOT_RUN |
| 全部 theta 动作进入 score | 保留原始 latent/active-agent mask，新增 mask 一致性检查 | 包含待命 Executor；不用 tanh 后动作替代 latent |
| 后级训练与主批次独立 | 独立 purpose 种子；先更新 phi，再采 main；正式采样前后核对身份 | 原后级 MC/单次 SGD 未改 |
| 状态充分 | 原快照保存 runtime/env/v2/guided/mission/physics/provider/controller/bridge 对象图和别名 | 静态清点；不能宣称真实完整性已证 |
| 未来创新重采 | 恢复过程状态后一次设置环境种子；策略噪声表独立寻址 | 不逐步 reseed；未知持久 RNG 阻塞 |
| pilot 与正式标签隔离 | 只拟合本周期独立 pilot；先冻结 tuple，再选索引和查询 | 固定 lambda，正式结果不回灌/调参 |

快照过程状态含消息/交付、地图/缓存、目标位置速度、流场相位、
跟踪历史、奖励记忆、任务时钟和已安装引导。仅重新绑定导航闭包；
不 reset 任务、目标、消息、地图或引导。对象图外的外部状态完整性尚无一般性证明。

N 为所有主任务数，q=1/N，固定 K 次有放回。抽到无交接保留零项；
重复索引也创建新的 draw 身份和配对，不缓存标签。
令 C_i = 1_handoff gamma^tau_i sum_{t<tau_i} score_it，冻结 ftilde_i=lambda*f_i：

    g_delta = sum_i C_i*ftilde_i/N
              + sum_k C_Jk*(delta_hat_k-ftilde_Jk)/(N*K*q_Jk)

条件于主批次和正式采样前信息 F，若 E[delta_hat|J=i,F]=mu_i，
每个残差查询期望为 sum_i C_i*(mu_i-ftilde_i)。
加预测项后均值为 sum_i C_i*mu_i/N。独立新标签和独立 draw 给出：

    r_i = mu_i-ftilde_i, b = sum_i C_i*r_i
    Cov(g_delta|F) =
      [sum_i C_i*C_i^T*(sigma_i^2+r_i^2)/q_i - b*b^T] / (N^2*K)

这是条件校正协方差，不是完整梯度方差，也不是 Bernoulli 纳入公式。
精确分数枚举覆盖均匀/非均匀 q、K=1/2、随机标签、lambda=0/.5/1；
浮点实现另按固定绝对容差 2e-12 对照。
有限 MDP 包含多次前缀动作、动作相关 tau、无交接、临近截止和不同后级终止长度；
新目标精确导数与 g0+校正期望一致。故意令后缀依赖 theta 的反例确实暴露缺项。
这不证明所有真实环境或扩展算法，也不证明效率提升。

## 4. 身份、旁路与预测器

行为身份读取实际对象，不仅构造配置：

- named_parameters/named_buffers 的 dtype、形状和精确字节，包括非持久 buffer、log_std。
- 每层 vars(module)：mean_epsilon、router.temperature、LayerNorm.eps、激活属性、train/eval 模式等；另读实际维度属性。
- 运行合同、latent/logstd/tanh 变换修订、显式旁路授权。
- 动态 hook、无法审计的普通属性类型直接拒绝。
- optimizer 状态和计数不参与身份；同时比较摘要与完整序列化字节，不用 allclose、KL 或 probe。

仅显式启用的新 HGR，phi0/phi1 完整身份相同、主轨迹完整新鲜、
当前 theta 身份匹配、边界记录一致、同 gamma 且未消费时允许旁路。
旁路不调用 pilot、fit、forward 或正式分支；prediction/correction/g_delta 精确为零，
仍执行 g0 的一次 SGD 和消费。direct_boundary_corrected 校正 V_phi1，绝不旁路。
一般改动后的 HGR 仍需 K>0。
日志含 skip_reason、K_requested/K_actual、双方身份、冻结预测摘要、
g0_norm、delta_g_norm、gfull_norm 和实际分项成本。

默认 zero 跳过拟合专用 pilot；ridge 仅诊断。
每周期清空有效性，n=0/1 回退零，全零标签确定返回零。
尺度为 sqrt((n*var+kappa*reference_scale^2)/(n+kappa))；
reference_scale=1 是数值参考，不是物理上界。
float64 dual ridge 用 solve，不显式求逆；n>=2 只是可计算门槛。
正式采样前冻结整个 ftilde tuple，预测项与残差项共用该值。
数值失败在正式查询前整批回退零并记录原因；非法原始状态/特征/回报/标签则报错，
phase1_failure.json 保留原因，不丢任务、不截断标签、不 nan_to_num。
拟合日志区分 pilot training MSE 与 independent_error=null，
记录模式、数据身份、尺度、alpha/kappa/lambda 和回退原因。

## 5. 随机源与配对范围

| 源 | 处理 |
|---|---|
| 策略高斯采样 | 每 pair 预生成 (400-tau,4,3) epsilon，按绝对步/agent 索引；分别计算动作和 latent |
| runtime.action_rng | 唯一许可的持久局部 Generator；分支注入 epsilon 后不依赖活跃 agent 的采样次数 |
| Torch CPU 全局 | map_module/path_planner 随机 top-k/多项式选择、waypoint 数、fallback；每分支环境种子独立 |
| Python random / NumPy 全局 | 原快照捕获/恢复，新分支隔离并给角色独立种子；审计路径未见额外未管理消费点 |
| 场景 NumPy default_rng | 具名 scenario 种子创建的局部生成器，属于初始分布，不是续分支过程状态 |
| 目标运动、流场相位 | 确定性更新的过程状态随快照保留，不重置成白噪声 |
| FixedReliableHandoff | 当前确定性延迟/事件，没有随机信道；事件随对象图恢复 |
| 未知持久局部 Generator | 恢复对象图递归检查，发现即报错 |
| CUDA | runtime 使用 CPU；隔离器保留调用方已初始化的 CUDA RNG；未验收 GPU runtime |

policy_crn 的准确范围是 **policy_only_environment_independent**：
策略 epsilon 对齐，环境角色种子相互独立，环境组件未逐时刻逐组件对齐。
independent 则策略与环境都按角色独立。两支串行，各自跑到真实终止，
一支早停不截断另一支。
具名 SHA256 输入包含 run_seed、cycle、purpose、draw/episode index 和规则修订；
共享 epsilon 键不含角色或策略 hash。独立环境键包含角色。
不同 draw/pilot/correction 分离；跳过 pilot/旁路不移动未来 main/suffix/正式选样流。
这是可复现伪随机流构造，不把“相同 seed”作为完整环境边缘正确性的证明。
G_plus 从 tau 相对折扣；gamma^tau 只在 C_i；不复用主轨迹旧后缀充当新标签。

## 6. 本地验收记录

环境：Windows，D:\anaconda\anaconda\envs\AUV\python.exe，Torch 2.11.0+cu126，CPU 合成测试。
沙箱内 Torch DLL 被拒绝，使用获准的同一现有环境运行限定测试，未扩大执行范围。
最终命令（PowerShell，在研究工作树）：

    & 'D:\anaconda\anaconda\envs\AUV\python.exe' -m unittest tests.test_hgr_phase1 tests.test_hgr_phase1_runtime_synthetic tests.test_hgr_phase1_legacy_fixture tests.test_hgr_mechanism tests.test_core_source_provenance tests.test_repository_metadata -v

结果：**34 tests，OK，16.232 秒**；本地记录，不是 CI。

| 验收 | 状态 | 说明 |
|---|---|---|
| T1 固定 K 期望/完整协方差 | PASS | 精确分数枚举 + 浮点实现对照 |
| T2 有限 MDP/反例 | PASS | 假设内相等；违背后缀隔离时暴露缺项 |
| T3 旁路 | PASS | 调用数、非法 K=0、完整性、g0/N、一次消费、direct boundary 不旁路 |
| T4 身份 | PASS | 参数/buffer/log_std/mean_epsilon/实际温度/模式；probe 不足；optimizer 无关 |
| T5 合成耦合 | PASS | 完整创新重放、带来源交换的反号、inactive、早停、相对折扣、声明范围 |
| T6 隔离/复现 | PASS | 具名流、全局恢复、合成 spawn；不冒充真实 spawn |
| T7 稳定预测 | PASS | 0/1/2、重复/常数/零/大尺度、无陈旧拟合、冻结、lambda=0、非法标签诊断 |
| T8 回归/状态/成本 | PASS | 固定 Git 基线源码合成 fixture，旧数值/流顺序一致；显式修订内存往返 |
| 27 条来源与元数据 | PASS | 8 项静态测试，历史摘要未重写 |
| Python compileall / Git diff --check | PASS | HGR 与新增测试的静态检查 |
| Linux bash -n / 新 CLI --help | PASS | 仅语法/参数解析 |
| Windows bat 实际启动 | NOT_RUN | 检查了失败退出逻辑；留给手动验收 |
| 旧 B0123 正式 source gate | FAIL / BLOCKED | 研究版源码不匹配；未绕过 |
| 真实 snapshot/continue/spawn | NOT_RUN | 留给用户 |
| 自然交接覆盖 | NOT_RUN | 手动没有交接时必须 NOT_EXERCISED/BLOCKED |
| 全梯度同成本效率 | NOT_RUN | 不从固定边界或单元测试推断性能 |
| 完整历史集成套件 | NOT_RUN | 含真实训练/恢复，按要求未整体运行 |

静态命令：

    python -m compileall -q chapter3_bser/models/hgr chapter3_bser/experiments/hgr tests/test_hgr_phase1.py tests/test_hgr_phase1_runtime_synthetic.py tests/test_hgr_phase1_legacy_fixture.py

    bash -n scripts/linux/run_hgr_phase1_acceptance.sh

    python -m chapter3_bser.experiments.hgr.phase1_acceptance --help

    git diff --check

旧 source gate 的只读检查（研究版当前失败）：

    python -c "from tools.ch3_baselines.provenance import production_sources; production_sources()"

本次真实环境步成本：**0**。合成替身步数只是计账测试。
若真实调用中途失败，诊断仅记录已完成调用成本，额外失败步数明确记为未知。

## 7. 用户手动真实验收

以下入口、配置和字段均是本次新增。进入研究工作树后执行；每条命令一行。
非空输出目录拒绝覆盖。

Windows PowerShell（已知 Python 环境）：

    & 'D:\anaconda\anaconda\envs\AUV\python.exe' -m chapter3_bser.experiments.hgr.phase1_acceptance --config configs/chapter3/hgr_phase1_zero.json --output outputs/chapter3/hgr_phase1/collision_terminal/manual_acceptance_01

Windows CMD（已激活正确环境，bat 检查 errorlevel）：

    scripts\run_hgr_phase1_acceptance.bat outputs\chapter3\hgr_phase1\collision_terminal\manual_acceptance_01

Linux（相同研究源码、正确 Python 环境；set -euo pipefail 保证 tee 失败传播）：

    bash scripts/linux/run_hgr_phase1_acceptance.sh outputs/chapter3/hgr_phase1/collision_terminal/manual_acceptance_01

固定计划：

- generator_seed=20260924、M20、train split、2 个预声明场景；清单在第一条任务前落盘，不按结果挑场景或重试。
- 新建 Actor，new phi.log_std 固定加 .01 作为诊断扰动；没有梯度训练、训练 checkpoint 或 ep100。
- 2 条 prefix 中按预声明顺序取第一个自然可靠交接；都没有则 NOT_EXERCISED/BLOCKED、退出码 2，不注入 Found。
- 6 对分支：基准、同进程重放、spawn 重放、连同随机源交换角色、independent、新的 policy_crn。
  重放不充当正式独立校正样本。
- 最坏预约 (2+2*6)*400=5600 环境步；重放/spawn 全计费；两支各自终止。
- 输出 plan.json、配置、场景清单、acceptance.json、本次新建决策快照，保留真实时钟/终止/步数、trace、源码/行为身份和耦合范围。
- FAIL、NOT_EXERCISED 或 spawn 超时返回非零，部分执行不记成功。

真实 PASS 只说明该预声明样本上的边界/恢复/续仿真检查通过。
降方差与全梯度效率实验另行手动执行：必须独立重新采主批次，
比较 HGR、直接新 phi1 MC、direct_boundary_corrected。
不要求 CRN 在任意场景都降方差；本次不宣称性能已经改善。

## 8. 手动验收记录模板

| 项目 | 填写 |
|---|---|
| 研究 HEAD、diff、source SHA256 | |
| 手动命令、新输出目录 | |
| 预声明场景是否全部执行、自然交接数 | |
| 状态 | PASS / FAIL / NOT_EXERCISED / BLOCKED |
| recorded_environment_steps / 5600 | |
| 失败调用额外步数是否未知 | |
| 原截止、实际终止、重放/spawn 一致性 | |
| coupling_scope、未知 Generator、剩余阻塞 | |
| 效率实验 | NOT_RUN，除非已有独立完整证据 |

未执行：真实训练、训练恢复、参数扫描、真实快照生成/恢复/续仿真、
真实成功率比较、完整梯度效率实验、1000 轮训练、正式论文评价。
