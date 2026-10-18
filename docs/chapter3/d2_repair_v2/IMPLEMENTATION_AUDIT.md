# D2 repair v2：实现审计与修复

## 验收结论

**`MECHANISM_VERIFIED`**。指定范围内的接口、数值更新、动作与回放对齐、事件时序和有界真实环境验证通过。尚未证明训练性能，也未满足 `READY_FOR_MANUAL_PILOT`：相关回归存在保留的旧来源门禁失败，历史 E0 黄金文件缺失。

本报告描述本次最新记录的本地验证，不是 CI。没有启动正式训练、恢复用户模型、修改用户 `outputs/`、提交或推送代码。实验性能不能由这些短机制场景推断。

## A. 版本与来源

- 实际分支：`codex/framework-audit-20261018-164056`。
- 基准及当前 HEAD：`6538c3e6a67e96d56940ab09fcff15cd9dc7a7d8`；修复仍在工作树，未提交。
- 初始受 Git 跟踪的工作树干净；已有 `tmp/`、`output/`、文献和 MAP-STP 未跟踪资料受到保留。完整初始状态见 [before.json](evidence/before.json)。
- 修复方法：`D2_B2_repair_v2`、`D2_B3_repair_v2`；配置与 checkpoint schema：`ch3.d2.implementation_repair.v2`。
- 独立来源清单：[d2_repair_v2_evolution.json](../../provenance/d2_repair_v2_evolution.json)。当前清单 SHA-256：`5dd3288e3aa3bf2e7634ffdc9e200965313acddd32362effa23b4dd949b1f171`。
- 当前完整生产代码 inventory SHA-256：`244636b8c253e446c6d4492a6bba951e9948b7924f0bd76d7a52c808909f153f`，匹配 `windows_existing` 配置。
- 来源按实际固定父提交的 Git 内容另列 `git_archive_preserved` 配置；它保留原本就在 Git 中的 CRLF，不把所有文件假定为 LF。

只修改两个既有生产文件，新增 8 个 v2 生产文件，其余 **302 个既有生产文件字节保持不变**。旧 B2/B3、repair v1、Linux Suite 方法映射、环境、规划器和历史来源清单没有改写。全部 27 条历史 provenance 记录保留，并通过历史基准与明确后继演进的检查。

| 既有生产文件 | 修复前 SHA-256 | 修复后 SHA-256 |
|---|---|---|
| `core/algorithms/maddpg.py` | `fbf983ec992613a45c10a2ced45725c17b2a412414d64167cb02b33ab8692a91` | `206c4b134ac16cfc903cdeabd6ee77de0ca5205dab36702abbf3bdd7ab40ef66` |
| `core/replay/ch3_buffer.py` | `e22412e43e589962582c2a3ce727b167a6689e60223f6d64fd71ef1e9e5cfa42` | `1ad6547e89c8c0291a3633f9a7b46f7c8a3b428577a6dee34115ba2bf04876cb` |

35 个代码、配置、测试、验证器及来源清单文件的新增/修改状态和前后完整 hash 见 [change_inventory.json](evidence/change_inventory.json)。新增文件的前版本 hash 为 null；新增生产文件另记录 v1 对应文件及其父版本 hash、变更原因。报告与证据的 hash 记录在 [acceptance.json](evidence/acceptance.json)，不自包含 hash。

六个既有测试文件只把后继 provenance helper 的导入改为 `tests.d2_repair_v2_review`。新 helper 先运行原 v1 的精确演进，再检查两个新生产 hash；没有删除旧断言、跳过来源测试或扩大豁免。`test_long_horizon_maddpg_v1.py` 的导入调整不代表重新采用该方法；本次未启用它。

### 新版本各文件职责

| 路径（相对 `chapter3_bser/experiments/d2_repair_v2/`） | 职责 |
|---|---|
| `config.py` | v2 schema、方法和修复开关校验；沿用原始任务、奖励及输入维度 |
| `runtime.py` | 显式复用 v1 已审计的事件和执行门控机制，输出 v2 身份 |
| `model.py` | 使用原 B2/B3 网络，检查门控、冻结、动作和团队终止的一致性；隔离 checkpoint 身份 |
| `replay.py` | 封闭普通 push 旁路；原子验证、写入与采样 execution 元数据 |
| `provenance.py` | 完整来源闭包校验、临时机制运行范围及正式运行验收门禁 |
| `train.py` | 复用父 Trainer 的场景、更新调度与外层循环；执行来源检查和失败记录 |
| `evaluate.py` | v2 checkpoint、独立评估场景和显式机制评估权限 |
| `__init__.py` | 独立命名空间 |

共享 learner 和 Replay 的通用校验放在 `core`；章节算法和执行元数据留在 Chapter 3。没有新增通用 Trainer、网络、算法模块、legacy import 或 sys.path 注入。

## B. 缺陷结论及作用范围

修复前独立反例见 [before_interfaces.json](evidence/before_interfaces.json) 和 [before_probes.log](evidence/before_probes.log)。本轮先读源码、保存审计快照并复跑探针，再进行修改；[PRE_REPAIR_AUDIT.md](PRE_REPAIR_AUDIT.md) 保留该阶段结论。

| 项目与准确位置 | 触发与修复前证据 | 本次处理及修复后证据 | 历史 B2/B3 与旧 Linux 的适用范围 |
|---|---|---|---|
| 航点时序：`core/env/uav_env.py:2487`；`d2_repair_v1/runtime.py:14` | 旧 `legacy_v1` 路径在到达事件设置前计算奖励，缺少该次 bonus；固定父版本已有显式 opt-in 修复 | 未再次改环境。v2 显式继承 `arrival_before_reward_v1`；真实到达只发一次，普通步其他奖励相同，发现及碰撞优先级、地图重规划均通过 | 旧配置仍使用旧时序；v2 开关仅修复内部搜索航点事件，不能把外部 D2 guidance 航点当作同一事件。旧 Linux 受影响程度未归因 |
| 冻结 Actor：`core/algorithms/maddpg.py:353`；`d2_repair_v1/model.py:30`；`d2_repair_v2/model.py:13` | 实际发现后冻结样本仍改变旧 B2/B3 搜索 Actor，最大参数差约 0.001001 | 复用已有逐样本门控并收紧元数据校验；修复版三个搜索 Actor 的参数、Adam 状态和更新计数不变，Critic 继续学习团队回报 | 旧 B2/B3 保留历史行为，真实机制重现；这是可达机制，仍无证据量化其对旧 Linux 差距的贡献 |
| 批长广播：`core/algorithms/maddpg.py:185` | 7 条 obs/action/done 配 1 条 reward/weight，旧 B2/B3 公共 learner 接受并写参数 | 全字段形状、agent 数、批长、有限值、二值标志及权重检查置于 target 和 optimizer 前；非法样本不改变训练状态 | 当前正常 Trainer 本来提供正确批长；不能将该外部接口反例当作旧 Linux 已触发的证据 |
| Replay 恢复：`core/replay/ch3_buffer.py:174` | 容量 3 写 5 条后恢复到容量 5，旧指针导致采样到未写入的 sentinel -9876 | 不支持容量迁移；容量、配置、环形指针、字段形状及数值在任何 mutation 前验证。同容量 round-trip 和继续写入通过；坏 state 完整保持目标对象不变 | B2/B3 和修复 Trainer 是 fresh-only，此问题不能直接解释旧训练性能 |
| 门控写入旁路：`d2_repair_v2/replay.py:15`、`:18`、`:54` | v1 继承的普通 push 可写无 execution 元数据的槽位；一次反例错误跳过 Actor | v2 普通 push 明确拒绝；完整 transition 预验证，当前/下一 gate 同一槽位写入，initialized bit 和采样索引检查；环绕对应关系通过 | v1 生产文件保持原状；其正常 Trainer 使用 push_transition，未证明旧路径调用过该旁路 |
| CPU 快照共享存储：`core/algorithms/maddpg.py:88`、`:472` | 获取 CPU state 后更新在线模型，快照 hash 随之变化 | 参数和嵌套 optimizer Tensor clone，init_dict 深复制；在线更新后快照不变，加载后训练状态一致 | 影响保留在内存中的 snapshot；同步即时 torch.save 不据此判为损坏，未证明旧 Linux 受影响 |
| step 数量截断：`core/algorithms/maddpg.py:156` | 原始 core.step 接受 3 个 agent 并返回 3 个动作 | 数量、特征形状、非空及有限值先校验；0/3/5 个 agent 拒绝，合法输入保持 | 正常 runtime 已约束 4×3；未证明该反例曾进入旧训练 |
| 诊断标志：`core/algorithms/maddpg.py:301` | update 后再 critic_only，last_actor_updated 仍为 true | critic_only 入口清除该标志；Actor 参数和 Adam 不动 | 仅修复日志语义，不据此解释任务性能 |

固定父版本已经修复的本地 DDPG critic 输入、终止 NaN target、可变 sample 缓存和 twin critic 梯度写入顺序仍经相关回归。它们不是本次新发现，也不能据此推定旧 B2/B3 使用了本地 DDPG 分支。

### 版本及行为隔离

旧路径仍为 `d2_suite_v1/run.py:80` → `DirectMCTrainer` / `DirectBoundaryTrainer` → 原 BaselineMissionRuntime → core MADDPG；原配置和原来源校验均保留。旧 Suite 未切换到 v1 或 v2，旧 checkpoint 不可改名载入 v2。

本次共享 core 改变非法输入的拒绝行为、快照存储独立性和诊断标志，不改变合法样本上的学习公式。全修复关闭时，完整宽度 B2/B3 与修复前 core 冻结副本的 loss、在线参数和 target 更新一致。原副本 hash 必须匹配父文件，不能依赖当前模块的参考函数自证等价。

旧 repair v1 的训练入口现在会因 core 精确 hash 演进拒绝当前工作树；这是保留旧实验身份的预期拒绝，并非 v1 来源检查被放宽。旧版本需要在其精确父版本环境运行，不能在当前目录冒充原实验身份。

## C. 学习链路验收

### 原始训练定义保持

保持 4×28D 局部观测、4×3D 归一化残差、124D 集中式 Critic，B2/B3 原网络及 B3 boundary encoder；gamma=0.95、tau=0.005、lr_actor=0.001、lr_critic=0.0005、hidden_dim=128、OU 噪声不变。batch=128、warmup=256、update_frequency=4、updates_per_train=2、replay_size=500000；原复合奖励、团队均值适配、成功及碰撞定义不变。

原 Critic target 具有既有 `[-10,10]` 截断，本轮未改。数值 oracle 同时检查截断前 Bellman 公式与最终截断值，避免把原有截断误报为公式缺陷。

### 真实环境轨迹

四种对象分别是旧 B2、旧 B3、v2 B2、v2 B3，均全新初始化。每个对象 7 个场景、147 个环境步，每场景不超过 32 步；每模型最多 4 轮完整多智能体更新。共 588 步，覆盖：未发现搜索、首次发现、等待合法交接、Executor 知情、搜索者冻结、成功、搜索碰撞和超时。

- near：第 1 步发现，第 2 步交接；团队未在搜索者冻结时提前终止。
- success：第 1 步发现，第 2 步交接，第 6 步成功。
- collision：合法无碰撞初始条件，第 1 步实际动力学碰撞终止；终止奖励为现有 -2。
- normal：运行 32 步仍未发现，真实超时。

上述场景采用真实环境允许的初始位置、目标、障碍与流场条件；不篡改 reward、done、next observation。它们是事件机制专用构造，不能代表正式场景性能。

独立 observer 记录 raw Actor、实际 OU sample、裁剪动作、执行门控后的动作、环境入参、先验/残差/合成加速度、速度差分、奖励分项、团队奖励和观测转移。非终止 next observation 与下一决策 observation 一致；Replay 按采样索引逐字段与真实 rollout 比较，包括 gate。修复版禁用动作严格为零；旧 B2/B3 的 16 次无效提议作为历史诊断保留，没有把它们改写为正确动作。

完整逐步原始追踪留在本机 `tmp/d2_repair_v2_20261018/trace_*.json`，路径及 hash 见 [trace_index.json](evidence/trace_index.json)；Git 可见证据只保存事件边界样例和紧凑汇总，避免提交原始轨迹。旧轨迹没有 v2 execution metadata，该差别明确保留。

| 对象 | 完整更新轮数 | 每 agent Actor 实际步数 | 全体 twin Critic optimizer 步数 | 冻结搜索 Actor 参数差 |
|---|---:|---|---:|---|
| 旧 B2 | 4 | 3,3,3,3 | 32 | 约 0.001001 |
| 旧 B3 | 4 | 3,3,3,3 | 32 | 约 0.001001 |
| v2 B2 | 4 | 2,2,2,3 | 32 | 0,0,0 |
| v2 B3 | 4 | 2,2,2,3 | 32 | 0,0,0 |

4 轮分别为 Critic-only、混合 batch、冻结 batch、有效 batch。对 B2 共 384 个参数 Tensor、B3 共 480 个参数 Tensor 逐轮核对软更新 `(1-.005)*old + .005*online`。参数归属、Adam 步数、冻结状态和模型计数一致。

### 独立数值与单因素验证

- 手算双 Critic 最小值与 Bellman target；terminal 不 bootstrap。terminal next-Q 为 NaN 时 target 仍有限；nonterminal NaN 在 Critic optimizer 前拒绝。
- 固定 Q/target 独立计算 Huber loss 和导数，用临时标量 Critic 验证方向及更新量；生产网络未因此改变。
- 已知 Q 对动作导数的 oracle 分别验证 B2/B3 Actor 朝提高 Q 的方向更新；混合 gate 与仅有效行参考损失一致。
- 每个 optimizer 只拥有所属网络，B3 encoder 属于 Actor optimizer；target encoder 与 target Actor 不加入在线 optimizer。boundary 当前与 next context 保持对应。
- 非有限 loss、target、梯度在对应 optimizer 写入前拒绝。Actor 阶段失败前已经完成的合法 Critic 更新不会自动回滚；Trainer 记录失败并抛出，不能继续标为成功。
- 全关、仅航点时序、仅冻结 Actor、仅保持/不可达残差门控、全部启用分别通过；全关合法更新与旧参考一致。
- 真实航点测试覆盖连续到达、无到达、发现与到达同一步、碰撞与到达同一步以及地图 revision 实际递增。发现后没有活动搜索路径，但固定三元素内部 waypoint Tensor 契约保留，不人为要求其变为 None。

### 原调度、checkpoint 与评估闭环

另外对 v2 B2/B3 各使用一个全新模型，按原 Trainer 调度各跑 8×32=256 步，满足原 warmup 和 batch 后执行恰好 2 轮更新；每 agent Actor 步数 2、twin Critic 总步数 16。保存的临时机制 checkpoint 记录配置 hash、模型 hash、完整来源身份、环境/episode/optimizer 计数及训练场景 ID/seed/fingerprint。

两个模型均完成来源验证后的保存、重载及 32 步独立场景评估；模型 hash 和优化器状态一致；篡改来源时保存被拒绝，旧 schema 被拒绝，机制 checkpoint 默认正式加载被拒绝。临时目录自动清理，没有读取或生成用户正式实验模型。

旧 B2/B3 的真实 Runtime→Replay→core update 链已验证，但原完整 CLI 的来源门禁仍拒绝当前目录；不把上述机制探针称作旧入口已经端到端通过。v2 的真实 Trainer 调度闭环单独通过。

## D. 来源和验证状态

详见 [VERIFICATION.md](VERIFICATION.md)。27 项定向、补充 checkpoint 1 项、最终来源 4 项通过；相关回归运行 108 项，101 条成功记录、1 条 failure 和 9 条 error 记录。subTest 和 setUpClass 导致记录数不能相加作为 testsRun。所有 traceback 原样保留，没有 skip。

9 条 SafeSearch 门禁记录（8 error、1 failure）延续父版本问题；新增第 9 条 error 来自旧 repair v1 检测到本次显式 core 演进。当前结果不能描述为“全部回归通过”。

另补查 21 项：PRRAC/Phase1C checkpoint 相关 18 项通过；旧 Suite 的 pipeline、Linux launch、evaluation 三个仅预览测试均在原 SafeSearch 来源门禁处报错，没有执行旧 Suite。该补充结果单列于 supplemental.json，不合并或覆盖上述回归记录。

首次干净归档校验因新清单错误假定统一 LF 而失败；保留 [clean_workspace_initial_failure.json](evidence/clean_workspace_initial_failure.json)。后按固定父 Git blob 修正**新**清单，当前及干净归档来源验证、干净目录 27 项测试均通过，未改旧文件或旧清单。

历史 `experiments/chapter3/e0_core_migration/golden_trace_manifest.json` 不存在，E0 验证未完成。没有生成替代数据。本次干净目录是固定父 `git archive` 加明确未提交修复覆盖，不是已提交修复版本的 clean-clone 发布验收；当前不能宣称仓库发布闭包全部完成。

## E. 剩余风险与未覆盖路径

1. 复合奖励回报与 Found、碰撞、惩罚发现时间、Success 的排序一致性尚未证明；权重及 gamma 未调。当前短场景不能代替训练后的逐场景排序分析。
2. 28D 局部观测和 124D 联合局部输入仍不含完整地图、剩余时间和完整历史动力学状态；价值估计的充分性没有保证。本次仅验证接口语义，不扩大输入。
3. 保持/不可达门控约束指定条件下的 RL 残差，不构成未知障碍条件下最终动力学的碰撞安全保证；流场、惯性与未知障碍仍可能导致碰撞。
4. 训练稳定性、收敛、最终任务成功率和公平性能对比均未验证。机制成功不代表 RL 必然优于 D2。
5. 未提供旧 Linux `linux_pipeline_02` 逐场景数据与原始轨迹；不能把已确认机制直接归因为旧性能差距。
6. 旧 Suite/SafeSearch 来源门禁仍失败；旧 repair v1 当前 core hash 门禁也失败。相应训练/检查点操作继续停止，不能借 v2 新清单豁免旧门禁。
7. 历史 E0 缺失；完整提交后的 clean-clone、Linux 复核、GPU 学习、长训练、旧 Replay 迁移和正式 checkpoint 恢复均未运行。v2 明确 fresh-only。
8. Torch 对小型 list-of-ndarray 转 Tensor 发出性能提醒，已原样保留日志；未观察数值错误，也未据此更改冻结的网络和调度。

## F. 正式训练保护

未指定 `--execute` 的 v2 训练入口只输出计划。正常执行及正式 checkpoint 保存/加载要求来源匹配且 `acceptance.json` 为 `READY_FOR_MANUAL_PILOT`，并逐项验证定向、相关回归、来源、机制、历史 E0 及证据文件 hash。本次 receipt 保持 `MECHANISM_VERIFIED`，所以正式入口拒绝执行。

有界验证需要明确 `mechanism=True` 或 `--mechanism-check`，配置必须满足每场景≤32 步、≤8 回合、总开始预算≤256，且每临时模型≤4 轮完整更新。产物标为 `mechanism_only=true`、`formal_use_allowed=false`；没有自动恢复功能。

本次完成了实现修复和可执行的机制证据；正式性能和所有来源/历史等价性验收仍需后续独立完成。
