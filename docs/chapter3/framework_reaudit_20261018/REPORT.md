# D2 / B2 / B3 底层复审（2026-10-18）

**结论：当前仍不能验收为“底层全部无错、可直接开始正式长实验”。** 本轮在正式网络宽度下取得了新的真实转移与数值公式核对结果，没有在所测修复版正常调用链中发现新增的奖励错位、动作写错、Bellman 计算或软更新错误；但另复现了三个需要收口的接口边界缺陷。它们在当前 B2/B3 trainer 中没有被证明触发，不能直接解释旧 Linux 的性能差距。

本轮是审计：**没有修改生产代码、训练配置、奖励权重、γ、历史来源清单或用户实验数据。** 本轮新增内容是报告、源码清单和可复现探针证据。没有开始正式训练，没有使用用户保留的模型；回归中的模型保存、加载及极短训练只使用测试创建的临时对象。

## 检查范围与证据强度

审计开始时的非 test、非输出、非 tmp Python 清单共 **384 文件、69,395 行**：core 44、chapter3_bser 260、scripts 20、tools 23，以及 docs 31、historical_verification 5、文献 1 个辅助脚本。这里真正的 core + Chapter 3 生产文件是 304 个，不能把辅助脚本也称为生产模块。

- 对清单全部进行 AST 解析、编译、内部导入与禁止的 legacy/path 注入模式检查；881 个内部绝对导入记录未发现缺失目标。
- 静态解析 4,882 个可解析的直接函数／类调用，检查必需参数、额外位置参数及未知关键字；没有新增候选。不包含任意动态分派、所有实例方法或 `**kwargs` 内容，因此不等于所有接口均获证明。
- 57 个配置 JSON 均可严格解析，没有 NaN/Infinity 常量或 JSON 语法错误。
- 深入复核 MADDPG 网络／优化器／target、回放、B3 boundary context、D2 guidance、物理残差、奖励适配、终止、训练入口、检查点与评估链。测试实现没有被当作生产算法源码。
- 另做独立于原专项测试的新接口反例、真实环境短轨迹和数值 oracle。完整回归范围、结果和失败原因见本目录 `evidence/regression.json`；没有宣称全仓全部测试通过。

`SOURCE_CATALOG.csv` 记录本轮初始文件身份，`evidence/static_summary.json` 记录静态范围。清单不是逐行语义证明，也不能排除所有未来输入、分支、设备和机器上的错误。

## 新复现的接口缺陷：尚未修改

### 1. MADDPG 批次形状检查不足，错误奖励会广播后继续学习

位置：`core/algorithms/maddpg.py:_prepare_sample`（177 行起）、`_compute_target_q`。

对 B2 和 B3 都传入 7 条观测／动作／done，但每个 agent 只给 1 条 reward，importance weight 也只有 1 条。当前 learner 接受样本、返回有限损失并实际改变参数，没有报错；`rew_i.view(-1,1)` 和单元素 weights 被广播到全部 7 行。这是具体的静默接口错误，不能用“网络维度正确”排除。

**当前可达边界：** 正常的 CH3ReplayBuffer / ExecutionReplay 每个字段按同一采样索引产生相同批长，本轮真实转移也满足该约束。没有证据表明旧 Linux trainer 生成过上述畸形批次。将来直接复用 core、替换 replay 或改训练接口时，则可能立即触发。

建议收口：在任何 optimizer 写入之前，按声明的 agent 数和 obs/action 维度，核对所有当前／后继观测、动作、奖励、done、weights 的批长与允许形状；拒绝空批次与会发生广播的单元素奖励。保持合法八项样本及 28D/3D/124D 输入不变。不能简单把 reward 扩展到 B 行来“兼容”。

### 2. 通用回放恢复到更大容量后，可能采样到从未写入的槽位

位置：`core/replay/ch3_buffer.py:load_state_dict`（173 行起）。

构造容量 3 的已环绕缓存，写入编号 0—4 的转移，将其状态恢复到容量 5 的缓存，再写入编号 5。当前代码保留旧的 next_idx，又把 filled_i 增加为 4，导致有效区包含一个未写入槽位。为明确区分未写入内容，探针预先将目标槽位置为 -9876；采样全部 4 行的编号为 `[-9876, 3, 4, 5]`，确认读到了该槽位，并覆盖了一个本该保留的旧样本。

**当前可达边界：** B2/B3 和 repair v1 都是 fresh-only，不恢复 replay。历史 Phase 1C 恢复入口另有 config hash 检查，会阻止正常配置变更；没有据此推断它的历史运行已被污染。缺陷位于可独立调用的共享 replay 恢复接口。

建议收口：不支持容量迁移时，应在修改任何缓存之前明确拒绝 max_steps 不匹配，同时检查 filled_i / next_idx 的环形不变量及源字段形状；若未来支持扩容，应显式按时间顺序重排并重建写指针，不能只复制旧 next_idx。

### 3. 修复版 replay 仍公开继承旧 push，可绕过执行门控元数据

位置：`chapter3_bser/experiments/d2_repair_v1/replay.py:ExecutionReplay`（14 行起）。

新训练器使用 `push_transition()` 存储动作及当前／后继执行门控，但类仍继承可直接调用的 `CH3ReplayBuffer.push()`。调用旧 push 会增加缓存长度而不写 gates；sample 随后读出对应未初始化的 bool 槽位。探针构造合法的全 false 残留门控，旧 push 写入普通搜索转移后，actor 被错误跳过。

**当前可达边界：** 本轮检查的新 Trainer 始终调用 `push_transition()`，没有走这个旁路。这是新接口尚未封闭，后续改 RL 模块时尤其容易误用；不能把它说成当前 trainer 已经随机跳过 actor。

建议收口：显式拒绝没有门控元数据的 public push，或让它必需提供完整门控；保留 `push_transition` 内部显式调用父实现的路径，并在 sample 时拒绝没有完整元数据的槽位。

## 其他已核实的使用风险

- **CPU 状态字典有共享存储。** `core/algorithms/maddpg.py:_recursive_to_cpu` 使用 `detach().cpu()`，CPU 张量不复制。取得 training_state_dict 后继续更新模型，之前返回的字典内容会随之变化。探针同时核实 data_ptr 相等及字典 hash 改变。这与 PyTorch 常见的浅 state_dict 语义相符，不能不加区分地判为即时同步 torch.save 路径有错；但它不是可保留的独立快照。未来需要回滚、分支比较或异步保存时，应复制张量并深拷贝初始化元数据。
- **core rollout 数量检查不完整。** core `step()` 对只有 3 个 agent 的观测会因为 zip 返回 3 个动作。实际 BaselineMissionRuntime 检查最终 4×3 动作形状并报错，所以正常完整任务不会静默吞掉一个 agent；直接复用 core 时仍应在入口核对数量、批长和有限值。
- **critic-only 后的诊断标志陈旧。** 先调用 update、再 update_critic_only，last_actor_updated 仍可能为 True。没有观察到 actor 被 critic-only 更新；这是标志未清零。repair Trainer 当前不调用 critic-only，因此不能用这个标志问题解释其实际 actor 更新次数。

这些风险单列，避免把没有证明触发的边界现象混成旧实验的因果结论。

## 修复版真实学习链的新验证

使用正式配置的 actor 隐藏宽度 128、critic 隐藏宽度 256，保持 γ=0.95、τ=0.005、原学习率、原复合奖励和 28D/3D/124D。仅将任务时限缩为 16 步，使用全新策略，不评价学习性能。

B2/B3 各运行普通目标和近目标场景，共 4 条完整的 16 步轨迹、64 个真实转移；近目标轨迹第 1 步发现，随后搜索者门控变为 0。

全部核对通过：

1. replay 当前观测与本步决策输入一致；非终止 next_obs 与下一转移的当前观测逐项一致。
2. 存储的 normalized residual 等于施加门控后的命令；禁用行的残差为零，并核对实际环境 `_last_residual_acc`。
3. 四个 learner reward 与最终 reward accounting 完全相同，都是同一个 team mean，没有再次平均或错误乘四。
4. terminal 的 next gates 全零，done 四者一致；这些短轨迹到达有限任务时限并按 timeout 终止。
5. 从各模型的 32 条真实样本重算 next action 投影、联合输入、twin-min、终止 mask、γ 和 ±10 截断，Bellman target 逐值一致。
6. 各模型各做一轮四智能体更新后，参数有限；将每个 target 参数与 `(1-τ)*旧 target + τ*online` 对照通过。
7. 纯 rollout 不改变策略／优化器状态。没有将这些步骤冒充训练收敛、正式性能试验或因果消融。

本轮真实普通轨迹未触发 SafetyBridge 的 UNVERIFIED_BRAKING 分支；近目标轨迹覆盖了冻结门控。保持／不可达门控的独立机制覆盖来自专项回归，不把它扩大描述为未知地图下的无碰撞证明。

## 旧入口与新修复版本的区别

`chapter3_bser/experiments/d2_suite_v1/run.py` 的 D2_B2 / D2_B3 仍导入历史 DirectMCTrainer / DirectBoundaryTrainer，评估也仍构建 BaselineMissionRuntime。旧 shell 脚本没有自动改成 repair v1。

这保留了旧实验身份，但意味着后续仅更新源码、照抄旧训练指令，**不会自动启用三个独立行为修复**：航点到达奖励时序、冻结 actor 屏蔽、保持／不可达时残差门控。后续修复实验必须明确使用 `d2_repair_v1` 配置和入口、新输出目录，并重新产生检查点；不能沿用旧 checkpoint 冒充修复结果。

本轮不改旧 Suite 的含义，也不增加正式实验执行。手动入口及修复范围见相邻 `d2_repair_v1/IMPLEMENTATION.md`。现在还不具备已验收的完整新长实验方案。

## 回归与尚缺的证据

最新本地回归完整执行了所选 **29 个模块**，unittest 报告 **104 项**，记录 **98 项成功、1 项断言失败、8 条错误**，没有 skip。错误含 subTest 和类初始化记录，数字不能简单相加。17 项实现修复专项全部通过；历史来源／冻结相关检查也通过。新真实链路探针和接口反例另列，不与这 104 项重复累加。

逐条检查全部失败／错误后，均含 `SafeSearch source inventory mismatch`：包括原 B2/B3 检查点评估入口、原 check-only／来源入口，以及两个旧基线的类初始化。断言失败原本期待 HGR 身份错误，却先被来源门禁拦截。没有把门禁失败改称“测试通过”。完整失败清单、成功项目及运行环境另存 `evidence/regression.json`，完整 traceback 在其记录的本地 tmp 文件中。此次复核耗时约 8 分钟，是本地结果，不是 CI。

普通沙箱先前尝试为 99 项、81 项成功、26 条错误，含临时文件／子进程权限问题；随后在允许这些本地有界检查的环境完整重跑，得到上面的 104 项结果。两次记录分开保留，没有跳过、豁免或通过修改历史哈希来消除失败。生产源码本轮未发生变化，见 `evidence/source_stability.json`。

即使上述底层接口补齐，仍然不能由代码检查保证 RL 优于 D2：

- 原复合奖励与 Found／发现前碰撞／失败惩罚发现时间没有已证明的排序等价性。旧 Linux 逐场景记录未在本机，真实排序尚未分析。
- γ=0.95 对远期发现的衰减仍在；本轮没有整体修改 γ 或重设其他奖励权重。
- critic 输入为联合局部观测，仍缺地图、剩余时间、上一加速度和控制器内部状态；这是保留契约下的学习风险。
- 保持时零残差没有消除惯性、流场和先验动作造成的碰撞可能。
- 原 MADDPG 实现的 twin critics、Huber、目标截断和队友当前策略动作属于现有变体；没有证据支持将它们直接作为错误删除。
- 本地仍缺历史 E0 `golden_trace_manifest.json`，不能声明历史黄金回放通过；既有前后对照不能替代它。
- 旧完整源码封存入口仍有 source inventory mismatch，不能把当前目录描述为全绿或正式发布就绪。

应先把已复现的三个接口边界收口、以专项反例验证拒绝错误输入且合法轨迹不变，再解决新实验入口的完整来源封存与缺失的黄金验证问题。之后的有界诊断与单因素对照才能回答“底层修复改善了多少”，长训练和正式性能比较仍由用户手动决定。
