# 3090 checkpoint 与 Phase2 策略来源核验

日期：2026-09-26。仓库：`E:\gym\code\WORKSPACE\AUV-Thesis`，分支 main。
HEAD：`16ae663fdc3d9d702f43e0f5de4bf0dfafaca7c3`；工作区已有配置修改，未 commit/push。
输入根目录：`E:\gym\code\WORKSPACE\3090结果\collision_terminal`。

## 结论

**发现真实训练权重，无需为了取得权重重新训练。已导出合规策略对，状态 POLICY_PAIR_READY。**
`frozen_phase2_source.pt` 尚未生成，真实 Phase2 未执行。策略差异极小，不能把输入就绪解读为有足够统计信号。

| 文件组 | 核验结果 | 能否给当前 Phase2 提供参数 |
|---|---|---|
| HGR-train100_seed2729_v1，7 个周期 checkpoint | hgr.complete_cycle.v1；HandoffPolicy；100 条主轨迹、7 次前/后级更新、69,976 环境步 | 可以，经明确的参数导出；不能直接当作 frozen source |
| B2_direct_mc_train100_seed2729_v1，ep100 | ch3.baseline.maddpg.v1；model_training_state | 不匹配当前策略架构 |
| B3_direct_boundary_train100_seed2729_v1，ep100 | ch3.baseline.maddpg.v1；boundary MADDPG | 不匹配当前策略架构 |
| B2/B3 final 与 smoke final | 文件已发现；未逐一读取其张量 | 不用作本次来源 |

真实权重判断来自 checkpoint 的张量、完成周期记录、累计成本、配置与来源清单交叉核对，
不是仅根据文件名判断。HGR 是 28D/3D、hidden_dim=128、expert_hidden_dim=128，M20/H=400。
ep48、ep64、ep96、ep100 的状态可严格装载为当前网络；用于导出的张量有限，权重摘要与保存记录一致。
CPU weights_only 读取不会恢复训练；本次没有恢复 Trainer、优化器、环境或 checkpoint RNG。

## 相邻 HGR 后级策略检查

固定输入为 `float32 linspace(-1,1,896).reshape(32,28)`，属于合成输入检查，不是假称真实轨迹。

| 相邻 episode | 后级权重差 L2 | 固定输入下 Gaussian 均值最大差 |
|---|---:|---:|
| 16 → 32 | 9.65e-9 | 7.45e-9 |
| 32 → 48 | 9.14e-9 | 6.05e-9 |
| 48 → 64 | 1.06e-6 | 3.36e-7 |
| 64 → 80 | 0 | 0 |
| 80 → 96 | 0 | 0 |
| 96 → 100 | 4.63e-14 | 0 |

ep100 是真实训练文件，但“最新”不代表适合观察非零后级修正。其最后一次更新哈希不同，
固定输入下 float32 输出却完全相同；这不能证明所有输入都相同，也不能据哈希不同声称有实质行为变化。

本次明确选择 **最新的、在上述固定检查中有非零分布差异的相邻周期 cycle4**，使用 ep48→ep64。
没有运行效率实验，没有按成功率、交接结果或梯度 MSE 选择策略。即使如此，变化仍然很小；
本次仅建立真实可审计输入，不保证正式实验的统计功效。

## 为什么可以恢复相同 θ

被审计的 Trainer 每个周期先保存 old=(θ,φ0)，再只更新 φ 得到 φ1；θ 更新发生在估计之后。
保存 checkpoint 时，φ1 保持不变，但 θ 已变为 θ_after。因此从两个最终策略整体直接配对会违反 θ 相同要求。

cycle4 的记录与实际张量逐项吻合：

| 用途 | 来源 | 权重 SHA256 |
|---|---|---|
| 共享 θ | ep48/cycle3 theta_minus | b7429dfe9ef364e7bec93486da9bf73e3f00cffa04962a205b2b0346bc73213f |
| φ0 | ep48/cycle3 phi | 8f2a548c9e44e7b1dc09106a0b98cd0a384483e0d7cb3ae881c0603c7def0dd6 |
| φ1 | ep64/cycle4 phi | f93d215c4e1264d8376d2c4ceecce09329bf87260e7754927adfb528922d5521 |

这恢复的是有记录支持的 cycle4 **前级更新之前**的真实策略对，没有随机扰动、补权重或通过训练制造差异。
不是任意把两个不同 θ 的策略修成一样。完整模块模式、属性及张量继续接受现有 behavior identity 检查。

## 来源与兼容性边界

原始 checkpoint 源码摘要：`767b1232abb19e3d4a1d9d2d4c173d0cc3417b0b17a893e23514e78d3db2608c`。
完整清单与仓库 Git `da4a8a64cba26b2adf7d248fd8486e517422628c` 的生产 Python 文件逐字节一致。
本次当前生产源码摘要：`100f369e2c1958a065905e379bd13ca63cdc7f76000352f2ee8fa267ff9aabca`。

历史来源与当前来源不同。已核对实际内容变化为 Phase1 的 runtime/train/estimator/policy 四个文件，
另有换行差异及新增 Phase1/Phase2 模块。policy 修改增加显式 CRN 噪声入口，未改变网络结构或 Gaussian 密度计算。
运行合同差异仅为 checkpoint_schema 和 Phase1 bypass 授权元数据。
已有 Phase1 legacy fixture 的策略输出、密度、梯度和合成周期兼容性测试本次全部通过。
这些检查不等于验证整个历史运行与当前 runtime 完全等价。

新增脚本只对已固定的两个文件 SHA256、历史源码和当前源码进行一次明确参数导入。
任何来源变化都拒绝，原有训练恢复和 source gate 不变。新文件描述当前策略对象，
同时保留原 checkpoint 完整来源清单及输入 SHA256；没有改写历史来源身份。

## 已生成文件与验证

- `outputs/chapter3/hgr_phase2/policy_pair_source.pt`
  - schema：hgr.phase2.policy_pair.v1。
  - 文件 SHA256：`fee275f981417e05cbc9ad4b98c677b9f27390d9052413f17b028f4402b49542`。
  - 来源准备成本：44,610 环境步，完整计入截至 ep64 的历史成本。
- `outputs/chapter3/hgr_phase2/policy_pair_source.audit.json`
  - 输入路径、文件 SHA、原始/当前完整源码清单、周期重建说明、差异诊断和限制。
  - 审计文件 SHA 被绑定在 policy-only 文件的 origin 中。

导出命令：`python -B -m scripts.export_hgr_phase2_policy_pair`（AUV 环境）。
现有 `load_policy_pair` 严格读取通过：完整 θ 行为身份相同，φ 行为身份不同，当前源码和运行合同匹配。
导出新增环境步 = 0，参数更新 = 0；没有生成场景或 frozen source，没有运行 Phase2。
所有输入和既有输出保留原样，新权重文件受 .gitignore 排除。

最新本地单元验证：**43 passed in 25.79s**，不是 CI。

```powershell
python -B -m pytest tests/test_hgr_phase2_policy_export.py tests/test_hgr_phase2_source_builder.py tests/test_hgr_phase2_gradient_efficiency.py tests/test_hgr_phase1_legacy_fixture.py -q -p no:cacheprovider
```

新增导出测试涵盖周期证据不匹配、错误张量/配置、相同 φ、未审核来源、文件 SHA 不符、
真实导出 schema 往返、相同 θ 的确切来源、RNG 隔离、输入不变和禁止覆盖。

下一步最小动作是运行既有 source builder，生成 10 个预声明场景及 frozen source。
正式梯度效率实验需单独发起；应把上述策略差异限制写入实验解释，不能从本次核验推出 HGR 收益。
