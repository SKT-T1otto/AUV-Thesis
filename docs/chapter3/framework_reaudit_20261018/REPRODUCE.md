# 本地复现说明

在仓库根目录，以 AUV Python 环境运行。下面都是本轮审计探针，不是正式训练命令；没有加载用户保留的模型。核心生产源码的身份见 SOURCE_CATALOG.csv。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -m docs.chapter3.framework_reaudit_20261018.reproduce.probe_interfaces
python -m docs.chapter3.framework_reaudit_20261018.reproduce.probe_live_chain
```

第一项**断言当前缺陷能够复现**，不是断言实现已经正确；未来修复后，对应反例断言应失败，并应改写为拒绝畸形输入的回归测试。第二项使用全新正式宽度网络、4 条 16 步临时任务和各一次四 agent 更新，核对真实转移与公式，不评价性能。

完整相关模块检查：

```powershell
python -m docs.chapter3.framework_reaudit_20261018.reproduce.run_checks
```

此项会执行既有测试中的临时模型保存／加载和有界短训练，预计数分钟，既有来源门禁失败会被原样记录。它不是全仓测试，不是 1000 回合实验。受限 Windows 沙箱可能因临时文件／子进程权限失败；本轮分开保留了受限尝试与权限复核结果，没有跳过测试。

新输出写在 `tmp/framework_reaudit_20261018`。本报告的 evidence 是本轮固定记录；复现脚本不重新发布或改写 evidence，也不接触 outputs。

可选静态清单：

```powershell
python -m docs.chapter3.framework_reaudit_20261018.reproduce.audit_static
```

本轮记录的 384 个 Python 文件是**本轮报告和复现脚本生成之前**的库存。复跑时 docs 下新增的这些审计脚本也会进入静态清单，文件数增加是预期的；core + chapter3_bser 的初始生产文件数是 304。
