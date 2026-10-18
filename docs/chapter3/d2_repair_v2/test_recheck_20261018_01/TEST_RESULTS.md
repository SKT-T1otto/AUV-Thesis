# 修复真实性测试：本轮实际结果

本轮结论：**`MECHANISM_VERIFIED`**。65 个 unittest 方法全部通过，failure/error/skip 均为 0。Python 入口报告退出码为 **2**：机制测试通过，额外的旧来源/E0 验收门禁仍阻塞；没有把这类拒绝记录为通过。正式训练资格和性能仍未建立。

## 版本与范围

分支 `codex/framework-audit-20261018-164056`，HEAD `6538c3e6a67e96d56940ab09fcff15cd9dc7a7d8`。只增加/调整测试、测试入口及文档；生产代码和原配置没有进一步变化。完整 v2 来源身份在运行前、每阶段前、运行后均通过两份独立校验；测试源文件运行中没有漂移。

Windows / Python 3.10.20 / Torch 2.11.0+cu126 / NumPy 2.2.6；CPU、单线程。属于最新记录的本地验证，不是 CI。

## 实际执行

```powershell
python -m tools.run_d2_repair_acceptance --execute --output tmp/d2_repair_acceptance/full_recheck_20261018_01
```

本次工具调用的 PowerShell 外壳未显式转发 `$LASTEXITCODE`，外层 exec 工具记录退出码 1；Python 的 results.json 记录分类码 2，各测试子进程均为 0。三者分开保留在 [execution_wrapper.json](evidence/execution_wrapper.json)，不能把外壳状态当成新增测试失败，也不能隐去它。手动重跑时直接查看 `$LASTEXITCODE` 和 results.json。

| 阶段 | testsRun | failure | error | skip | 子进程 exit | 秒 |
|---|---:|---:|---:|---:|---:|---:|
| contracts | 26 | 0 | 0 | 0 | 0 | 74.054 |
| negative_controls | 13 | 0 | 0 | 0 | 0 | 26.517 |
| retained_fixes | 6 | 0 | 0 | 0 | 0 | 2.552 |
| historical_records | 12 | 0 | 0 | 0 | 0 | 0.656 |
| runner_guards | 6 | 0 | 0 | 0 | 0 | 0.354 |
| runtime_chain | 1 | 0 | 0 | 0 | 0 | 187.829 |
| scheduler_checkpoint | 1 | 0 | 0 | 0 | 0 | 218.045 |

总耗时 536.674 秒。完整命令、解释器、平台、线程、环境变量、每个测试名称与 traceback 见各阶段 JSON；stdout/stderr 保留为同名 log。子进程退出码 0 不足以判定通过，父进程还检查测试数量、禁止 skip、必要产物、事件覆盖和实际预算。

## 负对照的实际结果

13 类缺陷共 17 个具体旧/新判据比较通过。每个记录都要求旧错误被同一判据识别、当前修复满足要求；不存在用“测试未触发”当作通过。详见 [observed_negative_controls.json](evidence/observed_negative_controls.json)。

涵盖批长广播、容量恢复、普通 push 旁路、CPU snapshot 别名、core.step 数量截断、critic-only 标志、航点时序、冻结 Actor、保持残差、terminal NaN、可变 sample 缓存、本地 DDPG 输入以及非有限梯度写入。B2/B3 相关学习行为分别测试；本地 DDPG 缺陷不用于解释 B2/B3 性能。

冻结副本在执行前检查完整 SHA-256。旧实现仅存在于测试进程内，没有修改生产函数，也没有将旧模型冒充新版本。

## 真实环境与学习更新

| 算法 | 修复版 | 环境步 | 完整更新轮数 | Actor optimizer 实际步数 | 冻结搜索 Actor 参数差 |
|---|---|---:|---:|---|---|
| b2 | False | 147 | 4 | [3, 3, 3, 3] | [0.0010013515129685402, 0.0010013328865170479, 0.0010013515129685402] |
| b3 | False | 147 | 4 | [3, 3, 3, 3] | [0.001001346274279058, 0.001001314609311521, 0.0010013356804847717] |
| b2 | True | 147 | 4 | [2, 2, 2, 3] | [0.0, 0.0, 0.0] |
| b3 | True | 147 | 4 | [2, 2, 2, 3] | [0.0, 0.0, 0.0] |

每对象 7 场景、147 步；每场景≤32 步，每模型≤4 轮完整更新。覆盖搜索未发现、首次发现、等待交接、Executor 知情、成功、搜索碰撞、超时和冻结。逐步核对 raw Actor、OU 噪声、clip、gate、环境入参、物理加速度、奖励分项、团队奖励、done、next observation 和 Replay 索引对应关系。

另外两个 v2 Trainer 模型分别执行 8×32=256 步、2 轮原调度更新，Actor 步数 [2,2,2,2]，twin Critic 总步数 16；临时 checkpoint 保存、加载、独立评估和错误来源拒绝全部通过。没有加载用户已有模型。临时模型文件随临时目录清理，逐步原始轨迹只在本轮 tmp 目录保留。

[observed_runtime_chain.json](evidence/observed_runtime_chain.json) 和 [observed_scheduler_checkpoint.json](evidence/observed_scheduler_checkpoint.json) 是实际运行产物，统一入口进一步核对了它们的内容。完整本地原始证据位于 `E:/gym/code/WORKSPACE/AUV-Thesis/tmp/d2_repair_acceptance/full_recheck_20261018_01`，hash 见 [results.json](evidence/results.json)。

## 未通过的额外验收门禁

- `legacy_baseline_source`：未通过；旧 SafeSearch 完整来源清单不匹配；全部差异见 JSON
- `legacy_repair_v1_source`：未通过；当前 core SHA 不属于旧 repair v1 身份，精确门禁拒绝
- `historical_e0`：未通过；Historical golden data missing

上述来源拒绝与完整 traceback 原样记录于 results.json。未启动被拒绝的旧训练/检查点操作，没有放宽 hash、删除来源文件或改写历史 manifest。历史 E0 没有重建替代数据。结果为 `MECHANISM_VERIFIED`，不提升为 `READY_FOR_MANUAL_PILOT`，也不修改上一轮 acceptance.json。

## 初次测试构造失败

新增负对照首次运行有 1 条 error：为原始 core.step 构造反例时误用了没有绑定联合 boundary context 的 B3 测试副本。已把这项负对照限定到其真实作用范围——原始 core/B2 接口；B3 网络、encoder、context 与动作对齐另由专门测试覆盖。修正的是测试构造，没有改变生产算法；最终完整运行通过。原始错误日志及说明保留于 [initial_failure.json](evidence/initial_failure.json)，不删除失败历史。

## 保留与局限

上一轮 31 个封存证据文件与 4 份原始轨迹 hash 均保持不变，详见 [preservation.json](evidence/preservation.json)。没有动用户 outputs、启动长训练、提交或推送。

测试支持“已列明修复在本轮覆盖条件下有效”，不能证明没有任何其他错误、RL 必然提升、复合奖励排序与论文指标一致或未知障碍下绝对安全。旧 Linux 逐场景数据仍未提供，不能完成其性能下降的因果归因。上轮更广的相关回归失败记录继续保留，没有将本轮 65 项选择性结果称作全仓库全绿。

设计、判据、预算和重跑方式见 [TEST_EXPERIMENT.md](../TEST_EXPERIMENT.md)。
