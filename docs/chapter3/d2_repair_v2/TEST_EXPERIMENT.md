# 修复真实性复查：可重复的有界测试实验

本轮新增测试实验只回答：**已列出的实现缺陷是否被当前修复版阻止，以及测试能否识别旧错误**。它不测收敛、论文排名或长期性能，不改生产代码、奖励、gamma 或网络，也不启动正式训练。

## 文件与执行入口

- `tests/d2_repair_v2_experiment.json`：机器可读的阶段、测试清单、数量和预算。
- `tests/test_d2_repair_v2_negative_controls.py`：13 类缺陷的旧实现负对照。
- `tests/test_d2_repair_acceptance_runner.py`：测试入口自身的来源、错误传播、证据保护和默认不执行保护。
- `tests/fixtures/ch3_buffer_before_repair_v2.py.txt`：从固定父提交提取的 Replay 冻结副本，运行前核对 SHA-256。
- `tools/run_d2_repair_acceptance.py`：统一入口；记录命令、环境、源码/测试身份、逐项结果、完整 traceback、退出码及证据 hash。

复用上一轮已有的接口、数值、消融和真实环境测试，避免为同一断言另写重复测试。旧 raw core、旧 Replay 和更早的 pre-v1 learner 只在测试对象中执行，不进入正式训练路径。

四个现有测试辅助/输出文件改用 `evidence_dir()`。统一入口为子进程设置新的证据目录；直接运行单项测试时则创建新的系统临时目录。不会再写死上一轮 `tmp/d2_repair_v2_20261018`，上一轮报告和证据不改写。

## 负对照判据

这里的“通过”包含两部分：当前修复版满足判据，冻结旧实现或显式关闭相应修复后**不满足同一判据**。若旧错误仍能通过检测，负对照测试本身失败。

| 缺陷 | 输入与独立判据 | 旧对照 |
|---|---|---|
| 批长广播 | 7 行 obs/action/done 配 1 行 reward/weight，必须在写参数前拒绝；B2/B3 分别检查 | 固定父提交的 learner 副本接受 |
| Replay 跨容量恢复 | 3 槽写 5 次，恢复到 5 槽必须拒绝并保持目标不变 | 冻结旧 loader 继续采样出未写槽位 sentinel |
| execution push 旁路 | 普通 push 必须拒绝，长度不变 | 原 v1 Replay 接受无 gate 转移 |
| CPU snapshot 别名 | 保存 snapshot 后改变在线参数，snapshot hash 必须不变；B2/B3 | 固定父 learner snapshot 随在线参数变化 |
| core.step 数量截断 | 原始 core 接口必须拒绝 3 个 agent | 旧 core 返回 3 个动作；不与 B3 的独立 boundary guard 混淆 |
| critic-only 陈旧标志 | Actor 更新后再 Critic-only，last_actor_updated 必须为 false；B2/B3 | 固定父 learner 保留 true |
| 航点到达时序 | 真实环境到达步 coverage 分项应比 legacy 对照恰好多一次原 bonus | 显式 legacy 时序漏发 |
| 冻结 Actor 更新 | 先建立 Adam 动量，再用真实冻结语义样本：Actor/Adam 不动，Critic 继续学习，team done=false；B2/B3 | 旧 B2/B3 Actor 继续变化 |
| 保持/不可达残差 | 真实 guidance 要求保持/不可达时，实际归一化残差严格为零 | 关闭门控后残差非零 |
| terminal NaN | terminal next-Q=NaN/Inf，最终 target 必须为有限终止奖励 -2 | 更早 pre-v1 的 NaN×0 公式产生 NaN |
| 可变 batch 缓存 | 同一 sample 对象修改 obs 后，拼接 Critic 输入必须反映当前内容 | 更早 pre-v1 按对象身份缓存返回旧输入 |
| 本地 DDPG 输入 | 28D obs+3D action 的本地 31D Critic 更新必须合法 | 更早 pre-v1 用联合 124D 输入导致维度异常；正常 B2/B3 不使用该分支 |
| 非有限梯度写入 | 第二 Critic 梯度为 NaN 时，两个 Critic 的 optimizer 写入前报错，参数均保持 | 更早 pre-v1 继续写入并污染第二 Critic |

对照不是对旧 Linux 性能的因果证明。部分 API 缺陷在正常旧 Trainer 中未证实可达；表中的 pre-v1 缺陷在上轮固定父版本前已经修复，本轮检验其保护是否仍有效。

## 分层实验

| 阶段 | 计划测试数 | 检查内容 |
|---|---:|---|
| contracts | 26 | batch、Replay、数值、Actor mask、航点、来源与单因素消融 |
| negative_controls | 13 | 上表旧缺陷能否被同一判据检测 |
| retained_fixes | 6 | 原有本地 DDPG、terminal NaN、梯度原子拒绝、缓存、奖励排序诊断与配置约束 |
| historical_records | 12 | 27 条历史记录、精确后继演进、冻结文件、输入维度和隔离 |
| runner_guards | 6 | 默认仅计划、旧目录保护、来源失败禁止模型阶段、子测试失败和测试身份漂移不能误报通过 |
| runtime_chain | 1 个综合测试 | 旧 B2/B3 与 v2 B2/B3 各 7 场景，发现前后、交接、冻结、成功、碰撞、超时；动作/奖励/Replay/更新对齐 |
| scheduler_checkpoint | 1 个综合测试 | v2 B2/B3 各按原 warmup/batch/频率完成 256 步、2 轮更新，临时保存/重载与独立评估 |

完整计划共 65 个 unittest 方法，部分包含 B2/B3 子用例。真实环境综合测试每场景≤32 步，每模型≤4 轮完整更新；Trainer 临时模型≤8 回合、总开始预算≤256。综合测试中的预算和值还会由父进程读取实际产物再次核对，缺失产物不能仅凭子进程 exit 0 判为通过。

运行前核对当前及独立验证器的完整来源身份；每阶段前再次核对；全部结束后核对源码和测试文件 hash 均未漂移。来源拒绝时不进入后续模型阶段。旧来源门禁和 E0 文件状态单列，不能将其错误吞掉或计作通过。

## 手动重跑

在仓库根目录的 Windows PowerShell 中，只打印计划：

```powershell
conda activate AUV
Set-Location E:\gym\code\WORKSPACE\AUV-Thesis
python -m tools.run_d2_repair_acceptance
```

执行完整有界测试实验：

```powershell
python -m tools.run_d2_repair_acceptance --execute
$testExit = $LASTEXITCODE
Write-Host "测试退出码：$testExit"
```

只执行较快阶段，明确不运行真实训练调度和综合轨迹：

```powershell
python -m tools.run_d2_repair_acceptance --execute --quick
```

默认每次使用新的 `tmp/d2_repair_acceptance/时间_随机ID`。也可以通过 `--output tmp/d2_repair_acceptance/新的名称` 指定**尚不存在**的目录。已有目录及 `outputs/` 均拒绝写入。正式模型不会被自动选择或加载。

## 退出码和解释

- `0`：只打印计划；或执行模式下所有所请求测试及额外验收门禁均通过。是否执行必须看 `executed`。
- `1`：源码漂移、测试 failure/error/skip、数量不符、超时或必要证据缺失；查看 results.json 及对应阶段日志。
- `2`：已运行机制检查通过，但完整验收仍受阻；quick 模式遗漏阶段也明确列为 NOT RUN。

`MECHANISM_VERIFIED` 只在完整 65 项及其必要产物通过后使用。quick 模式至多记为 `PARTIAL_CHECKS_PASSED`，不能冒充完整验收。本工具不运行历史 E0 黄金回放，只检查文件状态并保留其未验证事实；不写旧 acceptance.json、不授予正式训练资格，所有结果始终 `formal_training_allowed=false`、`performance_passed=false`。

奖励与任务指标排序、28D 的状态充分性、未知障碍的完整动力学安全、长期稳定性和旧 Linux 性能归因，仍不能由这些测试证明。

## 本轮实际结果

完整计划已实际运行：65 项全部通过，无 failure/error/skip；13 类负对照共 17 个比较通过。真实环境 588 步与两个原调度模型合计 512 步均通过预算和语义检查。Python 入口报告退出码 2，表示机制验证通过，但旧来源/E0 门禁仍阻塞；结论为 `MECHANISM_VERIFIED`。

逐项结果、初次测试构造失败、完整日志和退出码见 [本轮测试结果](test_recheck_20261018_01/TEST_RESULTS.md)。上一轮 IMPLEMENTATION_AUDIT.md、VERIFICATION.md 和 evidence/acceptance.json 保持原记录。
