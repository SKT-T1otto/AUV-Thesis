# 本地验证记录

这些是 2026-09-12 Windows 本地自动回归，不是 CI 或性能实验。环境为 Conda `AUV`、Python 3.10、Torch `2.11.0+cu126`，本轮测试使用 CPU，OMP/MKL 线程数为 1。命令中的实际解释器为 `D:/anaconda/anaconda/envs/AUV/python.exe`。

本地与通过 GitHub 连接器核对的远程默认 `main` 均为 `93a9c8fb53857051390265e3035061bf05402e25`。初始工作区干净；交付为未提交的代码与小型验证文档。没有正式 1000ep 训练、100 场配对评价、真实旧 checkpoint 的性能评价或 commit/push。

## 已完成的定向验证

| 验证 | 运行数 | 通过 | 失败 / 错误 | 日志 |
|---|---:|---:|---:|---|
| 严格协议专项 + 两套 provenance | 28 | 28 | 0 / 0 | `verification/acceptance_final.log` |
| 旧 BEDS-off worker 与 HEAD 逐项相等、checkpoint evaluator、指标、配对入口 | 23 | 23 | 0 / 0 | `verification/legacy_final.log` |
| 最后修改：公共 facade provenance、两套 40 文件冻结、负奖励配置、真实报告缓存及训练更新 | 13 | 13 | 0 / 0 | `verification/last_changes.log` |
| 真实 trainer 两个单步热启动回合、完整同协议恢复、最终冻结路径校验 | 2 | 2 | 0 / 0 | `verification/warmstart_resume_final.log` |
| 全部最终 provenance、冻结路径及 PRRAC 导入/网络/replay 隔离入口 | 12 | 12 | 0 / 0 | `verification/all_provenance_final.log` |
| 新增失败 JSON、不完整 CSV 和图表的防覆盖保护 | 1 | 1 | 0 / 0 | `verification/output_protection_final.log` |
| 最终输出保护与旧协议必需产物/缓存兼容 | 2 | 2 | 0 / 0 | `verification/output_compatibility_final.log` |
| 搜索价值原生 ON / shadow 无干扰、串行/并行一致性 | 2 | 2 | 0 / 0 | `verification/search_value_runtime_final.log` |
| 分支原始截止步、关闭搜索价值引导后的完整等价性 | 2 | 2 | 0 / 0 | `verification/late_legacy_rechecks.log` |
| 两种模式的原生残差追踪完整字典等价 | 1（2 个模式） | 1 | 0 / 0 | `verification/residual_trace_final.log` |
| 新报告真实 worker 单回合 + 缓存恢复 | 1 | 1 | 0 / 0 | `verification/report_integration_01.log` |
| 旧 metadata / 配对 / checkpoint / 更新接口兼容 | 19 | 19 | 0 / 0 | `verification/compatibility_01.log` |
| 单独两套 27 条 provenance 检查 | 6 | 6 | 0 / 0 | `verification/provenance_02.log` |
| 原 HEAD 快照上的事件与 Linux 资产测试 | 9 | 6 | 3 / 0 | `verification/immutable_head_baseline.log` |
| 当前事件、Linux 资产及训练更新测试 | 11 | 8 | 3 / 0 | `verification/runtime_final.log` |
| 历史证据及最终 v2 冻结路径检查 | 6 | 3 | 0 / 3 | `verification/historical_evidence_checks.log` |

上述运行之间有重复测试，不能相加宣称独立测试数量。专项物理夹具使用真实环境和 wrapper，受控设置障碍触发碰撞；collector 与 evaluator 测试执行真实 worker。报告和训练集成测试只把进程池替换为同步执行器，仍调用实际生产函数，没有用虚假的完成行替代环境输出。

## 全量首轮

`regression_02/summary.json` 与 `regression_02/unittest.log` 记录完整首轮：**571 个测试方法，553 个通过，15 条断言失败、4 条错误，0 跳过**。其中残差追踪的一个测试在两个模式子场景失败，所以 19 条失败/错误记录对应 18 个测试方法，不能直接用 571 − 19 计算通过方法数。unittest 耗时 6915.865 秒，完整 runner 计时 6918.592 秒。

全量首轮在源码修正期间启动，进程保留了早先载入的模块。它记录了未补齐的精确冻结演进、旧评价被加入额外协议/墙钟字段的兼容问题，以及一个缺少 runtime 字段的测试夹具。源码行号在运行中变化还使一项 `inspect.getsource` 检查读取到相邻函数。原始失败均保留，没有改写全量日志为通过。

最终代码用上表各项复验覆盖这些失败：旧 worker 与 HEAD 的逐项比较、完整 episode 字典等价、分支截止步、缓存夹具，以及所有 provenance/冻结入口。没有把这些定向结果冒充“最终树一次性全量全绿”。以下六个历史问题仍保留：事件枚举断言、两项旧 Linux 资产断言，以及三个历史证据缺失错误；冻结 E0 执行所需的 golden manifest 也仍缺失。

## 命令

每条 Python 命令执行前，本轮 PowerShell 设置如下。Git safe.directory 仅通过当前进程环境传入，没有修改全局 Git 配置。

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:GIT_CONFIG_COUNT='1'; $env:GIT_CONFIG_KEY_0='safe.directory'; $env:GIT_CONFIG_VALUE_0='E:/gym/code/WORKSPACE/AUV-Thesis'; $env:PATH='D:\anaconda\anaconda\envs\AUV;D:\anaconda\anaconda\envs\AUV\Library\bin;'+$env:PATH
```

实际运行的主要命令：

```powershell
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m tools.verify_collision_terminal --suite regression --output-dir docs/collision_terminal/verification/regression_02
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_collision_terminal_protocol tests.test_repository_metadata tests.test_core_source_provenance -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_beds_evaluation.BEDSEvaluationTests.test_off_worker_matches_prechange_HEAD_bit_for_bit tests.test_prrac_checkpoint_evaluator tests.test_prrac_evaluation_metrics tests.test_manual_prrac_pair -q
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_repository_metadata tests.test_core_source_provenance tests.test_phase1a_core_freeze tests.test_phase1a1_original_core_freeze tests.test_collision_terminal_protocol.CollisionEnvironmentTests.test_terminal_reward_configuration_is_validated tests.test_collision_terminal_protocol.CollisionTrainingTests.test_strict_evaluation_writes_complete_report_and_resumes_cache tests.test_prrac_training_smoke -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_collision_terminal_protocol.CollisionTrainingTests.test_real_warmstart_training_summary_and_same_protocol_resume tests.test_phase1c_v2_isolation.Phase1CV2IsolationTests.test_frozen_paths_have_no_worktree_diff_when_git_metadata_is_available -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_phase1c_v2_isolation tests.test_bser_v1_artifacts_frozen tests.test_ch3_e0_equivalence -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_prrac_isolation tests.test_repository_metadata tests.test_core_source_provenance tests.test_phase1a_core_freeze tests.test_phase1a1_original_core_freeze tests.test_phase1c_v2_isolation.Phase1CV2IsolationTests.test_frozen_paths_have_no_worktree_diff_when_git_metadata_is_available -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_search_value_audit_runtime -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_search_value_audit_branches.BranchTests.test_later_boundary_retains_acceptance_and_original_global_cutoff tests.test_search_value_guided_ranking.SearchValueGuidedRankingTests.test_worker_uses_checkpoint_head_without_update_and_disabled_equivalence -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_searcher_residual_trace.TraceMetricTests.test_native_evaluator_trace_no_op_both_modes -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m unittest tests.test_collision_terminal_protocol.CollisionTrainingTests.test_existing_strict_failure_and_plot_artifacts_are_protected tests.test_prrac_checkpoint_evaluator.PRRACCheckpointEvaluatorTests.test_incremental_outputs_and_resume_do_not_repeat_completed_combo -v
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m tools.verify_collision_terminal --suite golden --output-dir docs/collision_terminal/verification/golden_02
```

日志通过 `2>&1 | Tee-Object <新日志路径>` 保存，随后传递 `$LASTEXITCODE`。证据目录不能重用；复跑全量或 golden 时需要指定新的目录名。正式运行命令见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。

## 保留的历史失败与缺失输入

1. `test_bser_event_detection` 断言事件枚举为 8，原 HEAD 实际为 9。未改动事件算法或削弱该断言。
2. `test_linux_runtime_assets` 两项失败，原 HEAD 已有四个 search-value audit 脚本未纳入固定清单，其中 `bundle_search_value_audits.sh` 不含该检查要求的独立 Conda 初始化文本。本轮只将新增碰撞脚本及可配置 Conda 表达式加入检查；原失败保留。
3. `test_bser_v1_artifacts_frozen` 的历史清单存在，但其引用的 `experiments/chapter3/bser_e1_offline/aggregate_by_profile.csv` 不在 HEAD 中。没有伪造或回填历史 CSV。
4. `test_ch3_e0_equivalence` 缺少 `experiments/chapter3/e0_equivalence/equivalence_summary.json`。
5. `test_phase1c_v2_isolation` 缺少 `docs2/phase1c_v2_design/overlay_manifest.json`。
6. 冻结 Phase 0B-2 E0 执行还缺少 `experiments/chapter3/e0_core_migration/golden_trace_manifest.json`。只读 runner 返回非零，`golden_02/summary.json` 记录 0 条轨迹、`passed=false` 和缺失路径；不能报告 60/60 等价通过。

事件与 Linux 的 3 个断言失败已在 `git archive HEAD` 导出的独立临时快照中实际复现。它是 HEAD 快照基线验证，不是 clean-clone 发布验证。历史数据缺失通过真实读取和 `git ls-tree/cat-file` 结果确认。

## 调试过程与最终边界

保留了先前失败日志。`acceptance_02.log` 暴露测试对透明 wrapper 做 deepcopy 的错误；后续改为复制 core facade 并构造真实 wrapper。`acceptance_03.log` 中的报告测试误用了返回不完整模拟行的既有执行器；后续改为同步调用真实 worker。`warmstart_resume_integration.log` 首次恢复测试尝试在已含 episode 2 权重的目录恢复 episode 1，被正确的防覆盖检查拒绝；后续测试恢复到独立目录，保留权重保护。

`static_checks_final.json` 记录变更 Python/JSON 解析、Bash 语法、`git diff --check`、历史 manifest 和已跟踪 outputs 无 diff、checkpoint 被忽略及小型日志可见于 Git。PowerShell 启动器的 Conda `--help`、三个 Python 入口的 `--help` 和 PowerShell AST 解析也已实际通过。Linux 入口还在 Git Bash 中用临时 Conda/Python 桩执行，确认保留已有 CONDA_EXE、初始化指定环境、完整传参和返回约定退出码 37，见 `linux_launcher_stub.json`。没有在用户 Linux 主机启动训练或验证真实 Linux Conda 环境。

本轮没有用真实旧 1000ep checkpoint 验证性能，没有调优 -2 奖励或 256 次 critic 预热，也不建立安全性、收敛或成功率提升结论。28D 仍未包含完整剩余期限状态；保留的个体 shaping 也没有被证明等价于团队成功概率目标。

最终机器可读索引见 `verification/delivery_summary.json`，逐项关联首轮 19 条失败/错误记录、13 条最终复验通过记录与 6 项保留历史问题。
