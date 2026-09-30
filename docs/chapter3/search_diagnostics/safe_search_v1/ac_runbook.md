# V5：A+C 固定 20 场景开发补充实验

本轮用户明确要求验证 A+C。V5 的开关为起点一致性刷新开启、失败恢复关闭、路径安全开启。本轮使用与已完成 D2 完全相同的 B0、20 个原始场景、`12729 + 原索引` 种子及 400 步任务上限；不训练、不加载 checkpoint、不运行其余 80 场景。

2026-09-28 本地实际运行已完成：20/20 完整回合、6,315 个物理步、0 程序失败。Found 8/20，发现前碰撞 5/20，停滞加 Hold 10.97%，有效观察占比 42.52%；本轮独立开发门槛通过。完整解释、配对不确定性和残留调度问题见 [A+C 实验报告](ac_results_20260928/report.md)，机器证据见 [analysis.json](ac_results_20260928/analysis.json)。这不是正式论文验收，旧 V3 的历史选择保持不变。

## 冻结比较

[ac_experiment_plan.json](ac_experiment_plan.json) 在仿真前记录了场景、旧参考结果哈希和分析规则。V3 是主参考，V0/V4 是次参考，V1/V2 仅作附加描述。A+C 的单独开发门槛要求相对 V3：Found 不下降、发现前碰撞不增加、停滞加 Hold 的汇总占比严格下降、有效观察占比不下降，并通过新旧源码可比性检查。这不是原五组筛选的重写，也不构成统计或独立测试验收。

V5 必须收齐 20 个正常任务终止回合；碰撞和成功可早于第 400 步。程序错误、缺失轨迹、预算前缀不能变成未发现超时，也不能被剔除分母。

## 来源与前置检查

旧 D2 的 336 文件清单和来源说明逐字节保留为 `source_evolution_d2_v2.json`、`source_review_d2_v2.md`。新清单有 337 文件，来源 SHA-256 为 `44b35295cecfffeb08501acbc7a5994e62771fb131df0e9afb2737233133f496`，演进清单内部内容摘要为 `a5438f681bad3a1ebf895de53a5708a8c220a6887cf2889cfe48bf0130d20ca8`。

相对旧 D2，只有 `runtime.py` 的 V5 注册、`run_paired.py` 的显式接受名单、来源审阅说明发生改变，新增生产入口 `run_ac_development.py`。已有控制函数、三项修复模块、感知、动力学、奖励和任务契约保持原字节。独立审查见 [ac_source_diff_review.json](ac_source_diff_review.json)。原 D2 默认仍严格要求 V0–V4 的 100 个回合。

封存后安全搜索相关测试本地通过 140 项。测试日志保存在 `runs/safe_search_v1/ac_preparation_20260928_v1/tests_all_safe_search.txt`；其中子进程编排测试使用模拟子进程，不是重复执行真实 D2。历史旧 baseline 来源门禁问题仍按旧报告保留，不宣称全仓或 CI 通过。

启动完整 V5 前，先完成两个 30 步机制前缀，以及两条新源码旧变体完整回合对照：原索引 3/V3 与原索引 28/V4。对照要求同输入、同种子、完整有序物理签名摘要和任务字段一致，只有墙钟耗时不参与相等比较。它们均不进入 V5 的 20 回合分母。

## 本地执行入口

以下路径对应本机保留的参考和输入，复现必须使用全新输出目录。不要覆盖本轮原始文件。

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -B -m chapter3_bser.experiments.safe_search_v1.run_ac_development `
  --manifest '3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json' `
  --config configs/chapter3/hgr_train.json `
  --reference-dir runs/safe_search_v1/development_B0_20260928_v2 `
  --output-dir runs/safe_search_v1/development_AC_20260928_v1 `
  --workers 6 --execute
```

入口逐一验证参考的 100 个完整回合，保留旧来源身份。六个子进程只负责不同场景，不改变每个场景的模拟种子、控制或任务上限。旧参考无需重跑全部 100 回合，也不能伪装为在新源码下运行。

## 完整回合后处理

```powershell
& 'C:/Users/lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -B -m docs.chapter3.search_diagnostics.safe_search_v1.analyze_ac_development `
  --run-dir runs/safe_search_v1/development_AC_20260928_v1 `
  --reference-dir runs/safe_search_v1/development_B0_20260928_v2 `
  --reference-analysis runs/safe_search_v1/development_analysis_B0_20260928_v2/analysis.json `
  --comparability-json runs/safe_search_v1/ac_source_controls_20260928_v1/comparability.json `
  --output-dir docs/chapter3/search_diagnostics/safe_search_v1/ac_results_20260928
```

分析器要求全部 20 回合和两条源码对照均通过，重新对账新旧摘要、输入与全部轨迹，保留原选择 V3，单独输出 V5 的门槛及配对新增／丢失病例。场景是统计单位，120 个累计回合不等于 120 个独立样本。

所有配对区间采用 10,000 次 percentile bootstrap、种子 20260928。门槛使用汇总计数除以汇总曝光；JSON 中各率的配对区间使用逐场景率差的均值，不能混用。20 个场景已经用于开发诊断，且比较多个指标，结果不能代替新独立测试。

`runs/`、`outputs/`、`3090结果/` 和模型继续由 Git 忽略。只保留代码、测试、精简审查和紧凑结果。本次不执行 commit 或 push。
