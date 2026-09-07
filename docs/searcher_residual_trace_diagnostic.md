# Searcher residual 逐步轨迹诊断

只诊断，不设计新策略。未改变 Actor/Critic/PRRAC、BSER objective、C2、环境、reward、训练、residual scale 或 gating。原 evaluator 不传 `searcher_trace` 时不开启记录，也不新增输出。

## 文件

修改：

- `chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py`：默认 None 的两处只读 trace 调用，以及将可选记录列表交给现有 SEARCH diagnostics。
- `chapter3_bser/experiments/phase1c_prrac/search_continuity/diagnostics.py`：将已计算的逐-agent 值导出为普通 Python 数据；原聚合公式、计数、返回值不变。

新增：

- `chapter3_bser/experiments/phase1c_prrac/searcher_residual_trace.py`：只读 action/mission 记录器及字段定义。
- `chapter3_bser/experiments/phase1c_prrac/run_searcher_residual_trace.py`：来源核验、固定子集、原 evaluator 调用、spawn 与父进程写文件。
- `scripts/run_searcher_residual_trace_diagnostic.py`：从脚本位置定位仓库，无 sys.path 注入。
- `scripts/analyze_searcher_residual_trace.py`：纯标准库离线分析。
- `tests/test_searcher_residual_trace.py`：临时数据与短 native 夹具验证。
- 本说明文件。

已有 `analyze_searcher_residual_effect.py` 及对应测试不属于本次改动。

## 场景保真与执行

必须提供原实验根目录，内含 `full_prrac/` 和 `searcher_residual_off/`，各自包含完整 `evaluation_manifest.json`、`resolved_evaluation_config.json` 和 `episode_evaluation.csv`。

启动前重新计算原 manifest/config 的声明 hash，校验两份完整 100 场 manifest 完全一致，检查每个 episode 的 seed、checkpoint、max_steps、模式、B1、C2、native integration 和 runtime 身份。cases 必须恰好为 5 help + 10 hurt，并且等于从原 100 场 Success 结果重新求出的全部 discordant 集合。

恢复来源 manifest 中的完整 scenario 字典，按原顺序选取；保留原 canonical episode index 和 scenario seed。**不调用生成器，不重新生成 15 场，不根据 ID 猜参数。** `--max-per-transition` 只在完成上述完整核验后截取各组前 N 个，不改变 400 步截止。

新运行使用原已解析的各模式配置和原 `_evaluate_episode_job`。仍由原 `_apply_residual_mode` 处理 residual-off，Executor 不变。父进程只读加载 checkpoint 并传递普通数据/NumPy 快照；worker 各建独立 runtime，返回行数据，不共享 CSV append。

只允许 CPU；默认 workers=4，可为 1/2/3/4。完整运行 15×2=30 episodes；smoke N=1 为 2×2=4 episodes。两者 explore=false、training_update=false。非空输出拒绝覆盖，不支持 resume。

checkpoint 路径必须与历史 CSV 记录一致，不使用 basename 猜测同一个文件。新 manifest 记录本次读取的 SHA256；旧协议未记录 checkpoint 文件 SHA256，因此不声称历史字节级身份已获独立证明。加载时还校验 episode/config/runtime 身份，运行结束比较历史 Found/Contact/Success、episode length、found step 和搜索碰撞计数。出现重现差异时保留证据，状态为 `completed_with_mismatches`，退出码 2；离线分析拒绝将它作为有效机制报告。异常退出码 1，正常完成 0。

## 时间与字段合同

| 字段 | 来源与解释 |
| --- | --- |
| action `step=t` | 动作前物理状态；此行 collision 是动作 t 引起的 t→t+1 转移事件 |
| mission `step=t, state_step=t+1` | 同一转移后的任务/地图/Executor 状态；first Found 使用 state_step |
| position / navigation_target | 动作前运行时位置及实际安装的导航目标；不额外更新 tracker/provider |
| waypoint_cursor | `bridge.path_tracker.snapshot(agent).next_index`，为 base path 的公开游标，不冒充 C2 局部路径游标 |
| raw_residual | mode 处理前 `gated_residual_action`，与现有 episode raw norm 同源，维度无量纲 |
| applied_residual | 原 mode/continuity adapter 后实际传入 env 的 residual command，维度无量纲；R1 的 SEARCH agents 0/1/2 为 0 |
| prior_action | 环境已有 `_last_prior_acc`，物理加速度 |
| final_action | 环境已有 `_agent_acc`，经过加速度裁剪/完成-agent mask 后的实际加速度缓存 |
| residual_prior_cosine | 为兼容字段名；实际复用 Actor alignment_cosine：原始 **residual_mix 与 observation 9:12 导航单位向量**的对齐，不是 applied residual 与物理 prior acceleration 的 cosine |
| alignment 有效性 | 完全复用 episode diagnostics 的 route_active、导航长度和 residual_mix norm >1e-8、有限值检查；无效 cosine/negative_alignment 为 NA |
| residual_contribution_ratio | 若原统计输入是逐-agent ratio 才有值；当前 runtime 只暴露团队 scalar，因此此逐-agent 字段为 NA |
| mission team_residual_contribution_ratio | 直接读取 `last_residual_contribution_ratio_search`，即现有 episode 使用的团队均值；每步记录一次，不向三个 agent 广播 |
| collision/streak/waypoint_switch | 同一 SEARCH diagnostics 的已计算事件与计数，包含导致 Found 的最后一个 SEARCH 转移 |
| c2_active/state | 动作前现有 recovery snapshot，读取而不推进 C2 |
| allocation_change_event | public BSER assignment 的 ID/kind/final waypoint/planned path/reachable/hold 变化，排除纯 cursor 移动和 C2 overlay |
| executor_target_distance | 原 evaluator 已合法访问的真实目标诊断值，仅 audit-only；不进入 policy/BSER/observation |

**不能将 prior_action 与 applied_residual 直接相加**，两者单位不同；没有重新计算或改变环境的 residual scale。逐步 raw/applied norm、alignment 直接从原聚合计算位置输出，缺失不补零。当前团队 ratio 的事实限制会保存在 trace manifest，离线 ratio 窗口使用 mission 的原团队值。

## 离线分析

输入两份同一 trace manifest 下的 trace 文件夹。先校验 manifest 状态、各文件 hash、selected 身份、mode/seed、唯一键、连续 step、三个 Searcher 的 PRE_FOUND 覆盖和 mission 终态。两组 PRE_FOUND 长度可因 Found 时间不同而不同：只比较严格相同 `(scenario_id, step, agent_id)` 的交集，未配对尾部单独计数；不插值，不把缺失当成相等。

首个差异采用最早 step，再用最小 agent_id 打破并列。action/导航目标的数值差异容差为 1e-8；同一 agent 的欧氏位置距离使用 >=0.5、>=1.0 诊断阈值。每类首个分叉记录 step、agent_id 和 `*_context` JSON 列，包含 R0/R1 同一 agent 在精确前一步与当前步的 residual/alignment/route/collision/C2 值；团队比例另有明确命名的字段。

窗口定义为 `[max(0,event_step-20),event_step)`。缺少分叉或 Found 事件时窗口不可用，而不是改用整场。报告有效 alignment 样本数、ratio 有效 step 数及观察 step 数。

负对齐后的事件以**同一 agent** 的 t+1…t+5 为窗口；不足五步或必要字段缺失则 censored，不计入 none。动作当次 collision 另列，避免混淆“同时发生”和“后续发生”。组级先计算每个 scenario 的 rate，再等权汇总；agent-step 合计仅标为描述性 counts，不作为独立场景。

输出：

- `paired_first_divergence.csv`
- `residual_help_trace_summary.csv`
- `residual_hurt_trace_summary.csv`
- `searcher_residual_trace_summary.json`

报告明确：这是事后选择的机制诊断集，不是独立验证集；分叉后不是相同 counterfactual state；负对齐不必然有害；关注分叉前状态、首次分叉附近及 help/hurt 重复模式。没有自动“有必要修正/有害过度干预”的分类或因果结论。

## Linux 手工命令

在你的 Linux `AUV-Thesis-D1D2` 仓库根目录执行。以下假定原实验在 `outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2`，cases 在 `searcher_residual_analysis/`；若实际位置不同，替换路径。将 `<原checkpoint绝对路径>` 替换成原 episode CSV 的 checkpoint 值。可直接使用原 R1 的 resolved config 作为 OFF config，避免手工重建参数。全部命令为单行。

Smoke（完整验证 15 个 cases 后，每组取 1 个，仍为 400 步）：

```bash
python scripts/run_searcher_residual_trace_diagnostic.py --source-root "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2" --cases-csv "searcher_residual_analysis/success_transition_cases.csv" --checkpoint "<原checkpoint绝对路径>" --config "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/searcher_residual_off/resolved_evaluation_config.json" --workers 4 --device cpu --max-per-transition 1 --output-dir "searcher_residual_trace15_smoke_v1"
```

人工检查 smoke manifest 的 completed 状态及字段之后，完整 30 episodes：

```bash
python scripts/run_searcher_residual_trace_diagnostic.py --source-root "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2" --cases-csv "searcher_residual_analysis/success_transition_cases.csv" --checkpoint "<原checkpoint绝对路径>" --config "outputs/chapter3/phase1c_prrac/residual_role_pair100_ep100_v2/searcher_residual_off/resolved_evaluation_config.json" --workers 4 --device cpu --output-dir "searcher_residual_trace15_v1"
```

纯离线分析：

```bash
python scripts/analyze_searcher_residual_trace.py --full-trace "searcher_residual_trace15_v1/full_prrac" --searcher-off-trace "searcher_residual_trace15_v1/searcher_residual_off" --output-dir "searcher_residual_trace15_v1/analysis"
```

查看父进程逐 episode 原子更新的进度：

```bash
watch -n 5 cat searcher_residual_trace15_v1/progress.json
```

默认输出结构为 trace_manifest、selected_scenarios、progress，以及两个模式各自的 action trace、mission trace、episode_evaluation。analysis 文件夹由离线脚本显式生成，不在 runner 中自动开展下一步分析。

## 验证范围

只执行 compileall、专项 unittest 和短合成 native 夹具。未运行真实 checkpoint/正式 30 episodes、100×400、Linux 正式分析或训练。不把工程 no-op 通过当成机制假设成立。未 commit/push。

最新记录的本地验证（非 CI）：新增 17 项测试与已有 18 项相关回归测试分批通过；`python -m compileall scripts tests chapter3_bser` 和 `git diff --check` 通过。新增测试覆盖 canonical 子集/来源拒绝、spawn 四 worker 排序、30 份 job 的 mock 装配、离线输出与缺失时间步拒绝，以及两个模式各自的单步 native trace 开关等价性。单步等价性同时检查 evaluator payload、动作和 Python/NumPy/Torch 随机状态。
