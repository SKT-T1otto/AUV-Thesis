# Search Value D1 / D2 诊断工具

本工具只回答预测与候选选择价值问题，不改进在线算法，不训练，不更新 replay/optimizer，不改 Actor/Router/Expert/Gate/Head、reward、Executor、C2 或 BSER objective/接受阈值。28D / 3D / 124D 与 checkpoint schema 不变。

开发起点：main，HEAD `e08eec28bd3d5e96cfe8d8f91a6871f26595db60`，初始工作树干净。Head 训练参照 `38aee62d072a8ea109b18e3be7b0d8bd8ccf11aa`。每次真实运行重新记录实际 Git HEAD、dirty diff hash、版本差异、checkpoint 文件及 Actor/Head hash、配置/manifest hash 和依赖版本；本文的起点不是后续运行身份。

## 文件清单

| 文件（相对仓库根目录） | 目的 |
| --- | --- |
| chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py | 默认 None 的观察/单次提案审计生命周期钩子 |
| chapter3_bser/online/allocator.py | 默认不启用的只读实际生成记录，覆盖空候选早退；生成器/原求解与接受逻辑不变 |
| chapter3_bser/experiments/phase1c_prrac/search_value_audit/__init__.py | 固定 schema/参照/窗口常量 |
| 同目录 prediction_audit.py | D1 独立 CLI、冻结场景、OFF shadow、结果汇总 |
| 同目录 branch_audit.py | D2 历史选择/root 定位/单次分支/有效性门禁 |
| 同目录 runner.py | 输入核对、spawn 工作单元、原 evaluator 调用、异常记录 |
| 同目录 runtime_audit.py | 原候选捕获、A/B/C 当次提案替换、预测/安装/后缀记录 |
| 同目录 features.py | 34D 字段目录、同刻对齐、候选可区分性和逐轮 margin/context |
| 同目录 metrics.py | 标签/censoring、训练基线对比、校准/bootstrap、配对与事后参考 |
| 同目录 provenance.py | 只读 checkpoint、权重/场景/Git 身份、原子文件和进度 |
| 同目录 state_fingerprint.py | 完整运行状态清单和指纹，不实现存档恢复 |
| 同目录 analysis_bundle.py | 明确 allowlist 的小型证据打包 CLI |
| scripts/test_search_value_audit.bat | Windows compileall/专项回归入口 |
| scripts/linux/_search_value_audit_common.sh | 路径/参数/环境变量与文件检查 |
| scripts/linux/run_search_value_d1_audit.sh | Linux D1 手工 launcher |
| scripts/linux/run_search_value_d2_audit.sh | Linux D2 手工 launcher |
| scripts/linux/bundle_search_value_audits.sh | cwd 无关的分析包 launcher |
| tests/search_value_audit_support.py | 独立合成 checkpoint、九格闭环及 native 短夹具 |
| tests/test_search_value_audit_metrics.py | 标签/统计/特征/只读加载/进度/打包测试 |
| tests/test_search_value_audit_branches.py | 分支/接受/拒绝/C/顺序/截止/无真值评分测试 |
| tests/test_search_value_audit_drivers.py | 历史 joins/manifest/独立性/CLI/launcher 测试 |
| tests/test_search_value_audit_runtime.py | 原 native OFF/ON no-op 与审计分支 spawn 1/2 一致性 |
| docs/search_value_d1_d2_audit.md | 合同、限制、运行与交付说明（本文件） |

## 固定合同与解释范围

两个独立 CLI 都固定 M20、400 步全局上限、full_prrac、B1_ATOMIC_LAST_VALID/native、S2A1_C2_LOCAL_CONNECTOR、CPU、explore=false、training_update=false。只允许 workers=1/2。缺 Head、形状/schema 不兼容、其他 Search Value decision 通道开启或历史配置不符合合同时失败。旧输出目录非空则拒绝，未实现 resume。

历史协议没有保存 checkpoint 文件 SHA256：工具能核对路径 basename、episode、training config hash、schema/runtime，并重放历史任务结果；**不会声称历史 checkpoint 字节级身份已被独立证明**。当前实际读取文件的 SHA256 与前后参数 hash 会保存。这项来源限制需要结合历史文件保管记录人工审核。

D1 的场景排除核验与 D2 的有效配对门禁是数据/复现结论，不是性能结论。工程单测通过不代表它们已经在真实数据上成立。D1 为 diagnostic_only；D2 是历史干预筛选的开发集，不外推完整 M20 性能。没有 accuracy>95% 的通过门槛。

## D1：预测与特征审计

- 先生成/加载 seed=51729 的固定 30 场景 manifest，在任何 rollout 前落盘冻结；smoke 只运行其中前两个，但不缩短 400 步。正式诊断可直接使用 smoke 保存的 manifest。
- 与训练及历史 OFF/ON 场景核对 seed、障碍布局、初始 agents、目标初始状态与运动参数。复用 scenario_id 本身不等于重叠；相同 seed 或物理布局组合则保守拒绝。没有训练 manifest 时按原 config 重建，并核对 checkpoint 全部已完成 episode ID/seed、训练参照下的生成依赖代码。无法证实则标记 independence_unverified，不通过补抽“修复”分布。
- 实际控制始终 OFF+C2。私有冻结 Head/提取器只观察，不交给 runtime_factory。每个 PRE_FOUND 状态、下一次 action **之前**，为三个 Searcher 各记一条预测。只用原 observation 和已有 public state；不额外刷新 provider/map，不推进 live tracker，不引入未来信息。
- 标签在 episode 结束后计算。t 是动作前状态，首次 Found 为 F：`0 < F-t <= 50` 为正；确实观察满 50 步且未发现为负；提前结束、未来不可判断则 censored。行政剩余 `400-t >= 50` 才入主指标；尾部单列，已 Found 排除。三个共享团队标签不算三个独立场景。
- 训练常数概率仅取 checkpoint 中已占用、search_value_valid、stage_before=0 的唯一 `(episode_id, transition step)`，共享标签只计一次。不调用 replay.sample，不使用优先采样权重；ring 截断及原 trainer 尾部标签限制显式记录。
- Brier、非插值 AP（相同概率按组）、precision/recall/specificity/balanced accuracy/accuracy、正例比例、分布、固定十箱校准；按 Searcher、微平均、场景等权分别报告。相对训练常数的 Brier 差采用 scenario-cluster bootstrap（seed=61729，2000 次），每次保留场景全部时间与三个 Searcher。未定义指标 null+原因。
- 同时点比较安装后实际 observation+6D 与同一 assignment 的 candidate_feature；记录逐维差、两个概率、路径 hash、live tracker 游标、semantic/public/effective tracking waypoint、map revision 和 provider 更新时间。C2 有 plan 时归 overlay 层，即使 tracking 点恰好相同也不当无 overlay。跨 step/位置不一致时不做相减。
- 已查到的待诊断语义：候选预览使用新 PathTracker，live bridge 保留游标；实际导航距离使用 clamp_min(eps)，候选重建用原始距离；candidate_feature 只重建 6:12、15、17，其余任务/hold/progress 字段继承当前观测。当前提取器新增 synchronize_state，并在同 step refresh 同步；这些差异仅记录，不修改。34D 来源/尺度/更新说明写入 manifest。
- 第 18 维沿用原 obstacle sensor observation：M20 的 `_nearest_obstacle_distance` override 将距离截到 obstacle_sensor_range，再除以 10/clamp。审计不额外查询隐藏障碍几何，也不把未截断的距离送入 Head。
- 捕获每次实际生成的候选池，绝不再次调用生成器。精确 feature hash 与固定 atol=1e-6 的稳定首代表聚类同时记录；统计 endpoint/path/初始 tracking/feature 数量、同 feature 多 endpoint、预测范围/方差，以及原始边际收益差和 0.1×概率差的相对大小。candidate_scores.csv 另存每轮 greedy prefix/context，不把不同 prefix 的 marginal 混作同一评分。

每场景运行 bare（完全不传 hook）、fingerprint-only control、shadow 三次。control/shadow 的 action、完整状态边界/终态、guidance、RNG 和任务结果必须相同，bare 的任务结果也必须一致，否则非零退出且不发布有效 prediction_summary。

主要输出：audit_manifest.json、scenario_manifest.json、prediction_rows.csv、prediction_summary.json、calibration_bins.csv、feature_consistency.csv、candidate_generation.csv、candidate_representation.csv、candidate_scores.csv、training_label_baseline.json、no_op_validation.json、resolved_audit_config.json、progress.json。candidate_generation.csv 覆盖空候选与 solver 之前的早退；此时 margin 未定义，不伪造评分。D1 全程逐轮 candidate_scores.csv 留在本地，分析包使用汇总 candidate_representation.csv；D2 root 的 candidate_scores.csv 则是必需打包文件。可选 raw_features.npz 包含实际/安装后比较的 34D 向量，以 feature_id 关联；不默认入分析包。

## D2：公共前缀、一次干预、后续 OFF

1. 从历史 ON metrics 文件读取 **全部** accepted_search_change_count>0 场景，并与 OFF/ON manifests、episode CSV 逐项核对；数量由文件决定，不硬编码 25，不按结果选择。smoke 仅首个历史选择场景，找不到 root 也保留 mismatch。
2. 重放原 OFF/ON；各自做 bare/control/observer no-op 和历史结果核对。按 ON 决策顺序，用全新 runtime 从同 seed 重放 ON 前缀，到每个可能 root 的 `controller.initialize/step` **副作用之前**施加 A 的前缀 probe。首次真正安装的 search assignment/path/public/effective guidance 不同于 ON 的边界才是 root，候选 ID 不进入几何签名。
3. 指纹包含环境动力学与目标内部状态、地图/log-odds/belief/revisions、provider cache/刷新、controller/cache/hysteresis、原 allocator、bridge/tracker、installed guidance、C2 plan/失败记忆/重接标志、任务计数、全局及实例 RNG。真值只进入离线 hash，不进入评分接口。未知对象/循环引用失败，不用不完整 deepcopy/pickle 恢复。导航 hook 的 owner/bound runtime 校验为已列入清单的对象；纯计时/只供报告的诊断字段排除并在清单说明。状态 hash 不一致的 root 不入有效配对。
4. A 是同池原 BSER 完整 search 提案；B 是现有 weight=.1 引导提案；C 从 A 只替换一个 Searcher，保留其他 assignment 与 baseline standby，在原池/原正边际可行规则下排除 A/B 几何等价者，选原 objective 最高者，稳定 key 打破并列。部分重规划时，几何排除与 objective 均按原合并规则包含未变更的 Searcher，不把 solver 临时遗漏的冻结项误判为新方案。无 C 标记 C_UNAVAILABLE。
5. 所有提案经过原 controller 接受/拒绝和 guidance/C2 安装流程。记录 proposal_accepted、treatment_delivered、changed_agent_ids。B 改变多个 Searcher 时是完整 allocation 干预，不解释为单个候选的效应。
6. 当次决定后全部 OFF；自然重规划不被冻结，不追加 400 步。后续不同观测/轨迹/Executor 动作允许自然分化。A 尾部不要求等于整场历史 OFF；报告 ON 公共前缀与 OFF 的物理/action/guidance/RNG 是否相同。
7. A/A 重复，A/B/C 与反序 C/B/A，完整 root/prefix 一致，B 重现 ON 当次接受安装，干预后 guided=0、权重不变和全局截止均为门禁。异常退出 1，保留 mismatch 的完成运行退出 2，通过工程/复现门禁退出 0（不表示科学上有益）。

指标包含 root/剩余步数/pool/proposal/installed signatures、各候选 Head 值/原 marginal/aux/greedy prefix/context、接受与实施、guidance 有效持续步数（直到 effective search geometry 首次改变，包含 tracker 自然推进）、首个后续接受重规划、50 步 Found（与 D1 同 censoring）、最终 Found/Contact/Success、发现时间（未发现 null）、后缀 searcher 碰撞/最长连续碰撞、已知比例增量与移动距离。A 的 treatment_delivered 表示相对 root 前 assignment 的接受变更；B/C 另外明确是否相对 A 的实际控制发生变化。

A/B 按 root 统计 both/A_only/B_only/neither，连续量 B-A/分位数/逐场景/ties；失败门禁的 root 标 unavailable。拒绝或未实施提案单列。每个候选概率是共享团队标签的预测，不能相加成为团队概率，因此 allocation 概率排序一致率为 not_applicable。C 的事后最好结果只能作已测试候选中的诊断参考，不是在线 oracle 或理论上限。

主要输出：branch_manifest.json、historical_reproduction_check.json、decision_audit.csv、candidate_scores.csv、branch_outcomes.csv、paired_branch_comparison.csv、branch_summary.json、replay_validation.json、resolved_audit_config.json、progress.json。状态仅记录 hash/清单和紧凑 trace，不保存通用可恢复 snapshot。

## Windows 验证

在 PowerShell 运行（仅单测与合成夹具，不运行真实实验）：

```powershell
$env:AUV_AUDIT_PYTHON = 'D:\anaconda\anaconda\envs\AUV\python.exe'
& 'E:\gym\code\WORKSPACE\AUV-Thesis\scripts\test_search_value_audit.bat'
if ($LASTEXITCODE -ne 0) { throw 'Search Value audit verification failed' }
```

.bat 从自身位置定位仓库，依次执行 compileall、新增 test_search_value_audit*.py、用户列出的 Search Value/evaluator/provenance/runtime 测试及全部 test_s2a1*.py。不是全仓 suite。历史事件枚举/冻结证据缺失若出现在其他 suite，应单独报告，不修改断言或证据来凑绿。

### 最新记录的本地验证（2026-09-05）

| 验证范围 | 结果 |
| --- | --- |
| `python -m compileall -q chapter3_bser scripts tests` | 通过 |
| 新增诊断测试 | 分批共 28 项通过：metrics 12、branches 8、drivers 6、native runtime 2 |
| .bat 中列出的 6 个既有 Search Value/evaluator/provenance/runtime 模块 | 44 项通过 |
| `test_s2a1*.py` | 34 项通过 |
| Phase1C guidance、28D contract、Phase1B2 path tracking 及两个精确 provenance/registry 检查 | 7 项通过，27 条历史 provenance 记录保留并验证 |
| `git diff --check` | 通过；原 allocator 文件有 Git 的 CRLF/LF 提示，不是 whitespace error |

合计 113 项专项测试，不是单次全仓 discover，也不是 CI。最后新增的 partial-C 合并排除测试与 metrics/branches/drivers 一起补跑 26 项，全部通过；native OFF/ON no-op 和真实 runtime 的 A/B workers=1/2 全输出一致性已在 .bat 中通过。测试仅使用临时合成 checkpoint/场景；小型 native 夹具的步数不代表真实诊断的 400 步合同。

未运行：真实训练 checkpoint、Linux launchers 的实际执行、D1 两场 smoke/30 场诊断、D2 历史重现/首个 root smoke/全量分支、正式训练或全仓回归。Linux launcher 已做 LF/根目录/参数静态检查，CLI help 已验证。用户列出的其他历史失败项未作为本轮专项用例执行，未修改断言、skip、历史证据或其 hash。

因此工程状态为“Windows 专项与合成闭环验证通过”；真实数据独立性及历史复现门禁仍未验证；Head 是否具有预测/候选选择价值仍是证据不足。历史 checkpoint SHA256 缺失的来源限制不能由这些测试消除。

## Linux 手工启动

先激活用户自己的 AUV Python 环境。下面整段一次设置全部路径；未运行的输出目录名称可由用户预先换成新名称。不要复用旧 OFF/ON 输出作为 output-dir。

```bash
export AUV_AUDIT_REPO='/home/legion/AUV-Thesis/AUV-Thesis-SearchValue'
export AUV_AUDIT_PYTHON="$(command -v python)"
export AUV_AUDIT_CHECKPOINT="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/s2b_search_value_pilot_seed1_ep100_v1/checkpoints/phase1c_prrac_episode_0100.pt"
export AUV_AUDIT_TRAINING_CONFIG="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/s2b_search_value_pilot_seed1_ep100_v1/resolved_training_config.json"
unset AUV_AUDIT_TRAINING_MANIFEST
export AUV_AUDIT_OFF_OUTPUT="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/s2b_guided_pair100_off_ep100_v1"
export AUV_AUDIT_ON_OUTPUT="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/s2b_guided_pair100_on_w010_ep100_v1"
export AUV_AUDIT_WORKERS=1
export AUV_D1_SMOKE="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/d1_audit_smoke_v1"
export AUV_D1_OUTPUT="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/d1_audit_30_v1"
export AUV_D2_SMOKE="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/d2_audit_smoke_v1"
export AUV_D2_OUTPUT="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/d2_audit_all_v1"
export AUV_AUDIT_BUNDLE="$AUV_AUDIT_REPO/outputs/chapter3/phase1c_prrac/d1_d2_analysis_v1.zip"
```

D1 smoke（固定 30 manifest 中的前两个，仍为 400 步）：

```bash
export AUV_AUDIT_OUTPUT_DIR="$AUV_D1_SMOKE"
bash "$AUV_AUDIT_REPO/scripts/linux/run_search_value_d1_audit.sh" smoke
```

人工检查 smoke 的 provenance/no-op 后，正式 **30 场景诊断**（不是论文正式测试）：

```bash
export AUV_AUDIT_OUTPUT_DIR="$AUV_D1_OUTPUT"
bash "$AUV_AUDIT_REPO/scripts/linux/run_search_value_d1_audit.sh" diagnostic --manifest "$AUV_D1_SMOKE/scenario_manifest.json"
```

D2 首个历史干预场景/root smoke：

```bash
export AUV_AUDIT_OUTPUT_DIR="$AUV_D2_SMOKE"
bash "$AUV_AUDIT_REPO/scripts/linux/run_search_value_d2_audit.sh" smoke
```

人工检查历史重现和 root 门禁后，全量历史干预场景：

```bash
export AUV_AUDIT_OUTPUT_DIR="$AUV_D2_OUTPUT"
bash "$AUV_AUDIT_REPO/scripts/linux/run_search_value_d2_audit.sh" diagnostic
```

另开终端、设置相同 output-dir 查看百分比（耗时含每个独立 runtime 的公共前缀，首个单元未完成前 ETA=null）：

```bash
watch -n 5 --exec cat "$AUV_AUDIT_OUTPUT_DIR/progress.json"
```

一次性打包两份完成结果；required 文件缺失、仍在运行或超过小型证据限制时明确失败。不包含 .pt/replay/raw_features/full-state/failure_trace。包已存在则拒绝覆盖：

```bash
bash "$AUV_AUDIT_REPO/scripts/linux/bundle_search_value_audits.sh" --input-dir "$AUV_D1_OUTPUT" --input-dir "$AUV_D2_OUTPUT" --output "$AUV_AUDIT_BUNDLE"
```

查看与实际 CLI 一致的参数：

```bash
cd "$AUV_AUDIT_REPO"
"$AUV_AUDIT_PYTHON" -m chapter3_bser.experiments.phase1c_prrac.search_value_audit.prediction_audit --help
"$AUV_AUDIT_PYTHON" -m chapter3_bser.experiments.phase1c_prrac.search_value_audit.branch_audit --help
"$AUV_AUDIT_PYTHON" -m chapter3_bser.experiments.phase1c_prrac.search_value_audit.analysis_bundle --help
```

需要真实训练 manifest 时，unset AUV_AUDIT_TRAINING_CONFIG，设置 AUV_AUDIT_TRAINING_MANIFEST。workers=2 通过 spawn 运行，父进程逐单元原子落盘，最终输出按预声明顺序排序；不向子进程传活体 env/controller/CUDA tensor。D2 需要多次前缀/控制重放，不能按三个短后缀估计运行时间。

最终解读分三层：工程是否可运行；数据身份/排除/复现门禁是否成立；证据支持、有所反驳还是不足以判断 Head 价值。未运行真实 Linux 数据时，后两层仍待验证，不能由合成单测得出 Search Value 有益的结论。
