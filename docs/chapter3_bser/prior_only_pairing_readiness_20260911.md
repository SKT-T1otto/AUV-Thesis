# Prior-only 评价可靠性与手动配对准备汇总

日期：2026-09-11。工作区：`E:\gym\code\WORKSPACE\AUV-Thesis`。
本文可单独交给 GPT 分析。它描述本轮代码修补和本地验证，不是实验性能报告。

## 1. 本轮结果与边界

- 保留已有 Prior-only 核心实现：不调用 actor forward，生成 float32 的 `(4, 3)` 零残差，经过原有 residual mode 和 action adapter，再进入原有 `env.step()`。
- full_prrac 继续使用原有 actor 输出、gated residual 与 adapter。没有重新设计控制器。
- 没有修改 environment、reward、planner 决策、actor/critic 结构、replay buffer、checkpoint 加载或兼容逻辑；未实现 PVDRL 或新 Trust gate。
- 本轮没有训练，没有启动真实 episode、配对仿真或批量评价。只运行定向单元测试、PowerShell 语法检查，以及两个 BAT 的 `-PrepareOnly` 路径。
- 最终本地验证：51 项定向单元测试通过，耗时 34.025 秒。这不是 CI，也不是任务性能证据。
- 尚无本轮真实 trained checkpoint 的加载结果或两种 controller 的实际配对性能结果。不能据此判断 Prior-only 是否更优；`performance_passed` 不成立。

## 2. controller 身份与缓存核查

| 核查位置 | 原有保护或已确认风险 | 本轮处理 |
| --- | --- | --- |
| resolved config / resume | 原有 resolved_config_hash 已包含配置差异，正常使用同目录恢复时已有保护 | 保留哈希规则，额外显式核对 controller；不宣称原来所有恢复都不安全 |
| `_combo_key` | 原组合键没有 controller，无法独立区别同 checkpoint、同场景的两种控制器 | 增加 controller_mode |
| completed 缓存、episode/summary | 只依赖上层配置不足以发现混入或错误标记的行 | 使用前核对 progress、completed、episode、summary 及 trace/recovery 记录的 controller |
| activation 去重 | 原去重键会把仅 controller 不同的两条记录当成相同记录 | controller 加入去重键 |
| provenance | 原字段清单及跨文件匹配缺少 controller | 加入来源派生、组合身份和汇总匹配规则 |
| 结果汇总 | 通用汇总函数可能收到混合 controller 行 | 单次 evaluator 只允许一种 controller；混合输入报错，不合并平均 |
| 辅助 CSV/JSON | 部分派生表没有 controller 标识 | 导出时传播 controller；主指标、execution/search/recovery 汇总都可识别来源 |

兼容规则是明确的：**缺少**旧 controller 字段时按历史 `full_prrac` 解释；显式空白、null 或未知值是错误。旧 full_prrac completed 键在内存中补默认值后匹配，避免无谓重跑。旧行进入导出分组前使用副本规范化，避免将缺省值误写成显式 null。

本轮之前的 Prior-only smoke 虽然 episode 和 summary 已标记 prior_only，但其旧 progress/completed 尚无该字段。新版会拒绝恢复这种缺少身份的旧 prior_only 缓存；不会把它改写成新版证据。请保留 smoke，只在新的独立目录手动运行配对。

跨 controller 的比较由新的 `paired_evaluation.py analyze` 完成。既有 `paired_checkpoint_comparison.csv` 等文件仍是单一 controller 内的 checkpoint/消融比较，不应当作 Prior-only 对 full_prrac 的比较表。

## 3. `_trace_step` 的真实数值语义

依据 `core/env/uav_env.py` 的 `_actions_to_residual_acc()` 与 `_apply_agent_dynamics()`，现有计算顺序是：

1. 将归一化 residual command 限制在 `[-1, 1]`，已结束 agent 的 command 置零。
2. 按 XY/Z 加速度上限将 command 转成物理加速度，再乘 `_residual_scale`，形成 residual。
3. waypoint prior 本身经过其原有约束，再乘 `_prior_strength`，形成 prior。
4. 合成 `prior + residual`，再按轴限制加速度。
5. `_last_prior_acc` 与 `_last_residual_acc` 缓存的是第 2、3 步物理量；`_agent_acc` 是限幅后的量，后续已结束 agent 的加速度还会置零。

| trace 字段 | 保留的历史含义 |
| --- | --- |
| `navigation_prior_norm` | 执行器 agent 3 的 `_last_prior_acc[3]` 范数，已包含 prior strength |
| `residual_action_norm` | 执行器经过 adapter 的归一化 residual command 范数，不是 `_last_residual_acc[3]` 的物理范数 |
| `final_action_norm` | `norm(_last_prior_acc[3] + _last_residual_acc[3])`，是**合成后的限幅前物理加速度范数**，不是 `_agent_acc[3]` 的范数 |

因此不能把 trace 的 residual_action_norm 直接与 navigation_prior_norm 相除而解释为物理残差占比，也不能把 final_action_norm 当作实际施加加速度或速度。

本轮仅添加语义标记：

```json
{
  "final_action_norm_semantics": "pre_clip_prior_plus_scaled_residual_acceleration",
  "residual_action_norm_semantics": "normalized_command_after_adapters"
}
```

没有改变旧数值字段公式。合成夹具设置 prior=4、physical residual=3 时，历史 final_action_norm 仍是 7，与模拟缓存的限幅后范数不同。

## 4. Found 后兼容性验证

复用已有 public guidance、transition metadata、execution plan 测试夹具：

- POST_FOUND、CONTACT、HOLD、SUCCESS 下，actor_outputs 为空仍可调用 `_trace_step`。
- gate、alignment、router probability/prediction 保持缺失值，不伪造 actor 诊断；空 PRRACDiagnostics 也可汇总。
- stage 1/2 的零残差经过原 residual mode 和全部四个原 ExecutionVariant adapter 后，形状保持 `(4, 3)`、数值保持零。
- B3 SAFE_HOLD suppression 的原诊断标志保留，即使输入本来就是零。

这些都是夹具测试；没有观测真实 Found、Contact 或 Hold 轨迹，不能写成 Found 后任务成功或性能提升。

## 5. 配对 CSV 的解析与统计规则

新分析器读取 `episode_evaluation.csv`，不直接拿历史 summary 的条件率作为配对结论：

- CSV 布尔列 `found`、`success`、`contact_episode`、`hold_episode`、`collision_episode` 显式解析。忽略大小写和两侧空格，允许 `true/false/1/0`；其他非空值报错。禁止使用 `bool("False")`。
- 空白保留为 None/JSON null，不替换成 False 或零；scenario_id 保留字符串，避免丢失前导零。`success=true` 且 `found=false` 视为矛盾并报错。
- Found rate 分母为该 controller 的全部回合；Found 有缺失则该率不可用。
- Success rate 分母为该 controller 的全部回合；Success 有缺失则该率不可用。
- 条件成功率 = 该 controller 中 `Found=true 且 Success=true` 的回合数 / 该 controller 中 `Found=true` 的回合数。
- Found 分母为 0 时返回 null；Found 状态不完整或 Found 回合中的 Success 缺失，也返回 null。输出分子、分母与缺失计数供检查。
- 两种 controller 使用各自的 Found 回合，条件率分母可能不同。这不等于在“两边都 Found”的共同子集上比较。
- 率差方向固定为 `prior_only - full_prrac`；任一侧不可用则差值为 null。不输出优越性判断。

既有 evaluator 的历史 rate 分母为零处理方式没有静默改写；新的严格缺失值规则只用于新配对分析器。

分析前还检查：checkpoint 文件 SHA256 与准备时一致；共享配置快照 SHA256 一致；两个 resolved config 除 controller、输出路径与配置哈希外相同；关键输入与准备快照相符；manifest 完全相同；CSV 回合数正确；controller/checkpoint 身份正确；scenario_id + scenario_seed 唯一、两边相同且与 manifest 一致。发现不完整或混合结果时报错，不给出比较结论。

## 6. 手动运行前需要固定的配置清单

| 项目 | 默认值或要求 |
| --- | --- |
| 代码版本 | 两边必须使用同一工作树；运行期间不要修改源码。当前 HEAD 为 `8f7db5103f690c2612385a90d0511c5364ba5489`，本轮改动尚未提交，HEAD 本身不代表完整补丁 |
| checkpoint | 用户显式提供一个已训练的 PRRAC checkpoint；同一路径、同一 SHA256、同一训练来源 |
| 源 evaluation 配置 | 默认 `configs/chapter3/bser_phase1c_prrac_eval.json`，原文件不修改 |
| base candidate / profile | `ch3_v3_full_reference` / `M20_MOVING_UNKNOWN_MULTI` |
| split / scenario seed | `validation` / `1729` |
| 场景清单 | 同一生成规则；逐回合 scenario_id + seed 配对；最终两边 manifest 必须一致 |
| 回合数 | 1 场配对 = 每边 1 回合、合计 2；10 场配对 = 每边 10 回合、合计 20 |
| max_steps | 默认 400，不能只缩短其中一侧 |
| 维度 | observation/action/critic 固定 28/3/124，4 个 agent |
| controller / evaluation_mode | controller 分别 prior_only、full_prrac；两边 `modes=["full_prrac"]` |
| execution variant | 默认单一 `B0_LEGACY_V2_1`；必须匹配 checkpoint 的原有 runtime 合约 |
| checkpoint runtime | 默认 `dynamic_public_intercept_v2_1`；不自动转换 runtime、不绕过加载门禁 |
| recovery | 默认原有 baseline；配对要求只选择一个 recovery variant，避免回合数乘倍 |
| environment / reward / planner | 同一 checkpoint 和配置通过原有管线构造，不修改定义或动力学 |
| device / workers | 默认 cpu / 1；共享快照同时覆盖两边 worker 数 |
| explore / training_update | 两边均 false |
| trace | 共享原配置：enabled=true、only_found_failures=true、max_traces_per_checkpoint_mode=20 |
| 输出目录 | 每次创建 时间戳+随机后缀 新目录，各自包含 prior_only/ 与 full_prrac/，不复用 smoke |

原 evaluation 配置的 100 episodes / 4 workers 在**配对快照**中统一改为 1 或 10 / 1，不回写原配置。checkpoint_globs 清空，只允许显式指定的 checkpoint。两边共用 `paired_eval_config.json`，仅 controller 与输出目录通过命令行分开。

如果 checkpoint 是 native B1，需要用户提供与之兼容的评价 JSON，并通过 `-Config` 同时应用到两边。不能直接把默认 B0 改成可加载的假元数据。现有 `bser_phase1c_prrac_s1_search_diag_native.json` 含两个 evaluation modes，不能直接用于这组配对；配对输入要求单一 `modes=["full_prrac"]`。此轮没有替用户选择真实 checkpoint 或修改 runtime 配置。

脚本在正式手动 run 前复用原 checkpoint loader 验证兼容性；拒绝文件名含 `untrained_test_fixture`、测试 config_hash 或未完成训练 episode 的 checkpoint。已知 smoke fixture 即使只重命名，仍会被其测试元数据拒绝。此检查不能证明任意外部文件已充分训练，训练来源仍须由用户确认；PrepareOnly 不加载权重，也不声称通过该验证。

## 7. PowerShell 命令（本轮未执行真实评价）

将下方 checkpoint 替换为实际已训练文件。此机器 PATH 中发现异常无扩展名 conda 文件，因此显式设置已确认存在的 Conda 可执行路径；其他机器调整此路径。

```powershell
Set-Location 'E:\gym\code\WORKSPACE\AUV-Thesis'
$env:CRK_CONDA_EXE = 'D:\anaconda\anaconda\Scripts\conda.exe'
$checkpoint = 'E:\替换为真实训练目录\trained_prrac_checkpoint.pt'
$config = 'E:\gym\code\WORKSPACE\AUV-Thesis\configs\chapter3\bser_phase1c_prrac_eval.json'
```

只准备并检查命令/配置，不加载 checkpoint、不开始回合：

```powershell
& 'E:\gym\code\WORKSPACE\AUV-Thesis\scripts\run_prior_only_pair_1.bat' -Checkpoint $checkpoint -Config $config -PrepareOnly
& 'E:\gym\code\WORKSPACE\AUV-Thesis\scripts\run_prior_only_pair_10.bat' -Checkpoint $checkpoint -Config $config -PrepareOnly
```

用户决定执行后，手动选择下列命令。它们将启动真实评价，不能把它们作为本轮已运行结果：

```powershell
# 一组配对：两个 controller 各运行 1 episode
& 'E:\gym\code\WORKSPACE\AUV-Thesis\scripts\run_prior_only_pair_1.bat' -Checkpoint $checkpoint -Config $config

# 10 场配对：两个 controller 各运行 10 episodes
& 'E:\gym\code\WORKSPACE\AUV-Thesis\scripts\run_prior_only_pair_10.bat' -Checkpoint $checkpoint -Config $config
```

默认父目录 `outputs/chapter3/phase1c_prrac/paired/`。可用 `-OutputRoot 'E:\新的结果父目录'` 指定其他父目录；子目录始终新建。重复执行不会使用前一次 PrepareOnly 的目录，仍新建目录。脚本不提供隐式 resume。第一侧返回错误时停止，不运行第二侧。

每次成功手动运行输出：共享配置、`pair_plan.json`（初始计划）、`execution_started.json`（通过加载校验后写入）、两侧原 evaluator 全部产物、`paired_analysis.json`。计划中的 `execution_started=false` 是不可变的初始计划状态，不是实时进度；实际启动看单独标记，完成看两侧完整产物和分析结果。

已有完整配对目录的只读重新分析命令，不启动仿真、不覆写分析文件：

```powershell
& 'D:\anaconda\anaconda\envs\AUV\python.exe' -B -m chapter3_bser.experiments.phase1c_prrac.paired_evaluation analyze --pair-dir 'E:\替换为实际配对目录'
```

## 8. 本轮修改文件清单

以下为本轮修改/新增，不包括上一轮已存在的 Prior-only 配置、单回合 smoke 测试及旧汇总文件。

| 文件（相对工作区） | 修改原因 |
| --- | --- |
| `chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py` | 组合键、缓存、去重与导出识别 controller；保持 trace 数值并标注语义；不再改动已有 controller 选择逻辑 |
| `chapter3_bser/experiments/phase1c_prrac/evaluation_provenance.py` | controller 来源字段、旧缺省兼容与跨文件/恢复身份校验 |
| `chapter3_bser/experiments/phase1c_prrac/evaluation_metrics.py` | 拒绝跨 controller 汇总，保留原指标算法 |
| `chapter3_bser/experiments/phase1c_prrac/paired_evaluation.py` | 显式 checkpoint 的独立配对计划、手动执行入口与只读严格 CSV 分析 |
| `scripts/run_prrac_controller_pair.ps1` | Windows 入口、显式参数、Conda 路径和退出码处理 |
| `scripts/run_prior_only_pair_1.bat` | 1 场配对手动入口 |
| `scripts/run_prior_only_pair_10.bat` | 10 场配对手动入口 |
| `tests/test_prior_only_reliability.py` | controller 身份、缓存、模拟导出、Found 后 trace/adapter、限幅前含义测试 |
| `tests/test_manual_prrac_pair.py` | CSV 布尔/缺失、条件分母、输入一致性、新目录、PrepareOnly 与失败停止测试 |
| `docs/chapter3_bser/prior_only_pairing_readiness_20260911.md` | 本汇总与手动运行说明 |

上一轮文件保留：`configs/chapter3/prior_only_eval.json`、`scripts/run_prior_only_eval.bat`、`tests/test_prior_only_evaluation.py`、`docs/chapter3_bser/prior_only_gpt_handoff_20260911.md`。Git 的总 diff 包含上一轮未提交的核心实现，不能将总 diff 全部归为本轮新改动。

## 9. 可复现验证命令与证据

本轮最终定向测试命令（不包含真实 smoke class）：

```powershell
Set-Location 'E:\gym\code\WORKSPACE\AUV-Thesis'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& 'D:\anaconda\anaconda\envs\AUV\python.exe' -B -m unittest tests.test_prior_only_reliability tests.test_manual_prrac_pair tests.test_prior_only_evaluation.PriorOnlyControllerTests tests.test_prrac_evaluation_metrics tests.test_s2_0_evaluation_provenance tests.test_prrac_evaluation_information_boundary tests.test_execution_residual_suppression tests.test_execution_ablation_evaluator tests.test_repository_metadata tests.test_prrac_checkpoint_evaluator.PRRACCheckpointEvaluatorTests.test_incremental_outputs_and_resume_do_not_repeat_completed_combo
```

实际输出：`Ran 51 tests in 34.025s` / `OK`。测试中的 episode 数据来自 `_ImmediateExecutor` 合成夹具，真实 worker 未调用；临时 fixture checkpoint 只用于代码兼容性测试，不进入配对性能分析。

此外，PowerShell AST 语法检查通过；两个 BAT 都在系统临时目录使用不可加载的路径夹具完成 PrepareOnly，退出码 0。验证目录中仅有计划与配置，没有 controller 评价目录，也没有 execution_started.json。未使用 smoke checkpoint。沙箱内 Conda 启动遇到系统访问拒绝后，在授权工具环境中重复同一 PrepareOnly 检查成功。

`git diff --check` 通过；repository metadata 测试通过，历史 27 条 provenance 记录及其规则未修改。没有运行 git commit/push。

旧 smoke 证据完整保留，SHA256 复核与上一轮记录一致：

| 路径（`runs/prior_only_smoke_20260911/` 下） | SHA256 |
| --- | --- |
| `evaluation/smoke_checks.json` | `9c2e8b401174d83b682a0a32eec634e4879778d8ac008de40989e73cdceb61c2` |
| `evaluation/episode_evaluation.csv` | `f3edee336d62636491c25c4ebbcab01aabc9ff58ead6c9df9c2517ca0a4f5cef` |
| `evaluation/evaluation_summary.json` | `914765553064d89cde34fbd7661b51a101cc6dba497d8464d19ba13cf5552b07` |
| `untrained_test_fixture.pt` | `1f4c45b16bde374765afe19627292ba9c6661a2bc5df6661bbdc826c697281cf` |

旧 smoke 仅证明上一轮 prior_only 的 400 步流程能结束、shape=(4,3)、actor forward=0、residual=0、CSV=1 行。其 found=false、success=false 来自未训练测试权重；不能与 trained PRRAC 性能配对。本轮没有重跑或改写这些产物。

## 10. 交给 GPT 的分析任务

请审查 controller 身份是否在缓存、去重和汇总中闭合；限幅前范数和归一化 command 是否被正确区分；缺失值及条件成功率分母是否符合研究问题。请把本轮夹具结果与未来实际任务结果严格区分。只有用户手动运行并提供两侧原 CSV、manifest、resolved config、checkpoint 来源、pair_plan 和 paired_analysis 后，才讨论 Prior-only 相对 full_prrac 的性能差异；一组或 10 场配对都不能自动等同于正式论文验证完成。
