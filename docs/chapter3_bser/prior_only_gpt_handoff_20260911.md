Prior-only evaluation：交给 GPT 的工程与结果汇总

生成日期：2026-09-11，Asia/Shanghai。

这是供外部 GPT 分析的独立交接文件。后附代码差异、配置、测试源码和结果快照；无需能够访问本机路径即可进行初步审查。本文件只汇总已发生的工作，没有追加 episode、训练或代码修改。

当前结论：Prior-only evaluation 已实现，接口与单 episode smoke 验证通过；尚未进行 Prior-only 与训练后 PRRAC 的配对性能比较。不能据此判断任一控制器更优，也不能宣布第三章正式实验或 Phase 1C 完成。

项目位置：`E:\gym\code\WORKSPACE\AUV-Thesis`。

基准 Git HEAD：`8f7db5103f690c2612385a90d0511c5364ba5489`。实现改动仍在工作区，未 commit、未 push；该 HEAD 不是包含本次改动的新提交。交接范围为下表四个实现/测试文件，加上本汇总文件。

| 文件（相对上述项目根目录） | 改动原因 |
|---|---|
| chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py | 在评估路径选择 controller；零残差分支跳过 actor forward；兼容无 actor 输出的 trace；保存 controller_mode |
| configs/chapter3/prior_only_eval.json | 复制现有 evaluation 配置，仅增加 controller=prior_only |
| scripts/run_prior_only_eval.bat | 复用现有 PowerShell/conda AUV 启动脚本，并指定独立输出目录 |
| tests/test_prior_only_evaluation.py | controller 定向检查与一个真实、使用 spawn worker 的 smoke episode |

没有修改 environment、reward、actor 结构、critic、replay buffer、checkpoint 加载/兼容代码或 planner 决策；没有实现 PVDRL/PVDRLL 或新的 Trust gate；没有删除 PRRAC。28D observation、3D action、124D centralized critic 契约保持原样。未改动或删除用户保留的 outputs 文件；smoke 新产物位于 runs，checkpoint 被现有 Git ignore 规则排除。

本次研究问题是：在相同 scenario、seed、checkpoint 和 runtime 下，保留 prior、移除 residual 是否比保留 PRRAC residual 更好。本次只完成这个对照实验的 evaluation 基础，并未执行对照性能实验。

动作接口必须按当前项目的真实语义理解：

```text
full_prrac:
  public observations -> 原 actor -> gated residual commands
  -> 原 residual-mode adapter -> 原 execution-continuity action adapter
  -> env.step(commands)

prior_only:
  不调用 actor forward
  -> torch.zeros((4, 3), dtype=float32, device=评估设备)
  -> 同一 residual-mode adapter -> 同一 execution-continuity action adapter
  -> env.step(commands)

环境原有行为（未修改）：
  prior_acc = prior_strength * waypoint_prior_acc
  residual_acc = residual_scale * 各轴加速度上限转换(commands)
  final_acc = 按原加速度上限裁剪(prior_acc + residual_acc)
  已完成 agent 的屏蔽行为继续按原实现执行
```

因此不能把 prior 直接当作 env.step 的输入，否则会被现有环境再次解释为 residual。这里“a=a_prior”指使用原环境 prior 控制，并保留原有限幅/屏蔽规则。Actor 网络和测试 checkpoint 仍通过原路径构建、加载，prior_only 只跳过 forward，不绕过 checkpoint 验证。

默认 controller 为 `full_prrac`；新增配置和 CLI 可选 `prior_only`。为避免叠加其他消融或 oracle，prior_only 要求现有 `modes` 为 `["full_prrac"]`。无 actor 输出时，原有 gate/router/alignment 指标记为未观测，而非伪造预测。

新配置与原 `bser_phase1c_prrac_eval.json` 逐项比较，除新增 `controller` 外完全一致，包括空的 checkpoint 列表、scenario 设置、seed、100 episodes、400 max_steps、输出目录等。BAT 在启动时覆盖输出目录为独立的 prior_only_evaluation_v1，避免与原 PRRAC 输出冲突；不是改写 JSON 的其他参数。

实际 smoke 配置如下。episode 数量、workers、checkpoint 和输出位置是测试调用时的覆盖值，生产配置仍保留原值。

| 项目 | 实际值 |
|---|---|
| controller_mode | prior_only |
| evaluation_mode（既有消融字段） | full_prrac |
| profile / split | M20_MOVING_UNKNOWN_MULTI / validation |
| scenario_id / scenario_seed | unknown_validation_m20_0001 / 1729 |
| episode 数量 / 步数上限 | 1 / 400 |
| device / workers | cpu / 1 |
| execution_variant | B0_LEGACY_V2_1 |
| runtime | legacy / dynamic_public_intercept_v2_1 |
| search_recovery_variant | S2A_C0_BASELINE，恢复功能关闭 |
| checkpoint | 新建的 untrained_test_fixture.pt，来自已有测试 helper |
| explore / training_update | false / false |

最新记录的本地验证结果（不是 CI，也不是正式实验）：

| 检查 | 结果与证据边界 |
|---|---|
| 定向回归 | 28 tests，40.323 秒，OK；覆盖新增 controller、既有 evaluation metrics/provenance、replay、repository metadata，以及选定 checkpoint evaluator 检查 |
| PRRAC 默认动作一致性 | 对固定 4 组 28D 输入，新路径与原 `_policy_outputs` + stack 的动作及 actor 输出使用 torch.equal 比较，逐元素完全一致；不是完整双控制器 rollout 的等价实验 |
| 原 checkpoint 兼容规则 | 选定加载、旧 schema 拒绝、维度/架构拒绝检查通过；相关生产函数未修改 |
| 历史 provenance / replay | 相关检查通过，未修改历史 manifest 或弱化验证 |
| BAT 启动 | --help 返回 0，显示 --controller {full_prrac,prior_only}；未因此启动 episode |
| 真实 smoke | 1 test / 1 episode，997.007 秒，OK；真实 spawn 子进程和原环境完整运行 400 步 |
| 动作检查 | 400 次 adapter 调用及 env.step 输入均为 (4,3) 零残差，环境缓存的 physical residual 也为零 |
| Actor forward | 0 次；测试将 PhaseRoutedResidualActor.forward 临时替换为“调用即抛错” |
| 正常终止 | 到达 400 步上限，dones 正常返回；不是任务成功终止 |
| 文件 | episode_evaluation.csv 有 1 行；evaluation_summary.json 中 controller_mode=prior_only |
| 训练与 checkpoint | optimizer/replay sample/parameter update 均为 0；测试 checkpoint 评估前后 SHA256 相同 |

测试中的 adapter/step 观察器会先检查输入，再委托原方法；没有修改物理计算。真实 smoke 只运行了一次。本次生成交接文件时也没有重跑测试或 episode。

从原 episode CSV 提取的主要结果如下（空白值表示不可用/未发生，不能统一替换为 0）：

| 指标 | 值 |
|---|---|
| found / contact_episode / hold_episode / success | false / false / false / false |
| failure_stage | NOT_FOUND |
| episode_length / pre_found_step_count / post_found_step_count | 400 / 400 / 0 |
| episode reward | -20.87266796710901 |
| searcher raw / applied residual norm mean（pre-found） | 0 / 0 |
| search / executor residual contribution ratio（pre-found，旧统计口径） | 0 / 0 |
| collision_episode | true |
| searcher collision-bearing agent-steps | 153，全部属于 agent 0；agent 1、2 为 0 |
| agent 0 最大连续 collision streak | 152 步 |
| 搜索者第一次 / 最后一次 collision step | 246 / 400 |
| searcher_route_active_rate_pre_found | 1.0 |
| searcher assignment missing / unreachable step count | 0 / 0 |
| searcher assignment switch / tracking subgoal switch count | 8 / 45 |
| 已知地图比例 initial / end | 0.22375 / 0.68625 |
| target belief entropy initial / end | 6.683361045672447 / 5.967590228953221 |
| Found 时间、handoff readiness、post-found 条件指标 | 未发生 Found，对应数据不可用 |

153 是逐 agent、逐 transition 的碰撞标志计数，不是 153 次独立碰撞事件。连续碰撞与未发现目标值得后续定位，但仅凭这些汇总不能确定因果，也不能判定 planner、prior、地图更新或其他组件有 bug。route_active=1.0 表示引导任务处于活动状态，不保证沿途实际无碰撞或有有效进展。

必须避免下列误读：

- `controller_mode=prior_only` 是实际控制器选择；既有 `evaluation_mode=full_prrac` 在此表示未叠加其他消融，不意味着调用了 actor。
- `checkpoint_episode=12` 和 `checkpoint_config_hash="test-config-hash"` 来自原有测试 helper 的固定元数据。这个 checkpoint 没有训练 12 个 episode，也不是历史 episode 100/128 checkpoint。
- `recommended_checkpoint` 是旧汇总函数对唯一测试输入给出的机械选择，不是模型质量结论。`performance_passed=null`，没有性能通过证据。
- 原输出 `diagnostic_only=false` 是既有模式标记，不会把本次 smoke 转成正式实验。必须结合 checkpoint_kind 和本文件说明判断。
- router/gate/alignment 空值、全零 router confusion matrix 和 router_stage_counts 不代表整个 episode 没有任务阶段，而是 actor 未运行。
- `searcher_zeroed_step_count=0`、suppressed step count=0 也不否定零残差：这些旧字段统计 adapter 把非零输入改变/抑制的情况；prior_only 在源头就生成零值。实际零残差由 smoke 逐步断言确认。
- `smoke_checks.json` 的 found/success 来自 CSV 字符串，因此原文件保存为字符串 "False"。不能在 Python 中用 bool("False") 解析；应显式比较或使用类型转换。
- failure_trace.jsonl 为 0 字节是符合当前筛选条件的：配置 only_found_failures=true，而本 episode found=false。因此没有可用于重建这次持续碰撞的完整逐步轨迹。
- 这里只覆盖未发现目标的 Search 情形，没有实际覆盖 Found 后的 Intercept/Hold，也没有运行完整的 full_prrac 对照 episode。
- 未训练 checkpoint 不影响这个零 residual 分支的 actor 输出，但不能拿该夹具的 PRRAC 输出代表训练后 PRRAC 性能。只有一个场景，不能给出优劣结论或统计显著性。

已执行的 smoke 命令如下；若要再次留存结果，必须将 PRIOR_ONLY_SMOKE_OUTPUT 改为新的目录，不能覆盖本次证据：

```powershell
Set-Location 'E:\gym\code\WORKSPACE\AUV-Thesis'
$env:PRIOR_ONLY_SMOKE_OUTPUT = 'E:\gym\code\WORKSPACE\AUV-Thesis\runs\prior_only_smoke_20260911'
& 'D:\anaconda\anaconda\envs\AUV\python.exe' -B -m unittest tests.test_prior_only_evaluation.PriorOnlySmokeTests
```

后续使用真实、兼容的 PRRAC checkpoint 的示例命令（尚未执行；替换占位路径）：

```powershell
& 'E:\gym\code\WORKSPACE\AUV-Thesis\scripts\run_prior_only_eval.bat' --checkpoint '真实PRRAC checkpoint完整路径.pt' --episodes 1 --workers 1
```

原 JSON checkpoint 列表为空，因此不能省略实际 checkpoint 参数就直接评估。后续配对实验应固定同一 checkpoint、scenario manifest、seed、runtime、episode 数量和步数上限，只切换 controller 并使用分离的输出目录；新实验需要由用户明确选择执行。本文件不授权训练或启动批量实验。

请接收本文件的 GPT 分别回答下列问题，并明确区分代码事实、观测事实和推测：

1. 从附录 diff 和测试源码看，prior_only 是否满足“无 actor forward、零 residual、进入原 adapter、env.step 接口不变”？是否还有未覆盖的接口或日志问题？
2. `controller_mode` 与旧 `evaluation_mode`、fixture checkpoint 元数据是否足以造成后续分析误读？给出必要的解读约定，避免扩大实现范围。
3. 单场景未发现目标、agent 0 持续碰撞可以支持哪些结论，不能支持哪些结论？请不要把相关性直接解释为原因。
4. 在保留 environment、reward、actor/critic、checkpoint/replay 契约且不训练的前提下，下一轮最小配对评价应固定哪些变量、报告哪些既有指标？请仅提出方案，不假设它已获授权或已执行。
5. 请给出“实现验收”“smoke 证据”“性能证据不足”“需补充证据”四类结论。不要宣称 Prior-only 优于/劣于 PRRAC，也不要宣称 Phase 1C/正式论文实验完成。

以下为自动读取的文件快照。原 episode CSV 的完整一行在附录中以字符串 JSON 保留；字段值、空白值与原 CSV 一致。源码 diff 基于上述 HEAD，新增文件另附完整内容。本文件只嵌入文本与哈希，不包含 checkpoint 权重。

附录 A：证据文件 SHA256（从现有文件读取）

| 文件 | SHA256 |
|---|---|
| smoke_checks.json | 9c2e8b401174d83b682a0a32eec634e4879778d8ac008de40989e73cdceb61c2 |
| episode_evaluation.csv | f3edee336d62636491c25c4ebbcab01aabc9ff58ead6c9df9c2517ca0a4f5cef |
| evaluation_summary.json | 914765553064d89cde34fbd7661b51a101cc6dba497d8464d19ba13cf5552b07 |
| resolved_evaluation_config.json | 76d1897c1785c4fc784a55076a36ef02260b009cda6acf73c7cb29d1837a4ec7 |
| evaluation_manifest.json | 49a9515d77eaa26f51c3b088d5424e8c7535c79a31047d82bd24bc35a254661a |
| untrained_test_fixture.pt（权重未附） | 1f4c45b16bde374765afe19627292ba9c6661a2bc5df6661bbdc826c697281cf |

附录 B：evaluation 生产代码相对基准 HEAD 的完整 diff

```diff
diff --git a/chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py b/chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py
index 3920ed7..7422eb0 100644
--- a/chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py
+++ b/chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py
@@ -149,6 +149,7 @@ SUPPORTED_MODES = (
     "all_residual_off",
     "oracle_current_target_diagnostic",
 )
+CONTROLLER_MODES = ("full_prrac", "prior_only")
 OUTPUT_FILES = (
     "resolved_evaluation_config.json",
     "evaluation_manifest.json",
@@ -302,6 +303,7 @@ def _contains_tensor(payload: Any) -> bool:
 
 def _load_config(path: Path) -> dict[str, Any]:
     config = json.loads(Path(path).read_text(encoding="utf-8"))
+    _controller_mode(config)
     expected = {
         "method": METHOD,
         "implementation_version": IMPLEMENTATION_VERSION,
@@ -743,6 +745,27 @@ def _policy_outputs(
     return outputs
 
 
+def _controller_mode(config: Mapping[str, Any]) -> str:
+    mode = str(config.get("controller", "full_prrac"))
+    if mode not in CONTROLLER_MODES:
+        raise ValueError(f"unsupported evaluation controller: {mode!r}")
+    return mode
+
+
+def _controller_actions(actor, observations, device, controller_mode):
+    """Select residual commands; the existing environment owns the prior."""
+    mode = _controller_mode({"controller": controller_mode})
+    if mode == "prior_only":
+        if len(observations) != 4:
+            raise ValueError("PRRAC evaluation requires four public observations")
+        return [], torch.zeros((4, 3), dtype=torch.float32, device=device)
+    outputs = _policy_outputs(actor, observations, device)
+    actions = torch.stack(
+        [output.gated_residual_action.squeeze(0) for output in outputs]
+    )
+    return outputs, actions
+
+
 def _event_names(result: Any) -> list[str]:
     return [
         str(getattr(event, "value", event)).upper()
@@ -791,11 +814,15 @@ def _trace_step(
     prior = getattr(runtime, "_last_prior_acc", None)
     residual = getattr(runtime, "_last_residual_acc", None)
     final = None if prior is None or residual is None else prior[3] + residual[3]
-    actor_output = actor_outputs[3]
-    probabilities = actor_output.router_probabilities.detach().cpu().reshape(-1, 3)[0]
+    actor_output = actor_outputs[3] if actor_outputs else None
+    probabilities = (
+        None if actor_output is None
+        else actor_output.router_probabilities.detach().cpu().reshape(-1, 3)[0]
+    )
     empty = {agent_id: None for agent_id in range(3)}
     modes = empty if recovery_snapshot is None else recovery_snapshot.mode_by_agent
     return failure_trace_row(
+        controller_mode=str(info.get("controller_mode", "full_prrac")),
         checkpoint_episode=int(info["checkpoint_episode"]),
         execution_variant=str(info.get("execution_variant", "")),
         search_recovery_variant=str(info.get("search_recovery_variant", "")),
@@ -846,12 +873,12 @@ def _trace_step(
         navigation_prior_norm=(None if prior is None else float(torch.linalg.vector_norm(prior[3]).item())),
         residual_action_norm=float(torch.linalg.vector_norm(applied_actions[3]).item()),
         final_action_norm=(None if final is None else float(torch.linalg.vector_norm(final).item())),
-        trust_gate=float(actor_output.trust_gate.detach().cpu().reshape(-1)[0].item()),
-        alignment_cosine=float(actor_output.alignment_cosine.detach().cpu().reshape(-1)[0].item()),
-        router_probability_search=float(probabilities[0].item()),
-        router_probability_intercept=float(probabilities[1].item()),
-        router_probability_hold=float(probabilities[2].item()),
-        router_prediction=int(probabilities.argmax().item()),
+        trust_gate=None if actor_output is None else float(actor_output.trust_gate.detach().cpu().reshape(-1)[0].item()),
+        alignment_cosine=None if actor_output is None else float(actor_output.alignment_cosine.detach().cpu().reshape(-1)[0].item()),
+        router_probability_search=None if probabilities is None else float(probabilities[0].item()),
+        router_probability_intercept=None if probabilities is None else float(probabilities[1].item()),
+        router_probability_hold=None if probabilities is None else float(probabilities[2].item()),
+        router_prediction=None if probabilities is None else int(probabilities.argmax().item()),
         collision=bool(torch.as_tensor(getattr(runtime, "_collision_flags", False)).any().item()),
         installed_guidance_privileged=bool(
             installed_guidance.decision_reason
@@ -884,6 +911,7 @@ def _evaluate_episode_job(job: dict[str, Any], *, audit=None, searcher_trace=Non
     _seed_all(int(scenario["scenario_seed"]))
     config = copy.deepcopy(dict(job["config"]))
     info = dict(job["checkpoint_info"])
+    info["controller_mode"] = _controller_mode(config)
     mode = str(info["evaluation_mode"])
     execution_variant = parse_execution_variant(
         info.get("execution_variant", ExecutionVariant.B0_LEGACY_V2_1.value)
@@ -1000,12 +1028,11 @@ def _evaluate_episode_job(job: dict[str, Any], *, audit=None, searcher_trace=Non
             if audit is not None:
                 audit.before_action(state, observations, installed_guidance)
             with torch.no_grad():
-                outputs = _policy_outputs(actor, observations, device)
+                outputs, residual_actions = _controller_actions(
+                    actor, observations, device, info["controller_mode"]
+                )
                 for output in outputs:
                     diagnostics.observe_actor(output, [int(current_stage)])
-                residual_actions = torch.stack(
-                    [output.gated_residual_action.squeeze(0) for output in outputs]
-                )
                 mode_actions = _apply_residual_mode(
                     residual_actions, mode, current_stage
                 )
@@ -1611,8 +1638,12 @@ def _evaluation_summary(
     output: Path,
 ) -> dict[str, Any]:
     provenance = derive_unique_provenance(summary_rows)
+    controller_modes = {row.get("controller_mode", "full_prrac") for row in summary_rows}
+    if len(controller_modes) > 1:
+        raise ValueError("evaluation summary cannot mix controller modes")
     return {
         "schema": SUMMARY_SCHEMA,
+        "controller_mode": next(iter(controller_modes), "full_prrac"),
         **provenance,
         "method": METHOD,
         "implementation_version": IMPLEMENTATION_VERSION,
@@ -1929,6 +1960,7 @@ def run_evaluation(
     scenario_seed_override: int | None = None,
     workers_override: int | None = None,
     device_override: str | None = None,
+    controller_override: str | None = None,
     modes_override: Iterable[str] | None = None,
     execution_variants_override: Iterable[str] | None = None,
     search_recovery_variants_override: Iterable[str] | None = None,
@@ -1941,6 +1973,9 @@ def run_evaluation(
     assert_registered_ch3_method(METHOD)
     requested_checkpoints = tuple(checkpoints or ())
     config = copy.deepcopy(_load_config(config_path))
+    if controller_override is not None:
+        config["controller"] = controller_override
+    controller_mode = _controller_mode(config)
     if beds_variant_override is not None:
         from chapter3_bser.experiments.phase1c_prrac.beds import resolve_beds, validate_beds
         switches = {"baseline": (False, False), "early_only": (True, False),
@@ -1974,6 +2009,8 @@ def run_evaluation(
     if not modes or any(mode not in SUPPORTED_MODES for mode in modes):
         raise ValueError(f"unsupported PRRAC evaluation modes: {modes}")
     config["modes"] = list(modes)
+    if controller_mode == "prior_only" and modes != ("full_prrac",):
+        raise ValueError("prior_only requires modes=['full_prrac']")
     execution_variants = tuple(
         parse_execution_variant(value)
         for value in (
@@ -2242,6 +2279,7 @@ def run_evaluation(
                         search_recovery_config_hash=search_recovery_hash,
                         search_recovery_schema=search_recovery_schema,
                     )
+                    info["controller_mode"] = controller_mode
                     if bool(config.get("diagnostic_only", False)):
                         info["diagnostic_only"] = True
                     combo = _combo_key(info, manifest_hash)
@@ -2361,6 +2399,7 @@ def main(argv: list[str] | None = None) -> int:
     parser.add_argument("--scenario-seed", type=int)
     parser.add_argument("--workers", type=int)
     parser.add_argument("--device")
+    parser.add_argument("--controller", choices=CONTROLLER_MODES)
     parser.add_argument("--modes", nargs="+")
     parser.add_argument(
         "--execution-variants",
@@ -2388,6 +2427,7 @@ def main(argv: list[str] | None = None) -> int:
         scenario_seed_override=args.scenario_seed,
         workers_override=args.workers,
         device_override=args.device,
+        controller_override=args.controller,
         modes_override=args.modes,
         execution_variants_override=args.execution_variants,
         search_recovery_variants_override=args.search_recovery_variants,
```

源码快照：configs/chapter3/prior_only_eval.json（最后一个 helper 为既有文件，未修改）

```json
{
  "schema": "bser.phase1c.prrac.evaluation.v1",
  "method": "ch3_bser_rmaddpg_phase1c",
  "implementation_version": "bser.phase1c.prrac_v1",
  "architecture_version": "prrac.phase_routed_residual.v1",
  "checkpoint_schema": "bser.phase1c.prrac.training_state.v1",
  "base_candidate": "ch3_v3_full_reference",
  "profile": "M20_MOVING_UNKNOWN_MULTI",
  "observation_dim": 28,
  "action_dim": 3,
  "critic_dim": 124,
  "split": "validation",
  "scenario_seed": 1729,
  "evaluation_episodes": 100,
  "max_steps": 400,
  "explore": false,
  "training_update": false,
  "device": "cpu",
  "workers": 4,
  "modes": [
    "full_prrac"
  ],
  "execution_variants": [
    "B0_LEGACY_V2_1"
  ],
  "checkpoints": [],
  "checkpoint_globs": [],
  "execution_runtime_revision": "dynamic_public_intercept_v2_1",
  "execution_runtime": {
    "defer_stale_endpoint_invalid": true,
    "dynamic_public_target_enabled": true,
    "public_target_update_distance": 0.75,
    "public_target_update_min_steps": 20,
    "refresh_on_executor_handoff": true,
    "refresh_on_public_target_shift": true
  },
  "failure_trace": {
    "enabled": true,
    "only_found_failures": true,
    "max_traces_per_checkpoint_mode": 20
  },
  "output_dir": "outputs/chapter3/phase1c_prrac/evaluation_v1",
  "controller": "prior_only"
}

```

源码快照：scripts/run_prior_only_eval.bat（最后一个 helper 为既有文件，未修改）

```bat
@echo off
setlocal
rem Keep the paired JSON configuration identical; separate output at launch.
call "%~dp0run_phase1c_prrac_eval.bat" --config "%~dp0..\configs\chapter3\prior_only_eval.json" --output-dir "%~dp0..\outputs\chapter3\phase1c_prrac\prior_only_evaluation_v1" %*
exit /b %ERRORLEVEL%

```

源码快照：tests/test_prior_only_evaluation.py（最后一个 helper 为既有文件，未修改）

```python
"""Controller checks and exactly one real evaluation episode, with no training.

Set PRIOR_ONLY_SMOKE_OUTPUT to a new directory to retain smoke artifacts.
The checkpoint is a fresh, untrained test fixture, never an experiment result.
"""

from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.models.prrac.phase_routed_actor import PhaseRoutedResidualActor
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG
from tests.prrac_evaluation_support import ARCHITECTURE, LOSS, write_checkpoint


PRIOR_CONFIG = evaluator.ROOT / "configs/chapter3/prior_only_eval.json"
ORIGINAL_EPISODE_JOB = evaluator._evaluate_episode_job


def _checked_episode_job(job):
    """Spawn-safe observer around the real worker; all physics calls delegate."""
    original_step = evaluator.PRRACTrainingEnv.step
    original_adapter = evaluator.ExecutionContinuityActionAdapter.apply
    steps, adapter_calls, terminal = 0, 0, False

    def checked_adapter(self, actions, **kwargs):
        nonlocal adapter_calls
        if tuple(actions.shape) != (4, 3) or torch.count_nonzero(actions).item():
            raise AssertionError("prior_only must pass (4,3) zero residuals to the adapter")
        adapter_calls += 1
        return original_adapter(self, actions, **kwargs)

    def checked_step(self, actions):
        nonlocal steps, terminal
        if tuple(actions.shape) != (4, 3) or torch.count_nonzero(actions).item():
            raise AssertionError("env.step must receive (4,3) zero residuals")
        result = original_step(self, actions)
        if torch.count_nonzero(self.unwrapped._last_residual_acc).item():
            raise AssertionError("physical residual acceleration must remain zero")
        steps += 1
        terminal = all(bool(value) for value in result[2])
        return result

    with (
        mock.patch.object(PhaseRoutedResidualActor, "forward", side_effect=AssertionError("actor forward is forbidden")) as forward,
        mock.patch.object(evaluator.ExecutionContinuityActionAdapter, "apply", checked_adapter),
        mock.patch.object(evaluator.PRRACTrainingEnv, "step", checked_step),
        mock.patch.object(evaluator, "_trace_step", wraps=evaluator._trace_step) as trace,
    ):
        result = ORIGINAL_EPISODE_JOB(job)
        forward.assert_not_called()
        if not terminal or steps < 1 or adapter_calls != steps:
            raise AssertionError("episode must terminate normally through the existing adapter")
        if trace.call_count != steps or any(call.kwargs["actor_outputs"] for call in trace.call_args_list):
            raise AssertionError("trace must accept absent actor outputs on every step")
    checks = {
        "controller_mode": "prior_only",
        "episodes": 1,
        "steps": steps,
        "action_shape": [4, 3],
        "max_abs_residual": 0.0,
        "physical_residual_zero": True,
        "actor_forward_calls": 0,
        "adapter_calls": adapter_calls,
        "episode_ended": terminal,
        "checkpoint_kind": "untrained_test_fixture",
    }
    Path(job["config"]["output_dir"], "smoke_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    return result


class PriorOnlyControllerTests(unittest.TestCase):
    def test_config_diff_is_only_controller(self):
        baseline = json.loads(evaluator.DEFAULT_CONFIG.read_text(encoding="utf-8"))
        prior = json.loads(PRIOR_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(prior.pop("controller"), "prior_only")
        self.assertEqual(prior, baseline)
        self.assertEqual(evaluator._controller_mode(baseline), "full_prrac")

    def test_prior_skips_policy_and_preserves_adapter_input(self):
        observations = [torch.zeros(28) for _ in range(4)]
        with mock.patch.object(evaluator, "_policy_outputs", side_effect=AssertionError("no policy forward")):
            outputs, actions = evaluator._controller_actions(None, observations, torch.device("cpu"), "prior_only")
        self.assertEqual(outputs, [])
        self.assertEqual(tuple(actions.shape), (4, 3))
        self.assertEqual(actions.dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(actions).item(), 0)

    def test_full_prrac_actions_match_original_path_exactly(self):
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            learner = PRRACMADDPG(architecture=ARCHITECTURE, loss=LOSS)
            learner.prep_rollouts(torch.device("cpu"))
            observations = [torch.randn(28) for _ in range(4)]
            with torch.no_grad():
                original = evaluator._policy_outputs(learner, observations, torch.device("cpu"))
                expected = torch.stack([item.gated_residual_action.squeeze(0) for item in original])
                outputs, actual = evaluator._controller_actions(learner, observations, torch.device("cpu"), "full_prrac")
            self.assertTrue(torch.equal(actual, expected))
            for old, new in zip(original, outputs):
                for old_tensor, new_tensor in zip(old, new):
                    self.assertTrue(torch.equal(old_tensor, new_tensor))

    def test_unknown_controller_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported evaluation controller"):
            evaluator._controller_mode({"controller": "pvdrll"})

    def test_prior_cannot_enable_oracle_ablation(self):
        with self.assertRaisesRegex(ValueError, "prior_only requires"):
            evaluator.run_evaluation(config_path=PRIOR_CONFIG, modes_override=["oracle_current_target_diagnostic"])

    def test_summary_rejects_mixed_controller_rows(self):
        with mock.patch.object(evaluator, "derive_unique_provenance", return_value={}):
            with self.assertRaisesRegex(ValueError, "cannot mix controller"):
                evaluator._evaluation_summary(
                    checkpoint_paths=[], scenarios=[], modes=("full_prrac",), execution_variants=(),
                    summary_rows=[{"controller_mode": "prior_only"}, {"controller_mode": "full_prrac"}],
                    output=Path("unused"),
                )


class PriorOnlySmokeTests(unittest.TestCase):
    def test_one_episode_csv_and_summary(self):
        retained = os.environ.get("PRIOR_ONLY_SMOKE_OUTPUT")
        context = nullcontext(retained) if retained else tempfile.TemporaryDirectory()
        with context as directory:
            root = Path(directory).resolve()
            if retained:
                root.mkdir(parents=True, exist_ok=False)
            checkpoint = write_checkpoint(root / "untrained_test_fixture.pt")
            checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            output = root / "evaluation"
            with mock.patch.object(evaluator, "_evaluate_episode_job", _checked_episode_job):
                summary = evaluator.run_evaluation(
                    config_path=PRIOR_CONFIG,
                    checkpoints=[checkpoint],
                    output_dir=output,
                    episodes_override=1,
                    workers_override=1,
                )
            with (output / "episode_evaluation.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["controller_mode"], "prior_only")
            self.assertGreater(int(rows[0]["episode_length"]), 0)
            self.assertEqual(summary["controller_mode"], "prior_only")
            self.assertEqual(summary["scenario_count"], 1)
            self.assertEqual(summary["optimizer_update_count"], 0)
            self.assertEqual(summary["replay_sample_count"], 0)
            self.assertEqual(summary["parameter_update_count"], 0)
            saved = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["controller_mode"], "prior_only")
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checkpoint_hash)
            checks = json.loads((output / "smoke_checks.json").read_text(encoding="utf-8"))
            self.assertEqual(checks["steps"], int(rows[0]["episode_length"]))
            checks.update(csv_rows=1, csv_generated=True, checkpoint_unchanged=True,
                          success=rows[0]["success"], found=rows[0]["found"])
            (output / "smoke_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
            print(json.dumps({"smoke": checks, "artifacts": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()

```

源码快照：tests/prrac_evaluation_support.py（最后一个 helper 为既有文件，未修改）

```python
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch

from chapter3_bser.experiments.phase1c_prrac import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_SCHEMA,
    IMPLEMENTATION_VERSION,
    METHOD,
)
from chapter3_bser.experiments.phase1c_prrac import evaluate_prrac_checkpoints as evaluator
from chapter3_bser.models.prrac.prrac_maddpg import PRRACMADDPG


ARCHITECTURE = {
    "num_stages": 3,
    "encoder_hidden_dim": 8,
    "expert_hidden_dim": 8,
    "critic_hidden_dim": 16,
    "router_temperature": 1.0,
    "gate_initial_mean": 0.75,
    "alignment_scale_init": 1.0,
}
LOSS = {
    "router_ce_coef": 0.05,
    "gate_conflict_coef": 0.01,
    "gate_entropy_coef": 0.001,
    "residual_action_reg": 0.01,
}


def evaluation_config(*, scenario_count: int = 2) -> dict[str, Any]:
    config = copy.deepcopy(evaluator._load_config(evaluator.DEFAULT_CONFIG))
    config["evaluation_episodes"] = int(scenario_count)
    config["max_steps"] = 1
    config["workers"] = 2
    config["failure_trace"]["enabled"] = False
    return config


def checkpoint_payload() -> dict[str, Any]:
    training_config = json.loads(
        (evaluator.ROOT / "configs/chapter3/bser_phase1c_prrac_train.json").read_text(
            encoding="utf-8"
        )
    )
    learner = PRRACMADDPG(
        architecture=ARCHITECTURE,
        loss=LOSS,
        gamma=0.95,
        tau=0.005,
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "metadata": {
            "method": METHOD,
            "implementation_version": IMPLEMENTATION_VERSION,
            "architecture_version": ARCHITECTURE_VERSION,
            "config_hash": "test-config-hash",
            "completed_episode": 12,
            "observation_dim": 28,
            "action_dim": 3,
            "critic_dim": 124,
            "architecture": copy.deepcopy(ARCHITECTURE),
            "loss": copy.deepcopy(LOSS),
            "reward": copy.deepcopy(training_config["reward"]),
            "execution_runtime_revision": "dynamic_public_intercept_v2_1",
        },
        "prrac_training_state": learner.training_state_dict(),
        "completed_episode": 12,
    }


def write_checkpoint(path: Path, payload: dict[str, Any] | None = None) -> Path:
    torch.save(checkpoint_payload() if payload is None else payload, path)
    return path


def worker_jobs(path: Path, count: int = 2) -> list[dict[str, Any]]:
    config = evaluation_config(scenario_count=count)
    learner, payload = evaluator.load_prrac_checkpoint(path, config=config)
    scenarios, _ = evaluator._build_evaluation_manifest(config)
    metadata = dict(payload["metadata"])
    state = dict(payload["prrac_training_state"])
    info = evaluator._checkpoint_info(path, payload, "full_prrac")
    snapshot = learner.policy_snapshot()
    trace = {
        "enabled": False,
        "only_found_failures": True,
        "max_traces": 0,
    }
    return [
        {
            "episode_index": index,
            "scenario": scenarios[index],
            "config": config,
            "checkpoint_info": info,
            "architecture": metadata["architecture"],
            "loss": metadata["loss"],
            "gamma": state["gamma"],
            "tau": state["tau"],
            "reward": metadata["reward"],
            "policy_snapshot": snapshot,
            "failure_trace": trace,
            "device": "cpu",
        }
        for index in range(count)
    ]

```

结果快照：smoke_checks.json

```json
{
  "controller_mode": "prior_only",
  "episodes": 1,
  "steps": 400,
  "action_shape": [
    4,
    3
  ],
  "max_abs_residual": 0.0,
  "physical_residual_zero": true,
  "actor_forward_calls": 0,
  "adapter_calls": 400,
  "episode_ended": true,
  "checkpoint_kind": "untrained_test_fixture",
  "csv_rows": 1,
  "csv_generated": true,
  "checkpoint_unchanged": true,
  "success": "False",
  "found": "False"
}
```

结果快照：evaluation_summary.json

```json
{
  "activation_diagnostics_schema": "",
  "activation_diagnostics_schema_values": [
    ""
  ],
  "activation_steps_output": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\evaluation\\search_collision_recovery_activation_steps.csv",
  "architecture_version": "prrac.phase_routed_residual.v1",
  "checkpoint": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\untrained_test_fixture.pt",
  "checkpoint_config_hash": "test-config-hash",
  "checkpoint_config_hash_values": [
    "test-config-hash"
  ],
  "checkpoint_count": 1,
  "checkpoint_episode": 12,
  "checkpoint_episode_values": [
    12
  ],
  "checkpoint_runtime_revision": "dynamic_public_intercept_v2_1",
  "checkpoint_runtime_revision_values": [
    "dynamic_public_intercept_v2_1"
  ],
  "checkpoint_schema": "bser.phase1c.prrac.training_state.v1",
  "checkpoint_values": [
    "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\untrained_test_fixture.pt"
  ],
  "controller_mode": "prior_only",
  "evaluation_mode": "full_prrac",
  "evaluation_mode_values": [
    "full_prrac"
  ],
  "evaluation_modes": [
    "full_prrac"
  ],
  "evaluation_runtime_revision": "dynamic_public_intercept_v2_1",
  "evaluation_runtime_revision_values": [
    "dynamic_public_intercept_v2_1"
  ],
  "execution_variant": "B0_LEGACY_V2_1",
  "execution_variant_values": [
    "B0_LEGACY_V2_1"
  ],
  "execution_variants": [
    "B0_LEGACY_V2_1"
  ],
  "explore": false,
  "implementation_version": "bser.phase1c.prrac_v1",
  "manifest_sha256": "69d1838177568c8abacac08cf22852c8e51b52e265807434d64830ad3c49daa6",
  "manifest_sha256_values": [
    "69d1838177568c8abacac08cf22852c8e51b52e265807434d64830ad3c49daa6"
  ],
  "method": "ch3_bser_rmaddpg_phase1c",
  "optimizer_update_count": 0,
  "output_dir": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\evaluation",
  "parameter_update_count": 0,
  "performance_passed": null,
  "recommended_checkpoint": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\untrained_test_fixture.pt",
  "recommended_checkpoint_episode": 12,
  "replay_sample_count": 0,
  "report_schema": "bser.phase1c.prrac.evaluation_report.v2",
  "report_schema_values": [
    "bser.phase1c.prrac.evaluation_report.v2"
  ],
  "resolved_evaluation_episodes": 1,
  "runtime_integration_mode": "legacy",
  "runtime_integration_mode_values": [
    "legacy"
  ],
  "s2a1_activation_artifact_revision": "s2a1.activation_artifact.v1",
  "same_scenarios_for_all_checkpoints": true,
  "scenario_count": 1,
  "schema": "bser.phase1c.prrac.evaluation_summary.v2",
  "search_collision_recovery_config_hash": "28b7faa08a087ea1876fb0ef5708ff469947c29f7ec9971f38b3540581f6fd8b",
  "search_collision_recovery_config_hash_values": [
    "28b7faa08a087ea1876fb0ef5708ff469947c29f7ec9971f38b3540581f6fd8b"
  ],
  "search_collision_recovery_schema": "bser.phase1c.prrac.search_collision_recovery.v1",
  "search_collision_recovery_schema_values": [
    "bser.phase1c.prrac.search_collision_recovery.v1"
  ],
  "search_continuity_diagnostics_hash": "d7d784eb3299ecf84a09b1faf513cd89add0c0f02375644e8b79bb9e4fb56e3c",
  "search_continuity_diagnostics_hash_values": [
    "d7d784eb3299ecf84a09b1faf513cd89add0c0f02375644e8b79bb9e4fb56e3c"
  ],
  "search_recovery_variant": "S2A_C0_BASELINE",
  "search_recovery_variant_values": [
    "S2A_C0_BASELINE"
  ],
  "search_recovery_variants": [
    "S2A_C0_BASELINE"
  ],
  "selection_rule": "success_rate desc; success_if_found_rate desc; contact_if_found_rate desc; collision_episode_rate asc; mean_assignment_unreachable_if_found asc; checkpoint_episode asc",
  "training_update": false
}

```

结果快照：resolved_evaluation_config.json

```json
{
  "action_dim": 3,
  "activation_artifact_revision": "s2a1.activation_artifact.v1",
  "activation_diagnostics_schema": "",
  "architecture_version": "prrac.phase_routed_residual.v1",
  "base_candidate": "ch3_v3_full_reference",
  "checkpoint_globs": [],
  "checkpoint_runtime_revision": "dynamic_public_intercept_v2_1",
  "checkpoint_schema": "bser.phase1c.prrac.training_state.v1",
  "checkpoints": [
    "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\untrained_test_fixture.pt"
  ],
  "controller": "prior_only",
  "controller_factory_version": "prrac.controller_factory.v1",
  "critic_dim": 124,
  "device": "cpu",
  "evaluation_episodes": 1,
  "evaluation_runtime_revision": "dynamic_public_intercept_v2_1",
  "execution_continuity": {
    "checkpoint_runtime_revision": "dynamic_public_intercept_v2_1",
    "evaluation_runtime_revision": "dynamic_public_intercept_v2_1",
    "executor_cost_increase_threshold": 0.15,
    "public_target_update_distance": 0.75,
    "public_target_update_min_steps": 20,
    "schema": "bser.phase1c.prrac.execution_continuity.v1",
    "state_refresh_interval": 20
  },
  "execution_overlay_config_hash": "8ed1e291f69bcc117fb028aa786cc813a7add9d1d3ea0c30bef71c71f5563f42",
  "execution_runtime": {
    "defer_stale_endpoint_invalid": true,
    "dynamic_public_target_enabled": true,
    "public_target_update_distance": 0.75,
    "public_target_update_min_steps": 20,
    "refresh_on_executor_handoff": true,
    "refresh_on_public_target_shift": true
  },
  "execution_runtime_revision": "dynamic_public_intercept_v2_1",
  "execution_variants": [
    "B0_LEGACY_V2_1"
  ],
  "explore": false,
  "failure_trace": {
    "enabled": true,
    "max_traces_per_checkpoint_mode": 20,
    "only_found_failures": true
  },
  "generated_scenario_count": 1,
  "implementation_version": "bser.phase1c.prrac_v1",
  "manifest_sha256": "69d1838177568c8abacac08cf22852c8e51b52e265807434d64830ad3c49daa6",
  "max_steps": 400,
  "method": "ch3_bser_rmaddpg_phase1c",
  "modes": [
    "full_prrac"
  ],
  "observation_dim": 28,
  "output_dir": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\evaluation",
  "profile": "M20_MOVING_UNKNOWN_MULTI",
  "report_schema": "bser.phase1c.prrac.evaluation_report.v2",
  "requested_checkpoint_arguments": [
    "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\untrained_test_fixture.pt"
  ],
  "requested_config_path": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\configs\\chapter3\\prior_only_eval.json",
  "requested_evaluation_episodes": 1,
  "resolved_checkpoint_paths": [
    "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\untrained_test_fixture.pt"
  ],
  "resolved_config_hash": "ae54b9e7148dcfc86720f124d8b7344e7e908ba8b85658af9155c6a16e079a7c",
  "resolved_config_output_path": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\evaluation\\resolved_evaluation_config.json",
  "resolved_config_path": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\configs\\chapter3\\prior_only_eval.json",
  "resolved_device": "cpu",
  "resolved_evaluation_episodes": 1,
  "resolved_evaluation_modes": [
    "full_prrac"
  ],
  "resolved_evaluation_runtime_revisions": [
    "dynamic_public_intercept_v2_1"
  ],
  "resolved_execution_variants": [
    "B0_LEGACY_V2_1"
  ],
  "resolved_output_dir": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\evaluation",
  "resolved_runtime_integration_modes": [
    "legacy"
  ],
  "resolved_scenario_ids": [
    "unknown_validation_m20_0001"
  ],
  "resolved_scenario_seed": 1729,
  "resolved_search_recovery_variants": [
    "S2A_C0_BASELINE"
  ],
  "resolved_workers": 1,
  "runtime_integration_mode": "legacy",
  "s2a1_activation_artifact_revision": "s2a1.activation_artifact.v1",
  "scenario_seed": 1729,
  "schema": "bser.phase1c.prrac.evaluation_report.v2",
  "search_collision_recovery": {
    "c2_escalation": "collision_after_route_refresh_or_refresh_unreachable",
    "collision_rearm": "one_collision_free_search_transition",
    "egress_candidate_policy": "nearest_hop_known_free_v1",
    "enabled": false,
    "modify_actions": false,
    "modify_executor": false,
    "reuse_project_path_tracker_threshold": true,
    "schema": "bser.phase1c.prrac.search_collision_recovery.v1",
    "trigger": "collision_edge",
    "variants": [
      "S2A_C0_BASELINE"
    ]
  },
  "search_collision_recovery_config_hash": "28b7faa08a087ea1876fb0ef5708ff469947c29f7ec9971f38b3540581f6fd8b",
  "search_collision_recovery_schema": "bser.phase1c.prrac.search_collision_recovery.v1",
  "search_continuity_diagnostics": {
    "enabled": true,
    "schema": "bser.phase1c.prrac.search_continuity.v2"
  },
  "search_continuity_diagnostics_hash": "d7d784eb3299ecf84a09b1faf513cd89add0c0f02375644e8b79bb9e4fb56e3c",
  "search_continuity_diagnostics_schema": "bser.phase1c.prrac.search_continuity.v2",
  "search_recovery_variants": [
    "S2A_C0_BASELINE"
  ],
  "selected_scenario_count": 1,
  "split": "validation",
  "training_update": false,
  "workers": 1
}

```

结果快照：episode_evaluation.csv 完整一行，保留 CSV 字符串类型

```json
{
  "activation_artifact_revision": "s2a1.activation_artifact.v1",
  "activation_diagnostics_schema": "",
  "adjusted_episode_reward": "-20.87266796710901",
  "alignment_mean": "",
  "alignment_negative_rate": "",
  "assignment_unreachable_count_post_found": "0",
  "assignment_unreachable_rate_post_found": "",
  "candidate_tier_distribution": "{\"tier0\":0,\"tier1\":0,\"tier2\":0,\"tier3\":0}",
  "capture_contact_step_count": "0",
  "capture_full_hold_step_count": "0",
  "capture_hold_counter_max": "0",
  "capture_swept_min_distance": "",
  "checkpoint": "E:\\gym\\code\\WORKSPACE\\AUV-Thesis\\runs\\prior_only_smoke_20260911\\untrained_test_fixture.pt",
  "checkpoint_config_hash": "test-config-hash",
  "checkpoint_episode": "12",
  "checkpoint_runtime_revision": "dynamic_public_intercept_v2_1",
  "checkpoint_schema": "bser.phase1c.prrac.training_state.v1",
  "collision_episode": "True",
  "completion_reward_post_tanh": "0.0",
  "contact_bonus_count": "0",
  "contact_episode": "False",
  "controller_execution_target_at_handoff": "",
  "controller_mode": "prior_only",
  "controller_to_public_target_error_at_handoff": "",
  "controller_to_public_target_error_final": "",
  "controller_to_public_target_error_mean": "",
  "diagnostic_only": "False",
  "discovery_correction_count": "0",
  "effective_recovery_active_rate": "0.0",
  "egress_attempt_count": "0",
  "egress_attempt_count_agent_0": "0",
  "egress_attempt_count_agent_1": "0",
  "egress_attempt_count_agent_2": "0",
  "egress_failure_count": "0",
  "egress_failure_count_agent_0": "0",
  "egress_failure_count_agent_1": "0",
  "egress_failure_count_agent_2": "0",
  "egress_rejoin_count": "0",
  "egress_rejoin_count_agent_0": "0",
  "egress_rejoin_count_agent_1": "0",
  "egress_rejoin_count_agent_2": "0",
  "egress_success_count": "0",
  "egress_success_count_agent_0": "0",
  "egress_success_count_agent_1": "0",
  "egress_success_count_agent_2": "0",
  "environment_public_target_at_handoff": "",
  "episode_id": "0",
  "episode_index": "0",
  "episode_length": "400",
  "evaluation_mode": "full_prrac",
  "evaluation_runtime_revision": "dynamic_public_intercept_v2_1",
  "exact_public_target_plan_count": "0",
  "exact_public_target_unreachable_count": "0",
  "execution_continuity_event_counts": "{}",
  "execution_overlay_config_hash": "8ed1e291f69bcc117fb028aa786cc813a7add9d1d3ea0c30bef71c71f5563f42",
  "execution_variant": "B0_LEGACY_V2_1",
  "executor_collision_count_post_found": "0",
  "executor_collision_episode_post_found": "False",
  "executor_collision_max_streak_post_found": "0",
  "executor_distance_at_handoff": "",
  "executor_distance_to_intercept_at_received": "",
  "executor_distance_to_target_at_found": "",
  "executor_distance_to_target_at_received": "",
  "executor_final_distance_to_intercept": "",
  "executor_final_distance_to_target": "",
  "executor_first_collision_step_post_found": "",
  "executor_invalid_assignment_unreachable_count": "0",
  "executor_invalid_cost_increase_count": "0",
  "executor_invalid_count": "0",
  "executor_invalid_count_post_found": "0",
  "executor_invalid_query_unreachable_count": "0",
  "executor_invalid_rate_post_found": "",
  "executor_invalid_stale_snapshot_deferred_count": "132",
  "executor_last_collision_step_post_found": "",
  "executor_min_distance_to_intercept": "",
  "executor_min_distance_to_target": "",
  "executor_path_unreachable_count": "",
  "executor_replan_count": "2",
  "executor_residual_applied_norm_when_suppressed": "",
  "executor_residual_ratio_post_found": "",
  "executor_residual_ratio_pre_found": "0.0",
  "executor_residual_raw_norm_when_suppressed": "",
  "executor_residual_suppressed_step_count": "0",
  "executor_route_active_post_found_steps": "0",
  "executor_route_active_rate_post_found": "",
  "executor_route_inactive_post_found_steps": "0",
  "executor_target_received_step": "",
  "executor_validity_deferred_count": "132",
  "executor_validity_evaluation_count": "268",
  "expert_action_norm_hold": "",
  "expert_action_norm_intercept": "",
  "expert_action_norm_search": "",
  "explore": "False",
  "failure_reason_distribution": "{}",
  "failure_stage": "NOT_FOUND",
  "first_contact_step": "",
  "first_contact_to_success_steps": "",
  "first_full_hold_step": "",
  "forced_public_refresh_count": "0",
  "found": "False",
  "found_step": "",
  "found_to_first_contact_steps": "",
  "found_to_success_steps": "",
  "found_to_target_received_steps": "",
  "full_planning_refresh_count": "21",
  "gate_above_0_95_rate": "",
  "gate_below_0_05_rate": "",
  "gate_mean": "",
  "gate_p10": "",
  "gate_p50": "",
  "gate_p90": "",
  "gate_saturation_high_rate": "",
  "gate_saturation_low_rate": "",
  "graph_reconnect_attempt_count": "0",
  "graph_reconnect_attempt_count_agent_0": "0",
  "graph_reconnect_attempt_count_agent_1": "0",
  "graph_reconnect_attempt_count_agent_2": "0",
  "graph_reconnect_failure_count": "0",
  "graph_reconnect_failure_count_agent_0": "0",
  "graph_reconnect_failure_count_agent_1": "0",
  "graph_reconnect_failure_count_agent_2": "0",
  "graph_reconnect_success_count": "0",
  "graph_reconnect_success_count_agent_0": "0",
  "graph_reconnect_success_count_agent_1": "0",
  "graph_reconnect_success_count_agent_2": "0",
  "handoff_delay": "",
  "handoff_forced_refresh_count": "0",
  "hold_bonus_count": "0",
  "hold_episode": "False",
  "initial_planner_endpoint_fallback_count": "0",
  "installed_executor_tracking_target_at_handoff": "",
  "last_recovery_failure_reason_agent_0": "",
  "last_recovery_failure_reason_agent_1": "",
  "last_recovery_failure_reason_agent_2": "",
  "last_recovery_mode_agent_0": "NORMAL_SEARCH",
  "last_recovery_mode_agent_1": "NORMAL_SEARCH",
  "last_recovery_mode_agent_2": "NORMAL_SEARCH",
  "last_valid_route_active_step_count": "0",
  "last_valid_route_plan_count": "0",
  "local_connector_attempt_count": "0",
  "local_connector_attempt_count_agent_0": "0",
  "local_connector_attempt_count_agent_1": "0",
  "local_connector_attempt_count_agent_2": "0",
  "local_connector_collision_count": "0",
  "local_connector_collision_count_agent_0": "0",
  "local_connector_collision_count_agent_1": "0",
  "local_connector_collision_count_agent_2": "0",
  "local_connector_plan_count": "0",
  "local_connector_plan_count_agent_0": "0",
  "local_connector_plan_count_agent_1": "0",
  "local_connector_plan_count_agent_2": "0",
  "local_connector_reached_count": "0",
  "local_connector_reached_count_agent_0": "0",
  "local_connector_reached_count_agent_1": "0",
  "local_connector_reached_count_agent_2": "0",
  "manifest_sha256": "69d1838177568c8abacac08cf22852c8e51b52e265807434d64830ad3c49daa6",
  "map_known_fraction_at_found_or_end": "0.68625",
  "map_known_fraction_gain_pre_found": "0.4625",
  "map_known_fraction_initial": "0.22375",
  "max_steps": "400",
  "mean_target_prediction_error": "",
  "navigation_endpoint_switch_count": "0",
  "optimizer_update_count": "0",
  "parameter_update_count": "0",
  "path_changed_step_count": "0",
  "post_found_base_reward": "0.0",
  "post_found_collision_count": "0",
  "post_found_route_inactive_max_streak": "0",
  "post_found_route_inactive_step_count": "0",
  "post_found_route_inactive_terminal_streak": "0",
  "post_found_safe_hold_max_streak": "0",
  "post_found_safe_hold_step_count": "0",
  "post_found_safe_hold_terminal_streak": "0",
  "post_found_shaping_reward": "0.0",
  "post_found_step_count": "0",
  "pre_found_base_reward": "-20.87266796710901",
  "pre_found_step_count": "400",
  "privileged_oracle": "False",
  "proxy_distance_to_semantic_target_max": "",
  "proxy_distance_to_semantic_target_mean": "",
  "public_target_shift_max": "",
  "public_target_shift_mean": "",
  "public_target_shift_sum": "0.0",
  "public_target_update_accepted_count": "0",
  "public_target_update_event_count": "0",
  "reachable_proxy_active_step_count": "0",
  "reachable_proxy_plan_count": "0",
  "recovery_collision_count": "0",
  "recovery_duration_max": "0",
  "recovery_duration_mean": "",
  "recovery_duration_sum": "0",
  "recovery_effective_intervention_count": "0",
  "recovery_effective_intervention_count_agent_0": "0",
  "recovery_effective_intervention_count_agent_1": "0",
  "recovery_effective_intervention_count_agent_2": "0",
  "recovery_effective_intervention_episode": "False",
  "recovery_failed_endpoint_count": "0",
  "recovery_failed_pass_through_count": "0",
  "recovery_guidance_changed_step_count": "0",
  "recovery_max_collision_streak": "0",
  "recovery_no_egress_count": "0",
  "recovery_plan_active_step_count": "0",
  "recovery_state_non_normal_step_count": "0",
  "remaining_steps_after_found": "",
  "replay_sample_count": "0",
  "report_schema": "bser.phase1c.prrac.evaluation_report.v2",
  "reward": "-20.87266796710901",
  "route_refresh_attempt_count": "0",
  "route_refresh_attempt_count_agent_0": "0",
  "route_refresh_attempt_count_agent_1": "0",
  "route_refresh_attempt_count_agent_2": "0",
  "route_refresh_failure_count": "0",
  "route_refresh_failure_count_agent_0": "0",
  "route_refresh_failure_count_agent_1": "0",
  "route_refresh_failure_count_agent_2": "0",
  "route_refresh_identical_to_base_count": "0",
  "route_refresh_success_count": "0",
  "route_refresh_success_count_agent_0": "0",
  "route_refresh_success_count_agent_1": "0",
  "route_refresh_success_count_agent_2": "0",
  "router_accuracy": "",
  "router_argmax_accuracy": "",
  "router_balanced_accuracy": "",
  "router_confusion_matrix": "[[0,0,0],[0,0,0],[0,0,0]]",
  "router_precision_hold": "",
  "router_precision_intercept": "",
  "router_precision_search": "",
  "router_probability_hold": "",
  "router_probability_intercept": "",
  "router_probability_search": "",
  "router_recall_hold": "",
  "router_recall_intercept": "",
  "router_recall_search": "",
  "router_stage_counts": "[0,0,0]",
  "runtime_integration_mode": "legacy",
  "runtime_overlay_enabled": "False",
  "s2a1_activation_artifact_revision": "s2a1.activation_artifact.v1",
  "safe_hold_active_step_count": "0",
  "safe_hold_entry_count": "0",
  "scenario_id": "unknown_validation_m20_0001",
  "scenario_seed": "1729",
  "search_collision_recovery_config_hash": "28b7faa08a087ea1876fb0ef5708ff469947c29f7ec9971f38b3540581f6fd8b",
  "search_collision_recovery_schema": "bser.phase1c.prrac.search_collision_recovery.v1",
  "search_continuity_diagnostics_hash": "d7d784eb3299ecf84a09b1faf513cd89add0c0f02375644e8b79bb9e4fb56e3c",
  "search_continuity_diagnostics_schema": "bser.phase1c.prrac.search_continuity.v2",
  "search_recovery_active_rate": "0.0",
  "search_recovery_active_step_count": "0",
  "search_recovery_enabled": "False",
  "search_recovery_entry_count": "0",
  "search_recovery_entry_count_agent_0": "0",
  "search_recovery_entry_count_agent_1": "0",
  "search_recovery_entry_count_agent_2": "0",
  "search_recovery_variant": "S2A_C0_BASELINE",
  "search_residual_ratio_post_found": "",
  "search_residual_ratio_pre_found": "0.0",
  "searcher_applied_action_norm_pre_found": "0.0",
  "searcher_applied_residual_norm_mean_pre_found": "0.0",
  "searcher_assignment_missing_step_count_pre_found": "0",
  "searcher_assignment_switch_count_pre_found": "8",
  "searcher_assignment_unreachable_step_count_pre_found": "0",
  "searcher_collision_agent_count_pre_found": "1",
  "searcher_collision_count_pre_found": "153",
  "searcher_collision_count_pre_found_agent_0": "153",
  "searcher_collision_count_pre_found_agent_1": "0",
  "searcher_collision_count_pre_found_agent_2": "0",
  "searcher_collision_count_pre_found_total": "153",
  "searcher_collision_episode_pre_found": "True",
  "searcher_collision_max_streak_pre_found": "152",
  "searcher_collision_max_streak_pre_found_agent_0": "152",
  "searcher_collision_max_streak_pre_found_agent_1": "0",
  "searcher_collision_max_streak_pre_found_agent_2": "0",
  "searcher_collision_streak_pre_found_agent_0": "152",
  "searcher_collision_streak_pre_found_agent_1": "0",
  "searcher_collision_streak_pre_found_agent_2": "0",
  "searcher_distance_travelled_pre_found": "99.4633927655383",
  "searcher_distance_travelled_pre_found_agent_0": "37.651626878011605",
  "searcher_distance_travelled_pre_found_agent_1": "43.83316235688277",
  "searcher_distance_travelled_pre_found_agent_2": "17.978603530643923",
  "searcher_first_collision_step_pre_found": "246",
  "searcher_hold_rate_pre_found": "0.0",
  "searcher_hold_rate_pre_found_agent_0": "0.0",
  "searcher_hold_rate_pre_found_agent_1": "0.0",
  "searcher_hold_rate_pre_found_agent_2": "0.0",
  "searcher_hold_step_count_pre_found": "0",
  "searcher_hold_step_count_pre_found_agent_0": "0",
  "searcher_hold_step_count_pre_found_agent_1": "0",
  "searcher_hold_step_count_pre_found_agent_2": "0",
  "searcher_last_collision_step_pre_found": "400",
  "searcher_raw_action_norm_pre_found": "0.0",
  "searcher_raw_residual_norm_mean_pre_found": "0.0",
  "searcher_residual_alignment_valid_count_pre_found": "0",
  "searcher_residual_alignment_zero_navigation_count_pre_found": "0",
  "searcher_residual_alignment_zero_residual_count_pre_found": "0",
  "searcher_residual_contribution_ratio_mean_pre_found": "0.0",
  "searcher_residual_negative_alignment_count_pre_found": "0",
  "searcher_residual_negative_alignment_rate_pre_found": "",
  "searcher_residual_off_enabled": "False",
  "searcher_residual_suppressed_agent_step_count_pre_found": "0",
  "searcher_residual_suppressed_env_step_count_pre_found": "0",
  "searcher_residual_suppressed_step_count_pre_found": "0",
  "searcher_route_active_rate_pre_found": "1.0",
  "searcher_route_active_rate_pre_found_agent_0": "1.0",
  "searcher_route_active_rate_pre_found_agent_1": "1.0",
  "searcher_route_active_rate_pre_found_agent_2": "1.0",
  "searcher_route_active_step_count_pre_found": "1200",
  "searcher_route_active_step_count_pre_found_agent_0": "400",
  "searcher_route_active_step_count_pre_found_agent_1": "400",
  "searcher_route_active_step_count_pre_found_agent_2": "400",
  "searcher_route_inactive_step_count_pre_found": "0",
  "searcher_tracking_subgoal_switch_count_pre_found": "45",
  "searcher_waypoint_switch_count_pre_found": "53",
  "searcher_zeroed_step_count": "0",
  "semantic_target_update_count": "0",
  "stage_critic_losses": "{\"hold\":null,\"intercept\":null,\"search\":null}",
  "stage_td_errors": "{\"hold\":null,\"intercept\":null,\"search\":null}",
  "success": "False",
  "success_if_found": "False",
  "success_step": "",
  "target_belief_entropy_at_found_or_end": "5.967590228953221",
  "target_belief_entropy_delta_pre_found": "-0.7157708167192256",
  "target_belief_entropy_initial": "6.683361045672447",
  "target_belief_peak_at_found_or_end": "0.007098506670445204",
  "target_belief_peak_delta_pre_found": "0.005846942192874849",
  "target_belief_peak_initial": "0.001251564477570355",
  "target_prediction_error_at_delivery": "",
  "target_prediction_map_fallback_count": "0",
  "target_shift_forced_refresh_count": "0",
  "terminal_bonus_count": "0",
  "tier0_count": "0",
  "tier1_count": "0",
  "tier2_count": "0",
  "tier3_count": "0",
  "tracking_waypoint_delta_norm_max": "0.0",
  "tracking_waypoint_delta_norm_mean": "",
  "tracking_waypoint_delta_norm_sum": "0.0",
  "training_update": "False"
}
```
