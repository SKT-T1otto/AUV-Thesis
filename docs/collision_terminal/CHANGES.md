# 文件与作用

以下为本轮工作区变更，未 commit/push。路径相对仓库根目录；历史 outputs 和历史 manifest 没有修改。

| 文件 | 作用 |
|---|---|
| `core/env/task_protocol.py`（新增） | 协议及检测/奖励版本、配置校验、闭集线段/AABB 检测、权威 EpisodeOutcome、终止 metadata |
| `core/env/uav_env.py` | 显式协议构造参数、reset 初始碰撞检查、实际运动段检测、冻结处理、终止优先级、最终团队奖励、重复 step 拒绝 |
| `core/env/mission_env.py` | 公共只读 `get_episode_result()` 委托 |
| `chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/reward_adapter.py` | 最高优先级 terminal override，最终 learner 奖励与原始诊断分离 |
| `chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/training_env.py` | wrapper 汇总采用权威终止结果 |
| `chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/train_phase1c_v2.py` | 共享环境工厂显式校验并传递新协议配置 |
| `chapter3_bser/experiments/phase1c_prrac/training_env.py` | PRRAC 转移 metadata 写入协议、terminated/truncated/reason |
| `chapter3_bser/experiments/phase1c_prrac/transition_protocol.py` | 新 metadata 字段与旧格式默认解释；保持三阶段标签 |
| `chapter3_bser/experiments/phase1c_prrac/replay_adapter.py` | 协议隔离、状态恢复校验、严格终止 dones/success 约束 |
| `chapter3_bser/models/prrac/prrac_maddpg.py` | 仅为可 bootstrap 行计算 next Q，终止 target 等于 reward |
| `chapter3_bser/experiments/phase1c_prrac/train_phase1c_prrac.py` | collector 终止短路、worker 配置、完整 resume 校验、actor warmstart、critic-only 预热、checkpoint 身份、进度与汇总、严格新目录保护 |
| `chapter3_bser/experiments/phase1c_prrac/checkpoint_transfer.py`（新增） | 完成 checkpoint 路径、SHA256、真实源配置、完整 Actor 键/buffer/形状检查及来源成本 |
| `chapter3_bser/experiments/phase1c_prrac/evaluate_prrac_checkpoints.py` | 显式跨协议只读评价、目标协议建环境、终止短路、CSV/汇总/图、缓存 SHA 身份、异常退出 |
| `chapter3_bser/experiments/phase1c_prrac/evaluation_metrics.py` | 严格任务指标、失败原因、完整性与配对/排名校验 |
| `chapter3_bser/experiments/phase1c_prrac/evaluation_provenance.py` | 协议、checkpoint SHA 进入跨文件来源一致性检查和组合身份 |
| `chapter3_bser/experiments/phase1c_prrac/evaluation_trace.py` | 未 Found 的严格碰撞也可保存有上限的诊断 |
| `chapter3_bser/experiments/phase1c_prrac/task_metrics.py`（新增） | 完成/异常分母、条件成功率 null、首次碰撞角色/阶段汇总、只读终端规划快照 |
| `chapter3_bser/experiments/phase1c_prrac/search_continuity/aggregation.py` | 严格协议失败漏斗使用实际 terminal reason |
| `chapter3_bser/experiments/phase1c_prrac/search_collision_recovery/aggregation.py` | 严格协议失败漏斗使用实际 terminal reason |
| `chapter3_bser/experiments/phase1c_prrac/paired_evaluation.py` | 支持任意正整数场数及 100 场计划，保留原配对验证，记录协议与安全完成指标 |
| `chapter3_bser/experiments/phase1c_prrac/collision_terminal_cli.py`（新增） | evaluate / train / warmstart / resume 四种明确入口 |
| `configs/chapter3/bser_phase1c_prrac_collision_terminal_train.json`（新增） | 新严格训练配置，1000 episodes、400 steps、原架构与控制参数 |
| `configs/chapter3/bser_phase1c_prrac_collision_terminal_eval.json`（新增） | 新严格评价配置，100 episodes、400 steps、相同终端奖励 |
| `scripts/run_collision_terminal.ps1`（新增） | 可配置 Conda 环境、完整传参、退出码传递 |
| `scripts/run_collision_terminal.bat`（新增） | Windows BAT 包装入口 |
| `scripts/linux/run_collision_terminal.sh`（新增） | Linux 环境初始化与 exec 启动，不改旧训练脚本 |
| `tests/test_collision_terminal_protocol.py`（新增） | 物理/环境/wrapper/replay/critic/transfer/collector/evaluator/报告缓存的真实有界回归 |
| `tests/test_repository_metadata.py` | 精确核对本轮 evolution 文件及公共 facade，保留全部历史元数据检查 |
| `tests/test_core_source_provenance.py` | 27 条记录全部保留；为 uav_env 的明确演进固定历史与当前哈希 |
| `tests/test_phase1a_core_freeze.py` | 为两个授权环境文件固定历史、当前与 AST 哈希；仍检查原 40 个文件 |
| `tests/test_phase1a1_original_core_freeze.py` | 为两个授权环境文件固定历史与当前哈希；仍检查原 40 个文件 |
| `tests/test_phase1c_v2_isolation.py` | 对同两个环境文件校验 HEAD 和当前精确哈希，其他冻结路径仍要求无 diff |
| `tests/test_prrac_isolation.py` | 对环境及 v2 结果 wrapper 固定历史和当前哈希；保留旧网络/replay/导入方向检查 |
| `tests/test_linux_runtime_assets.py` | 原资产检查纳入新脚本及显式 Conda 可执行文件表达式；未屏蔽现有失败 |
| `tools/verify_collision_terminal.py`（新增） | 全量 unittest 与只读冻结 E0 验证，要求新的证据目录，缺输入返回非零 |
| `docs/provenance/collision_terminal_evolution.json`（新增） | 本轮用户授权的逐文件演进；27 条来源中的 uav_env、Phase 1A 公共 facade、PRRAC 冻结 wrapper 分开记录 |
| `AGENTS.md` | 记录本轮明确授权及历史保护边界 |
| `README.md` | 添加实现文档入口，说明旧状态文字属于历史记录 |
| `docs/collision_terminal/IMPLEMENTATION.md`（新增） | 行为对照、调用链、兼容规则、四种执行方式、Windows/Linux 完整命令与结果路径 |
| `docs/collision_terminal/VERIFICATION.md`（新增） | 真实本地验收结果、失败归因、命令和限制 |
| `docs/collision_terminal/CHANGES.md`（本文件） | 变更清单 |
| `docs/collision_terminal/verification/`（新增） | 小体积本地回归日志与 JSON 证据，包括失败过程；不是训练输出或模型 |

## Schema 与新增参数

原 training/checkpoint/replay schema 仍通过原校验；新增的显式身份字段构成兼容边界，缺字段仅可解释为 legacy。没有将历史 checkpoint metadata 改写成新协议。

| 字段 / 参数 | 语义 |
|---|---|
| `task_protocol` | legacy_nonterminal_v1 / collision_terminal_v1 |
| `collision_detection_revision` | endpoint_rollback_v1 / segment_closed_aabb_v1 |
| `terminal_reward_revision` | legacy_shaping_v1 / team_failure_override_v1 |
| `collision_terminal_reward` | 严格协议最终团队奖励，默认 -2.0；有限、负值、与 wrapper clip 兼容 |
| `--allow-protocol-transfer` | 仅显式 legacy → strict 只读评价；其他架构、维度及执行 runtime 检查继续生效 |
| `--init-actors-from` | 底层 trainer 的 Actor 热启动路径，与 `--resume` 互斥 |
| `--critic-warmup-updates` | 热启动 critic-only 更新轮数，默认 256，允许 0；已有 replay/warmup 条件仍生效 |
| `--episodes` | 正整数；严格配对支持每控制器 100 场，准备阶段不执行 |
| `--prepare-only` | 配对计划生成；不加载权重、不开始回合 |
| `EpisodeOutcome` | episode_terminated/truncated、reason、terminal_step、首次碰撞信息、success/mission_complete |
| `resolved_training_config` | 新 checkpoint 内嵌真实规范化配置；旧 checkpoint 可从原 run sidecar 验证 |
| `initialization` | 热启动来源 SHA、协议、episode、环境步/更新/成本；未知成本 null |
| `learner_update_rounds` | 训练汇总中的采样更新轮数，对应 checkpoint 的 update_step |
| `optimizer_update_count` | 既有每 agent learner 更新调用计数；不等于采样轮数或所有 optimizer.step 的总数 |
| `critic_only_warmup_updates_executed` | 实际预热采样更新轮数，最多为配置预算 |
