# 碰撞协议维护与验收

2026-09-13 开始核对：Windows 工作区为 `E:/gym/code/WORKSPACE/AUV-Thesis`，origin 为 `https://github.com/SKT-T1otto/AUV-Thesis.git`，分支 `main`，本地 HEAD 与当时远程默认分支最新提交均为 `06167b5807a05b29ee0f42e21556015c17cfd5c1`，工作区干净。远程通过 GitHub 连接器实时核对。该提交已发布新协议；旧文档的“未提交”是原交付时点事实。本次维护修改未 commit/push。

## 修复与整理

严格协议统一使用 `task_metrics.strict_outcome` 和 `boolean`。success、obstacle_collision、timeout、running 显式映射；缺失/空白原因或必需标志缺失为 INCOMPLETE，未知原因或明确矛盾抛出含场景、原因的 ValueError。CSV 的 False 保持 false，空白保持未知。单回合、三类 funnel、summary、CSV 缓存恢复、checkpoint 排名和配对分析复用校验；不完整回合不认证完整统计，异常不会静默删除。

当前 BEDS 缺省/关闭配置等价检查保留，去掉浮动 HEAD 与旧源码 exec。另用完整提交 `93a9c8fb53857051390265e3035061bf05402e25` 做碰撞改造前 legacy 对照；该提交已经包含 BEDS。参考源码使用 git archive，当前源码单独复制，两边分别以独立目录和 Python 子进程运行。测试 Actor seed 991、同一明确场景/配置/runtime、每边 1 回合最多 4 步；完整动作、状态、观测、奖励、碰撞、任务事件及规划控制诊断逐元素/字节比较。身份字段精确单独核验，详见 `tests/legacy_baseline.json`。

14 份小测试合为 3 组，所有原 TestCase/方法体和种子保留：目标性质、规划接口、当前离线 smoke/协议。4 项固定样本/报告/overlay 检查迁入 `historical_verification/`，其余 v2 当前检查保留。共享 `_transitions`、`_ImmediateExecutor` 和导航 fixture 已移入辅助模块，辅助模块没有 TestCase。原 583 个静态测试方法全部映射，另新增 3 个方法；当前 582、历史 4，实际发现和执行数以本次日志为准。

保留生产 v2、search/execution continuity、collision recovery、docs2、provenance 等目录，它们仍参与当前调用链、信息边界或来源检查。没有改动碰撞几何、奖励/成功定义、网络、规划算法、训练超参数或用户 outputs/runs/checkpoints。

## 日志归档和恢复

仅移出 `acceptance_02.log`、`acceptance_03.log`、`warmstart_resume_integration.log` 三份过期调试日志。原始内容与固定提交 `06167b5807a05b29ee0f42e21556015c17cfd5c1:path` 的 Git blob 已验证一致（记录 LF/CRLF 的无损转换）。原路径、工作树字节 SHA256、Git blob ID/内容 SHA256 和恢复规则均在 [cleanup_manifest.json](cleanup_manifest.json)。原最终验收日志、交付汇总和未解决问题的失败证据仍保留。

恢复到新的独立目录：用 Python `subprocess.check_output(['git','show', commit+':'+path])` 读取记录中的固定对象，若 `restore_newlines` 为 `LF_to_CRLF` 则执行 `data.replace(b'\n', b'\r\n')`，验证 `hashlib.sha256(data).hexdigest()` 等于记录的 `sha256` 后，以二进制写到新目录。不要用 PowerShell 文本重定向改变日志编码，也不要覆盖现有文件。

## 本次验收

最终验收已实际完成，日志为 [logs/final_01](logs/final_01/summary.json)，机器汇总为 [verification_summary.json](verification_summary.json)。2026-09-13 UTC 06:29:24 至 07:18:43，总耗时 2958.589 秒；这是最新本地记录，不是 CI。

| 套件 | 发现 | 实际运行 | 通过 | 失败 / 错误 / 跳过 | 阻塞 |
|---|---:|---:|---:|---:|---:|
| current（180 个模块） | 582 方法 | 582 | 582 | 0 / 0 / 0 | 0 |
| 固定 legacy 对照 | 1 对照 | 1（两个独立进程，各 4 步） | 1 | 0 / 0 / 0 | 0 |
| historical | 4 方法 | 1 | 1 | 0 / 0 / 0 | 3 |
| golden | 60 条预期轨迹 | 0 | 0 | 未执行 | 缺少冻结输入 |

current 的发现 ID 与实际执行 ID 完全一致，subTest 失败记录为 0。原 583 个方法的去向、通过/阻塞状态全部验证，新增 3 个方法通过；没有靠减少原断言取得通过。碰撞专项 24/24、分类指标 6/6、全部 27 条 provenance、原 40 个 core 文件冻结，以及真实 spawn、checkpoint/warmstart/resume、100 场配对计划生成均在完整当前套件中执行通过。未运行 100 场性能评价。

180 个模块的原始日志已按完成顺序无损合并为 [current/unittest.log](logs/final_01/current/unittest.log)，每段原路径、字节位置、长度和 SHA256 在 cleanup manifest 的 `current_verification_log_merge` 中，合并前后逐段校验一致。各模块方法、时间和退出码仍在 [current/methods.json](logs/final_01/current/methods.json)，没有重写测试结果。

最终状态：`current_suite_passed=true`、`legacy_compatibility_passed=true`、`historical_suite_status=blocked_missing_historical_input`、`golden_status=blocked_missing_historical_input`、**`all_required_verification_passed=false`**。`all` 的实际退出码为 1，原因是历史阻塞。

被测源码、测试、配置、脚本和必要清单的前后哈希一致，交付时再次核对为：

`ca3ae8bd48249eefc36082407a2f60d985e14b8a0821466de8fdd4dfd9ff584d`

测试期间没有修改被测树；本收尾输出目录不在源文件哈希范围内。环境为 Windows 10.0.26200、AUV Conda Python 3.10.20、PyTorch 2.11.0+cu126、NumPy 2.2.6，测试设备 CPU。完整依赖版本和实际参数记录在机器汇总中。未在用户 Linux 主机运行。

legacy 使用场景 `unknown_validation_m20_0001`，场景 seed 1729、Actor seed 991。两边完整行为无差异；当前新增的 legacy 协议身份值、每个仓库模块的来源目录及精确新增导入清单另行核验通过。

剩余阻塞的准确位置：

- Phase 1A-v1：固定提交 `93a9c8fb53857051390265e3035061bf05402e25` 中缺少 `experiments/chapter3/bser_e1_offline/` 下 19 个清单指定的历史产物 Git 路径（包括 `aggregate_by_profile.csv`、`per_instance_results.csv` 等）；完整逐项路径在 [historical/summary.json](logs/final_01/historical/summary.json)。
- CH3 E0：`experiments/chapter3/e0_equivalence/equivalence_summary.json` 和 `per_trajectory_results.csv` 缺失。
- v2 overlay：`docs2/phase1c_v2_design/overlay_manifest.json` 缺失。
- golden：`experiments/chapter3/e0_core_migration/golden_trace_manifest.json` 缺失，实际执行轨迹数为 0。

已只读检查现有源码备份、DATA_TO_KEEP/保留数据目录及本地 21 个提交的相关历史，未找到可恢复的上述输入。没有伪造 CSV/manifest，也没有改写旧哈希；“已备份”不能替代可访问且身份匹配的证据。

## 文件变化与工作区

不计本轮新增收尾材料，Git 已跟踪/未忽略文件从 **759 → 751**，字节数从 **5,029,495 → 5,020,196**，净减少 8 个文件、9,299 字节。三份归档日志原工作树内容共 52,224 字节；另删除 517 个缓存文件、2,880,915 字节，以及 5 个完成使命的一次性编辑脚本。新增真实日志会占用额外空间，不能把上述数字当作含验收材料的整个磁盘目录总量，也不代表训练加速。

本轮生产修改集中在 PRRAC 的 `task_metrics.py`、`evaluation_metrics.py`、`evaluation_provenance.py`、`evaluate_prrac_checkpoints.py`、`paired_evaluation.py` 和两个搜索聚合模块。训练 smoke、checkpoint evaluator、BEDS safe-standby 测试文件仍保留其有效行为断言；仅共享 helper 移出。v2、搜索/执行连续性目录和 provenance 等仍有当前依赖，保留理由、引用和逐文件映射见 cleanup manifest。

`git diff --check` 退出码 0。`git diff --stat` 为已跟踪的 46 个文件、711 行新增、1167 行删除；新增未跟踪文件另列于 [最终 Git 记录](logs/git_final.log)。工作区有本轮未提交修改，无 commit/push；core、配置、模型、控制器、历史 provenance 和用户 outputs/runs/checkpoints 没有 diff。

Windows 和 Linux 都从仓库根目录、已激活的 AUV 环境运行（每次换新的输出目录）：

```powershell
conda activate AUV
python -B -m tools.verify_collision_terminal --suite all --workers 4 --output-dir docs/collision_terminal/maintenance/logs/windows_manual_01
```

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CRK_CONDA_ENV:-AUV}"
python -B -m tools.verify_collision_terminal --suite all --workers 4 --output-dir docs/collision_terminal/maintenance/logs/linux_manual_01
```

`current`（或兼容别名 `regression`）只覆盖当前功能；`legacy` 为固定完整源码对照；`historical` 为固定历史证据；`golden` 为冻结轨迹。`all` 汇总四者，历史缺失仍返回非零。运行器记录完整方法 ID、subTest 记录、实际退出码、时间、环境版本，以及排除本收尾输出目录的代码树前后哈希。0 项发现必须失败。

Windows PowerShell/BAT 碰撞入口使用显式 `-CondaEnv AUV` 后再传 mode，以匹配已有参数绑定契约。省略该参数后直接传 positional mode 会占用 CondaEnv 位置；本轮没有扩展修改生产启动器。Bash 参数和退出码在 Git Bash 中以测试桩验证，不能据此声称用户 Linux Conda/CUDA 主机已经跑通。
