# CRK-Thesis-v2 Cross Platform Audi

审计日期：2026-08-24（Asia/Shanghai）
审计 commit：`83ededb056eaf8f1b51414335aab515f31f453e0`（`fix: use spawn workers for Phase 1C-v2 training`）
审计方式：只读静态扫描、AST/JSON 解析、Git 元数据检查、文件编码/EOL 检查、文件大小盘点；未运行测试、训练、GPU 作业或 episode 实验。

## 1. Summary

总体风险：**HIGH**

结论：当前 commit 不能认定为 “Windows PASS / Linux RTX3090 PASS”。`core` 与大部分算法实现使用 `pathlib` 和显式 device 传递，具备继续维持单代码仓库的基础；但 Linux 3090 训练仍有已复现的 Tensor 进程间传输故障，现有环境锁又是 Windows 专用，Linux 启动与测试覆盖不完整。

最关键结论如下：

- Phase1C-v2 已在 `train_phase1c_v2.py:758-763` 使用显式 `spawn`，这解决了“CUDA 初始化后 fork”这一类风险，但没有解决 PyTorch Tensor IPC。
- `_collect_episode()` 仍在每一步构造并返回多个 CPU `torch.Tensor`（`train_phase1c_v2.py:303-315`），主进程通过 `ProcessPoolExecutor` 接收整个 transitions 列表（`:779-784`）。预先存在、未跟踪的 `3090/worker_failure_episode_0001.json:31` 明确显示当前代码在 Linux 上失败于 `torch.multiprocessing.reductions.rebuild_storage_fd`，随后出现 `received 0 items of ancdata` 和 `BrokenProcessPool`。
- v1 trainer 更高风险：主进程可能先把模型移动到 CUDA（`train_phase1c.py:589-591`），然后用 Linux 默认 `fork` 创建 executor（`:629`），worker 同样返回 Tensor transitions（`:227-234`）。
- `configs/environment_lock/conda_explicit_spec.txt` 是 `win-64` exact spec；`conda_environment_full.yml` 固定了 Windows build、`ucrt/vc` 和 `D:\...` prefix，不能作为 Linux 环境创建文件。
- 建议继续使用一个 Git 代码仓库。应增加的是薄的 Linux 运维层（Linux 环境清单、`.sh` launcher、GPU preflight/CI），不是复制一套 Linux Python 代码，也不是新增顶层业务 `runtime/` 实现。

扫描计数口径：

- 必扫目录内目标扩展名：248 个文件（213 `.py`、27 `.json`、2 `.yml`、3 `.ps1`、3 `.bat`；0 `.yaml`、0 `.sh`）。
- 全仓库 Git 已跟踪目标扩展名：308 个文件（223 `.py`、77 `.json`、2 `.yml`、3 `.ps1`、3 `.bat`）。
- 另扫描 5 个被忽略的 experiment checkpoint JSON 和 2 个预先存在的未跟踪 `3090/*.json`；代码/配置/脚本目标文件去重总数为 315。
- 编码/EOL 检查覆盖 410 个已跟踪相关文本文件；计入上述 7 个本地 JSON 后，相关文本审计覆盖 417 个唯一文件。
- AST 静态解析：213/213 Python PASS；必扫目录 JSON 解析：27/27 PASS；`3090` JSON 解析：2/2 PASS。

问题数以本报告表格中的独立审计事项计：**P0 4，P1 8，P2 9，共 21 项**。

## 2. P0 Issues

| 文件 | 行号 | 问题 | 影响 | 建议 |
|---|---:|---|---|---|
| `chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/train_phase1c_v2.py` | 303-315, 742-784 | worker 输入包含 policy state Tensor，输出包含每步多组 CPU Tensor；`spawn` 仍通过 PyTorch storage/file-descriptor IPC 传输 | 当前 Linux 3090 证据已复现 `rebuild_storage_fd` → `received 0 items of ancdata` → `BrokenProcessPool`，训练不能启动完成 | worker 边界改为普通可序列化 payload（NumPy/连续数组/单块批量数据）；不要返回数千个独立 Tensor storage。保留 `spawn`，并增加真实多进程回归 |
| `chapter3_bser/experiments/phase1c_bser_rmaddpg/train_phase1c.py` | 227-234, 589-591, 629-649 | CUDA device 初始化/模型迁移可能发生在 executor 之前，executor 未指定 context，worker 返回 Tensor 列表 | Linux 默认 `fork` 与已初始化 CUDA 不兼容，同时存在与 v2 相同的 Tensor IPC/FD 风险 | 显式 `spawn`，并移除 Tensor 跨进程返回；若 v1 已冻结，则在 Linux GPU 入口 fail-fast 禁止该并行路径 |
| `configs/environment_lock/conda_explicit_spec.txt` | 3-24 | exact spec 明确为 `win-64`，URL 固定 Windows 包 | Linux conda 无法用该文件创建可用环境 | 保留为 Windows provenance snapshot，并另建 Linux 3090/CUDA 环境清单；不要把两平台 exact spec 混成一个文件 |
| `configs/environment_lock/conda_environment_full.yml` | 8-26, 50 | 固定 Windows build string、`ucrt`、`vc`、`vs2015_runtime` 和 `D:\...` prefix | Linux solver 无法解析或会创建错误环境 | 将其明确标注为 Windows lock；提供 Linux lock，移除 Linux 文件中的 Windows build/prefix，并记录 CUDA/PyTorch 安装源 |

## 3. P1 Issues

| 文件 | 行号 | 问题 | 影响 | 建议 |
|---|---:|---|---|---|
| `configs/environment_lock/pip_freeze.txt` | 26-28 | `torch/torchvision/torchaudio` 使用 `+cu126` local version，但文件未记录 wheel index/source 或平台约束 | Linux 重建环境可能解析失败或拿到非预期构建 | 记录 Linux wheel index/conda channel、Python/driver/CUDA compatibility 和验证命令 |
| `scripts/` | N/A | 只有 3 组 `.ps1` + `.bat`，没有任何 `.sh` | Linux 缺少受控的 Phase1C train、v2 train、diagnostic eval 入口 | 为三组入口提供薄 `.sh` launcher；调用同一 `python -m ...` 模块，不复制 Python trainer |
| `chapter3_bser/experiments/phase1c_bser_rmaddpg/train_phase1c.py`; `.../phase1c_bser_rmaddpg_v2/train_phase1c_v2.py` | 49-53; 60-64 | seed helper 未显式记录 `torch.cuda.manual_seed_all`、cuDNN benchmark/deterministic 状态；`warn_only=True` 允许不确定算法继续 | Windows CPU 与 Linux GPU 结果不能宣称 bitwise 一致；GPU 重现实验审计信息不足 | 与 `core/runtime/engine.py:405-419` 的确定性协议统一，并在 manifest 中记录实际 device、CUDA/cuDNN 与确定性标志 |
| `core/algorithms/maddpg.py`; `core/runtime/engine.py`; `phase_aware_replay.py` | 66-76; 169-175; 16-22 | 请求 `cuda` 但 CUDA 不可用时静默回退 CPU | RTX3090 环境配置错误可能被误报为“训练通过”，实际长任务跑在 CPU | 正式 GPU 模式应 fail-fast，并在启动日志/summary 记录 requested 与 resolved device |
| `tests/test_phase1c_v2_isolation.py` | 50-55 | 读取不存在的 `docs2/phase1c_v2_design/overlay_manifest.json` | 测试在 Windows/Linux 都会直接 `FileNotFoundError` | 恢复受控 evidence，或在经过 provenance 审查后修正测试/文档契约 |
| `tests/test_ch3_e0_equivalence.py` | 8, 13, 20 | 依赖不存在的 `experiments/chapter3/e0_equivalence/equivalence_summary.json` 与 `per_trajectory_results.csv` | repository-integrity 测试在两平台都不能通过 | 恢复应跟踪的 compact evidence；不要用重新训练代替历史证据 |
| `tests/test_phase1c_v2_training_smoke.py`; `tests/test_phase1c_v2_isolation.py` | 138-193; 38-39 | smoke 只测 replay/update/checkpoint，未执行真实 `ProcessPoolExecutor`；isolation 只要求 Windows launcher 存在 | 当前 Linux ancdata 故障无法被测试系统捕获 | 增加 CPU spawn 多 worker IPC smoke 和 Linux CUDA preflight；至少验证 worker 返回值不含 Tensor storage |
| `chapter3_bser/experiments/plot_e1.py` | 9 | 直接 import `matplotlib.pyplot`，未像其他绘图/训练模块一样显式使用 `Agg` | 无显示器的 Linux 节点上绘图入口可能受 backend/`DISPLAY` 影响 | 在后续修复阶段统一 headless backend 或由 Linux launcher 设置 `MPLBACKEND=Agg` |

## 4. P2 Issues

| 文件 | 行号 | 问题 | 影响 | 建议 |
|---|---:|---|---|---|
| `scripts/run_phase1c_train.ps1`; `run_phase1c_diagnostic_eval.ps1`; `run_phase1c_v2_train.ps1` | 21; 26; 26 | conda fallback 硬编码 `D:\anaconda\anaconda\Scripts\conda.exe` | Windows 换机器/换安装位置时入口失效 | fallback 改为参数/环境变量；保留 `Get-Command conda` 为首选 |
| 同上 3 个 PowerShell 文件 | 16; 21; 21 | 手工用 `;` 拼 `PYTHONPATH` | 这些脚本若由 Linux PowerShell Core 执行会使用错误 path separator | 明确限定为 Windows launcher，Linux 使用 `.sh`/`os.pathsep` 对应入口 |
| 47 个已跟踪文件（详见 EOL 小节） | N/A | 当前 Windows worktree EOL 与 `.gitattributes` 目标不一致；`run_pilot.py` 为 mixed EOL | 易产生无意义 diff、脚本行为与本地工具差异 | 后续单独做受控 `git add --renormalize` 评审；本次禁止格式化，未处理 |
| `configs/environment_lock/torch_runtime.json`; historical `docs/**/*.json`; `tests/test_repository_metadata.py` | 3, 22; 多处; 158 | 包含本机绝对路径/历史验证路径；test 中的 `E:\...` 是禁止字符串断言而非运行路径 | 作为 provenance 可接受，但若误当运行配置会混淆平台状态 | 标注 snapshot/provenance；运行时不得读取这些路径。测试命中按 false positive 处理 |
| `configs/chapter3/bser_phase1c_train.json`; `bser_phase1c_v2_train.json` | 17; 18 | 默认 device 为 `cpu` | 同一配置不会自动使用 RTX3090，但可用 CLI `--device cuda` 覆盖 | 保留安全默认或提供 Linux launcher 显式 GPU 参数，并记录 resolved device |
| 5 个 chapter experiment executor + 2 个 `tools/` executor | 146/385/207/357/528; tools 40/344 | 未显式统一 start method；当前 worker 主要返回标量/字典/NumPy，且未见父进程 CUDA 初始化 | Windows 使用 spawn、Linux 默认 fork，存在低等级行为/性能差异 | 后续统一 context；优先处理 trainer，低风险 CPU 工具可后置 |
| `outputs/`; `experiments/` | N/A | 本地分别约 4.78 GiB/34.96 MiB；大量 checkpoint/raw diagnostics 不适合传输 Linux 节点 | 手工复制工作区会浪费空间并混入历史结果 | 保留本地且不删除；只通过 Git 传输 compact evidence，checkpoint 用明确的外部 artifact 流程 |
| `.gitattributes` | 5-16 | 已覆盖当前文件类型，但没有 `*.sh eol=lf`；仓库当前无 100755 文件 | 新增 Linux launcher 时可能丢 LF/shebang executable bit | 新增 `.sh` 时同时定义 LF，并在 Git 索引确认 mode 100755；本次无 `.sh`，故不是当前阻断 |
| `3090/resolved_training_config.json`; `3090/worker_failure_episode_0001.json` | N/A; 31 | 审计开始前即为未跟踪且未被 ignore 的本地证据 | 可能被误提交，也可能是需要保留的关键失败证据 | 按 workspace 规则分类为 **REVIEW_REQUIRED**；本次未移动、删除、修改或加入 Git |

## 5. Windows/Linux Compatibility Matrix

|模块|Windows|Linux|状态|
|---|---|---|---|
|core|静态解析 PASS；路径/device 设计总体可移植|静态可移植；正式 CUDA 模式需避免 silent CPU fallback|CONDITIONAL PASS|
|chapter3_bser|常规 CPU 算法/环境可运行；Phase1C Windows launcher 齐全|大部分 CPU 模块可移植；v1/v2 trainer 有 P0|FAIL（训练路径）|
|configs|实验语义配置使用相对路径；Windows lock 可用作本机快照|缺少可安装的 Linux lock；现有 full/explicit lock 不可用|FAIL|
|scripts|3 组 `.ps1`/`.bat` 入口|0 个 `.sh`；PowerShell 内容为 Windows 语义|FAIL|
|tests|98 个 test module 可由 Python 发现，但 2 个 integrity fixture 缺失|无真正 Windows-only test；同样有 2 个缺失 fixture，且无 CUDA multiprocessing 覆盖|FAIL / COVERAGE GAP|
|experiments|compact tracked evidence 可保留；本地 ignored 产物较大|Git clone 可避免大产物；手工复制不建议|CONDITIONAL PASS|
|docs2|文档本身可移植|`overlay_manifest.json` 缺失导致 integrity test 失败|FAIL（完整性）|
|runtime|顶层 `runtime/` 不存在；实际共享实现位于 `core/runtime/`|无需为 Linux复制业务 runtime|N/A / ARCHITECTURE OK|
|Phase1C-v2|`spawn` 与 Windows 模型一致；尚无本次动态验证|当前 3090 失败证据确认 Tensor IPC P0|FAIL|

“PASS”仅指静态兼容性判断；本次未执行动态测试，因此不得把本表当作 CI 或 Linux RTX3090 验收结果。

## 6. Required Changes

### 必须修改（P0/P1 gate）

1. 重构 Phase1C-v2 worker 边界，禁止逐 transition 的 `torch.Tensor` 跨进程传输；保留 spawn，并用真实多 worker 测试验证。
2. 对 v1 trainer 采用同一安全策略，或在 Linux CUDA 正式入口明确禁止不安全的 v1 并行路径。
3. 提供 Linux RTX3090 可创建、可追溯的环境文件，记录 PyTorch CUDA wheel/channel、Python、CUDA/driver compatibility。
4. 恢复或审查两个缺失的 repository-integrity evidence：E0 compact evidence 与 Phase1C-v2 overlay manifest。
5. 增加 Linux CPU spawn smoke 和 Linux CUDA preflight；GPU 正式模式 requested/resolved device 不一致时必须失败。

### 建议修改

1. 为 v1 train、v2 train、diagnostic eval 增加 `.sh` launcher；共享同一 Python 模块和 JSON 语义配置。
2. 对所有 process pool 显式选择 context，并按 worker payload 类型分级测试。
3. 统一 Phase1C 与 `core.runtime.engine.set_ch3_determinism()` 的 seed/provenance 协议。
4. 将 Windows absolute fallback 参数化；统一 headless plotting；受控处理 EOL normalization。
5. 新增 `.sh` 时补充 `.gitattributes` 和 executable bit 验证。

### 无需修改

- 不需要拆分 Windows/Linux 两套 `core`、环境动力学、reward、observation、action 或 mission semantics。
- 不需要为 `training_env.py`、`phase_aware_replay.py`、`reward_adapter.py` 建立平台分支；静态扫描未发现 OS 路径或平台 API。
- 不需要新增顶层业务 `runtime/` 层；现有 `core/runtime/` 仍应是共享实现。
- 不应删除或重写历史 checkpoint、outputs 或 experiment evidence。

## 7. Files Safe To Keep Single Version

以下文件/目录应继续维持单版本，不应复制成 `_windows` / `_linux`：

- `core/**/*.py`，包括 `core/runtime/`、`core/env/`、`core/algorithms/`、`core/replay/`、`core/mapping/`。
- `chapter3_bser` 的算法、controller、online、integration、events、diagnostics 模块。
- Phase1C-v2 的 `training_env.py`、`phase_aware_replay.py`、`reward_adapter.py` 与 `phase1c_common/transition_schema.py`。
- `configs/chapter3/*.json` 与 `configs/scenarios/*.json` 的实验/场景语义配置。
- 绝大多数 `tests/test_*.py`；测试逻辑应同一份，在不同 runner/环境执行。
- Python trainer 也应在修复 IPC 边界后继续保持单版本，而不是创建 Linux trainer 副本。

## 8. Files That May Need Platform Layer

| 区域 | 建议平台层 | 边界 |
|---|---|---|
| `scripts/` | Windows 保留 `.ps1/.bat`；Linux 增加 `.sh` | 只负责环境激活、device/preflight、日志与 `python -m` 调用，不复制训练逻辑 |
| `configs/environment_lock/` | 分别维护 Windows 与 Linux lock/spec | 实验语义 JSON 仍共享；平台依赖清单分开 |
| GPU 作业入口 | Linux 3090 launcher 或 scheduler wrapper | 可设置 `CUDA_VISIBLE_DEVICES`、线程数、文件描述符 preflight、MPL backend；不能用 wrapper 掩盖 Tensor IPC 代码缺陷 |
| CI/验证 | Windows CPU job + Linux CPU spawn job + Linux CUDA preflight | 结果应记录 commit、requested/resolved device、torch/CUDA/cuDNN |
| 顶层 `runtime/` | **不建议创建** | 共享 runtime 已在 `core/runtime/`；平台差异属于启动/环境层 |

## 9. Path Compatibility Scan

运行时路径结论：核心 Python 和实验 Python 主要用 `Path(__file__).resolve()`、仓库相对路径与 `/` 风格 JSON 路径；未发现可执行 Python 中的 `C:\`/`D:\`/`E:\`、`/home/`、`/mnt/` 硬编码运行路径。

| 文件 | 行号 | 类型 | 等级 | 判断 |
|---|---:|---|---|---|
| `scripts/run_phase1c_train.ps1` | 21 | Windows absolute fallback | P2 | Windows launcher 本机路径，不影响 Python/Linux，但换 Windows 机器会失败 |
| `scripts/run_phase1c_diagnostic_eval.ps1` | 26 | Windows absolute fallback | P2 | 同上 |
| `scripts/run_phase1c_v2_train.ps1` | 26 | Windows absolute fallback | P2 | 同上 |
| `configs/environment_lock/conda_environment_full.yml` | 50 | Windows prefix | P0 | Linux 环境创建阻断 |
| `configs/environment_lock/conda_environment_history.yml` | 8 | Windows prefix | P2 | history snapshot；不可直接当 Linux lock |
| `configs/environment_lock/torch_runtime.json` | 3, 22 | Windows installation snapshot | P2 | provenance 信息，不得作为运行路径 |
| `tests/test_repository_metadata.py` | 158 | Windows path literal | 无问题 | 该字符串用于 `assertNotIn`，不会访问路径 |
| `3090/worker_failure_episode_0001.json` | 31 | `/home/legion/...` | 无新增问题 | 失败 traceback 证据，不是配置路径；文件为 REVIEW_REQUIRED |

历史 `docs/**/*.json` 中还存在 `C:\tmp`、`C:\Users\...`、`D:\...` 远程验证记录，均为 provenance snapshot，不是运行路径。

## 10. Multiprocessing Audi

| 文件 | 函数/位置 | start method / CUDA 时序 | 返回值 | 风险与建议 |
|---|---|---|---|---|
| `phase1c_bser_rmaddpg_v2/train_phase1c_v2.py` | `_collect_episode`, `run_training` | 显式 spawn；父进程可先迁移 learner 到 CUDA | 大量 CPU Tensor transitions + scalar dict | **P0**。spawn 已正确但不充分；改普通数组/批量 payload，增加真实 IPC test |
| `phase1c_bser_rmaddpg/train_phase1c.py` | `_collect_episode`, `run_training` | 默认 context；父进程先按 config 迁移 device | 大量 CPU Tensor transitions | **P0**。CUDA-after-fork 与 Tensor IPC 双重风险 |
| `phase1c_bser_rmaddpg/run_phase1c.py` | `_episode`, `run_preflight` | 默认 context；worker 内 CPU load/rollout | metrics + step rows | P2。未见 CUDA-before-fork/返回 Tensor，建议统一 spawn |
| `phase1b1_pilot/run_pilot.py` | `_seed_worker`, `run` | 默认 context，CPU | 标量字典/列表 | P2。Windows/Linux context 不同，当前 CUDA 风险低 |
| `phase1b2_pilot/run_pilot.py` | `_worker`, `run` | 默认 context，CPU | metric dict | P2，低风险 |
| `phase1b3a_diagnosis/run_diagnosis.py` | `_diagnostic_case_worker`, `run` | 默认 context，CPU | metric + recorder | P2，关注 recorder 体积/可 pickle 性 |
| `phase1b_online/run_e2.py` | `_episode_worker`, `run` | 默认 context，CPU | metrics/events/replans | P2，低风险 |
| `tools/run_phase1b1_seed_parallel.py` | `run_seed`, `main` | 默认 context | `Path` | P2，低风险 |
| `tools/run_core_golden_e0.py` | `_run_profile`, `run` | 默认 context，CPU | dict/NumPy-normalized evidence | P2，低风险；不应在本次审计运行 |

判断汇总：

1. Linux 默认 fork 风险：v1 trainer 为 P0；其余未显式 context 的 CPU experiment/tool 为 P2。
2. CUDA 初始化后创建 worker：v1 trainer 可能发生；v2 使用 spawn，不继承父进程 CUDA context。
3. Tensor 跨进程：v1/v2 trainer 均存在，且 v2 有当前 Linux 失败证据。
4. worker 返回大量 Tensor：v1/v2 均是。以 v2 每 step 约 10 个 Tensor storage、默认 400 step 估算，单 episode 可产生约 4000 个独立 storage 对象，FD/ancillary-data 压力显著。
5. 是否需要 spawn：所有 trainer 需要；但 v2 证明 spawn 不是 Tensor IPC 的完整修复。

## 11. PyTorch CUDA and Seed Audi

- 未发现硬编码 `cuda:0`、`.cuda()` 或 `to("cuda")` 的正式调用。
- `config["device"]`/CLI device 已贯穿 v1/v2 learner；worker 环境明确使用 CPU，这是合理的“CPU rollout + GPU learner”结构。
- 正式配置默认 `device: "cpu"`，Linux launcher 必须显式选择 CUDA；正式 GPU 模式应拒绝 silent fallback。
- `core/runtime/engine.py:405-419` 已有较完整的 Python/NumPy/Torch/CUDA/cuDNN determinism 协议。
- Phase1C v1/v2 自有 `_seed_all()` 只调用 Python/NumPy/`torch.manual_seed` 和 `torch.use_deterministic_algorithms(..., warn_only=True)`，未记录 CUDA RNG/cuDNN flags。现代 PyTorch 的 `torch.manual_seed` 会覆盖设备 RNG，但当前实现仍不足以证明 Windows/Linux bitwise 等价。
- 跨 Windows CPU 与 Linux GPU 应追求协议可复现和统计一致，不应承诺浮点 bitwise 一致。

## 12. Encoding and Shell Audi

编码：

- 410/410 个 Git 已跟踪 `.py/.json/.yaml/.yml/.md/.sh/.ps1/.bat` 文件均可严格解码为 UTF-8。
- 未发现 UTF-16、UTF-8 BOM 或非法 UTF-8。
- EOL 汇总：309 LF、95 CRLF、6 mixed（全仓已跟踪相关文本）；必扫范围中有 47 个 worktree/attribute 目标不一致。
- 唯一必扫 Python mixed-EOL 文件：`chapter3_bser/experiments/phase1b1_pilot/run_pilot.py`。
- Git index 对上述文本均为 LF；`.gitattributes` 对 Python/config/docs 强制 LF，对 `.ps1/.bat` 强制 CRLF。当前 EOL 漂移主要与本机 `core.autocrlf=true`/历史 checkout 有关，Linux clean checkout 的索引内容不会携带 Windows-only script EOL 到 Python 文件。

Shell：

| 实验入口 | Windows | Linux | 结论 |
|---|---|---|---|
| Phase1C v1 train | `.ps1` + `.bat` | 缺 `.sh` | P1 |
| Phase1C diagnostic eval | `.ps1` + `.bat` | 缺 `.sh` | P1 |
| Phase1C-v2 train | `.ps1` + `.bat` | 缺 `.sh` | P1 |

## 13. Environment Dependency Audi

实际源码第三方依赖集中为 `torch`、`numpy`、`matplotlib`；未发现 `scipy`、`pandas`、`gym/gymnasium`、`opencv` 等未锁定 import。`torchvision` 与 `torchaudio` 在扫描源码中没有 import，属于环境快照附带包。

| 文件 | Windows | Linux | 判断 |
|---|---|---|---|
| `conda_explicit_spec.txt` | 可重放 win-64 基础环境 | 不可用 | P0 |
| `conda_environment_full.yml` | 当前 Windows AUV 快照 | Windows build/prefix 阻断 | P0 |
| `conda_environment_history.yml` | 最小 Windows history | prefix 需清理/不能当 lock | P2 |
| `pip_freeze.txt` | 记录当前 pip 包 | CUDA wheel source 不完整 | P1 |
| `torch_runtime.json` | RTX3060 Laptop/Windows runtime snapshot | 不是 RTX3090 验收配置 | P2 provenance |

## 14. Outputs, Experiments, and Git Audi

### 文件大小/保留分类

| 区域 | 文件数 | 大小 | Git 状态 | 分类 |
|---|---:|---:|---|---|
| `outputs/` | 65 | 4.78 GiB | 整体 ignored | **KEEP_LOCAL / DO_NOT_TRANSFER_BY_GIT**；本次不删除 |
| `experiments/` | 25 | 34.96 MiB | 19 个 compact/figure evidence tracked；5 checkpoint JSON + `step_diagnostics.csv` ignored | tracked evidence **KEEP**；ignored raw/checkpoint **KEEP_LOCAL** |
| `3090/` | 2 | 小型 JSON | untracked、未 ignored | **REVIEW_REQUIRED** |

最大本地产物为 v2 dry-run checkpoint，约 500.8 MiB；v1 restart checkpoints 单文件最高约 405.4 MiB。没有已跟踪文件达到 5 MiB，当前 commit 没有明显 tracked large-file 阻断。

### `.gitignore` / `.gitattributes

- `.gitignore` 正确覆盖 `outputs/`、`**/_checkpoints/`、`*.pt/*.pth/*.ckpt` 和指定 raw diagnostics。
- `.gitattributes` 正确定义主流文本 EOL 和 checkpoint/image binary 类型。
- 当前没有 `.sh`，也没有 mode `100755` 的已跟踪文件；新增 Linux launcher 时必须补 executable bit 验证。
- 当前未配置 Git LFS。因大模型/checkpoint 已被 ignore，现阶段不是 P0；如未来需要版本化大 artifact，应使用独立 artifact store 或经审查的 LFS 策略。

## 15. Test System Classification

`tests/` 必扫文件共 104：98 个 `test_*.py`、2 个 Python utility、4 个 JSON fixture。

### 1. Runtime tests（85 个 test module）

除下列 13 个 repository-integrity/provenance module 外，其余 85 个 `test_*.py` 归为 runtime/algorithm/contract tests。它们主要覆盖 core env/training smoke、BSER solver/controller、observation/action contract、Phase1B、Phase1C reward/replay/checkpoint。静态扫描未发现 Windows-only API，可在 Linux Python 环境运行；但本次没有动态执行。

### 2. Repository integrity / provenance tests（13 个）

- `test_bser_v1_artifacts_frozen.py
- `test_ch3_e0_equivalence.py`（当前缺 evidence，静态 FAIL）
- `test_clean_clone_standalone.py
- `test_core_has_no_legacy_imports.py
- `test_core_import_graph.py
- `test_core_source_provenance.py
- `test_core_without_legacy_directory.py
- `test_no_legacy_write.py
- `test_phase1a1_original_core_freeze.py
- `test_phase1a_core_freeze.py
- `test_phase1c_v2_isolation.py`（当前缺 overlay manifest，静态 FAIL）
- `test_repository_metadata.py
- `test_verify_external_archive.py

这些测试依赖 Git metadata、`docs/`/`docs2/` provenance、历史 compact evidence 或 archive contract；在 Linux 可运行，但 clean clone 必须包含对应 evidence。

### 3. Windows-only tests

严格意义的 Windows-only test：**0**。

`test_phase1c_v2_isolation.py:38-39` 只断言 `.ps1/.bat` 存在，属于 Windows artifact coverage，但测试本身可在 Linux 运行。当前没有测试执行 PowerShell/batch，也没有测试要求 Linux `.sh` 存在。

## 16. Phase1C-v2 Special Review

审计目标实际目录为 `chapter3_bser/experiments/phase1c_bser_rmaddpg_v2/`；任务文本中的 `rmaddpdg_v2` 是拼写差异，仓库内不存在该目录。

### 1. 是否存在 Windows/Linux 行为差异

存在。Windows 的 `ProcessPoolExecutor` 原生采用 spawn；当前 v2 已在两平台显式 spawn，start-method 差异已缩小。但 Linux 仍通过 Unix file descriptor/ancillary data 重建 Torch storage，当前失败证据表明 Tensor IPC 行为仍与 Windows 实际结果不同。脚本与环境也明显分平台：只有 Windows launcher，环境 lock 为 Windows。

### 2. 是否存在 CUDA multiprocessing 风险

存在，等级 P0。显式 spawn 避免继承父进程 CUDA context；worker 本身固定 CPU，这是正确方向。剩余故障来自 Tensor storage 跨进程传输，而不是 worker 使用 CUDA。当前 traceback 正好落在 `rebuild_storage_fd`。

### 3. 是否需要 Linux runtime wrapper

需要一个薄的 Linux **launcher/job wrapper**，用于环境激活、GPU/driver/device preflight、线程/FD 检查、日志和 scheduler 集成；但不需要新增或复制业务 runtime 层。wrapper 不能修复当前 Tensor IPC P0，只能在代码修复后提供可靠运行入口。

### 4. 是否需要修改代码

需要修改 `train_phase1c_v2.py` 的进程边界/返回 payload，并为该边界增加测试。`training_env.py`、`phase_aware_replay.py`、`reward_adapter.py` 静态扫描未发现平台专用路径、fork 或 worker 创建逻辑，不需要为跨平台目的拆版本；任何后续修改仍必须保持既有 reward/observation/action/mission semantics。

## 17. Audit Validation and Created Files

唯一创建的文件：

- `docs2/cross_platform_audit/cross_platform_audit_report.md

唯一创建的目录：

- `docs2/cross_platform_audit/

审计开始前 Git 状态：

```tex
?? 3090/resolved_training_config.json
?? 3090/worker_failure_episode_0001.json


CH3/CH4/CH5 审计前 byte-tree 聚合 SHA-256：

| Legacy repo | 文件数 | 聚合 SHA-256 |
|---|---:|---|
| CH3 | 369 | `cb3fe0ffbb82c76871cff538eb4724b43b5cb9bafbb9c969b9678ce1a116e3d9` |
| CH4 | 103 | `2a1b5e11e5070fd9611e4bd1af7f2a76c65e5a965a864d9eaa79edc82936cd71` |
| CH5 | 306 | `2dbf797efe594c4b99cbf0d173f9bf2e9c6397f965748192ef488b9b12dc89d6` |

结束复核结果：三组文件数和 digest 与审计前完全一致，确认 CH3、CH4、CH5 byte-for-byte unchanged。

最终 `git status --short`：

```tex
?? 3090/resolved_training_config.json
?? 3090/worker_failure_episode_0001.json
?? docs2/cross_platform_audit/cross_platform_audit_report.md


`git diff --exit-code` = 0，`git diff --cached --exit-code` = 0。结论：tracked working tree 与 index unchanged；没有 source/config/test/script/README diff。状态不为空的原因仅为审计开始前已有的两个 `3090` JSON，以及本任务要求创建的审计报告。
