# 固定 20 场景完整回合开发评估：执行与复现

本文记录 2026-09-28 的 D2 开发评估及其复现入口。第一批 `development_B0_20260928_v1` 因路径记录器越界而中断：原索引 10 的 V3 在已保存 280 步后报错；另外三个子进程受控停止。该批数据完整保留，不计为已完成评估，也不将程序错误归为物理碰撞或未发现超时。

修复只更新独立观察器：保留真实 tracker 索引与原运动停滞代理，无法与当前 guidance 对齐的几何指标明确为空。控制、奖励、感知、动力学和任务规则均未变。失败场景 V3 的有／无观察器完整 400 步对照已通过：401 个有序签名逐项一致，均于 281 步 Found，400 步超时；旧已保存 280 步公共轨迹也一致。结果见 `development_verification_v2.json`。新批 `development_B0_20260928_v2` 已完成全部 100 回合，共 27,790 个物理步，程序失败数为零，来源与输入前后核验通过。旧回合只用于签名交叠验证，不进入新分母。最终结果见 [完整报告](development_results_20260928/report.md) 和 [紧凑证据](development_results_20260928/compact_evidence.json)。

V0/V1/V2/V3/V4 的 Found 分别为 25%/40%/40%/35%/40%，发现前碰撞分别为 45%/35%/50%/25%/35%。按预设安全优先工程规则选中 V3；其停滞仍为 37.41%，不是性能验收或最优证明。下一轮候选方案已另行记录，不改写本轮结果。

本次用户已明确授权直接执行固定 20 场景开发评估。范围为 B0、V0–V4、每回合保留原有 400 步任务上限，共 100 回合、最多 40,000 个物理步。碰撞或任务完成可提前正常终止；无须强行运行到第 400 步。此次不训练、不加载 checkpoint、不执行其余 80 场景，不把开发评估标为论文独立测试或 `performance_passed`。

## 已冻结的实验条件

固定原始零基索引：

```text
0,3,7,10,22,28,31,40,42,46,62,63,67,76,77,78,86,91,95,97
```

索引来自既有 `docs/chapter3/search_diagnostics/3090_20260928/experiment_plan.json`。入口需要完整原始 100 场景 manifest，并校验上述 20 场景的 ID、场景种子和内容哈希；不能传入重排或重新编号的 20 场景文件。环境创新种子固定为 `12729 + 原始零基索引`。

| 变体 | 启用内容 |
|---|---|
| V0 | 原 B0 路径，安全搜索干预关闭 |
| V1 | 连续位置与规划状态的一致性刷新 |
| V2 | V1，加失败退避与分配恢复 |
| V3 | 路径几何安全检查与失效制动 |
| V4 | V1、V2、V3 的全部干预 |

每个子进程串行运行同一场景的五个变体，默认四个独立场景进程并行。每个子进程的 OMP、MKL、OpenBLAS 线程数均为 1；模拟器种子不依赖调度顺序。配置仍用仓库 `configs/chapter3/hgr_train.json` 中的公共运行条件；文件名称不意味着启动 HGR 训练，入口只构造 B0 的零残差控制器。

## 来源封存与本地验证

本次启动时 `framework_sources()` 通过，记录如下：

| 项目 | 值 |
|---|---|
| 本地匹配的整套来源配置 | `windows_existing` |
| 文件数 | 336 |
| 本地来源集合 SHA-256 | `8220c352e2b605c74e275701094195d77fbee53777e5554091f59556d157a33c` |
| 本次演进清单 SHA-256 | `6e8247919258eb35f96d0307dde93d9e1cbab3530b9b8bf02083737399207bec` |
| 冻结历史迁移记录 | 27 条全部保留 |

清单同时包含经审阅的 Git/Linux 字节表示。Linux 应匹配其中完整的一套来源配置；不要求其集合哈希等于上面的 Windows 哈希，不能逐文件混配或修改预期哈希来消除失败。运行中禁止修改封存代码、配置、计划、来源审阅说明和输入 manifest。历史清单与 D1 来源记录未被重写为本次结果。

修复后最新记录的本地验证为以下 69 项测试通过，属于本机验证，不是 CI：

```bash
python -B -m unittest \
  tests.test_safe_search_coverage \
  tests.test_safe_search_coverage_integration \
  tests.test_safe_search_development \
  tests.test_safe_search_analysis \
  tests.test_safe_search_runner \
  tests.test_safe_search_provenance \
  tests.test_safe_search_telemetry -v
```

覆盖观测器只读取原始目标负证据更新实际写入的时间戳，并检查可选的逐智能体归因。它不向控制器回传信息。V0 的观测非干扰检查另有 `--verify-v0 --search-coverage` 入口，比较每步状态、观测、奖励、导航和 Python/NumPy/torch CPU 随机状态。

## 本地实际启动配置

本机使用 `D:/anaconda/anaconda/envs/AUV/python.exe`。下列命令对应修复后批次的输入、配置、输出和并行度；输出目录已经创建，不能用同一目录重跑。需要重新执行时，另选一个全新目录并保留原运行证据。6 个进程只改变独立场景的调度；种子、控制参数和任务上限不变。

```powershell
Set-Location 'E:/gym/code/WORKSPACE/AUV-Thesis'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -B -m chapter3_bser.experiments.safe_search_v1.run_development `
  --manifest '3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json' `
  --config 'configs/chapter3/hgr_train.json' `
  --output-dir 'runs/safe_search_v1/development_B0_20260928_v2' `
  --workers 6 --execute
```

这里只读本地保留结果中的场景 manifest；不会写入 `3090结果`。代码和 Linux 入口没有对该结果文件夹的运行时依赖。

## Linux 同条件复现

先将本次代码及来源清单同步到新的 Linux checkout。此前贴出的实际目录为 `/home/legion/AUV-Thesis/AUV-Thesis-SafeSearch`。`MANIFEST` 必须替换为该机器已经准备好的、与冻结计划一致的 B0 原始 100 场景 manifest 的绝对路径；不需要上传整个 `3090结果` 文件夹。外部 manifest 放在仓库外或 `runs/` 中，不能向受保护的 `configs/` 随意添加 JSON。

```bash
cd /home/legion/AUV-Thesis/AUV-Thesis-SafeSearch
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate AUV
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg

MANIFEST=/absolute/path/to/existing/B0/evaluation_manifest.json
CONFIG="$PWD/configs/chapter3/hgr_train.json"
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="runs/safe_search_v1/development_B0_${RUN_ID}"
test -f "$MANIFEST" || exit 1

python -B -c 'from chapter3_bser.experiments.safe_search_v1.provenance import framework_sources; s=framework_sources(); print(s["checkout_profile"], len(s["inventory"]["files"]), s["inventory"]["sha256"])' || exit 1

python -B -m chapter3_bser.experiments.safe_search_v1.run_development \
  --manifest "$MANIFEST" \
  --config "$CONFIG" \
  --output-dir "$RUN_DIR" \
  --workers 4 --execute
```

`--execute` 是显式实验入口，不会恢复 checkpoint。该入口固定五个变体、20 个原始场景和完整任务上限；不需要再手动传索引或 `--full-episodes`。不要并行启动第二份相同实验来覆盖同一目录。

## 运行进度、完整性与分析入口

顶层目录包含以下文件；尚未完成时，不应把局部计数当作最终 20 场景成功率：

- `identity.json`：全部预定场景、来源集合、输入哈希、种子规则、100 回合与 40,000 步预算。
- `progress.json`：已完成、正在运行、未启动的原始场景索引及程序错误。
- `logs/scene_NNNN.log`：每个子进程的进度和异常。
- `scene_NNNN/identity.json`、`episodes.json`、`summary.json`：该场景五个变体的身份和完整回合结果。
- `scene_NNNN/episode_NNNN/VN/step_trace.jsonl`：运动、规划查询、控制诊断和实际目标观察足迹。
- 顶层 `episodes.json`、`summary.json`：全部子运行通过核验后的 100 条结果和配对汇总。
- `failure.json`：程序失败记录。失败回合不会被悄悄剔除分母，也不会被改标为物理碰撞或任务超时。

只有同时满足以下条件，分析入口才接受整个开发评估：场景元信息和内容哈希与冻结计划一致；20 × 5 条唯一结果全部达到任务终止；每回合请求的是完整 400 步任务；子运行与顶层的来源和输入身份一致；来源前后核验通过；没有程序失败；逐回合摘要与子汇总一致；轨迹步数与终止摘要一致。

完整性通过之后，在 Linux 当前 shell 使用一个全新的分析目录：

```bash
python -B -m chapter3_bser.experiments.safe_search_v1.analyze_development \
  --run-dir "$RUN_DIR" \
  --output-dir "runs/safe_search_v1/analysis_B0_${RUN_ID}" \
  --d1-mechanisms-passed
```

这里 `--d1-mechanisms-passed` 引用本轮修复已记录的 D1 机制验证。四进程运行的墙钟时间不满足预设串行计时比较条件，不能额外传 `--serial-wall-time-comparable`，也不能用并行耗时裁决变体性能。

本次本地后处理已经完成。以下命令只读取完整结果，不启动模拟或训练；复现导出时使用新的输出目录，避免覆盖已封存文件：

```bash
ANALYSIS_DIR="runs/safe_search_v1/analysis_B0_${RUN_ID}"
WINDOW_AUDIT="runs/safe_search_v1/window_audit_B0_${RUN_ID}.json"
EVIDENCE_DIR="docs/chapter3/search_diagnostics/safe_search_v1/development_results_${RUN_ID}"

python -B -m docs.chapter3.search_diagnostics.safe_search_v1.development_trace_audit \
  --run-dir "$RUN_DIR" --output-json "$WINDOW_AUDIT"

python -B -m docs.chapter3.search_diagnostics.safe_search_v1.development_compact_evidence \
  --analysis "$ANALYSIS_DIR/analysis.json" --window-audit "$WINDOW_AUDIT" \
  --output-json "$EVIDENCE_DIR/compact_evidence.json"

python -B -m docs.chapter3.search_diagnostics.safe_search_v1.development_figures \
  --analysis "$ANALYSIS_DIR/analysis.json" --window-audit "$WINDOW_AUDIT" \
  --output-dir "$EVIDENCE_DIR/figures"
```

图表需要原 AUV 环境中的 matplotlib。紧凑导出会再次核对 100 回合、终止步、覆盖计数、配对结果与逐窗口汇总，排除轨迹和网格位置等原始数据。本地额外历史审计见 `development_results_20260928/baseline_audit.json`；它依赖本地保留的旧结果，只作交叠核验，不是新克隆运行评估的前置条件。

## 诊断定义与解释边界

Found 表示首次发现目标；任务成功还包含后续交接与执行，二者分别统计。发现前碰撞导致未发现终止，需要区分碰撞者是搜索者还是待命执行者；仅凭角色分解不能断言控制机制。未发现且在原有 400 步上限终止的回合才是未发现超时。

搜索曝光步数表示发现前实际经历的物理步。停滞与 Hold 占比的分母为三个搜索者的发现前 agent-step；停滞是既有窗口内稳定目标下的运动代理指标。增加存活时间或 Hold 时间本身不能证明搜索有效。

新增目标信念观察足迹来自原始负证据更新实际盖戳的有效网格中心。初始化已观察单元单独记账，不增加物理步数；Found 当步可计入，之后排除。有效观察步数是当步存在新单元或间隔至少一个配置重访半衰期的再次观察的代理指标，不是检测概率、真实连续体积覆盖或障碍地图覆盖。静止连续观察会记作重复证据，不能计作持续新增覆盖。缺少有效观测数据时指标保留为空，不能补成 0。

逐搜索者足迹和重复覆盖仅在重建的三个搜索者并集与实际时间戳一致时发布；归因校验失败会将归因指标留空，同时保留有效的实际并集足迹。不可达查询量反映规划接口/候选请求失败，不能直接当作物理地图不可达率。

配对统计的独立单位是 20 个场景，五个变体共 100 回合不等于 100 个独立样本。开发场景包含已知诊断案例；分析中的 10,000 次配对 bootstrap、95% 区间和新增/丢失发现配对只作为开发比较。其余 80 场景仍未执行；本轮不会自动选择并运行它们，也不会宣称新的独立泛化证据。

## 原始结果保留与 Git

已检查 `.gitignore`：`runs/`、`outputs/`、`/3090结果/`、checkpoint 和模型权重均受忽略；本次 `git check-ignore` 确认 D2 原始输出和本地 B0 manifest 命中这些规则。`git ls-files` 检查未发现这些目录或模型扩展名的已跟踪文件。

Git 中仅保留实现、测试、精简来源记录和经核验的紧凑结果说明。不要强制添加 `runs/`、原始轨迹、整个 `3090结果` 或模型文件。本次未执行 `git commit` 或 `git push`。此运行说明不属于 336 文件封存集合，补充说明不会改变已完成实验的来源身份。
