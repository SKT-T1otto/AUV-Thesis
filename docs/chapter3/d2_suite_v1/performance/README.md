# D2 Suite 性能修复与新实验操作

本目录记录 2026-10-08 Windows 本地性能修复。结果属于最新本地验证，
不是 CI、Linux 3090 实测或正式论文实验。完整测量见同目录 benchmark_summary.json
及 verification.json；源码边界见 [source_review.md](source_review.md)。

## 使用新源码

旧 `linux_pipeline_01` 只保留归档。不要覆盖旧输入、日志、模型，也不要修改
旧计划的 source hash。新源码必须新建 `linux_pipeline_02`，全部学习组从头训练。
对旧 B2 seed 2729/3729 的建议是保留历史记录，退出新版本正式比较；新计划重训
2729/3729/4729，B3/HGR 也保持同一源码身份。旧运行若尚未结束，不要在其正在
使用的源码目录同步改动；先让它结束，或在独立的新源码 checkout 准备新计划。
同步必须包含新增的 `d2_performance` 包、benchmark 工具、演化 manifest 和
source review，不能只复制 tracked diff。结果报告见 [RESULTS.md](RESULTS.md)。

默认仍为每进程 CPU 单线程，串行 seed。`--learner-device cpu|cuda` 和
`--cpu-threads N` 在 prepare 时固化到配置；CUDA 仅用于 B2/B3，HGR 保持 CPU。
不会根据 `cuda_available` 自动切换。小网络设备结论以端到端测量为准。

训练中请读取 `training_metrics.jsonl`、`episodes.jsonl`；D2 HGR 使用
`cycles.jsonl`、`episodes.jsonl`、`branches.jsonl`。已完成的每条记录独立可读，
中断后忽略没有换行符的末尾半条记录。完整损坏记录会报错。正常结束时一次性生成
原名 `.json` 数组，checkpoint 仍保存必要的完整 metadata 并执行加载校验。
周期 checkpoint 不要求重写几十 MB 历史 JSON。

## Windows 验证

在仓库根目录，使用已有 AUV 环境，不安装或改变依赖：

```powershell
$AuvPython = 'D:\anaconda\anaconda\envs\AUV\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& $AuvPython -m unittest tests.test_d2_target_prediction_cache tests.test_d2_mapping_performance tests.test_d2_performance tests.test_d2_performance_provenance
& $AuvPython -m pytest tests -q
& $AuvPython -m tools.benchmark_d2_performance --output runs/d2_performance/review_cpu1 --label review_cpu1
& $AuvPython -m tools.verify_collision_terminal --suite golden --output-dir runs/d2_performance/review_golden
```

每次指定全新的 benchmark/verification 输出目录。benchmark 默认两个已有 M20
开发场景、每场景 5 步热身后测 100 步并继续至自然终止/400 步，以及两个真实训练
回合；不会运行 400000 步，不会恢复旧 checkpoint。开发场景不作为正式实验数据。

冻结 E0 输入目前缺失，最后一条命令会明确报告 blocked。必须找回原始
`experiments/chapter3/e0_core_migration/golden_trace_manifest.json`，其历史 SHA256 为
`b51987de18e93051fcdd7bb88025308f46bdada8fcc6b56a7b84f010ebbe54b9`；不能用当前源码重造
golden 后称作历史等价验证。

若要用同一工具复测旧代码，在当前仓库根目录另开全新临时 clone：

```powershell
$OldCheckout = Join-Path $PWD 'runs/d2_performance/review_old_checkout'
$OldResult = Join-Path $PWD 'runs/d2_performance/review_old_result'
if ((Test-Path $OldCheckout) -or (Test-Path $OldResult)) { throw '请选择全新的输出目录' }
git clone --no-hardlinks . $OldCheckout
if ($LASTEXITCODE -ne 0) { throw 'git clone 失败' }
git -C $OldCheckout checkout --detach 08200428351d0de47e6078094a58927feed611a3
if ($LASTEXITCODE -ne 0) { throw '旧 commit checkout 失败' }
Copy-Item tools/benchmark_d2_performance.py (Join-Path $OldCheckout 'tools/benchmark_d2_performance.py') -ErrorAction Stop
Push-Location $OldCheckout -ErrorAction Stop
try {
  & $AuvPython -m tools.benchmark_d2_performance --output $OldResult --label review_before
} finally {
  Pop-Location
}
```

旧 clone 仅增加诊断工具，production 源码仍来自固定旧 commit。不要与新版本 benchmark
并发执行，也不要复用已有目录。

## Linux：复用原场景，新建 linux_pipeline_02

以下命令在同步并检查最终代码后、Linux 仓库根目录执行。旧实验目录是输入来源，
不会写入旧目录。若旧目录放在其他位置，只改 `OLD_RUN`。先确认旧进程已结束。
若要更改设备或线程数，先完成文末目标主机设备复测，再 prepare 固定选择。

```bash
conda activate AUV
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
OLD_RUN="$PWD/outputs/chapter3/d2_suite_v1/collision_terminal/linux_pipeline_01"
RUN="$PWD/outputs/chapter3/d2_suite_v1/collision_terminal/linux_pipeline_02"
test -f "$OLD_RUN/inputs/train.json"
test -f "$OLD_RUN/inputs/evaluation.json"
test ! -e "$RUN"

python -m unittest tests.test_d2_target_prediction_cache tests.test_d2_mapping_performance tests.test_d2_performance tests.test_d2_performance_provenance
python -m tools.benchmark_d2_performance --output runs/d2_performance/linux_review_cpu1 --label linux_review_cpu1
python -m tools.verify_collision_terminal --suite golden --output-dir runs/d2_performance/linux_review_golden

bash scripts/linux/run_d2_suite.sh prepare \
  --output "$RUN" \
  --train-manifest "$OLD_RUN/inputs/train.json" \
  --eval-manifest "$OLD_RUN/inputs/evaluation.json" \
  --learner-device cpu --cpu-threads 1
python -m chapter3_bser.experiments.d2_suite_v1.linux check --root "$RUN"
bash scripts/linux/run_d2_full_suite.sh --root "$RUN"
```

上述准备和预览不启动正式训练。人工检查本次改动和预览后，再执行：

```bash
LOG="$RUN/nohup_pipeline_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
nohup bash scripts/linux/run_d2_full_suite.sh --root "$RUN" --execute \
  > "$LOG" 2>&1 < /dev/null &
EXPERIMENT_PID=$!
printf 'PID=%s\nLOG=%s\n' "$EXPERIMENT_PID" "$LOG"
tail -f "$LOG"
```

完整入口依次执行 9 个训练任务、10 个评估任务、统一汇总。每 seed 仍是
400000 actual environment steps，完整回合/周期结束可能略超预算；默认正式评估
仍为每任务 100 回合。此任务没有执行上述正式启动命令。

## Linux 3090 上复测设备

Windows 可用 GPU 是 RTX 3060 Laptop，不能代替 3090 主机测量。先在目标主机
完成独立的下列短测，再决定是否在新 prepare 时改变默认参数。各命令顺序运行，
不要让其他训练争抢资源；记录完整场景、seed、Torch/CPU/GPU 型号和源码身份。

```bash
for n in 1 2 4 8; do
  python -m tools.benchmark_d2_performance --compute-only --threads "$n" \
    --output "runs/d2_performance/linux_compute_cpu${n}"
  python -m tools.benchmark_d2_performance --threads "$n" --scenes 0 --training-episodes 1 \
    --output "runs/d2_performance/linux_end_to_end_cpu${n}" --label "linux_cpu${n}"
done
python -m tools.benchmark_d2_performance --compute-only --device cuda --threads 1 \
  --output runs/d2_performance/linux_compute_cuda
python -m tools.benchmark_d2_performance --device cuda --threads 1 --scenes 0 --training-episodes 1 \
  --output runs/d2_performance/linux_end_to_end_cuda --label linux_cuda
```

并行 seed 暂不启用；没有目标主机并发测量，不推荐凭逻辑核数选择 2/3 路并发。
现有 launcher 的运行目录锁继续生效，同一 run 只启动一个入口。
