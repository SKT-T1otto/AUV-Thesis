# 全程 Linux：D2 评估、三组训练与评估、统一汇总

默认全流程现已整合为 [一个脚本](../pipeline/README.md)：先训练全部 9 个模型，再进行全部 10 个评估。
下文保留按阶段手动调用的可选方式。

本流程在同一 Linux 仓库、同一个新实验目录里完成，不需要回 Windows 运行任何阶段。
先同步本次完整代码，再准备目录；完成准备后不要更新源码或移动目录。此前旧版本已准备的
计划绑定旧源码身份，不能直接混用本次新入口。保留旧目录，用新目录重新准备。

## 入口和任务数量

| 脚本 | 默认任务 | 是否需要学习检查点 |
|---|---|---|
| `evaluate_d2_reference.sh` | D2 单独评估 100 个完整回合 | 不需要 |
| `train_d2_suite.sh` | B2、B3、HGR 各 3 个种子，共 9 个训练任务 | 从头训练 |
| `evaluate_d2_trained.sh` | 上述 9 个最终模型各评估 100 个完整回合 | 必须有对应最终训练回执与检查点 |
| `summarize_d2_suite.sh` | 四组汇总、六种组间配对、训练成本 | 校验已保存的产物 |

三组训练默认每个任务 400000 实际环境步；完整周期停止规则和 HGR 分支计步不变。
三组当前仍使用 CPU，CUDA 可用不代表自动启用 GPU。D2 无需训练。
四组共 1000 个完整评估回合，共用同样的 100 个评估场景及创新种子。
单独的 D2 结果只计算一次，配对时复用，不当作三个独立重复。

## 1. 准备一次

在 Linux 仓库根目录执行。把两个场景路径换成已经确定的真实清单路径，不重新生成场景。

```bash
conda activate AUV
RUN="$PWD/outputs/chapter3/d2_suite_v1/collision_terminal/linux_full_01"
TRAIN_MANIFEST=/absolute/path/to/train.json
EVAL_MANIFEST=/absolute/path/to/evaluation.json

bash scripts/linux/run_d2_suite.sh prepare \
  --output "$RUN" \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST"

"${PYTHON:-python}" -u -m chapter3_bser.experiments.d2_suite_v1.linux check --root "$RUN"
```

默认使用当前 AUV 环境的 Python，可统一设置 `PYTHON=/absolute/path/to/python`。
准备和预检不启动训练或评估。`RUN` 必须是新目录。

## 2. 预览三个执行阶段

```bash
bash scripts/linux/evaluate_d2_reference.sh --root "$RUN"
bash scripts/linux/train_d2_suite.sh --root "$RUN"
bash scripts/linux/evaluate_d2_trained.sh --root "$RUN"
```

不带 `--execute` 均为只读预览，不创建模型、不启动仿真。
训练前最后一条正常显示 `ready: false` 和 9 个 `missing_training_jobs`。
只有所选任务的训练回执、最终检查点与全部绑定产物通过校验后，训练后评估才显示
`ready: true`。缺失任务只在预览中列出；显式执行时会在任何本次评估开始前报错。
产物损坏直接报错，不当成“还没训练”。

## 3. 顺序完成全部实验

```bash
bash scripts/linux/evaluate_d2_reference.sh --root "$RUN" --execute && \
bash scripts/linux/train_d2_suite.sh --root "$RUN" --execute && \
bash scripts/linux/evaluate_d2_trained.sh --root "$RUN" --execute && \
bash scripts/linux/summarize_d2_suite.sh --root "$RUN" --output "$RUN/report_01"
```

这四条顺序执行，任一步失败都停止后续阶段。也可以分阶段手动运行。
D2 可放在训练前或训练后执行，但最终汇总需要它的完成结果。评估入口不会启动训练。
每次显式评估自动保存 `launch_logs/<时间和随机标识>/console.log`、`launch.json`；
逐回合结果、进度、身份和汇总保存到 `jobs/<组名和种子>/evaluation/`。
完成任务以 `receipts/` 的可验证完成记录为准。

SSH 后台运行整个顺序流程时，用下面命令**替代**上面的前台流程，不要同时启动两份：

```bash
LOG="$RUN/nohup_full_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
nohup bash -c '
  set -euo pipefail
  bash scripts/linux/evaluate_d2_reference.sh --root "$1" --execute
  bash scripts/linux/train_d2_suite.sh --root "$1" --execute
  bash scripts/linux/evaluate_d2_trained.sh --root "$1" --execute
  bash scripts/linux/summarize_d2_suite.sh --root "$1" --output "$1/report_01"
' _ "$RUN" > "$LOG" 2>&1 < /dev/null &
EXPERIMENT_PID=$!
printf 'PID=%s\nLOG=%s\n' "$EXPERIMENT_PID" "$LOG"
tail -f "$LOG"
```

停止 `tail` 不会停止后台实验。所有阶段保持同一源码、目录、场景和配置。
同一目录按顺序运行；新训练／评估入口有互斥锁，不要绕过它并发调用旧通用入口。
终止进程可能留下 `running` 状态和锁；核实进程已停止后才能手动移除失效锁。

## 4. 三组训练完成后的单独评估

如果训练已通过本次版本完成，不必重跑 D2 或训练：

```bash
bash scripts/linux/evaluate_d2_trained.sh --root "$RUN"
bash scripts/linux/evaluate_d2_trained.sh --root "$RUN" --execute
bash scripts/linux/summarize_d2_suite.sh --root "$RUN" --output "$RUN/report_01"
```

只有需要部分模型时才显式选择子集，例如：

```bash
bash scripts/linux/evaluate_d2_trained.sh --root "$RUN" --arms D2_B3 --seeds 2729 --execute
```

默认不会挑子集：完整执行三组的所有已配置种子。不能用历史 HGR/B2/B3 检查点替代当前计划
绑定的训练结果，也不能按评估分数选择检查点。D2 专用入口不接受训练种子或组名覆盖。

有校验通过的评估回执会跳过，不重复计入。没有回执的旧评估目录会拒绝覆盖；中断现场保留，
不自动重启部分回合或恢复训练。需要重跑时另建目录，不手工拼接不同计划的结果。
原通用 `evaluate_d2_suite.sh` 仍可一次评估全部四组，但本指南使用分阶段入口。

## 5. Linux 上查看结果

最终 `report_01/` 包含：

- `report.md`：四组主要结果与解释。
- `groups.csv`、`jobs.csv`：分组及逐模型结果。
- `paired.csv`：相同场景上的六种组间配对。
- `training_costs.csv`：实际环境步、超预算量和训练耗时。
- `summary.json`：完整机器可读结果及 `suite_complete`。

只有 `suite_complete: true` 才表示四组评估全部结束。
途中可向新的报告目录汇总进度；缺失任务不记为失败，未完成组的率保持空值。
主指标仍是 Found 率、发现前碰撞率和失败惩罚发现时间；同时保留完整任务指标。
所有评估跑到完整任务终止，不在 Found 时截断。

本次 38 项本地测试通过，包含完整临时 smoke 链路；记录见 [local_verification.json](local_verification.json)。
Bash 脚本和指南命令通过语法检查，三个新入口通过帮助检查。验证在 Windows／Git Bash 完成，
尚未在原生 Linux 执行，不代表正式性能结果。
结果及检查点继续保存在被忽略的 outputs 下，不上传 GitHub。
