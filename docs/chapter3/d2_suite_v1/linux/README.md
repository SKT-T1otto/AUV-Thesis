# Linux 训练与统一评估

最新总入口见 [单脚本流程](../pipeline/README.md)，一次启动自动完成全部训练后再进行全部评估。

全部实验在 Linux 完成时，请使用最新的 [全程 Linux 操作指南](../linux_evaluation/README.md)。
其中新增 D2 专用评估、三组训练后评估和独立汇总入口；下文保留通用入口说明。

需要从头训练 **3 组**：D2_B2、D2_B3、D2_HGR。D2 是冻结规划器加零残差策略，
无可训练模型。默认每组 3 个种子（2729、3729、4729），共 9 个训练任务。
每个任务预算 400000 实际环境步；HGR 的分支采样计入预算，完整回合／周期结束后停止，
报告实际步数和超预算量。预算、方法与四组配对规则没有因 Linux 迁移而变化。

当前原生训练器使用 **CPU**。机器装有 CUDA／3090 并不意味着这三组会在 GPU 上训练。
预检单独报告 `training_device=cpu` 和 `cuda_available`。本次未改设备选择、网络、
训练算法或 D2 行为；不应预期仅换 Linux 就一定加速。

## 1. 在 Linux 新目录准备

把本次最新代码同步到 Linux 仓库，激活已有 AUV 环境，进入仓库根目录。
`configs/environment_lock/environment_linux_cuda.yml` 是仓库已有环境清单，
本次未更改依赖版本或安装环境。下面命令使用当前环境的 Python；也可设置
`PYTHON=/absolute/path/to/python`，Python 预检时使用同一解释器。

沿用已经确定的训练／评估场景清单，替换下面两个绝对路径。不要重新生成已确定的场景，
也不要拿训练清单改名充当评估清单。所有方法共用准备时保存的同一份场景副本。

```bash
conda activate AUV
RUN="$PWD/outputs/chapter3/d2_suite_v1/collision_terminal/linux_01"
TRAIN_MANIFEST=/absolute/path/to/train.json
EVAL_MANIFEST=/absolute/path/to/evaluation.json

bash scripts/linux/run_d2_suite.sh prepare \
  --output "$RUN" \
  --train-manifest "$TRAIN_MANIFEST" \
  --eval-manifest "$EVAL_MANIFEST"

"${PYTHON:-python}" -u -m chapter3_bser.experiments.d2_suite_v1.linux check --root "$RUN"
bash scripts/linux/train_d2_suite.sh --root "$RUN"
```

默认训练预览列出 9 个任务，不创建模型或启动回合。若要改预算／种子，先在临时目录
创建配置副本，再通过 `prepare --config /path/to/copy.json` 指定；不要把未封存的新配置
放进 `configs` 或源码目录。每次使用新实验目录，不修改已准备的 `inputs` 或 `plan.json`。

Linux 必须重新 `prepare`：旧 Windows 计划含 Windows 绝对路径，且代码身份已更新。
复制场景原文件没有问题；不要直接搬运 Windows 已准备的目录继续训练。完成准备后，
保持源码、配置、场景和实验目录位置不变，直到训练及评估全部完成。

## 2. 手动启动训练

前台运行三组、三个种子，按顺序完成：

```bash
bash scripts/linux/train_d2_suite.sh --root "$RUN" --execute
```

SSH 下希望断开连接后继续，使用以下命令**替代**前台命令，不能同时启动两份：

```bash
CONSOLE="$RUN/nohup_train_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
nohup bash scripts/linux/train_d2_suite.sh --root "$RUN" --execute > "$CONSOLE" 2>&1 < /dev/null &
TRAIN_PID=$!
printf 'PID=%s\nLOG=%s\n' "$TRAIN_PID" "$CONSOLE"
tail -f "$CONSOLE"
```

训练本身串行；停止 `tail` 不会停止后台训练。新入口同时保存
`launch_logs/<时间和随机标识>/console.log` 与 `launch.json`，后者包含平台、解释器、
依赖版本、CPU 设备、计划身份、启动和结束时间、成功／失败状态。
原生逐回合进度和检查点仍在 `jobs/<组和种子>/train/`。

需要分组／分种子启动时：

```bash
bash scripts/linux/train_d2_suite.sh --root "$RUN" --arms D2_B2 --seeds 2729 --execute
```

`--arms` 允许 D2_B2、D2_B3、D2_HGR；D2 会被拒绝。多个参数值用空格分隔。
同一目录的新训练／评估入口使用 `.d2_linux.lock` 互斥锁，不要绕过它同时运行旧通用入口。
进程异常被杀死可能留下锁和 `running` 状态；先核查锁中主机与 PID，确认进程已停止才
手动移除失效锁，程序不会自动抢占。启动日志的 complete 不是四组全完成证明；
以每个任务的校验回执和汇总完成状态为准。

已完整结束且回执通过校验的任务会跳过；中途失败的任务目录保留并拒绝覆盖，
**不自动恢复检查点**。需要重跑失败任务时，在新目录准备相同配置和场景，
再用 `--arms`、`--seeds` 选择任务；不同目录的结果不要手工拼接成一个已完成计划。

## 3. 训练完成后评估四组并汇总

在**同一 Linux 机器、同一目录、同一代码版本**上完成评估：

```bash
bash scripts/linux/evaluate_d2_suite.sh --root "$RUN"
bash scripts/linux/evaluate_d2_suite.sh --root "$RUN" --execute
bash scripts/linux/run_d2_suite.sh summarize --root "$RUN" --output "$RUN/report_01"
```

评估预览列出 10 个任务：D2 一次，加 9 个训练模型；默认共 1000 个完整回合。
学习组必须先有对应最终训练检查点的校验回执。D2 可提前单独评估：

```bash
bash scripts/linux/evaluate_d2_suite.sh --root "$RUN" --arms D2 --execute
```

Found、发现前碰撞、惩罚发现时间为主指标；完整任务指标继续记录。
评估不要仅在 Found 时截断。最终检查点固定由预算结束规则选择，不能按评估结果挑选。

## 4. 带回 Windows 分析

将 **整个实验目录原样复制**到 Windows，包含 plan、inputs、receipts、jobs 和检查点；
不要只复制 CSV 或改写配置里的绝对路径。可在 Windows 做离线校验和汇总：

```powershell
conda activate AUV
python -m scripts.run_d2_suite summarize --root "E:/path/to/copied/linux_01" --output "E:/path/to/copied/linux_01/report_windows_01"
```

这里的汇总按保存的身份与原始字节校验，不在 Windows 重新加载模型执行评估。
源配置里的 Linux 路径要原样保留，不能改成 Windows 路径。报告目录必须是新目录。
原始结果、模型和 3090 结果不提交 Git。本次本地入口验证不是正式实验或性能结论。

## 最新本地验证

34 项本地测试通过，记录见 [local_verification.json](local_verification.json)。
覆盖预检、命令分发、日志、互斥锁、迁移路径拒绝、离线汇总、三种完整来源字节配置和历史元数据保护。
Bash 脚本通过 Git Bash 语法与帮助入口检查；没有原生 Linux 正式运行或性能结论。
