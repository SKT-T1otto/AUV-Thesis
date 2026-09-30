# 一个脚本：先完成全部训练，再完成全部评估

正式总入口：`bash scripts/linux/run_d2_full_suite.sh --root "$RUN" --execute`。

执行顺序固定为：

1. B2、B3、HGR 各 3 个种子，共 **9 个训练任务**，串行完成。
2. 核验全部训练完成回执和最终 checkpoint；全部通过才能进入评估。
3. D2 评估一次，再评估上述 9 个模型，共 **10 个评估任务**。
4. 自动生成四组汇总、配对比较和训练成本报告。

**D2 不需要 checkpoint**：使用冻结 D2 规划器和零残差策略。另外 9 个模型分别加载当前
计划绑定的最终训练 checkpoint。总入口仍将 D2 放在全部训练之后，符合统一阶段顺序。
它不会按每个模型“训练完马上评估”，也不会提前运行 D2。

预算保持不变：默认每个训练任务 400000 实际环境步，不是 100 ep；完整回合／周期结束
可能略微超出预算，HGR 分支采样计入。训练仍使用 CPU。默认每个评估任务 100 个完整回合，
共 1000 个评估回合。任务数随已冻结配置的种子数变化，总入口不接受临时组别或种子筛选。

## 准备一次

同步最终代码，在 Linux 仓库根目录激活 AUV 环境。使用真实的已有场景清单路径。
本次源码身份已更新，旧版本准备的计划不能直接用于新入口；保留旧目录，另准备新目录。

```bash
conda activate AUV
RUN="$PWD/outputs/chapter3/d2_suite_v1/collision_terminal/linux_pipeline_01"

bash scripts/linux/run_d2_suite.sh prepare \
  --output "$RUN" \
  --train-manifest /absolute/path/to/train.json \
  --eval-manifest /absolute/path/to/evaluation.json
```

prepare 只冻结输入，不启动训练／评估，也不会重新生成场景。所有阶段使用同一目录中的
冻结场景和配置。准备后不更新源码、不修改输入、不移动目录。

## 一个命令预览，一个命令执行

只预览计划，不创建模型、不启动回合：

```bash
bash scripts/linux/run_d2_full_suite.sh --root "$RUN"
```

手动启动整个流程，后续无需再次输入评估或汇总命令：

```bash
bash scripts/linux/run_d2_full_suite.sh --root "$RUN" --execute
```

默认预览应显示 train 9 个任务、evaluate 10 个任务。已完成且校验通过的任务标记为
`verified_complete`，运行时会跳过。新入口的完整阶段顺序与旧通用 `run --stage all`
不同；需要本次“先完成所有训练”的顺序时请使用 `run_d2_full_suite.sh`。

SSH 后台运行时，以下命令替代前台执行命令，不能同时启动两份：

```bash
LOG="$RUN/nohup_pipeline_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
nohup bash scripts/linux/run_d2_full_suite.sh --root "$RUN" --execute \
  > "$LOG" 2>&1 < /dev/null &
EXPERIMENT_PID=$!
printf 'PID=%s\nLOG=%s\n' "$EXPERIMENT_PID" "$LOG"
tail -f "$LOG"
```

默认使用当前环境的 Python，也可统一设置 `PYTHON=/absolute/path/to/python`。
停止 tail 不会停止后台任务。

## 日志、报告及失败处理

每次显式启动创建一个全新的目录：

```text
<RUN>/launch_logs/pipeline_<时间和随机标识>/
  console.log
  pipeline.json
  report/
    report.md
    summary.json
    groups.csv
    jobs.csv
    paired.csv
    training_costs.csv
```

控制台会打印实际目录。`pipeline.json` 记录当前阶段、开始／结束时间、失败原因、
各阶段结果及报告位置；状态 complete 仅在训练、评估、汇总全部通过后写入。
`report/summary.json` 的 `suite_complete: true` 表示四组评估齐全。
主指标仍是 Found、发现前碰撞和失败惩罚发现时间；完整任务指标继续保留。

整个流程持有同一把运行锁，训练、评估和汇总之间不会释放它；不要同时调用其他入口操作
这个目录。任何阶段异常都会停止后续阶段，保存失败日志并以非零状态退出。
如训练失败，所有本次评估均不启动；如评估失败，不生成最终汇总。

重复启动会重新校验并跳过完整任务，同时创建新的报告目录，不覆盖历史报告。
未完成任务的已有目录仍会拒绝覆盖，不自动恢复 checkpoint 或重启部分评估。
强制终止可能留下运行锁和 running 状态；先核实原进程已停止，再处理失效锁，不能自动抢占。
训练和评估原始结果仍在 jobs/、receipts/，不会自动删除或上传 GitHub。

本次 22 项本地测试通过，包含真实临时检查点的完整 smoke 链路，以及阶段顺序、失败停止、
来源保护和历史记录检查。总入口 Bash 语法、帮助入口和 4 段文档命令语法检查通过。
记录见 [local_verification.json](local_verification.json)。验证在 Windows／Git Bash 完成；
未启动正式实验，不代表正式性能结果或原生 Linux 运行成功。
