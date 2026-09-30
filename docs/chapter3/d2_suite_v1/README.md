# D2 四组统一实验接口

本接口固定同一版 `d2_v1` 搜索规划，比较 D2、D2＋B2、D2＋B3、D2＋HGR。
新增实验编排和只读诊断，并修复 B2/B3 场景清单协议大小写兼容；不改变 D2、环境、奖励、网络和训练算法。
历史 B0/B1、D/V/R/F 结果保留。接口测试不是性能实验，也不证明任何一组优于另一组。

| 组名／命令参数 | 搜索规划 | 学习模块 | 评估动作 |
|---|---|---|---|
| D2 | 冻结 D2 | 无，不训练 | 零残差，保留规划器先验动作 |
| D2_B2 | 冻结 D2 | 独立 direct MC MADDPG | 确定性 actor |
| D2_B3 | 冻结 D2 | 独立 boundary MADDPG | 确定性 actor |
| D2_HGR | 冻结 D2 | 原生 HGR 完整训练周期 | 默认随机策略，固定评估采样种子 |

B2/B3 不加载 HGR 权重。HGR 随机策略与 MADDPG 确定性 actor 的差异会记录在结果中；
这是四个完整方法的比较，不是仅改变一个梯度估计器的严格消融。HGR Phase 1/2 的旧专项脚本
不属于本接口，不能混入本组结果。需要 HGR 确定性均值评估时，必须在准备前的配置副本里
明确设定 `hgr_policy_mode=deterministic_mean`，另建实验，不事后挑选最优模式。

## 默认预算与配对规则

配置：`configs/chapter3/d2_suite_v1/experiment.json`。

- 三个训练种子：2729、3729、4729；共 9 个从头训练任务。
- 每个训练任务 400000 **实际环境步**。HGR 主轨迹、后缀训练、pilot 和 correction
  分支全部按原生计数计入；开始的完整回合／周期跑完再停止，因此报告实际用量和超预算步数。
- 主轨迹计数上限也设为 400000，仅作保护上限，不是计划训练 40 万回合。
  这不是旧版“每组 1000 主轨迹”的预算。不能据此宣称实际计算量严格相等；还要看步数与墙钟时间。
- 每 100 主轨迹设检查点边界，采用预算结束时的最后完整检查点。禁止根据评估集挑最好检查点。
- 每个模型使用相同顺序的 100 个评估场景、相同 `12729 + 场景序号` 环境创新种子。
  三种学习方法使用同一训练场景池，但保留原生采样调度，不宣称训练轨迹逐回合相同。
- D2 只评估一次；另 9 个模型各评估 100 回合，共 1000 个完整评估回合。
  同一个 D2 结果被复用作配对对照，不算三个独立重复。
- 均为 M20、400 步、`collision_terminal_v1`、`team_mean_v1`、gamma=0.95、28D/3D。
  不在 Found 时截断，保留执行阶段结果作为次要指标，以维持完整任务和 HGR 训练契约。

## Windows 启动

先激活已有 AUV Python 环境，在仓库根目录执行。`prepare` 只冻结输入，不训练或评估。
下面两个场景参数替换为实际的已有清单路径；需要 train / validation 正确标签，不能把同一清单重命名成两份。

```powershell
conda activate AUV
$run = "outputs/chapter3/d2_suite_v1/collision_terminal/formal_01"
python -m scripts.run_d2_suite prepare --output $run --train-manifest "实际训练场景清单.json" --eval-manifest "实际评估场景清单.json"
python -m scripts.run_d2_suite run --root $run
```

预检列出 19 个阶段任务：D2 评估一次，另三组×三种子的训练与评估。
确认计划后，由你手动执行完整实验：

```powershell
python -m scripts.run_d2_suite run --root $run --execute
python -m scripts.run_d2_suite summarize --root $run --output "$run/report_01"
```

如果需要一套新场景，明确改用下列准备命令（与已有清单方式二选一）：

```powershell
python -m scripts.run_d2_suite prepare --output $run --generate-scenes
```

默认一次生成并冻结 200 个训练场景（生成种子 82729）和 100 个评估场景（92729）。
生成后所有组都读取副本；实验中不再重新生成场景。已有清单检查 ID、seed 和去除身份字段后的
场景内容是否与训练集重叠，但不能自动证明这些评估场景从未被以前的人工调参看过。
历史用于 V/R/D/F 调优的场景应标为开发评估，不能当作全新盲测证据。

## Linux 启动

当前统一总入口见 [一个脚本完成训练、评估与汇总](pipeline/README.md)：先全部训练，再全部评估。

**全流程在 Linux 完成的最新命令见 [D2 与训练后评估指南](linux_evaluation/README.md)。**

需要重新训练的是 D2_B2、D2_B3、D2_HGR 三组，共 9 个训练任务；D2 不训练。
新增的独立训练／评估入口、日志、互斥锁及完整命令见 [Linux 训练指南](linux/README.md)。
当前训练器实际使用 CPU，CUDA 可用不等于启用了 GPU。

在新克隆仓库根目录，激活已安装依赖的 AUV 环境。下列脚本自动定位仓库，不依赖兄弟仓库。
Windows 和 Linux 均调用同一个 Python 实现。

```bash
conda activate AUV
RUN=outputs/chapter3/d2_suite_v1/collision_terminal/formal_01
bash scripts/linux/run_d2_suite.sh prepare --output "$RUN" --train-manifest /path/to/train.json --eval-manifest /path/to/evaluation.json
bash scripts/linux/run_d2_suite.sh run --root "$RUN"
bash scripts/linux/run_d2_suite.sh run --root "$RUN" --execute
bash scripts/linux/run_d2_suite.sh summarize --root "$RUN" --output "$RUN/report_01"
```

默认使用当前环境的 `python`；可设置 `PYTHON=/absolute/path/to/python`。
准备目录含本机绝对路径，在实际运行机器上重新 `prepare`，不要把 Windows 准备目录直接搬到 Linux 继续训练。

## 分组运行、重复启动和中断

```powershell
python -m scripts.run_d2_suite run --root $run --arms D2 --stage evaluate --execute
python -m scripts.run_d2_suite run --root $run --arms D2_B2 --seeds 2729 --stage train --execute
python -m scripts.run_d2_suite run --root $run --arms D2_B2 --seeds 2729 --stage evaluate --execute
```

默认串行。不要让两个进程同时操作同一个实验目录。重复完整启动会校验并跳过有完整回执的任务；
校验包括计划、场景、源代码、结果和最终检查点哈希。没有回执的已有任务目录视为未完成，拒绝覆盖。
不隐式恢复检查点。中断任务保留现场；重跑需新建准备目录，或另行明确设计恢复流程。
修改预算／种子／评估模式须复制配置并在新目录准备；不要修改已冻结输入。
配置副本应放在临时目录，或 `outputs` 下新建的配置目录，通过 `prepare --config` 指定。
不要把副本放入 `configs`、`scripts` 或生产源码目录：这些目录属于精确来源清单，
增加文件也需要重新审查封存。准备好的单个实验目录则由 `plan.json` 独立固定参数和输入。

## 输出与分析口径

`plan.json` 记录源版本、共同任务条件、场景来源、配置哈希与配对规则；`inputs/` 保存输入副本；
`jobs/<组名和种子>/train/` 保存原生训练结果；`evaluation/` 保存逐回合结果、身份、汇总与进度；
`receipts/` 保存可验证完成记录。原始数据和模型留在已忽略的 `outputs` 下，不提交 Git。

汇总生成 `summary.json`、`groups.csv`、`jobs.csv`、`paired.csv`、`training_costs.csv` 和 `report.md`。

- 主指标：Found 率、发现前碰撞率、失败惩罚发现步数（未发现统一记 400）。
- 辅助指标：仅发现样本的发现步数、两方法都发现时的配对步数差；不能用它们掩盖失败样本。
- 搜索诊断：发现前持续步数，三搜索者每步位移都不超过 0.02 时的停滞步数，
  持续步数减停滞步数得到的运动搜索步数。后者是运动代理量，不等于有效新增覆盖时间。
- 同一步发生发现和碰撞时，保守计入发现前碰撞。完整任务碰撞率和成功率作为次要结果。
- 六种组间配对逐训练种子给出新增发现／丢失发现数量和指标差；差值方向均为右组减左组。
  三种子均值和种子间标准差不当作置信区间，不重复计数 D2，不声称统计显著。
- 同时报告训练实际步数、超预算量、主轨迹数、墙钟时间及 HGR 原生成本明细。
- 未完成任务的成功率保持空值，不能按失败计入分母。可以在运行途中生成新的汇总目录查看完成情况。

本次验证范围与结果记录于 `local_verification.json`；正式性能实验仍需你手动运行后再分析。
