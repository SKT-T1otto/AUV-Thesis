# BSER D0--D3：贡献拆解与 Windows 手动运行

本实验回答：BSER 是否改善实际搜索；收益来自响应加权、待命位置还是二者联合。允许结论为有收益、负收益或证据不足。代码完成不代表算法有效，开发实验不直接作为硕士论文的正式有效性结论。

## 分组

| 组 | 搜索路线评分 | 发现前待命 | 对照含义 |
| --- | --- | --- | --- |
| D0 | 纯探测概率 | 固定执行者初始位置 | 同一框架的无 BSER 对照 |
| D1 | 从固定初始待命点计算响应加权探测 | 同 D0 | D1-D0：响应加权是否有收益 |
| D2 | 先按纯探测选择搜索路线 | 再为既定搜索集合选最佳待命点 | D2-D0：待命优化增量 |
| D3 | 联合选择搜索路线与待命点 | 联合贪心得到的待命点 | D3-D0：完整贡献；D3-D2：联合的额外贡献 |

D1 响应权重从初始锚点计算，不从每步漂移位置替换。固定锚点仍由原控制器返回/保持，物理漂移与风险进入结果。局部规划按原触发范围保留未受影响的搜索者、执行者。

共同条件：B1 V4 的 A+B+C 搜索修复、原候选数量/最小间距、tau=18、0.75 航点阈值、相同感知/目标运动/未知障碍、400 步上限、任一智能体碰撞全队终止、原奖励/28D/3D/124D 接口、零残差。无训练、无模型加载。不使用 R2/R4 制动和收紧阈值。

共同实现修正：在当前状态重算旧分配分数；实际保留旧路线后重算提案分数；部分规划将未受影响搜索者固定。D2 稳定处理后基于实际搜索集合重选待命点。全规划路径探测模型保持原定义，未改为剩余路径或时间折扣模型。

保持同一事件/冷却规则，搜索阶段统一在合格事件接受原子可行且确有变化的提案，不使用相对增益门槛，避免待命变化被纯探测目标零增益抑制。事件时刻仍可因轨迹不同而不同。D0 不是历史 B0 的重命名，D3 也不是逐步等价的历史 B1；论文应说明这是共同实现修正后的贡献实验。旧 B0/V/R 结果原样保留，只作历史参考。

## 样本与运行前固定的指标

固定原20场景索引：`0,3,7,10,22,28,31,40,42,46,62,63,67,76,77,78,86,91,95,97`。同一场景输入与创新随机种子 `12729+原始索引`。默认80完整回合，统计单位是20个场景。Found 后继续原执行阶段至真实终止，本轮首先看发现前表现。

主比较 D3-D0；机制比较 D1-D0、D2-D0、D3-D2，不能只挑有利结果。主指标：Found 比例、发现前碰撞比例、全场景发现步数惩罚均值（未 Found 记400）。辅助看仅 Found 回合时间、共同 Found 场景时间差、停滞/保持步占比、有效观测步占比、未 Found 超时、执行者移动距离、到达待命步数、响应加权改变搜索选择的提案数、候选缺失/锚点不可达计数。代理指标不能直接当真实探测概率。

保留成功率、Found 时执行者距离、handoff 距离/延迟、Found 到成功时间为次指标。共同有观测回合的配对均值有选择偏差，不能单凭这些证明全任务收益，也不能因为 Found 不升就否定潜在响应收益。

汇总给出配对差值、Found 赢/输/平、场景配对 bootstrap 95% 区间（5000次，种子2929）、Found exact McNemar 双侧 p 值。次比较未作多重校正，均为开发诊断，不能据此自动宣称显著优于基线。未完成组的完整 Found/碰撞率为 null；程序错误、中断不算物理失败。

判读：D3 优于 D0 且优于 D2 才支持联合机制的额外贡献；D2 改善而 D3≈D2 则主要支持待命优化。若 D1/D3 很少改变纯搜索选择，检查响应权重及候选区分度；若改变较多但 Found 更差，检查目标偏好、全路径探测模型、时序和待命迁移代价。若只有响应改善，应写成响应优化而不是发现率优化，并独立确认其安全代价。

20个重复使用开发场景未检出收益，不等于证明算法无任何效果。先定位问题，再做预先固定的修正和独立确认；不能反复挑种子或只报告有利场景。`performance_passed` 保持 null。

## Windows 命令

默认清单只读使用本地3090结果，不上传 GitHub。先检查计划：

```powershell
conda activate AUV
Set-Location E:\gym\code\WORKSPACE\AUV-Thesis
python -B -m chapter3_bser.experiments.bser_effect_v1.run_windows --output-dir runs/bser_effect_v1/development_20260929_v1
```

显示 `planned_episode_runs: 80`、`PLAN_ONLY` 后手动启动：

```powershell
python -B -m chapter3_bser.experiments.bser_effect_v1.run_windows --output-dir runs/bser_effect_v1/development_20260929_v1 --workers 4 --execute
```

中断后保持代码/输入/组别不变，续跑：

```powershell
python -B -m chapter3_bser.experiments.bser_effect_v1.run_windows --output-dir runs/bser_effect_v1/development_20260929_v1 --workers 4 --execute --resume
```

workers 可降为2，不改变种子。不加 resume 拒绝覆盖非空目录；算法/输入变化必须新目录。移动清单时加 `--manifest <路径>`，续跑保持该路径。`--arms D0,D3` 只用于另开目录的子集调试，不是完整 D0123。

```powershell
Get-Content runs/bser_effect_v1/development_20260929_v1/progress.json
```

完成标志：`BSER_EFFECT_FINISHED`；progress 的 completed=planned=80、experiment_complete=true；summary 的 source_and_input_verification_passed=true。

子任务在 `jobs/D_Dx_索引/attempt_N` 保存 identity、result、step_trace、planning_audit、initial_queries；父目录 `attempt_N.log` 每20步报告进度。总表 `episodes.json`、`summary.json`。proposal 是提案，decision/trace 才表示实际执行；不能把提案代理收益当实际收益。额外重评分查询进入日志，不应误当失效重规划。

本轮没有运行完整开发实验。手动结束后，将结果目录交给后续分析即可。
