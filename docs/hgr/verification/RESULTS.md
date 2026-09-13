# 团队目标与 HGR 实现验收

2026-09-13 最新记录的本地验收，不是 CI。算法实现、有界真实集成和机制检查完成；正式实验及性能结论尚未完成。[机器可读状态](acceptance_summary.json) 与 [完整实施说明和命令](../README.md) 配套使用。

| 状态 | 值 |
|---|---|
| implementation_complete | true |
| integration_passed | true |
| mechanism_checks_passed | true |
| formal_experiments_completed | false |
| performance_claims_supported | false |
| all_required_verification_passed | false：历史原始输入缺失 |

## 固定源码树与回归

基线 HEAD 为 `0a5bd3a4efdb68f2cca8721ffe0e8dbf2d54e265`，实现仍在工作区，未 commit 或 push。

- 验收源码树 SHA-256：`cebb5de89775406ca1dd3bb26ae2ed2f80fdc90c90f350ea5541803b94495e45`。
- HGR checkpoint 所记录的生产 Python 源码 SHA-256：`0f29d031b818d1c87ca238dade78cde3bb8a8ff9f07f5180cecdac35160cbae8`。
- 初次运行、中断后恢复及最终检查的源码树一致。测试生成产物与本说明不计入生产源码身份。

| 检查 | 实际结果 |
|---|---|
| 当前代码完整回归 | 183 个模块、600/600 个方法通过，0 失败、0 错误、0 跳过；发现/执行列表完全匹配 |
| HGR 数学机制 | 8/8，通过枚举/有限差分、错误预测器校正、一般 q、折扣、无交接/重复样本、梯度抵消及参数隔离检查 |
| 团队奖励实链路 | 5/5，包含真实 PRRAC collector→replay→critic 更新及旧 Actor 初始化/跨目标恢复拒绝 |
| HGR 真实集成 | 4/4，包含两个完整生产外循环、续训、两个随机基线、固定评价、真实 M20 和 spawn 恢复 |
| 固定历史行为对照 | 固定提交 `93a9c8fb53857051390265e3035061bf05402e25` 与当前完整源码副本，各 1 个场景/4 步；逐字段严格一致，依赖隔离通过 |
| 历史套件 | 发现 4 项，1 项通过，3 项因原始输入缺失 blocked |
| E0 golden | 缺少冻结 manifest，0 条轨迹运行，blocked |

全量运行在用户中断时结束，原 runner 的最终进程报告未保留下来。恢复时严格核对固定树、600 项发现列表，以及 178 个模块的逐方法名称、数量和完整 `OK` 结尾，保留其中 550 项已通过记录；其余 5 个模块/50 项正常补跑，退出码均为 0。原 178 个模块的子进程退出码明确记为不可用，未补造。[恢复清单](final_02/continuation/recovery_inventory.json)、[恢复记录](final_02/continuation/recovered_methods.json)、[补跑结果](final_02/continuation/new_methods.json) 和[最终方法列表](final_02/current/methods.json) 可逐项检查。

## 真实更新与成本

主 HGR 集成为两个合法短任务外循环，共 108 个环境步、12 次快照恢复、6 次预测器更新，前后 Actor 各更新 2 次。前阶段参数变化范数分别为 `0.003317084` 和 `0.003253160`；旧梯度、预测和校正均进入同一次 SGD。正式标签保留负值，未制造非零奖励或标签。[梯度和成本分项](final_02/integration/cycles.json) 与[真实分支记录](final_02/integration/branches.json) 保留了策略、快照与随机流身份。

第一轮 checkpoint 续训到第二轮后，前后参数哈希和累计成本与连续运行一致。直接 MC 和直接边界校正各完成一轮真实更新，分别使用 16 和 30 个环境步；这些成本、续训成本和其他集成检查不包含在主 HGR 的 108 步中。上述短任务不能用于推断正式性能。

测试 checkpoint 存在于专用、Git 忽略的输出目录，已再次只读加载并验证当前源码/参数身份，见[持久路径](final_02/integration/retained_checkpoint.json)及[读回结果](final_02/integration/retained_checkpoint_readback.json)。历史用户模型及训练输出未改写或删除。

## 历史缺口与剩余限制

[历史输入清单](final_02/historical/summary.json) 记录：固定历史提交缺少 19 个 `experiments/chapter3/bser_e1_offline/` 产物；本地缺少 E0 的 `equivalence_summary.json`、`per_trajectory_results.csv` 和 `docs2/phase1c_v2_design/overlay_manifest.json`。[Golden 检查](final_02/golden/summary.json) 缺少 `experiments/chapter3/e0_core_migration/golden_trace_manifest.json`。

这些检查没有被标为通过，也没有生成替代历史数据。因此总验收退出码为 1，`all_required_verification_passed=false`；[最终汇总](final_02/summary.json) 中当前回归与历史行为兼容均为 true。27 条历史来源记录与历史哈希仍保留，并通过当前来源检查；只登记了用户授权的精确 registry 演进。

首版 HGR 使用 CPU 串行采集，默认超参数未调优。正式 1000 主轨迹训练、跨方法性能比较和论文创新性/性能优势结论均未执行或建立。Windows/BAT/Linux 的训练、评价、初始化和恢复命令见[实施说明](../README.md)。
