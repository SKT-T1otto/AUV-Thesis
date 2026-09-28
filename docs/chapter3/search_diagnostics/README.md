# 安全搜索诊断工具

目标是解释 Found 损失，不改变搜索、控制、奖励或终止行为。三个层次分别为：导入结果的事实分解、查询/运动的逐步采集、原实现上的有界机制验证。原始 `3090结果` 和用户保留的 `outputs` 只读。

本次结果见 [3090 诊断结论与修复顺序](3090_20260928/diagnostic_findings.md)，包含总体分桶、两个复现机制和本地验证记录。

下一步见 [代码实施与分阶段实验方案](3090_20260928/next_step_plan.md) 及 [固定实验分组和场景清单](3090_20260928/experiment_plan.json)。后者是待实现的实验规格，不能直接传给现有 runner。

## 离线分析

在仓库根目录执行，无第三方依赖、无模型加载：

```powershell
python -B -m scripts.analyze_search_failures --source-root "3090结果/collision_terminal" --output-dir "docs/chapter3/search_diagnostics/new_analysis"
```

输出目录必须不存在或为空，并位于输入目录外。生成 `diagnosis.json`、`episode_diagnosis.csv` 和 `report.md`。工具验证终止/发现时序、完整性、唯一身份、汇总一致性、诊断一对一关联，记录输入字节哈希，并消除相同 summary/episodes/config 的重复导出。

- B0/B1：同验证场景可做配对描述；相同 seed 不保证不同轨迹的随机事件完全相同。
- B2/B3：本次导入的是持续更新策略的训练轨迹，不是固定 checkpoint 的评价。
- HGR：只选 `purpose=main` 的完整主轨迹；pilot、suffix_training 单列。`scenario_id` 可能重复，身份使用 `dataset_id`。
- 完整搜索暴露步数不是有效搜索时间。没有逐步覆盖证据时，停滞和有效时间保持 null。
- B0 `allocation_reasons` 中少量细分原因来自 anchor 查询，不能套到汇总的搜索不可达次数上。

## 短探针

需要有 PyTorch 的 AUV 环境。只支持不加载网络的 B0/B1，每次最多 100 物理步。开启对照核验时运行两个相同长度的前缀。工具调用原有 `framework_sources()` 校验；不提供跳过来源检查的选项。

```powershell
python -B -m scripts.run_search_diagnostic_probe --run-dir "3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1" --episode-index 0 --steps 30 --current-source-probe --verify-noninterference --output-dir "runs/new_search_probe"
```

`--live-public-map` 额外采集最新公开占据图，要求同时开启 `--verify-noninterference`。新快照仅进入诊断，不刷新控制器持有的快照，也不参与动作计算。对照核验逐步比较公开物理状态、观测、奖励记录、安装的 guidance，以及 Python/NumPy/Torch CPU RNG。

若当前来源 gate 失败，停止物理探针。可在运行记录所指提交的独立 worktree 中复制这些 **scripts 下的诊断 Python 文件** 后运行，冻结生产代码和 gate 均保持原样；不复制到受保护的 `tools/ch3_baselines` 清单。报告保留两边源码身份，来源字节相同也不单独证明跨机器的完整轨迹相同。

探针输出：

- `identity.json`：输入、生产来源、脚本、场景、seed、预算和用途。
- `initial_queries.json`：初始化查询，避免混入逐步查询统计。
- `step_trace.jsonl`：每次真实路径查询的原因计数和有限样本；实际位置/速度、语义目标/跟踪目标、活动线段横向偏差、转角、地图快照年龄、拒绝原因与是否保留分配。
- `summary.json`：有界结果与诊断不干预核验；不输出正式评价通过结论。
- `failure.json`：运行中的异常。来源 gate 在创建输出前失败时，错误在控制台返回。

```powershell
python -B -m scripts.summarize_search_trace --probe-dir "runs/new_search_probe" --output "docs/chapter3/search_diagnostics/new_trace_summary.json"
```

可添加 `--historical-run-dir "3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1"`，离线核对终止步/碰撞坐标，并审计原路径与真实障碍、初始公开占据体素的交集。真实障碍只用于事后审计，从未反馈给在线控制。

后续 B2/B3/HGR 的已获准运行可复用 `QueryTap`、`capture_runtime` 和 `TransitionObserver`；本工具不自行加载它们的 checkpoint 或触发训练。

## 指标边界

1. 发现前碰撞包含三个搜索者和执行者；同一步碰撞/发现遵守原严格终止优先级。
2. 查询失败按实际调用者细分。`no_start_connector`、`invalid_start`、图分量不连通分别统计，不相互替代。
3. 拒绝重分配且分配哈希不变只能证明保留分配，不能单独证明路径已经危险。
4. `motion_stall_proxy` 要求固定窗口内分配稳定、剩余路径进展和净位移都低于阈值。它只是运动停滞代理；绕圈和目标搜索覆盖仍需进一步证据。
5. 新占据体素出现时间不是精确障碍表面首次被传感器看到的时间。碰撞归因必须把速度、航迹、公开地图和控制决策时间对齐。
6. 不将少数诊断场景的机制结论扩展成全部 100 回合的因果占比。

## 验证

```powershell
python -B -m unittest tests.test_search_failure_diagnostics -v
python -B -m unittest tests.test_search_endpoint_diagnostic tests.test_repository_metadata -v
```

第一组无模拟器依赖；第二组运行原图查询的合成连接检查和既有历史来源检查。工具未改动 `core`、Chapter 3 生产代码、冻结 B0、模型或历史 provenance。
