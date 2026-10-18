# D2 Suite 第二轮严格语义等价性能优化

2026-10-10，Windows CPU1；latest recorded local verification，**不是 CI，也不是正式实验验收**。

> 后续机器漂移复测已完成：固定性能核、8 个全新进程、4 组交错配对，完整 rollout 吞吐提升 11.03%，有界真实训练提升 10.98%。当前性能判断以 [REVALIDATION.md](REVALIDATION.md) 为准；下文旧计时保留为历史记录。新复测包含一项略低于 95% CPU/墙钟占比目标的记录，未删除任何测量。

保留的运行时改动只有 `OnlineUnknownMapTaskPlanner` 两处 CPU 标量访问。
原生 planner 的逐边代价计算才是本轮主要热点；TravelCostService A* 的绝对耗时很小。
base graph 复用候选通过精确对照，但端到端仅提高 0.86%，已全部撤回。
没有追加 A* 缓存、数组化、raycast、replay 或日志优化。

HEAD：`c468663acf05fa11b41eb068169ae44cc79a2376`。修改前 `git status --short` 为空，工作树 clean。
最终 HEAD 不变；没有 commit、push、连接 Linux、覆盖正式输出、恢复任何既有正式实验 checkpoint。
**未运行正式 400000-step 实验，未运行 HGR 长训练。**

## 1. 方法、工作负载与证据边界

- 原 HEAD 用 `git archive` 保存至独立基线目录；其生产文件保持原样。
- seed=2729、`learner_device=cpu`、`cpu_threads=1`，Torch 2.11.0+cu126、NumPy 2.2.6。
  CUDA 可用性只用于环境信息记录，计算设备明确为 CPU。
- 本地不存在 Linux 正式冻结 `train.json`；本次使用仓库 M20 开发验证 manifest 的
  `unknown_validation_m20_0001`、`unknown_validation_m20_0002`，不冒称已覆盖正式 slow IDs。
- 最终 B2/B3 对照统一读取独立基线目录中的同一份 manifest 原始字节：SHA256
  `9e02e857f16402a249ff84cef5ae16fa8bea3c416521340144bd953b8e8934bc`。
  Windows 工作树和 archive 的换行会使原始文件 hash 不同，因此不跨这两种字节版本比较。
- rollout 不更新 learner；每个 scene 重置相同模型种子。真实训练保留原 warmup、回放采样与更新节奏，
  两回合自然结束于 400、366 步，合计 766 步、256 次更新；环境 innovation seed 为 2730、2731。
- 训练回合 1 在 step 276 Found，覆盖较长 Search；回合 2 在 step 26 Found。
  rollout scene 0 在 259 步碰撞、没有 Found；scene 1 在 332 步自然结束。
  所有自然终止都参与计时，没有为了性能提前结束。
- baseline profile 在生产修改前完成。第一次工作树 baseline 完成两场 rollout 后，训练被 source gate
  拒绝（诊断工具已改动但当时尚未登记来源演进）；该记录不当作训练证据。
  随后在独立 pristine baseline 通过外部诊断 harness 完成 B2/B3 真实训练，并重新做精确对照。
- 基线与候选串行运行，未与完整回归并发。cProfile 用于热点定位，不用于宣称端到端吞吐。
  最终额外相邻运行一组 before/after 完整 rollout，使用完全相同的最新版诊断 harness。

完整数值、summary 和奖励签名见 [benchmark_summary.json](benchmark_summary.json)。
原始 benchmark、奖励数组、pstats、临时 checkpoint 位于 Git 忽略的
`runs/d2_round2_20261010/`；可见 Git 的交付只有源码、测试、来源记录及紧凑证据。

## 2. 源码审计：forced refresh 与实际变化

`SearchController → start_status/path safety → PlanningViews.refresh → snapshot(force=True)
→ extract_planning_state → build_planning_graph` 的调用链存在。
`PlanningViews.refresh` 每个 runtime step 最多一次 forced snapshot，逻辑事件数不能减少。

1. **只有动态端点需要变化的条件**：grid geometry、valid/occupied 内容、occupancy probability、
   knowledge mode、role 速度/成本参数全部相同，而 agent position/velocity/navigation target 改变。
   例如重复观测已在 logodds clip 上饱和，或没有新的有效 occupancy 改变量。
   同步的 belief/pheromone 等状态仍可能变化，不能把整个 PlanningState 当成不变。
2. **概率、拓扑与代价不是同一个变化条件**：`integrate_obstacle_scan` 依原顺序累加 free/occupied
   votes，再 clamp；logodds 差值大于 `1e-9` 才计 changed cell。changed_count>0 时刷新 sigmoid 概率、
   masks、revision，并清空 geodesic cache。`valid_mask = ~known_occupied_mask`，只在跨 occupied
   分类阈值时变化；free/unknown 分类变化不一定改变 valid。概率即使没跨阈值，仍可改变 entropy/risk
   及 role adjacency planning cost。拓扑取决于有效节点与 edge legality，component labels 取决于该拓扑。
   geometry/cost 参数在本次 episode 中固定，但不能据此设计不完整的通用 cache key。
3. **map_revision 不等于完整图内容标识**：revision 变化可以仅由概率变化引起；不同 runtime 相同
   revision 更不代表内容相同。不可仅按 valid mask 或 map revision 复用 online adjacency。
4. **现有缓存覆盖**：`_TOPOLOGY_CACHE` 保存邻接目标列表，key 包含 planner 类、shape、valid bytes hash
   与 online/layout 标识；不是完整加权图缓存。`_ORACLE_ADJACENCY_CACHE` 只覆盖非 online 图。
   `planner._geodesic_cache` 保存原生路径/邻接等，map refresh 时清空；图构建隔离并恢复这个 cache。
   PublicGeometry 的 16 项缓存复用节点/边几何合法性，仍需处理本次 adjacency、components、endpoints
   和 graph hash。PlanningViews 另按最终 graph hash 复用过滤图。
5. **其余既有缓存**：segment-cell LRU=8192、sampled-segment LRU=32768、physical-time LRU=32768。
   本轮没有扩大容量、改变生命周期，也没有新增全局缓存。

## 3. 基线与候选 profiling

同一 seed、同一 scene 0、初始化之后前 100 个真实 step，开启相同 opt-in cProfile/查询诊断。
下表 inclusive 时间可能嵌套，不能相加。self 是 cProfile 原始 tottime，不是工具边界相减估计。

| 函数 | 调用数 前/后 | 前 inclusive s | 后 inclusive s | 前 self s | 后 self s | inclusive 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| build_planning_graph | 12/12 | 9.915 | 10.928 | 0.035 | 0.041 | 0.907× |
| _topology | 12/12 | 6.568 | 7.018 | 0.322 | 0.347 | 0.936× |
| _component_labels_from_topology | 12/12 | 0.036 | 0.041 | 0.034 | 0.039 | 0.882× |
| _build_role_adjacency | 24/24 | 2.110 | 2.438 | 0.983 | 0.940 | 0.866× |
| _build_endpoint | 192/192 | 0.128 | 0.158 | 0.028 | 0.033 | 0.807× |
| _graph_hash | 12/12 | 0.973 | 1.155 | 0.740 | 0.881 | 0.842× |
| filter_planning_state | 8/8 | 2.318 | 2.758 | 0.952 | 1.147 | 0.841× |
| query | 123/123 | 0.482 | 0.529 | 0.005 | 0.007 | 0.912× |
| _astar | 107/107 | 0.343 | 0.369 | 0.038 | 0.039 | 0.930× |
| _heuristic | 11137/11137 | 0.254 | 0.274 | 0.086 | 0.095 | 0.930× |
| _core_cell_path | 366/366 | 27.917 | 23.263 | 3.207 | 3.845 | 1.200× |
| _edge_time | 217390/217390 | 14.546 | 7.613 | 5.574 | 1.932 | 1.911× |
| _physical_edge_time | 218721/218721 | 8.154 | 4.149 | 7.580 | 2.375 | 1.965× |

profile 总 self：59.231 → 56.410 秒。
TravelCostService：query=123、A*=107；expanded nodes 合计 1728、均值 16.1495、最大 70，
同一 immutable graph 对象的重复 `(role,start,goal)` 查询 27 次，前后逐次 expansion 序列一致。
这里 expansion 指非 stale、非 goal 的邻接行展开。基线 graph 的 9.915 s 中，topology 约占 66.2%、
role adjacency 21.3%、hash 9.8%、endpoint 1.3%、components 0.36%。
本轮没有新增 exact A* cache，所以不能把重复查询数称作 cache hit。

主要实测热点是环境 planner `_core_cell_path` 内的 `_edge_time` 与 `_physical_edge_time`，
两项约 1.91×、1.97×。这并不意味着全任务接近两倍。
原始排序结果见 [baseline_profile_cprofile.txt](baseline_profile_cprofile.txt)、
[scalar_profile_cprofile.txt](scalar_profile_cprofile.txt)。

## 4. 保留与撤回的修改

保留 `OnlineUnknownMapTaskPlanner` CPU float32/float64 的 live NumPy view 读取：

- grid displacement 仍按 tensor 原生精度做相同逐元素减法；传给**原有** physical-time cache 的值不变，
  原有 Torch norm 与 role 算术不变。
- probability 的 NumPy scalar 转 Python float，与原 tensor `.item()` 得到相同值；entropy/risk 公式、
  运算顺序、physical cost、A* heap/parent/tie-breaking/邻居顺序全部不变。
- 每次调用重新取得 view，读取最新 in-place 内容及 replacement tensor；没有概率快照或过期缓存。
- CUDA、其他 dtype 保留旧 Torch 路径；不新增第三方依赖。

float32/64、27 个 cell 全配对、三种 role，逐值 `float.hex()` 对照原 Torch reference；另验证原地修改、
替换 tensor、RNG 不消耗及 A* cell sequence/cost/expansion count。

撤回/不实施：

| 方案 | 证据与决定 |
| --- | --- |
| base graph / dynamic endpoint 拆分 | 完整输入 key 的有界不可变缓存候选，59 次 build 仅 3 hits、56 misses；两场 rollout 231.306 → 229.331 s，吞吐 +0.861%，exact PASS。复杂度不值得，完整撤回；`planning_graph.py` 与 HEAD 一致。 |
| TravelCostService A* memo/heuristic cache/ndarray | 100 步 A* 总量仅约 0.34 s；即使全消除也不是主要收益。未改 dict、heap 或算法。 |
| endpoint index | endpoint 构建与 query 总量较小；不改 `np.allclose` fallback/connector failure semantics。 |
| PublicGeometry 更大缓存 | 已有节点/边几何缓存；无充分证据支持另一层状态缓存。 |
| frozen-searcher raycast | 本采样 raycast 约 2.21 s；优先级低于原生 planner。未跳过任何 observation vote 或地图更新。 |
| replay/logging | 本轮不是主要瓶颈，未改 sampling、multinomial、JSONL 或更新频率。 |

## 5. 端到端性能：真实训练与完整回合分别报告

倍数=before/after；吞吐提升=(倍数−1)，耗时下降=(1−after/before)，二者不是同一个百分比。
真实训练计时含初始化、真实 rollout、replay、learner；最终 checkpoint 单独计时，不混进训练回合。

| 训练 | 步数 | 前 s | 后 s | 前 step/s | 后 step/s | 倍数 | 吞吐提升 | 耗时下降 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B2 | 766 | 364.599 | 315.615 | 2.101 | 2.427 | 1.1552× | 15.52% | 13.43% |
| B3 | 766 | 361.173 | 334.421 | 2.121 | 2.291 | 1.0800× | 8.00% | 7.41% |

两组共四个训练回合合计 725.772 → 650.036 s，累计吞吐提升 **11.65%**（耗时下降 10.44%）。这是本地样本的合计值，不是 Linux 预测，也不是重复测量置信区间。

| 训练 | 回合 | 步数 | Found step | 前 s | 后 s | 倍数 |
| --- | --- | --- | --- | --- | --- | --- |
| B2 | 1 | 400 | 276 | 236.277 | 202.040 | 1.169× |
| B2 | 2 | 366 | 26 | 128.322 | 113.575 | 1.130× |
| B3 | 1 | 400 | 276 | 229.846 | 212.633 | 1.081× |
| B3 | 2 | 366 | 26 | 131.327 | 121.788 | 1.078× |

训练热点汇总如下，每列均为两回合累计秒数。它们可能嵌套，不能加总为 wall time；learner 是 model update + target update。

| 项目 | B2前 s | B2后 s | B2倍数 | B3前 s | B3后 s | B3倍数 |
| --- | --- | --- | --- | --- | --- | --- |
| env.step | 214.383 | 168.246 | 1.274× | 209.976 | 186.138 | 1.128× |
| controller | 49.213 | 46.349 | 1.062× | 46.664 | 50.173 | 0.930× |
| snapshot | 56.687 | 54.092 | 1.048× | 55.234 | 57.240 | 0.965× |
| build graph | 57.047 | 54.489 | 1.047× | 55.571 | 57.810 | 0.961× |
| endpoint | 0.688 | 0.662 | 1.040× | 0.690 | 0.742 | 0.930× |
| adjacency | 11.452 | 11.518 | 0.994× | 11.498 | 12.926 | 0.890× |
| route query | 5.473 | 5.338 | 1.025× | 5.501 | 5.803 | 0.948× |
| TravelCost A* | 3.542 | 3.528 | 1.004× | 3.590 | 3.836 | 0.936× |
| learner | 18.820 | 18.148 | 1.037× | 21.836 | 22.941 | 0.952× |

相邻补测的**完整 rollout**（含第一步至自然终止，排除单独初始化，两个 scene 合计 591 步）：
99.153 → 75.075 s，5.961 → 7.872 step/s，
1.3207×，吞吐 +32.07%，耗时下降 24.28%。
这组完整回合计时和两组真实训练都有独立 exact PASS。

| 完整 rollout scene | 自然长度 | 前 s | 后 s | 倍数 |
| --- | --- | --- | --- | --- |
| 0 | 259 | 75.404 | 52.597 | 1.434× |
| 1 | 332 | 23.749 | 22.478 | 1.057× |

此前另一组完整回合为 224.267 →
76.912 s（2.916×），
虽然 exact PASS，但 graph hash、TravelCost A*、网络前向等未改动代码也整体变快约 2 倍。
同一旧源码随后的 control_before 又明显快于此前基线，说明机器运行条件发生了实质变化。
因此这组大倍率**不归因于本轮源码优化**，保留于 JSON 的 timing_anomaly 字段。
上述 control 对照使用两边完全相同的外部启动方式，控制了入口差异；它仍是本机当前状态下的样本，
与较早真实训练结果分别报告，不合并成一个跨机器状态的加速率。

控制复测中，未改动的 graph hash 仍快约 9.02%，网络前向快约 7.87%；系统噪声没有完全消除。因此完整回合的 +32.07% 是观测值，不能严称为净源码收益；不据此宣布稳定达到 25% 以上的性能档位。同一旧源码两次运行的 transition/reward/summary/counters 也经独立 exact compare 确认为一致。

| 较早批次 rollout（排除前5步） | 前 s | 后 s | 吞吐变化 |
| --- | --- | --- | --- |
| B2 | 214.045 | 213.627 | +0.20% |
| B3 | 232.298 | 201.366 | +15.36% |

Windows 批次波动明确存在：未直接修改的 graph/A*、learner 在 B3 批次也有变慢，
较早 rollout 的 B2 接近持平、B3 +15.36%。不能选择最好的一场作为总体收益，也不能把批次差异当成 B2/B3 算法差异。
目前证据支持“局部热点明显改善，端到端收益中小、依赖工作负载”；不支持第二轮再获数倍加速。
小于 10% 的场合继续堆缓存不值得，因此只保留这段很小、已逐值验证的改动，并停止扩展范围。
Linux 的真实收益尚未知，必须由用户在独立目录按最后一节的命令复核。

## 6. Exact regression

对照工具不使用浮点 tolerance；实际比较完整奖励数组而不只比较奖励 hash。
transition hash 覆盖 observations/actions/rewards/next_observations/dones 的 dtype、shape 与原始 bytes。
summary 递归比较仅排除 `wall_seconds`；来源身份与 profiling 时间另行保存。
模型 hash 覆盖 `training_state_dict()`，包含 learner 与 optimizer 状态。
训练第二回合的 transition/reward signature 是从第一回合开始累计的原 benchmark 定义，不是独立第二回合签名。

| 检查 | B2 | B3 |
| --- | --- | --- |
| transition hash / actions / obs / next_obs / dones | 完全一致 | 完全一致 |
| reward arrays | 完全一致 | 完全一致 |
| model / optimizer state hash | 完全一致 | 完全一致 |
| Found / handoff / collision / termination | 完全一致 | 完全一致 |
| 全部 safety counters / failure semantics | 完全一致 | 完全一致 |
| route query / A* calls / forced refresh | 完全一致 | 完全一致 |
| global_step=766 / optimizer_updates=256 | 完全一致 | 完全一致 |

| 组 | 回合 | transition SHA256（前后相同） | model SHA256（前后相同） |
| --- | --- | --- | --- |
| B2 | 1 | `e4d3ae9719f15d417cc6dc6ac51db131a65030c0b852cea9d7545d30307dad9b` | `8b294c1c73c53ad5a88f897778a8bc5d876b8bb95c0ce0d9591158458d7bccaf` |
| B2 | 2 | `d165d615a11b278e4776c48cc3aef5ffb71efdb0df56b3133161f3587cbf10e2` | `6a699da6f2eaaac9269ecc9d575a5aa5e548c5d13bfc78a2fc3f94fbda3b20b1` |
| B3 | 1 | `ce52fcc7fabae6089ddaa0e6257ea0a0d7cd348b8c6418ac69f60ca3ed2b80e7` | `df34ac6dcf072edf6dba2ccf01750249333b5b955c88423e29c19962226e76e0` |
| B3 | 2 | `375dfbdbb7c00fb2b3142966d94cec6c4ea4a175e43daa7c75380b2a5e31a6ca` | `1a06e26c56aba3ae3e78494b3ec86e6640d6c2b165a2b033a184cb451afe35bc` |

每组两回合调用数均为：env.step=766、snapshot=794、build graph=73、role adjacency=146、
endpoint=1168、controller=764、route query=2128、TravelCost A*=1954。
第一训练回合 path_safety_checks=828、forced_planning_refreshes=30、safety_route_queries=105，
完整两回合 safety 字典及所有失败类别都随 summary 保存在 JSON 中并逐项比较，未减少逻辑查询或刷新。

## 7. 测试、来源与未闭合验证

完整现有测试集：**1125 tests，0 failures，0 errors，0 skipped**，
另有 **1344 subtests passed**；JUnit 共 2469 条记录（包含子测试）。
四个隔离进程运行全部已收集测试，累计进程时间 4823.89 s，
实际 wall 1416.91 s；见 [pytest.log](pytest.log)、
[pytest_inventory.json](pytest_inventory.json)。没有与 benchmark 并发。
四个 pytest 子进程退出码均为 0。初版汇总控制器把 JUnit 的子测试记录数误当主测试数，
其计数断言报错；随后独立按完整 collection、逐组通过日志与原 XML 核对，得到上述主测试/子测试拆分。
没有改写任何原始测试结果，后续汇总器已修正此计数口径。
此前针对 scalar/mapping 的 8 项测试已通过；完整 suite 覆盖这些测试以及新增 comparison/provenance 检查。
本轮净新增 8 项测试（scalar 3、comparison 3、round2 provenance 2），没有删除、跳过或弱化旧检查。
首次全回归因旧 `final_checkout` fixture 仍使用第一轮 manifest 失败，已中止并保留诊断日志。
修复 fixture 的 successor 链，并将新 review 文档存成跨平台稳定的 LF 字节后，重新执行完整 1125 项；
旧 manifest 和历史断言均未改写。最终测试结果来自修复后的全量执行。

全部 27 条历史 provenance 记录和所有旧 manifest 保留原字节；新增
`docs/provenance/d2_performance_v2_evolution.json` 精确记录 successor。
源 gate 校验 parent 原 hash、每个 profile、逐文件 before/after、review 文档、范围与 aggregate；
核心 planner 的新 raw/AST hash 另以测试常量钉住，不能靠改 manifest 绕过。
该 gate 仅接受三处既有来源文件改动和一处新增 verifier，没有目录级豁免。

冻结 Phase 0B-2 E0 命令实际尝试，但原
`experiments/chapter3/e0_core_migration/golden_trace_manifest.json` 缺失。
结果为 **blocked_missing_historical_input**，实际 golden 执行轨迹为 0。
没有重造 golden，没有声明 E0 通过或 `performance_passed`，也没有把短测称为正式 thesis evaluation。
该限制不等于当前精确 D2 对照或全量本地测试失败；它仍是独立、未闭合的历史验证门槛。

执行的主要命令：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:AUV_HGR_EVIDENCE_DIR='runs/d2_round2_20261010/test_hgr_evidence'
$env:HGR_ENTRY_EVIDENCE_DIR='runs/d2_round2_20261010/test_hgr_entry'
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m pytest tests -q --junitxml=runs/d2_round2_20261010/pytest.xml --basetemp=runs/d2_round2_20261010/pytest_tmp
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -m tools.verify_collision_terminal --suite golden --output-dir runs/d2_round2_20261010/golden
```

Python 的 ctypes/Torch DLL 在当前文件沙箱内会被拒绝访问；这些本地验证使用已获自动批准的 Python 执行权限，
没有改系统配置。Git safe.directory 仅对当前命令临时设置，没有改全局 Git 配置。

## 8. Linux 3090 机器 CPU1 最终验证命令

本节命令**没有在 Linux 执行**。运行中的正式源码和输出继续保留。
准备两个独立目录：baseline 为原 HEAD 的 pristine checkout；optimized 包含本轮全部新增/修改文件。
只读原 frozen manifest，复制到全新 A/B 结果目录；不使用旧模型、不恢复 checkpoint、不运行 400k。

```bash
export BASELINE=/absolute/path/to/pristine-c468663
export OPTIMIZED=/absolute/path/to/round2-optimized
export MANIFEST=/absolute/path/to/outputs/chapter3/d2_suite_v1/collision_terminal/linux_pipeline_02/inputs/train.json
export AB_OUTPUT=/absolute/path/to/new-round2-ab-results
export PYTHON=python
bash "$OPTIMIZED/docs/chapter3/d2_suite_v1/performance_round2/linux_ab.sh"
```

[linux_ab.sh](linux_ab.sh) 自动从 manifest 解析 scenario_id→index，先验证每个 ID 唯一存在：
普通 `unknown_train_m20_0072` 配慢场景 `unknown_train_m20_0200`；
普通 `unknown_train_m20_0089` 配慢场景 `unknown_train_m20_0120`。
两个组合均执行 B2/B3 的 baseline→optimized，seed=2729、CPU1、各两回合真实训练，并运行 `--require-training` exact compare。
脚本用相同的优化版 diagnostic harness 外部加载到两个 checkout；基线封存的 tool/生产文件不被覆盖，
也不修改 `sys.path`。manifest 原始字节、scenarios、seed、线程、模型起点完全相同。
源码来源 gate 仍在各自 checkout 正常校验；任何 exact 差异均以非零退出码停止。
为了控制机器负载，可在现有正式进程结束后或独占 CPU 时运行；无需为本次 Windows 修改暂停现有任务。

**不建议为了本轮优化重新跑当前正式 9 个 training jobs。** 当前证据没有改变科学结果，速度收益不足以抵消重跑成本；
本轮 source identity 已改变，也不能自动把旧来源绑定的 checkpoint 接到新源码。
保留现有 Linux 实验连续性；未来新实验是否采用本轮代码，等 Linux exact A/B 的实际收益再决定。

## 9. 文件改动与最终 Git 状态

| 文件 | 作用 |
| --- | --- |
| `core/mapping/path_planner.py` | 唯一运行时优化：逐边 grid/occupancy 的 CPU 标量读取。 |
| `tools/benchmark_d2_performance.py` | 诊断时间细分、opt-in A* expansion/repeat、完整 episode 计时及 bounded profile 参数。 |
| `tools/compare_d2_performance.py` | fail-closed 精确对照，缺签名/缺训练/重复 case 均拒绝。 |
| `tests/test_d2_scalar_planning.py` | 原 Torch reference、live view 与 A* 精确回归。 |
| `tests/test_d2_benchmark_comparison.py` | 对照工具的负例与 metadata/clock 白名单测试。 |
| `chapter3_bser/experiments/d2_performance/round2_provenance.py` | 精确 successor 来源校验。 |
| `chapter3_bser/experiments/safe_search_v1/provenance.py` | 接入新 successor；保留全部旧 gate。 |
| `docs/provenance/d2_performance_v2_evolution.json` | 新范围、profile、before/after 与 review hash。 |
| `tests/d2_performance_review.py` | 历史常量保留，追加获授权的精确 current raw/AST pin。 |
| `tests/test_d2_round2_provenance.py` | 篡改来源、parent、范围、其他 profile 时必须拒绝。 |
| `tests/test_safe_search_provenance.py` | 测试复制与多 profile 验证包含新 successor。 |
| `tests/test_ch3_baseline_provenance.py` | 当前 checkout fixture 跟随最新 manifest，逐级验证第一轮到第二轮的 before/after。 |
| 本目录 `source_review.md`、本报告、JSON、profile 文本、pytest 日志、`linux_ab.sh` | 可追溯审查、紧凑证据、人工 Linux A/B 命令。 |

`planning_graph.py` 的试验改动已撤回；正式配置、网络、奖励、RNG、地图更新与安全策略均无 diff。
以下 `git diff --stat` 只显示 tracked diff；新文件在其后的 `git status --short` 显示，尚未 staging。

```text
 .../experiments/safe_search_v1/provenance.py       |   6 ++
 core/mapping/path_planner.py                       |  19 +++-
 tests/d2_performance_review.py                     |  13 +++
 tests/test_ch3_baseline_provenance.py              |   8 +-
 tests/test_safe_search_provenance.py               |   6 ++
 tools/benchmark_d2_performance.py                  | 107 ++++++++++++++++++++-
 6 files changed, 148 insertions(+), 11 deletions(-)
```

```text
 M chapter3_bser/experiments/safe_search_v1/provenance.py
 M core/mapping/path_planner.py
 M tests/d2_performance_review.py
 M tests/test_ch3_baseline_provenance.py
 M tests/test_safe_search_provenance.py
 M tools/benchmark_d2_performance.py
?? chapter3_bser/experiments/d2_performance/round2_provenance.py
?? docs/chapter3/d2_suite_v1/performance_round2/
?? docs/provenance/d2_performance_v2_evolution.json
?? tests/test_d2_benchmark_comparison.py
?? tests/test_d2_round2_provenance.py
?? tests/test_d2_scalar_planning.py
?? tools/compare_d2_performance.py
```

最终 HEAD：`c468663acf05fa11b41eb068169ae44cc79a2376`。所有输出 checkpoint/raw benchmark 保持 ignored，正式 retained outputs 没有改动。
