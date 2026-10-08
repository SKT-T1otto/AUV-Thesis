# D2 Suite 性能修复：本地测量与验证

日期：2026-10-08。基线 commit：`08200428351d0de47e6078094a58927feed611a3`。
候选改动尚未 commit。所有结果属于最新本地验证，不是 CI、论文正式结果或 Linux 3090 测量。
本机为 i9-12900H / RTX 3060 Laptop；Torch 2.11.0+cu126。
原始轨迹、checkpoint、计时日志留在被忽略的 `runs/d2_performance/`；
可审查的紧凑证据是 [benchmark_summary.json](benchmark_summary.json)、
[regression.json](regression.json) 与 [verification.json](verification.json)。

## 1. 已确认的瓶颈与修复

按旧版补测的 400 步真实训练回合，主要计算成本依次为环境、控制器/图、learner；长历史日志另测。下述 inclusive 时间存在嵌套，不能相加。

1. 环境步进 986.51 秒（初始批次 258.79 秒）：目标截获预测、原生导航 A*、重复线段采样、格子转换与连通分量。
   后期 10 步诊断中原积分器共调用 5547 次（包含实际目标推进），预测占 9.69/14.29 秒。新增精确物理子步缓存，
   每个 miss 仍调用原积分器，预测长度、积分与反射公式未改；缓存命中复用相同物理输入的结果。
   保留采样点和算术顺序，批量转换格子；按完整几何内容缓存分量、线段结果和采样格子；
   缓存原 Torch 精度的物理边时间。没有缓存概率相关路线，也没有减少环境步或规划决策。
2. 控制器 353.77 秒；完整图提取/构图累计 342.01 秒，分布于控制器与 snapshot 调用中（初始批次分别 111.83/106.28 秒）。
   D2 的实时安全视图仅复制最新公共占用与 agent 状态，正式图仍由原 provider 和原刷新条件提供。
3. Learner 更新及 target 更新约 10.07 秒（初始批次约 3.13 秒）；不是原主要瓶颈。
   设备与线程数现在显式配置，由端到端测量选择。
4. 日志在短回合中成本很小；19 万行历史的独立微基准见第 6 节。改为紧凑 JSONL 逐条追加，
   正常结束才一次性生成兼容 JSON 数组。checkpoint metadata/加载校验保留。

## 2. 实验定义与语义

reward、gamma、actor/critic、28D/3D/124D、batch=128、warmup=256、update_frequency=4、
updates_per_train=2、400000 步预算、collision_terminal_v1、team_mean_v1、BSER 候选、
D2 目标函数、V4 安全恢复、Found/handoff/mission、动力学、随机事件及场景分布均未修改。
CPU1 下两个完整 rollout 和两个真实训练回合：所有 transition、奖励、summary（仅排除 wall_seconds）
及训练模型状态哈希均与旧版本精确一致。global_step=766，optimizer_updates=256。
400 步回合 Found=276、接收目标=277、timeout、无碰撞；366 步回合 Found=26、接收目标=27、success。
CUDA 保留原 CPU OU 噪声随机数流，但 GPU 算术及多线程运算仍须按下表单独检查。

## 3. 统一 before/after 测量

固定 seed 2729、已有 M20 开发场景 0/1，同一工具、相同真实更新节奏。
各场景先热身 5 步，计时 100 步后继续到自然终止；完整区间已经包含 warm100，不能重复相加。
训练耗时包含回合初始化与日志，最终 checkpoint 单独测量。
早期迭代保留；最后旧版补测与最终 CPU1 对照紧邻、顺序执行，完整回归后再跑最终硬件矩阵。
旧版补测来自原 commit 的独立 clone，
仅复制相同 benchmark 工具，production source identity 与初始基线一致。
本机不同时间批次存在明显计时变化，故同时披露初始和补测基线；单次测量不构成统计置信区间。

| 项目 | 初始旧基线 | 旧版本补测 | 最终修复版 |
|---|---:|---:|---:|
| 场景 0 warm100 秒 | 203.184 | 466.383 | 107.614 |
| 场景 0 warm100 steps/s | 0.492 | 0.214 | 0.929 |
| 场景 0 自然回合区间秒（扣除前 5 步） | 393.404 | 1178.756 | 312.713 |
| 场景 1 warm100 秒 | 68.208 | 248.495 | 58.715 |
| 场景 1 warm100 steps/s | 1.466 | 0.402 | 1.703 |
| 场景 1 自然回合区间秒（扣除前 5 步） | 175.936 | 666.538 | 128.048 |
| 真实训练回合 1 秒 | 402.242 | 1456.285 | 452.597 |
| 真实训练回合 2 秒 | 264.148 | 870.954 | 247.109 |
| 766 步训练合计秒 | 666.390 | 2327.239 | 699.705 |
| 训练 steps/s | 1.149 | 0.329 | 1.095 |
| 最终 checkpoint 保存及加载验证秒 | 0.547 | 1.529 | 1.594 |

相对初始基线总加速 **0.95×**；相对旧版补测 **3.33×**。
未达到 5× 整体目标。初始旧版 666.39 秒、相邻旧版 2327.24 秒的明显变化，说明不能忽略批次条件。
相邻两次首回合 learner+target 耗时约 10.07/10.34 秒；完整计时仍是单次样本，没有统计置信区间。
此比例仅对应固定场景的 B2 小规模训练；B3/HGR 正式吞吐量及 24 小时/seed 目标未验证。

400 步回合保留 warmup，因此有 74 次更新；第二回合 366 步有 182 次更新。
不能把首回合耗时直接当成用户所述稳定阶段 400 步/200 updates 的成本。

### A–M 分类：第一个 400 步训练回合

| 分类（inclusive 秒） | 初始旧基线 | 旧版本补测 | 最终修复版 |
|---|---:|---:|---:|
| A rollout | 0.362 | 1.066 | 1.076 |
| B env.step | 258.787 | 986.511 | 282.820 |
| C snapshot | 12.563 | 38.969 | 73.727 |
| D extract | 106.372 | 342.284 | 74.167 |
| E build graph | 106.282 | 342.005 | 74.123 |
| F controller | 111.828 | 353.767 | 82.818 |
| G geometry/path/segment | 0.715 | 1.951 | 2.004 |
| G filter graph（独立列） | 5.740 | 17.076 | 17.115 |
| H route query | 1.375 | 4.906 | 5.050 |
| H A*（已含在 route） | 0.915 | 3.306 | 3.394 |
| I guidance | 0.630 | 1.837 | 1.558 |
| J replay | 0.034 | 0.103 | 0.104 |
| K update+targets | 3.127 | 10.067 | 10.342 |
| L JSON/JSONL | 0.007 | 0.013 | 0.157 |

M checkpoint 见上一表；嵌套关系详见 JSON 中的 `self_seconds`，不应把上表各行相加。

| 调用数 | 初始旧基线 | 旧版本补测 | 最终修复版 |
|---|---:|---:|---:|
| D_extract_planning_state | 328 | 328 | 52 |
| E_build_planning_graph | 328 | 328 | 52 |
| E_endpoint | 5248 | 5248 | 832 |
| H_route_query | 995 | 995 | 995 |
| H_astar | 841 | 841 | 841 |

权威 full_planning_refresh=52、path_safety_checks=828、safety_route_queries=105 均未改变。

## 4. 设备与线程矩阵

每一行独立进程，场景 0、5 步热身、100 步窗口、自然 rollout、一个真实 400 步训练回合。
矩阵与上面的双场景实验有不同缓存历史，只在矩阵行之间比较设备。微基准使用固定合成 replay，
3 次热身、20 次测量；一次 update 包括四个 agent 更新和 target 更新；CUDA 同步计时。

| 设备/线程 | replay sample 毫秒 | update 毫秒 | warm100 秒 | 400 步训练秒 | task/safety 事件一致 | 奖励最大绝对差 |
|---|---:|---:|---:|---:|---|---:|
| cpu/1 | 0.443 | 38.131 | 28.486 | 119.347 | True | 0 |
| cpu/2 | 0.409 | 31.503 | 28.508 | 117.070 | True | 8.9639798e-09 |
| cpu/4 | 0.410 | 28.638 | 29.370 | 118.326 | True | 0 |
| cpu/8 | 0.484 | 32.993 | 31.701 | 390.169 | True | 0 |
| cuda/1 | 1.141 | 30.831 | 109.431 | 387.186 | True | 8.9639798e-09 |

同一设备矩阵的真实回合内部耗时如下，用于区分环境波动和 learner 成本。

| 设备/线程 | env.step 秒 | learner+target 秒 |
|---|---:|---:|
| cpu/1 | 69.633 | 2.960 |
| cpu/2 | 68.475 | 2.448 |
| cpu/4 | 69.436 | 2.355 |
| cpu/8 | 247.376 | 6.807 |
| cuda/1 | 234.543 | 6.566 |

CUDA 子操作可能异步，整回合与微基准有同步。各行环境耗时也有波动，
不能把所有墙钟差异都归因于设备或线程；最快行仅表示这次顺序测量的结果。

补充回放采样压力检查：原容量 500000，分别填入 256 和 400000 条合成零值记录。
只调用原 replay.sample，不推进环境、不做 optimizer update；CPU 存储，batch=128，3 次热身及 20 次测量。
同一填充规模下比较各设备/线程的全部采样索引和结束 CPU RNG 状态，确保设备建议不依赖改变抽样序列。

| 已填充记录 | 设备/线程 | sample 毫秒 | 索引和 RNG 与 CPU1 一致 |
|---:|---|---:|---|
| 256 | cpu/1 | 1.000 | True |
| 256 | cpu/2 | 1.208 | True |
| 256 | cpu/4 | 0.846 | True |
| 256 | cpu/8 | 1.301 | True |
| 256 | cuda/1 | 3.019 | True |
| 400000 | cpu/1 | 26.693 | True |
| 400000 | cpu/2 | 24.458 | True |
| 400000 | cpu/4 | 22.390 | True |
| 400000 | cpu/8 | 22.589 | True |
| 400000 | cuda/1 | 27.136 | True |

设备建议只从任务/安全事件一致且抽样索引/RNG 一致的配置中选择；逐帧浮点差异见紧凑证据。
相对 CPU1，最终模型状态 hash 不同的矩阵配置：cpu2、cpu4、cpu8、cuda1。
线程数/设备变化产生了数值差异，不能称为逐 bit 等价；任务/安全事件和奖励误差已分别披露。
源码修改前后的精确等价结论限定在相同 CPU1 配置。

本机矩阵最快配置为 **cpu2**；差距与每行原始耗时见表，单次样本不代表统计置信度。软件默认仍为 CPU1，不自动选择 CUDA。
Linux 3090 必须在目标主机重跑 README 中同一矩阵；此 Windows 3060 测量不能作为 3090 实测。
推荐 seed 并发数为 **1**。当前只测单进程，未测 2/3 个 seed 的资源争抢，不默认新增并行入口。

## 5. 400000 步外推与剩余瓶颈

双回合样本直接按步数外推约 **101.5 小时/seed**；这是本机短样本估算，不能用作正式完成时间承诺。
将每回合补足到 0.5 update/step，使用该回合实测 replay+update+target 平均时间补偿 warmup：

| 最终版样本 | 原 updates | 补偿后 400000 步外推小时 |
|---|---:|---:|
| 回合 1，400 步 | 74 | 130.7 |
| 回合 2，366 步 | 182 | 75.1 |

后续设备矩阵的最快配置 cpu2，单个 400 步样本直接外推为 **32.5 小时**，补足更新频率后为 **33.7 小时**。
它与双场景对照处于不同计时批次且缓存历史不同，不能替换统一 before/after 的分母；
这些估算不是置信区间，均未验证 Linux 3090 或正式 400000 步完成时间。

估算包含每回合初始化，但没有模拟长程场景分布、长期系统波动、全部周期 checkpoint 和正式评估。
剩余主要成本仍在 CPU 环境/原生导航与必要的权威图构建；不能凭 learner 微基准推断整体收益。
最终 10 步 cProfile 及原始/中间 profile 保存在 benchmark_summary.json，供定位剩余 native A*、
代价计算、占用扫描与图处理成本；profiler 本身的 elapsed 不作为吞吐量。

| 最终 profile，10 步 | 总采样秒 | 原生 A* 秒 | 感知/地图更新秒 | 目标预测秒 |
|---|---:|---:|---:|---:|
| 早期 | 6.691 | 4.702 | 1.100 | 0.000 |
| 后期 200–210 步 | 4.758 | 2.730 | 0.908 | 0.438 |

同一个后期窗口，目标预测累计时间从中间版的 9.685 秒降为 0.438 秒；`advance_target_state` 总调用数从 5547 降为 248（包含实际目标推进）。
预测请求数和预测步数定义不变，缓存 miss 仍执行原始积分器。
前 210 步累计预测物理子步缓存命中 66100 次、miss 1391 次，容量上限 16384。
后期路线诊断 53 次请求中，已有局部缓存命中 33 次；按完整内容检查可再消除的重复计算为 0 次，因此没有新增整条路线缓存。
后续性能工作应优先处理原生 A* 的边代价、边有效性查询与必要图构建；同时保留路径排序和安全语义。
原 replay.sample 的成本还随已填充容量增长：上面的 400000 条压力样本不能被小回放池的短测覆盖。
本次没有改变采样实现；前述外推没有加入实际长程回放池增长成本。

## 6. 日志验证

合成 190000 行旧历史，连续测量 3 次、每次增加 200 行：平均加速 **30.30×**。
旧 JSON 文件约 54.4 MB；这是合成测试，不是用户已有 66 MB 文件。新日志每条独立追加并关闭文件，
完整行可读取；只忽略末尾没有换行符的半条，完整损坏行报错。不宣称突然断电 fsync 持久性。

## 7. 回归、provenance 与边界

- 最终候选完整回归 1117 项通过，0 skipped、0 失败；分成四个独立进程，未与 benchmark 并发。
  完整回归后仅删除测试 fixture 中一处重复赋值，该模块 3 项再次通过；生产源码没有再改动。
- 中间版本首次回归的 13 个旧契约/fixture 失败已修复；原失败记录保留，最终候选重新跑了全部模块。
  另有六项预测缓存测试，包含与旧源码积分器/碰撞函数定义的固定 AST 对照。
- 新鲜 Git clone 加未提交候选覆盖后，source closure/provenance/Linux preview 等 21 项通过。
  未创建 commit，因此不称作已提交 release 的 clean-clone 验证。
- 历史 27 条 provenance 记录及历史 manifest 保留；新增精确 performance evolution 与两个精确 core 文件 hash 钉住。
- 与历史 commit 93a9c8fb53857051390265e3035061bf05402e25 的 4 步 legacy 对照精确通过。
- 冻结 E0 golden 验证实际尝试过，但缺原 `experiments/chapter3/e0_core_migration/golden_trace_manifest.json`，
  实际执行轨迹为 0；没有重造 golden，也没有报告 60/60。完整验证尚未通过，performance_passed 未建立。
- 没有启动正式 400000 步实验、恢复历史 checkpoint、commit 或 push。

## 8. Windows 与 Linux 操作

完整可复制命令见 [README.md](README.md)：Windows 定向/全回归、固定 benchmark、冻结 E0 验证；
Linux 复用旧场景 manifest，新建 `linux_pipeline_02`、check、preview，以及单独列出的人工正式启动命令。
运行中的旧源码目录不要覆盖。同步需包含全部新增文件。正式命令本次未执行。

旧 D2_B2 seed 2729/3729 保留归档，不删除、不与新版本混用。采用新版本后，建议所有正式学习组
2729/3729/4729 从头重训，保持统一源码身份；旧 linux_pipeline_01 不能改 hash 后继续。

## 9. 修改文件清单

- `chapter3_bser/experiments/baselines/common/checkpoint.py`
- `chapter3_bser/experiments/baselines/common/train.py`
- `chapter3_bser/experiments/d2_performance/__init__.py`
- `chapter3_bser/experiments/d2_performance/compute.py`
- `chapter3_bser/experiments/d2_performance/logging.py`
- `chapter3_bser/experiments/d2_performance/options.py`
- `chapter3_bser/experiments/d2_performance/planning.py`
- `chapter3_bser/experiments/d2_performance/provenance.py`
- `chapter3_bser/experiments/d2_suite_v1/cli.py`
- `chapter3_bser/experiments/d2_suite_v1/linux.py`
- `chapter3_bser/experiments/d2_suite_v1/plan.py`
- `chapter3_bser/experiments/d2_v1/assembly.py`
- `chapter3_bser/experiments/hgr/train.py`
- `chapter3_bser/experiments/safe_search_v1/provenance.py`
- `core/env/target_motion.py`
- `core/mapping/path_planner.py`
- `docs/chapter3/d2_suite_v1/performance/README.md`
- `docs/chapter3/d2_suite_v1/performance/RESULTS.md`
- `docs/chapter3/d2_suite_v1/performance/benchmark_summary.json`
- `docs/chapter3/d2_suite_v1/performance/regression.json`
- `docs/chapter3/d2_suite_v1/performance/source_review.md`
- `docs/chapter3/d2_suite_v1/performance/verification.json`
- `docs/provenance/d2_performance_v1_evolution.json`
- `tests/d2_performance_review.py`
- `tests/test_ch3_baseline_provenance.py`
- `tests/test_core_source_provenance.py`
- `tests/test_d2_mapping_performance.py`
- `tests/test_d2_performance.py`
- `tests/test_d2_performance_provenance.py`
- `tests/test_d2_target_prediction_cache.py`
- `tests/test_hgr_phase1_legacy_fixture.py`
- `tests/test_hgr_phase1_runtime_synthetic.py`
- `tests/test_linux_runtime_assets.py`
- `tests/test_openmp_runtime_env.py`
- `tests/test_phase1a1_original_core_freeze.py`
- `tests/test_phase1a_core_freeze.py`
- `tests/test_repository_metadata.py`
- `tests/test_safe_search_provenance.py`
- `tools/benchmark_d2_performance.py`
