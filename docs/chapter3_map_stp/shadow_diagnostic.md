# 同快照、同终点、同候选池的地图影子诊断 v1

2026-10-17。独立入口：`chapter3_bser.experiments.map_stp_shadow_v1.run`。现有生产代码、基线和历史 provenance 清单不修改；无训练、无 checkpoint、无 Actor 调用，影子路线安装次数固定为零。本协议在读取本轮评分结果前冻结在 `configs/chapter3/map_stp_shadow_v1.json`。

## 数据与执行边界

从历史固定 20 场开发索引中按固定间隔选第 0/3/7/11/15/19 项，即原始场景 0、10、40、63、78、97。使用原 D2（V4 安全链）零残差 runtime，种子 `12729+原始索引`，仅取 0/10/30 步公开状态；最多每场 30 步、总计 180 物理步。Found 或终止立即停止，不补选场景，不执行完整评估回合。

采集与评分分离。采集器只把 `PlanningStateView`、各 Searcher 的既有分配终点、公开车辆常量、dt/工作空间/探测阈值/反射规格等白名单字段写成 JSON gzip，不保存环境对象或 simulator checkpoint。评分器不能读取 scenario manifest，不接收真实目标初状态、初速度、运动 seed、真实障碍或未来轨迹。保存的 snapshot 有独立字节哈希和公开状态哈希。

快照通过独立只读 `extract_planning_state` 提取，不强制刷新基线 provider，不调用基线重分配。采集前后检查 allocation hash 和 Torch RNG 不变。没有用影子结果改变后续采集动作。

## 共享候选与执行模型

对每个快照和既有 Searcher 终点，使用原 `TravelCostService.query` 路线作为参考。通过最多 12 个公开栅格中间点产生有界替代路线，限制 planning cost 不超过原 query 的 1.4 倍，排除重复、循环和公开几何违规路线，按几何差异保留最多 4 条。不是全空间 K-shortest 或不同拓扑走廊完备搜索。

所有评分共用相同候选池、同起终点、同 dt、同 60 步（12 秒）时域、同 detector 和同目标样本。三 Searcher 的团队组合最多 4³=64。缺少合法分配、不可连接或没有可行路线的情况记录并保留在诊断分母；不偷偷用目标真值补路径。

Searcher 轨迹由现有 `Vehicle.step` 与 `PathTracker(threshold=.75)` 预测，起始速度来自公开观测，零残差、未来流场估计为零。预测轨迹必须满足同一公开几何筛选。它不预测未来安全替换、地图刷新和上层重规划，也没有声称真实闭环轨迹精确。这些预测模型假设对四组完全相同。

每一组同时报告原高斯核的静态分数作为背景参考，但它不参与四组的“纯时间机制”比较，因为它使用另一种 detector/占据权重和未进一步条件化的原 belief。

## 公开目标假设及限制

从当前位置栅格信念抽样，去除当前已知占据扩张盒及三个 Searcher 当前探测范围内的质量，再归一化。记录保留质量，不把这一步藏在“精确后验”表述里。它使用当前未发现这一公开事件，但不重新运行或改写现有环境 belief 更新。

速度采用明示模型：大小 U(0.25,0.55)，方向为球面各向同性，且与当前位置独立。这个分布匹配生成器的速度/方向先验形式，**不是在非零时刻根据历史负观测推断的速度后验**。它是位置边际的近似 lift。方向、速度不读取当前场景的真实样本。

未来轨迹调用共享 `simulate_target_trajectory` 和原反射实现，仅使用当前已知占据体素盒。未知障碍导致的反射未建模；这是地图条件预测，不是隐藏真实地图上的精确未来。模型错误不能算作环境发现性能。

每个快照使用 384 个选路样本及另 384 个独立复核样本，基础 seed 为 1729 和 92729，均加 `原始索引*1000+step`。候选之间共用样本。复核不重新选路。没有在本轮数据上调整参数或挑 seed。

## 四种主评分

发现事件都使用 core swept 判据的同一代数，半径为该角色 sensor radius 加现有 detect_eps_bias。向量化实现逐项与 core 标量函数做契约测试；没有新增遮挡或软探测模型。

设目标假设 m，搜索者 i 的候选路线在第 k 步的事件为 D_i,k,m，权重为 1/M，折扣率 λ=0.04/秒，时域 H=60。

1. **static**：把每个假设的目标固定在其当前初位置，计算团队在 H 内至少发现一次的比例。沿途空间覆盖具有相同 detector，去除高斯核与硬阈值差异这一混杂。
2. **arrival_discount**：沿用 stationary 覆盖，每个能覆盖该假设的 Searcher 按其路线名义终点到达时间折扣，团队取最大值再求均值。单 Searcher 时就是 static×exp(-λ ETA)。该定义是显式终点 ETA 代理，不等同于原 early_discovery 的多智能体贪心实现。
3. **time_marginal**：目标按同一批反射轨迹运动；每步 q_k 为团队发生发现的**未条件化边际概率**，再按独立近似计算 `Σ exp(-λkdt) q_k Π_(j<k)(1-q_j)`。团队每步先对同一目标做事件 OR，不把多个 Searcher 当作独立目标。
4. **correlated_first**：对同一批目标轨迹计算 T_m=min{i,k:D_i,k,m=1}，得 `mean(exp(-λT_m dt) 1[T_m<=H])`。每条轨迹仅计一次最早团队发现；精确性只限于给定离散样本和预测轨迹，不延伸到真实后验/环境。

额外记录不带折扣的发现概率、stationary-first 折扣收益，以及 marginal 独立模型的累计发现率，便于区分移动预测、时间折扣和重复事件相关性的影响。

## 比较和事前继续门槛

逐 agent 记录完整候选与四组排序。团队穷举同一组合池，各方法在选路样本上固定选出一个组合，然后用独立复核样本评价同一 correlated-first 目标。评分相同（1e-12 容差）时保持组合顺序，优先原 query 路线，不通过浮点噪声制造换路。

继续“下一阶段有界执行验证”的筛选门槛同时要求：至少 12 个多路线 agent 案例、6 个多组合团队快照、4 个场景；correlated-first 相比 static 的团队选择变化比例至少 20%；按场景等权的独立复核收益差至少 0.005；至少 60% 的团队快照差值为正。其余输出为路线机会不足或暂不集成。门槛是机制筛选的实际量级约定，不是统计显著性或论文通过标准。

同目标样本的配对 bootstrap（400 次，独立 seed）只度量 Monte Carlo 复核误差，不代表不同环境场景的泛化置信区间。场景内多个快照不是独立实验样本；汇总同时显示逐场景均值。即便通过门槛，也只允许建议下一阶段，不能得出真实 Found 提升或自动启动完整评估。

## 运行命令

从仓库根目录执行，必须使用新的输出子目录（已有目录不覆盖）：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONUTF8='1'
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -B -m chapter3_bser.experiments.map_stp_shadow_v1.run collect `
  --output-dir runs/map_stp_shadow_v1/collection_20261017_v1
& 'D:/anaconda/anaconda/envs/AUV/python.exe' -B -m chapter3_bser.experiments.map_stp_shadow_v1.run score `
  --snapshots runs/map_stp_shadow_v1/collection_20261017_v1 `
  --output-dir docs/chapter3_map_stp/shadow_20261017_v1
```

原始公开快照留在 Git 忽略的 runs 中。Git 可见证据包括配置、原始输入身份、源码逐文件哈希、采集计数、候选/评分与独立复核汇总以及解释报告。执行代码和源身份在采集与评分之间必须一致；失败输出保留，不覆盖成成功结果。
