# BSER 最后一版配对评估：F0—F6

本轮只修改规划与候选筛选，不训练、不加载模型。正式场景由用户手动运行。
助手只运行合成夹具 smoke、接口及来源校验；通过这些检查不代表 Found 改善。

## 七组设置

| 组 | 设置 | 比较目的 |
|---|---|---|
| F0 | 直接调用原 bser_effect_v1 的 D2 | 原样对照 |
| F1 | D2 搜索保护＋发现时刻响应＋运动风险筛选＋候选池枚举 | 完整方案 |
| F2 | 搜索保护、风险、枚举保留；只优化近程搜索，待命按 D2 顺序选择 | 联合响应是否有额外价值 |
| F3 | F1 去掉全路径及近程搜索不退化约束 | 搜索保护作用 |
| F4 | F1 的发现时刻响应换成静态待命点、全路径覆盖评分 | 时间响应模型作用 |
| F5 | F1 关闭新增运动风险筛选；原 V4 路径安全仍在 | 新增风险筛选作用 |
| F6 | F1 的枚举换成单轮坐标贪心，其余评分和约束相同 | 求解方式作用 |

所有组：原 V4 路径安全、事件及冷却、0.75 航点阈值、零残差、400 步完整终止回合。
不启用 R2/R4 制动、0.25 阈值或新 Hold 控制，不更改 Found 后执行链路。
F6 是按智能体顺序、每个待命点一轮的坐标贪心，起点为同一候选池的 D2 搜索方案；不是旧 D3。
F4 与 F1 使用相同的公开安全过滤图和连接检查，只改变响应评分模型。

## Windows：先运行固定 20 场景开发评估

在 PowerShell 中进入本仓库，使用已有 AUV 环境：

```powershell
conda activate AUV
cd E:\gym\code\WORKSPACE\AUV-Thesis
python -B -m chapter3_bser.experiments.bser_final_v1.run_windows --output-dir runs/bser_final_v1/development_v1
```

上面只校验并打印计划，不创建仿真、不运行回合。默认使用本地已有的
`3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json`。
它保持原 100 场景清单中的 20 个固定开发索引。原始 3090 结果无需上传 GitHub。
若文件在别处，用 `--manifest 完整路径` 指定同一份清单。

正式启动这 **20 × 7＝140 回合**：

```powershell
python -B -m chapter3_bser.experiments.bser_final_v1.run_windows --output-dir runs/bser_final_v1/development_v1 --workers 4 --execute
```

中断后，代码、场景和参数没有变化时续跑：

```powershell
python -B -m chapter3_bser.experiments.bser_final_v1.run_windows --output-dir runs/bser_final_v1/development_v1 --workers 4 --execute --resume
```

workers 控制独立 CPU 进程，并非 GPU 训练。内存紧张时改成 2。
新运行必须使用新目录；不要用旧 D、V、R 结果目录。运行期间不要修改代码或清单。
程序错误会中断并保留已完成回合，不会被计为超时或碰撞。续跑只重做未完成的作业。

开发组完成后先检查 F0 是否复现历史 D2，以及模型错误、回退比例、耗时和七组开关。
历史 D2 为 12/20 Found、3/20 发现前碰撞；这只是本地历史参考，不能代替逐场景核对。
如果发现实现错误，修复后使用新结果目录；不要把修复前后的记录拼接成完整实验。

## 独立确认场景接口（仍由用户手动执行）

开发检查完成、代码冻结后，生成新的 100 场景清单。此命令只准备场景，不跑实验：

```powershell
python -B -m chapter3_bser.experiments.bser_final_v1.scenarios --count 100 --output-dir runs/bser_final_v1/confirmation_scenarios_v1
```

生成种子固定为 20260930，环境创新种子为 492929＋场景索引。清单检查与原 100 场景的
ID、种子和内容不重合，并绑定当前完整源码身份。count 可在运行前决定样本量；
不得根据显著性边跑边增加。100 场景不保证能稳定识别 5 个百分点的小差异。
这个检查不声称排除了所有其他历史训练数据的重合。

预览计划：

```powershell
python -B -m chapter3_bser.experiments.bser_final_v1.run_windows --stage confirmation --manifest runs/bser_final_v1/confirmation_scenarios_v1/evaluation_manifest.json --output-dir runs/bser_final_v1/confirmation_v1
```

手动启动 **100 × 7＝700 回合**：

```powershell
python -B -m chapter3_bser.experiments.bser_final_v1.run_windows --stage confirmation --manifest runs/bser_final_v1/confirmation_scenarios_v1/evaluation_manifest.json --output-dir runs/bser_final_v1/confirmation_v1 --workers 4 --execute
```

续跑在同一条命令末尾加 `--resume`。准备清单后更改源码会被拒绝，避免确认组混用版本。
脚本不会自动串联开发与确认运行，也不会自动决定是否保留 BSER。

## 固定评分定义及计算预算

每次原有规划事件触发时，使用同一候选池计算 D2 参考，预览原航点稳定机制后再比较。
保留未受影响智能体的分配；恢复事件沿用原来的直接安装规则。

20 步预测使用当前公开位置、速度、角色常数和原控制器。初始扰动估计为零，后续由已观测
速度残差估计 XY 扰动，不读取真实流场。预测固定地图、目标信念及高层路径，不模拟未来
地图更新、目标运动和新规划事件。首个控制指令通过克隆的真实 V4 bridge 预览；随后是固定
路径的短程跟踪近似。记录一步预测与真实运动的误差，不能据此宣称 20 步预测精确。

令 Q(t,z) 为预测至 t 步对格点 z 的累计覆盖核。单智能体时间维度取最大覆盖，智能体间
使用原有补集乘积；同一位置反复观测不会被当作独立发现试验。

- 近程分数：20 步内 `sum_z b(z) * (Q(t,z)-Q(0,z))` 的平均值，越早新增覆盖越好。
- 动态响应分数：`sum_t,z b(z) * (Q(t,z)-Q(t-1,z)) * exp(-T(e(t),z)/18)`。
  执行者位置 e(t) 是发现时刻预测位置，不假设它已到达待命点。
- F4 静态响应：原全路径覆盖核乘以从待命点出发的响应权重；近程搜索约束仍保留。
- 风险逐智能体、逐分量比较：已知阻挡段数、未知区域中点加权行程、剩余路径最大偏离。
  名义及四个 XY 扰动样本取最大值；不允许跨智能体抵消风险。未知暴露是代理指标。

完整版本要求原全路径分数、近程分数、响应分数不低于同状态 D2，风险不增加；然后至少
近程提升 3% 或响应提升 5%。F2 只要求近程提升 3%。F3 去掉两项搜索不退化约束。
分母小于等于 1e-10、公开快照版本不一致、连接不可靠等情况回退 D2。
这些是固定工程阈值，不是效果保证，也不是论文统计验收阈值。

单次规划最多 256 个候选组合、24 条独立轨迹预测、80 次响应图查询；单次连续起点连接
最多检查 128 个格点。缓存只在同一次规划内有效。预算耗尽会舍弃未遍历完整候选池的
临时优胜方案并回退 D2，避免不同电脑速度改变策略。未连接成功不能解释成真实地图无路。

## 结果与日志

结果目录中：

- progress.json：已完成/计划回合数和当前调用耗时。
- summary.json：七组指标、配对差异、计算耗时、回退和实际安装情况。
- episodes.json：逐回合完整结果。
- jobs/F_Fx_*/attempt_*/planning_audit.json：D2 参考、候选拒绝、选择、计算次数及预测误差。
- jobs/F_Fx_*/attempt_*/step_trace.jsonl：连续物理步记录。

主要比较 F1 对 F0、F1 对 F2；F1 对 F3—F6 解释机制。主要指标为 Found、发现前碰撞率、
失败按 400 步计入的发现时间。Found 条件均值、响应距离及执行成功率是次要指标。
配对 bootstrap 以场景为单位；其他比较的 p 值未作多重校正，属于探索性证据。
耗时、fallback、proposal 都不能单独证明有效；尤其要区分提出改进与实际安装改进。

## 本地 smoke

```powershell
python -B -m unittest tests.test_bser_final tests.test_bser_final_integration tests.test_bser_final_runner -v
```

集成模块只使用合成夹具，共 18 个物理步；不会启动固定 20 场景或任何完整评估回合。
最新本地验证记录保存在同目录 local_verification.json，不表述为 CI。
