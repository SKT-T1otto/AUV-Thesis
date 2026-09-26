# HGR 后续决策实验

本入口用于判断是否值得继续研究当前 HGR 校正路线。它不启动完整 HGR 训练，也不改变 reward、environment、planner、gamma、Actor 架构、原梯度估计式或 Phase1 随机源协议。

本轮交付范围为代码、单元检查和 preflight；**没有执行本文的真实后级更新或主轨迹/续分支实验**。只有用户显式选择 `--execute` 后，入口才运行真实任务。默认模式只检查输入并展示预算，不创建输出目录或计划文件。

## 为什么需要这一步

前一次 10 场景 debug 使用的真实策略对变化极小：后级参数差 L2 约 `1.06e-6`，40 对续分支的最大绝对回报差约 `1.59e-8`。校正消耗 19,244 个环境步，对同一批 HGR 主梯度的平均平方误差改变约为 `-1.54e-8`。这支持“该策略对的校正很弱”，不能据此证明 HGR 数学机制总体无效。

因此，本入口先从同一个合法起点独立产生两个真实、单次后级更新，再测量每个策略对的完整梯度及成本。来源、批量和阈值都在看到新结果前确定；不选择成功案例，不随机扰动参数，不挑变化较大的更新，也不把旧主梯度计作免费数据。当前不加入跨版本 predictor、Bernoulli 纳入或自适应预算。

## 固定配置与来源

入口：`python -B -m scripts.hgr_decision_experiment`。

配置：`configs/chapter3/hgr_decision_experiment.json`，schema 为 `hgr.decision_experiment.v1`。

| 字段 | 默认值与约束 |
|---|---|
| runtime_config | `configs/chapter3/hgr_phase1_zero.json` |
| snapshot_source | 既有 `outputs/chapter3/hgr_phase2/frozen_phase2_source.pt`，只读 |
| starting_policy | 来源中的 `new_policy` |
| snapshot_count | 10；使用来源中的全部固定场景和既有顺序 |
| update_replicas | 2；每次从相同合法起点独立复制 |
| suffix_batch_size | 4；必须等于 runtime 原有 `suffix_training_episodes_per_cycle` |
| macro_repeats | 每个策略对 2 次独立完整主批次 |
| correction_draws | 每个 HGR 主批次总计 K=8；必须等于 runtime 原值 |
| random_source_revision | `hgr.phase1.named_streams.v1` |
| seed | 92729 |
| max_environment_steps | 60,800 个物理环境转移，非墙钟时限 |
| output_dir | `outputs/chapter3/hgr_decision/collision_terminal/decision_01`；禁止覆盖非空目录 |

生产 `core/` 和 `chapter3_bser/` Python 源码保持原样，既有来源的严格 source identity 校验继续执行。独立实验脚本及相关输入的哈希写入计划；独立脚本不进入生产源码扫描，不代表它可以不记来源。不得更新旧 provenance 摘要来使校验通过。

## 一次执行的精确定义

1. 读取并严格验证冻结来源、运行配置、全部 10 个场景与起点策略。冻结 `theta_minus`。
2. 对两个 update replica 分别从同一个起点复制策略，生成新的、预先定义随机流下的自然训练场景，各采 4 条完整初始分布任务。调用原 `update_suffix` 和原后级 plain SGD 一次，形成各自的 `phi0/phi1`。未交接任务的零贡献和完整分母保留。
3. 两个 replica 互不继承更新结果。这不是连续训练两轮，也不对同一 batch 多次更新。记录实际后级梯度、参数变化和有效交接，检查前级参数保持不变。
4. 若某 replica 的完整行为身份精确相同，按严格零更新规则记录旁路，不再为该 replica 采额外效能主轨迹。若身份不同，无论参数差或 KL 多小，都进入比较；没有“小差异视为零”的近似旁路。
5. 对每个发生变化的策略对，运行两次独立 macro。每次使用全部 10 个固定来源场景，分别采 10 条 HGR 旧策略完整主轨迹和 10 条 Direct New MC 新策略完整主轨迹。不同方法的主轨迹随机流独立。
6. **每个 macro 的 HGR 主批次 N=10，整个批次共有 K=8 次均匀有放回抽样**。不是每场景 K=8。沿用原 `prefix_losses`、零预测器和完整主批次分母；重复索引保留重数，无交接抽样贡献零且不重抽。
7. 每个有效抽样从对应完整交接快照运行旧、新续分支，使用 Phase1 `policy_crn`：旧新分支共享策略噪声，环境创新流独立。各分支运行到各自真实成功、首次碰撞或原 H=400 截止，不强制相同长度。
8. 保存完整成本、行为身份、来源和向量摘要，按每个策略对单独统计。失败时保留 traceback 和已完成部分的成本；不删掉失败证据后重跑到同一目录。

固定来源场景用于主梯度比较；后级更新任务来自自然训练生成器。两者不能互换，也不能用主比较结果挑选后级训练样本。

## 预算及 diagnostics-only

最坏物理步数采用 H=400 预约：

```text
2 replicas × [4 suffix-training tasks × 400
              + 2 macros × ((10 HGR main + 10 Direct main) × 400
                            + 2 branch sides × 8 draws × 400)]
= 60,800 environment steps
```

每条续分支实际剩余时域不超过 400，因此上式保守覆盖所有预声明查询。提前终止、无交接及精确零更新旁路可降低实际成本。预算不得通过截断主轨迹、取消耗时长的分支、重抽无交接或事后只保留部分结果来满足。该上限不约束恢复、规划、统计或序列化墙钟时间。

`--diagnostics-only` 使用同一配置里的 **2 个 update replica、每个 4 条任务**，只进行真实后级更新诊断，不采 macro 主轨迹和效能续分支。最坏成本为 `2 × 4 × 400 = 3,200` 步；不另设隐含的 4-replica 默认值。仍然需要 `--execute` 才会真实运行，单独加 `--diagnostics-only` 只预检查诊断计划。

若先执行 diagnostics-only，之后另行执行完整实验，应使用新的输出目录，两个执行的实际成本分别记录。完整执行不是从诊断结果续训，也不把诊断输出选择成更有利的起点。

## 统计与决策解释

每个 replica 对应独立生成的 `phi1`，其梯度目标可能不同。按策略对分别估计 macro 梯度方差；不能把两个 replica 混在一起，当成同一估计器的重复样本方差。每对仅有 2 个 macro，精度有限。

需要报告：

- 校正向量的信号能量，并扣除重复估计噪声的估计贡献。去偏后的信号能量允许为负，不裁成零后伪装存在信号。
- 同一 HGR 主批次内部 `g0 → g_full` 的配对变化，以独立 Direct New MC 向量作新目标锚点。锚点本身有 MC 噪声，不能称为真梯度。
- HGR 与 Direct 的完整主梯度方差、完整物理步数和后级来源准备成本。
- `cost × variance` 作为效率代理；本入口没有实跑相同实际环境预算下的两条学习过程，不能将该代理写成等实际预算性能胜负。

本入口不采额外昂贵 reference。它是有预算上限的研究筛查，不证明训练成功率、系统性能优势、总体无偏性或算法收敛。

阈值预先固定为：`relative_signal_tolerance=0.001`、`noise_scale_multiplier=2.0`、`min_paired_queries_per_replica=2`。其中 0.1% 和 2 倍噪声尺度是事前研究筛查容忍度，**不是置信界、显著性检验或普遍有效的理论常数**。不能看到结果后修改阈值再把同一结果称为预声明验证。

脚本使用以下决策标签，具体触发条件及中间量同时保存在结果中：

| 标签 | 解释边界 |
|---|---|
| STOP_CURRENT_HGR | 当前预声明路线没有给出值得继续投入的校正证据；不等于 HGR 数学恒等式被否定 |
| CONTINUE_SIGNAL_STUDY | 当前证据支持进一步研究校正信号；不代表已证明任务性能有效，也不自动授权下一阶段 |
| INCONCLUSIVE | 噪声、重复数或诊断条件不足，不能据本次排序；不自动扩充预算 |
| NOT_EXERCISED | 相关真实校正没有被实际检验，不能当作成功的性能验证 |

精确零更新 replica 必须明确记录。参数改变但没有足够有效续分支的 replica 不能被解释为已验证“校正无用”。不按结果筛除 replica，也不把不同新策略的信号合并制造总体优势。

非零更新的实际筛查公式为 `center=||mean(g_delta)||/RMS(g0)`、`noise=2*sqrt(trace(S_delta)/R)/RMS(g0)`。若所有非零更新都有至少 2 对有效查询，且 `center+noise<0.001`，建议停止当前路线；若任一预声明更新满足 `max(0,center-noise)>=0.001`，建议继续独立信号验证。其余为证据不足。这里的上下尺度仅是操作性规则，绝不是 95% 置信界；两次 macro 不足以排除稀有事件或证明普遍没有信号。

## 输出与失败审计

- `plan.json`：执行前声明的全部后级场景、评价场景、随机流、配置、预算及来源身份。
- `decision_results.json`：持续更新的原始记录、分策略对统计、完整成本与决策。执行状态 `PASS` 只代表流程完成。
- `replica_*_training_records.pt`：后级学习原始记录，供核对梯度；不是 Trainer checkpoint。
- `replica_*_policy_pair.pt`：本次单步更新的策略审计张量，使用独立 schema；不自动覆盖或替换已有 Phase2 source。
- `replica_*_suffix.npz`、`replica_*_macro_*.npz`：实际梯度、折扣分解、参数增量及各方法向量，附 SHA256。

每条完整轨迹和每侧已完成续分支的元数据及时保存。若后续调用失败，保留已完成记录、累计已知成本、失败调用、完整 traceback，并标注失败调用内部步数是否未知。不自动恢复、清理或覆盖此目录。`parameter_updates`/`suffix_sgd_calls` 记录原 SGD 调用次数，`changed_suffix_replicas` 单独记录实际行为改变数量；零梯度调用不冒充有效学习。

## 手动命令

### Linux 克隆与冻结来源兼容性

除已提交的代码和配置外，需要单独传输
`outputs/chapter3/hgr_phase2/frozen_phase2_source.pt`。该文件被 Git 忽略，
本实验不依赖历史 Trainer checkpoint 或中断运行的输出目录。
现有文件 SHA256 为
`4a60f4228ef8f8982f9f9ff0959a06574384fe043141e4dba6100e97d893278d`，
绑定的生产源码身份为
`100f369e2c1958a065905e379bd13ca63cdc7f76000352f2ee8fa267ff9aabca`。

首次提交曾将六个历史 CRLF/混合换行源码文件规范化为 LF，使全新克隆的
源码字节与冻结来源不一致。`.gitattributes` 现在仅对这六个精确路径禁用
换行转换，并将原始字节纳入 Git；其余 Python 文件继续使用 LF。
这保留现有来源身份，不修改算法、冻结文件或历史 provenance，也不放宽校验。
后续 Linux 更新即可取得匹配的字节，无需逐个手动传输源码文件。

在 Linux 仓库根目录执行（预检查不会启动 rollout）：

```bash
git pull --ff-only origin main
conda activate AUV
sha256sum outputs/chapter3/hgr_phase2/frozen_phase2_source.pt
python -B -m scripts.hgr_decision_experiment --config configs/chapter3/hgr_decision_experiment.json
```

只有取得 `PREFLIGHT_PASS` 后才手动选择 `--execute`。输出目录必须为空或不存在；
旧输出保留，不能覆盖。同一 Git 提交并不保证不同平台的浮点运算逐位一致。

仓库根目录下，默认只做 preflight：

```powershell
python -B -m scripts.hgr_decision_experiment --config configs/chapter3/hgr_decision_experiment.json
.\scripts\run_hgr_decision_experiment.bat
```

显式选择完整执行：

```powershell
python -B -m scripts.hgr_decision_experiment --config configs/chapter3/hgr_decision_experiment.json --execute
.\scripts\run_hgr_decision_experiment.bat configs\chapter3\hgr_decision_experiment.json --execute
```

显式选择只运行同配置的后级诊断：

```powershell
python -B -m scripts.hgr_decision_experiment --config configs/chapter3/hgr_decision_experiment.json --execute --diagnostics-only
.\scripts\run_hgr_decision_experiment.bat configs\chapter3\hgr_decision_experiment.json --execute --diagnostics-only
```

BAT 第一个参数是配置，第二个参数只能是 `--execute`，第三个参数可为 `--diagnostics-only`。BAT 使用已激活环境的 `python`，或 `PYTHON` 环境变量指定的可执行文件，并保留 Python 退出码。上述执行命令是人工入口示例；本轮未自动运行。

既有 debug 的结论和证据继续保存在 `docs/chapter3/hgr_phase2/analysis/debug_10scene_20260926/`，不覆写为本实验的结果。

## 本次本地验证

以下合成及接口回归测试：**81 passed in 47.56s**。这是本地记录，不是 CI 或真实环境实验。

```powershell
python -B -m pytest tests/test_hgr_decision_experiment.py tests/test_hgr_suffix_diagnostics.py tests/test_hgr_decision_statistics.py tests/test_hgr_phase2_gradient_efficiency.py tests/test_hgr_phase2_config_validation.py tests/test_hgr_phase1.py -q -p no:cacheprovider
```

最终只读预检查为 `PREFLIGHT_PASS`，实际环境步为 0，目标输出目录未创建。生产源码身份仍为 `100f369e2c1958a065905e379bd13ca63cdc7f76000352f2ee8fa267ff9aabca`。既有 policy pair、frozen source 和 debug 结果 SHA256 与此前审计一致。

现有 Phase2 配置测试原先假设实际 debug 目录尚未运行；本次将来源验证放入临时配置/目录，并继续断言非空历史目录拒绝覆盖。真实配置及历史输出没有因此改动。
