# CH3 B0–B3 baseline framework 交付报告

日期：2026-09-19。工作目录：`E:\gym\code\WORKSPACE\AUV-Thesis`。开始时 HEAD：`c7be3fdbf9e7305308ed5aac3ea3a896dfc86b26`，分支 main。结果为最新记录的本地验证，不是 CI。

已新增独立注册表、B1 零残差运行时、B2/B3 原生算法配置和训练入口、四方法统一评价、来源保护及文档。没有复制 HGR 算法，没有调参或更改任务。本轮只新增文件，既有跟踪文件没有修改。

## 1. 新增文件

| 类别 | 新增路径 |
|---|---|
| 工具，5 个 | `tools/ch3_baselines/registry.py`；`bser_prior.py`；`framework_provenance.py`；`run_baseline.py`；`run_training.py` |
| 配置，5 个 | `configs/chapter3/baselines/baseline_registry.json`；`search_prior_eval.json`；`bser_prior_eval.json`；`direct_mc_train.json`；`direct_boundary_train.json` |
| Windows 入口，3 个 | `scripts/run_ch3_baseline_eval.bat`；`scripts/train_ch3_direct_mc.bat`；`scripts/train_ch3_direct_boundary.bat` |
| Linux 入口，3 个 | `scripts/linux/run_ch3_baseline_eval.sh`；`scripts/linux/train_ch3_direct_mc.sh`；`scripts/linux/train_ch3_direct_boundary.sh` |
| 测试，1 个 | `tests/test_ch3_baseline_registry.py` |
| 文档与紧凑证据，6 个 | `docs/chapter3/baselines/baseline_design.md`；`linux_commands.md`；`FRAMEWORK_IMPLEMENTATION_REPORT.md`；`framework_protected_b0.json`；`framework_verification_results.json`；`framework_delivery_manifest.json` |

共 23 个新增文件。该表各行中省略目录的文件均位于同一行首文件目录；逐文件完整相对路径及 SHA256 另见 `framework_delivery_manifest.json`。清单不自包含自己的哈希。

## 2. 修改文件

无既有跟踪文件修改。`core/`、`chapter3_bser/`、HGR 的 train/cli/runtime/evaluation/provenance、既有 B0 的四个 Python/配置/旧脚本均未改变；不更新历史 B0 文档、验证结果或旧交付清单。不修改或删除用户保留的 outputs，不提交或推送 Git，不访问 Linux/SSH。

## 3–6. 实现状态、可运行性与未运行项

| 方法 | 实现状态 | 可直接运行什么 | 是否需要训练 | 本轮执行 |
|---|---|---|---|---|
| B0 SearchPrior | 复用冻结 B0，新增统一注册/配置/身份适配 | 提供现有 manifest 后即可完整评价，无 checkpoint | 否 | 既有单元回归；同场景两步物理对照 |
| B1 BSERPrior | 新增原 MissionRuntime 的零动作源，保留原 joint allocator | 提供现有 manifest 后即可完整评价，无 checkpoint | 否 | 禁止学习调用的两步物理对照；完整任务接口验证见下文 |
| B2 DirectMC | 已有 `stochastic_direct_mc`，仅新配置与路由 | check-only；独立 checkpoint 就绪后评价 | 是，后续手动 | check-only、mock 路由/输出/方法拒绝测试；未实际训练或载入学习权重 |
| B3 DirectBoundary | 已有 `direct_boundary_corrected`，仅新配置与路由 | check-only；独立 checkpoint 就绪后评价 | 是，后续手动 | check-only、mock 路由、原生 label 选择纯函数测试；未实际训练或载入学习权重 |

B2/B3 的论文名称与原生 config/checkpoint method 分开保存。原生 checkpoint 保留生产 evaluator 所需的 method/schema；HGR 权重或另一 baseline 权重不能作为本方法通过验证。B3 的原生 label 选择只使用当前 suffix，测试确认不请求旧 suffix。

本轮未执行：真实训练、任何 1000 episode 运行、正式 100 场评价、真实 B2/B3 checkpoint 评价、Linux 评价/训练、HGR checkpoint 复验。用户提供的 `hgr_main_001000_cycle_000070.pt` 状态未被本轮重新认证。Windows 对应实际 HGR run config 和指定正式 manifest 均未找到，故本地使用明确标记的默认 reference 和临时合成场景。

## 7. 最新本地验证

- 新框架第一批 18 项测试通过，211.432 秒：方法注册和隔离、仅三个训练配置字段变化、实际 reference 参数继承、四方法相同场景和任务输入哈希、碰撞规则拒绝修改、原 B0 来源保护、mock native evaluator 输出、错误保留和不完整结果率值为 null、预算提前停止不标记训练完成，以及三套 Windows 入口的 help/退出码。
- 两步物理测试保持 max_steps=400；B0/B1 初始 belief 和 agent positions 完全一致。B1 确实调用原生 `solve_joint_greedy`，allocator 为 `BSEROnlineAllocator`；B0 不调用 joint solver。两者命令残差和物理残差均严格为零，实际 prior 非零，网络 forward、Gaussian sampling、predictor 和 optimizer 的禁止调用检查通过。
- 相关回归 39 项通过，121.992 秒：原 B0 的 12 项单元/入口合约，以及 27 项生产元数据、观察、路径/局部规划、公开交接和碰撞协议回归。
- B1 完整合成任务测试通过，462.922 秒（episode console 460.67 秒）：任务上限保持 400，实际执行 400 步，原规则终止为 TIMEOUT。Found=1、发布字段 handoff_event_step=1、合法接收和 handoff_decision_step=2；全程 residual=0、optimizer_update_count=0，实际 prior 非零，统一五个结果文件完整。测试没有要求成功，也没有为超时调参或改变场景。
- 三个新 Linux 脚本和 11 条文档内单行 Bash 命令通过本地 Git Bash `bash -n`；这不是 Linux 执行验证。基线工具及新测试共 10 个 Python 文件通过 AST parse，baseline JSON 可解析。

新测试结果均写临时目录，测试退出后清理。mock checkpoint 输入被明确标记为非权重文件；mock Trainer 只返回计数，不创建实际网络、轨迹或更新。合成任务只是接口证据，不是正式成功率证据。

合计 58 项通过：新框架 19 项（先运行 18 项，再单独运行新增完整任务 1 项）加既有回归 39 项；失败、错误均为 0。没有重复声称本轮重新运行旧 B0 的完整任务测试，其旧证据保持历史身份。

复验命令：`python -B -m unittest tests.test_ch3_baseline_registry -v`。相关回归的完整命令和数值见 `framework_verification_results.json`。

## 8. 冻结与 provenance

196 个生产 Python 文件保持原聚合 SHA256：`7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491`。27 条历史 source provenance 完整保留并通过原测试。新增 `framework_protected_b0.json` 记录既有七个 B0 文件字节哈希；外层框架来源单独收集，不替换 HGR source identity。

评价输出统一五个必需文件，并保留 checkpoint 原生评价子目录、来源前后状态、manifest 和 reference 哈希。非空输出目录拒绝运行。正式统计只有所有请求场景完成且最终来源检查通过才发布；learned 中断时总消耗不可精确获知，明确置 null，另给完整场景步数下界。

## 9. 下一步 Linux 实验

完整独立单行命令见 `linux_commands.md`：先运行 B0/B1，B2/B3 可先执行 `--check-only`；用户随后选择启动时独立训练，再用各自训练 summary 中的实际 checkpoint 评价。同一冻结 manifest、N、评价 seed、任务配置和网络合同必须保留，训练的总环境成本及耗时同时报告。

当前完成的是 baseline 代码与实验入口建设；不建立性能优劣结论。
