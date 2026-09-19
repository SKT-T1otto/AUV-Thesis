# Basic Search-Prior 实现记录

实际工作路径：`E:\gym\code\WORKSPACE\AUV-Thesis`。开始时分支 `main`，HEAD `a51a234e4ca9575c9707409c6b75a26380ad2a5c`，与用户指定冻结提交一致；工作区原本干净。没有 pull/reset/clean/commit/push、SSH、模型加载或训练。

## 新增文件

| 文件 | 作用 |
| --- | --- |
| `tools/ch3_baselines/__init__.py` | 独立外层包 |
| `tools/ch3_baselines/basic_search_prior.py` | 搜索分配器、控制器注入、零残差任务运行时 |
| `tools/ch3_baselines/evaluate.py` | CPU 串行评价、输入保护、任务记账和异常保存 |
| `tools/ch3_baselines/provenance.py` | 冻结生产来源核对、独立工具/配置来源清单 |
| `configs/chapter3/baselines/basic_search_prior_v1.json` | 新方法身份与方法干预合同 |
| `scripts/run_ch3_basic_prior_eval.bat` | Windows 手动入口 |
| `scripts/linux/run_ch3_basic_prior_eval.sh` | Linux 手动入口 |
| `tests/test_ch3_basic_search_prior.py` | 定向数学、重规划、信息边界、输入保护和真实接口测试 |
| `docs/chapter3/baselines/BASIC_SEARCH_PRIOR_V1.md` | 方法说明、比较边界和单行命令 |
| `docs/chapter3/baselines/frozen_production_source.json` | 新增的完整冻结生产 Python 字节来源记录 |
| `docs/chapter3/baselines/IMPLEMENTATION_REPORT.md` | 本记录 |
| `docs/chapter3/baselines/verification_results.json` | 最新本地验证的紧凑记录 |
| `docs/chapter3/baselines/delivery_manifest.json` | 交付文件清单与 SHA256，不包含自身 |

仅新增以上文件；没有修改既有生产 Python、配置、测试、历史 provenance 或用户保留的 outputs。

## 接入结论

分配器在第一次 controller.initialize 之前注入，完全覆盖初始化/完整/局部搜索规划。没有先跑联合 BSER 再修改分数字段。`standby=None` 与 Searcher 分配解耦，不再触发默认“丢弃所有搜索分配”的 fallback。局部条件边际收益保留未受影响的路径；方案比较和稳定化后的分值都来自 search-only。

Executor anchor 来自每回合合法 reset 后的初始实际位置。Found 不会改变该目标；原公共信息接收事件才进入既有执行路径。待命路径不可达时继续使用原 bridge 保持机制，不选择 belief 峰值。环境中独立的旧 PSE standby 开关也在构造时关闭，作为显式待命干预记录。其余环境构造参数由同一 reference 解析链生成并逐项比对，奖励、地图、动力学、400 步上限与成功条件保持原定义。

每步环境输入残差和物理 `_last_residual_acc` 都检查严格为零，`use_residual_prior=True`。实际物理先验仍可非零。测试禁止所有 PyTorch Module forward、HGR actions、Gaussian sampling、SGD/Adam update、在线联合 BSER 和旧 PSE standby 方法调用。未构造任何网络、critic、replay、predictor、optimizer 或 checkpoint。

特别核查了原 handoff 字段：`_publish_detection` 同时设置 found_step 和 handoff_step；因此继承 HGR 映射的 handoff_event_step 是发布字段，可与 Found 同号。实际接收另存 executor_target_received_step，首次合法接收后的控制决策另存 handoff_decision_step。初次本地测试错误地要求发布步严格晚于 Found；已根据真实源码修正断言，没有改动任务通信协议。

## 来源与产物保护

本次记录的完整生产源聚合 SHA256：`7a8dda612fb68c0a63bcc38b04ee0973ac80488bddd6c9fffe1fbf9b2f23f491`。启动、逐场、结束均核对冻结清单；所有 27 条历史 provenance 仍由原 tests.test_repository_metadata 校验，未更改历史哈希或放宽例外。

外层工具自身的 Python、Chapter 3 JSON 配置和两种启动脚本单独记录清单及哈希。reference、manifest 原始文件和选中场景内容也独立记录；没有复制 HGR pairing_sha256。交付清单另外覆盖测试及文档。来源变化会使任务失败并保留异常场景，不转成正常 timeout。

本地使用 `configs/chapter3/hgr_train.json`，标签为 default_config_reference；指定 HGR 实际 run 配置在 Windows 不存在，没有核查 Linux 的 run config/正式 manifest。测试使用临时合成/故障夹具，产物随临时目录清理；不写入保留的 outputs，也不形成正式性能数据。正式输出预定为 `outputs/chapter3/baselines/collision_terminal/basic_search_prior_eval100_seed12729_v1`，本轮没有启动它。

Windows、Linux 运行及 Linux 动态查看进度的全部单行命令见 [BASIC_SEARCH_PRIOR_V1.md](BASIC_SEARCH_PRIOR_V1.md)。脚本均不安装依赖、不生成 manifest、不加载模型、不支持 resume。

## 验证范围

最新结果记录在 [verification_results.json](verification_results.json)，属于本地验证，不是 CI。原有回归范围：来源 metadata、28D 观测、Phase 1B.2 路径跟踪/局部 BSER/Executor invalid、公开目标重规划和锁定、不可达保持、刷新协议、严格碰撞几何和环境终止分支。没有运行包含训练的 HGR integration suite。

针对性测试覆盖 F_search 公式、响应权重独立性、standby=None 下完整分配、原子局部重规划、Found/handoff 边界、anchor fallback、无候选恢复、固定接收者信息下的隐藏真值隔离、配置/manifest/目录保护、独立来源变化、三类任务结果、程序异常、零残差真实物理接入及启动脚本。

合成完整任务保留 400 步任务上限与原接触/保持规则，最多执行到原终止；不是固定 100 场验证集。碰撞测试只在独立故障夹具中注入物理接触，确认原碰撞分支先于 capture 且不能继续 step。成功/超时终止也由原终止回归与汇总单元夹具覆盖。没有降低成功率的验收条件，不声称优于或劣于 HGR。

Windows 模块 --help 与批处理 --help/错误退出码已经本地检查。Linux 脚本通过 Windows Git Bash 的 bash -n 语法检查；未执行 Linux 正式评价。跨平台浮点/依赖差异不作逐项轨迹一致承诺。

最终本地结果：新基线 15 项测试通过（437.086 秒），原有定向回归 27 项通过（105.003 秒）。最后一次合成完整任务在 161/400 步判为 success，Found/发布字段为 1；合法接收决策晚于 Found、与原接收字段一致的断言通过。零残差与非零先验断言通过，所有禁止调用守卫通过。早先因错误理解 handoff 发布字段产生的单条测试失败已修正并完整复核；不是修改协议以让测试通过。Windows 指定的实际 run 配置和正式 manifest 均不存在，因此正式命令仍需用户准备该冻结文件。所有 196 个生产 Python 文件字节哈希保持不变。

最终状态：

```text
implementation_complete = true
targeted_tests_passed = true
frozen_production_source_unchanged = true
formal_100_episode_evaluation_started = false
new_training_started = false
baseline_performance_verified = false
```
