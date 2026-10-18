# 修改前审计快照（2026-10-18）

目标分支 codex/framework-audit-20261018-164056，HEAD 6538c3e6a67e96d56940ab09fcff15cd9dc7a7d8。已跟踪文件无修改，用户未跟踪的 tmp、output、文献及 MAP-STP 原始记录保留。304 个生产文件的修改前 SHA-256、历史清单、原配置、工作树状态见 evidence/before.json。

已独立重新阅读生产代码并运行父提交的接口反例：B2/B3 的 7 行观测配 1 行奖励/权重被广播并更新；跨容量 replay 恢复读到预填的未写入槽位；旧 push 绕过 gates；CPU snapshot 共享存储；step 对 3 agent 返回 3 动作；critic-only 标志陈旧。原始结果见 evidence/before_interfaces.json。本轮尚未修改生产实现。

航点显式 arrival_before_reward_v1、冻结 actor 和动作门控在 v1 已有实现，先保留其合法路径并扩展边界验证，不重复重构。历史 legacy_v1 环境时序和 Suite 映射保持原样。

最小修复：共享 MADDPG 增加声明维度/批长/有限性与八项元数据校验、快照复制、rollout 数量校验、critic-only 标志复位；共享 replay 对相同容量恢复做完整预验证并拒绝容量迁移；独立 v2 封闭无门控写入口和未初始化槽位，复用原学习公式/架构并增加门控一致性检查。v1 生产源码和历史 manifests 不覆盖。

v2 采用独立配置/模型状态/来源身份，复用原 Trainer 的通用更新与循环，不将旧 Suite 重定向。训练和保存/加载均先校验新完整来源清单；机制检查点标明 mechanism_only，不允许用于正式性能声明。

父版本最新记录：29 个模块/104 项，98 成功、1 failure、8 error，全部未通过项含 SafeSearch 来源门禁；这是已有记录而非本轮重跑结果。历史 E0 golden_trace_manifest.json 缺失。本轮将原样重跑并保留这些阻塞，不能修改旧清单或伪造黄金数据。新来源尚未建立，当前不得生成正式检查点。
