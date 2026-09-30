# B1 配对迁移验证（固定 20 场景）

目的：检查 B0 中的 A+C 是否能迁移到 B1 联合分配器。保留现有 A/C 参数，不训练、不加载检查点。原始 100 场景清单中的固定 20 个开发场景，使用原始索引对应的随机种子，每个版本独立运行至真实任务终止，最长 400 步。

| 版本 | A：规划状态一致性 | C：已知障碍路径检查 | B：候选失败处理 |
| --- | --- | --- | --- |
| V0 | 关 | 关 | 关 |
| V3 | 关 | 开 | 关 |
| V5 | 开 | 开 | 关 |

运行入口明确使用 `B1_bser_prior` / `ch3_baseline_bser_prior`，对应 BSER 联合搜索与待命分配；四个智能体残差动作均为零。`hgr_train.json` 提供共用环境设置，并不调用 HGR 策略训练。

## CLI 结构与跨平台复现前提

下面使用 Bash 展示调用结构，**不是已通过验证的全新 Linux 克隆即用命令**。本次冻结的前测、源码差异评审和离线分析入口对应实际 Windows 运行；新 Linux 来源身份需要单独记录和核验，离线分析中的前测绑定也需另立明确的平台版本。不得修改本次冻结计划、历史结果或其哈希来绕过校验。

在具备 AUV 依赖的环境中，从仓库根目录执行。清单路径须指向本地保留的原始文件；原始结果和模型不提交 Git。输出目录必须不存在。历史 B0 两组结果作为只读参考，其文件哈希须符合 `b1_experiment_plan.json`，不可伪造或改写来源身份。

```bash
python -B -m unittest discover -s tests -p 'test_safe_search*.py' -q

python -B -m chapter3_bser.experiments.safe_search_v1.run_paired \
  --manifest '3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json' \
  --config configs/chapter3/hgr_train.json \
  --baseline B1_bser_prior --variants V0,V3,V5 \
  --episode-indices 3,28 --steps 30 --seed 12729 \
  --verify-v0 --search-coverage \
  --output-dir runs/safe_search_v1/b1_mechanism_new

python -B -m chapter3_bser.experiments.safe_search_v1.run_b1_development \
  --manifest '3090结果/collision_terminal/B0_search_prior_eval100_seed12729_v1/evaluation_manifest.json' \
  --config configs/chapter3/hgr_train.json \
  --b0-reference-dir runs/safe_search_v1/development_B0_20260928_v2 \
  --b0-ac-dir runs/safe_search_v1/development_AC_20260928_v1 \
  --output-dir runs/safe_search_v1/development_B1_new \
  --workers 6 --execute

python -B -m docs.chapter3.search_diagnostics.safe_search_v1.analyze_b1_development \
  --run runs/safe_search_v1/development_B1_new \
  --output runs/safe_search_v1/analysis_B1_new
```

严格的历史来源校验还要求 B0 子运行身份中记录的配置与清单身份匹配；跨平台搬运已有目录时，路径和源码字节身份可能不同。不要通过修改历史结果解决校验失败；需要在目标机器保留可验证的原路径身份或重新建立明确记录的平台迁移证据。这是复现入口说明，当前实际验证为本地 Windows AUV 环境，未宣称新 Linux 克隆已运行。

## 判读

主比较为 V5−V3，辅比较为 V5−V0、V3−V0。分别报告 Found、完整任务成功、发现前碰撞、搜索者停滞与 Hold、有效观测、查询失败和执行者运动。所有 60 个终止回合都收齐才输出完整判断；程序错误不会被算成超时或从分母排除。

汇总比例用总计数除以总暴露步数；配对区间使用 20 个场景的差值进行 10,000 次 bootstrap。B0 与 B1 的优化增益再作场景配对比较，用于描述迁移差异。这 20 个场景已用于开发，结论不替代独立测试，也不证明 HGR 有效。
