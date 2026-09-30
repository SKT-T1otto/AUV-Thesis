# B1 V1/V2/V4 补充实验

用户在 2026-09-29 要求补齐这三个版本。沿用上一轮固定20场景、B1原生方法、零残差、400步上限、碰撞终止协议。V1=A，V2=A+B，V4=A+B+C。不修改算法，不训练，不加载检查点。

实验编排位于本目录，调用已审查的 `chapter3_bser.experiments.safe_search_v1.run_paired`。该入口原本就支持 B1 和三个开关组合，因此无需演化338文件的运行源码清单。编排脚本、冻结计划及历史参考文件的原始字节哈希另行写入新运行身份，在运行前后校验；不是放宽运行源码检查。

从仓库根目录在具备依赖的本地 AUV Python 环境执行：

```powershell
python -B -m unittest discover -s tests -p 'test_safe_search*.py' -q
python -B -m docs.chapter3.search_diagnostics.safe_search_v1.run_b1_remaining --output-dir runs/safe_search_v1/development_B1_remaining_new --workers 6 --execute
python -B -m docs.chapter3.search_diagnostics.safe_search_v1.analyze_b1_remaining --run runs/safe_search_v1/development_B1_remaining_new --output docs/chapter3/search_diagnostics/safe_search_v1/b1_remaining_results_new
```

输出目录必须不存在。计划 `b1_remaining_plan_20260929.json` 绑定本地真实历史B1参考和清单；不可改写历史结果或哈希使检查通过。上述命令描述本次本地工作流，不宣称全新Linux克隆已验证。跨平台搬运需保留并验证独立的路径/源码身份。

每个场景×版本独立子进程；共60个真实终止回合，程序错误使实验失败且保留记录，不从分母剔除。旧B1 V0/V3/V5只读复用，合并后120回合，但独立配对单位仍为20场景。旧B0六版本只读复用，不重复模拟。

先冻结比较：V1−V0（A），V2−V1（A上加B），V4−V5（A+C上加B），V4−V2（A+B上加C）。另报V4−V0、V4−V3、V2−V0。全部报告Found、完整成功、发现前碰撞、停滞+Hold、有效观测，并保留查询和执行阶段诊断。使用场景配对10,000次bootstrap（seed20260928），描述性区间不作多重校正。

原始运行写入Git忽略的runs；不提交3090结果、模型或轨迹。不写入outputs，不提交或推送Git，不运行HGR或剩余80场景，不把开发结果标记为正式论文验收。
