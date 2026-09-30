# Windows 手动运行 Found 搜索修复实验

目标：优先发现能力与搜索安全，发现后结果继续记录，暂不用于选型。固定原20开发场景、400步、seed=12729+原始索引，不训练或加载模型。

|组|B0起点|B1起点|新增机制|
|---|---|---|---|
|R0|V5=A+C|V4=A+B+C|原始对照，直接调用原类|
|R1|V5|V4|固定Hold参考点|
|R2|V5|V4|固定Hold+转弯减速及预测制动|
|R3|V5|V4|固定Hold+按智能体恢复|
|R4|V5|V4|三项全部启用|

新R组不覆盖旧V组。各R组与同基线R0配对；R2−R1看制动，R3−R1看恢复，R4−R3/R2看组合。20场景是开发集，不用于声称泛化或正式论文验收。

## PowerShell

```powershell
conda activate AUV
Set-Location 'E:\gym\code\WORKSPACE\AUV-Thesis'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$manifest = '.\3090结果\collision_terminal\B0_search_prior_eval100_seed12729_v1\evaluation_manifest.json'
$out = '.\runs\found_search_v2\development_20260929_v1'
```

先检查来源、场景和参数，不创建模拟器或运行回合：

```powershell
python -B -m chapter3_bser.experiments.safe_search_v2.run_windows --manifest $manifest --output-dir $out --workers 4
```

看到 `PLAN_ONLY` 后手动启动：

```powershell
python -B -m chapter3_bser.experiments.safe_search_v2.run_windows --manifest $manifest --output-dir $out --workers 4 --execute
```

共 **2×5×20=200** 完整回合，最大80,000物理步。四个子进程各限制一个计算线程，不依赖GPU训练。每回合打印Found与终止原因；最终需出现 `FOUND_SEARCH_FINISHED` 且summary.json的experiment_complete=true。

若先只比较整体效果，用独立目录运行R0/R4共80回合；该组合不能区分单项贡献：

```powershell
python -B -m chapter3_bser.experiments.safe_search_v2.run_windows --manifest $manifest --output-dir '.\runs\found_search_v2\paired_R0_R4_20260929_v1' --arms R0,R4 --workers 4 --execute
```

可加 `--baselines B0` 或 `--baselines B1` 分开运行，每个基线五组100回合。输出须为runs下的新目录。实验中不要修改代码、配置或来源manifest。

## 中断续跑

Ctrl+C终止本次启动的子进程并保留已完成记录。相同命令、路径和代码增加 `--resume`：

```powershell
python -B -m chapter3_bser.experiments.safe_search_v2.run_windows --manifest $manifest --output-dir $out --workers 4 --execute --resume
```

校验通过的已完成回合才复用，未完成任务另建attempt。允许调整workers；组别、代码或输入变化后必须另开输出目录。

## 供后续分析的文件

保留整个输出目录，重点包括：

- identity.json：来源、参数、种子、场景身份。
- episodes.json：逐回合Found、碰撞、覆盖、完整任务结果及轨迹位置。
- summary.json：分组指标、逐场景配对差值。
- jobs/.../attempt_*/step_trace.jsonl：完整轨迹，含Hold参考点、命令点、预测状态、恢复原因。
- jobs/.../attempt_*.log和interruption_*.json：程序日志；程序错误不计作超时。

发现时间评分：Found回合记实际步数，未Found记400；另报成功发现条件下的时间。碰撞保持原口径：碰撞步≤Found步也算发现前碰撞。同一步Found与碰撞可能同时为真，两指标不是天然互斥分类。

不删除物理失败回合、不只比较成功回合速度、不把预测通过当成真实安全保证。发现后成功率保留为次要记录。
