# Linux 手动实验入口

以下均为后续手动命令，本轮未执行。先将本次新增文件部署到已有 `/home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR` checkout，并使用原 AUV Python 环境。没有新建 Linux 环境或安装依赖步骤。实际 HGR config 和冻结 manifest 路径沿用用户实验目录；本轮未连接 Linux 核查文件。

每条命令独立为一行，不依赖前一 shell 的变量。输出目录须为新目录或空目录；已存在结果时另选 run_id，不覆盖。四种评价传入相同 manifest、100 场和 seed 12729。

## 无学习方法：直接评价

B0：既有 Search-only + Prior-only，通过统一入口记录论文方法名。

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/run_ch3_baseline_eval.sh --baseline B0_search_prior --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --manifest /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/final_manifests/ch3_validation_M20_seed12729_n100.json --episodes 100 --seed 12729 --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B0_search_prior/collision_terminal/eval100_seed12729_v1
```

B1：BSER joint + Prior-only。

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/run_ch3_baseline_eval.sh --baseline B1_bser_prior --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --manifest /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/final_manifests/ch3_validation_M20_seed12729_n100.json --episodes 100 --seed 12729 --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B1_bser_prior/collision_terminal/eval100_seed12729_v1
```

## 独立训练方法：先只检查计划

以下 `--check-only` 不创建 Trainer、网络或结果目录，不进行训练。参数完整继承实际 HGR config，仅 method、algorithm、output_dir 改为对应 baseline。

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/train_ch3_direct_mc.sh --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B2_direct_mc/collision_terminal/train_seed2729_v1 --check-only
```

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/train_ch3_direct_boundary.sh --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B3_direct_boundary/collision_terminal/train_seed2729_v1 --check-only
```

## 用户后续选择启动时：原生训练

以下两条会启动实际训练，应在正式计划确定后手动运行。本任务没有运行它们。episode 数和总环境步数预算取自实际 reference，不由脚本临时改写；不存在恢复已有 HGR checkpoint 的步骤。

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/train_ch3_direct_mc.sh --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B2_direct_mc/collision_terminal/train_seed2729_v1
```

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/train_ch3_direct_boundary.sh --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B3_direct_boundary/collision_terminal/train_seed2729_v1
```

训练完成后审阅原 `summary.json` 的 `completed_main_trajectories`、`costs`、`actual_total_environment_steps` 与 `latest_checkpoint`，以及 `baseline_training_identity.json` 的状态和来源。因总步数预算停止不等于已完成请求的 main trajectory 数。

## 使用各自训练结果评价

以下从各自新训练目录的原生 summary 读取实际 `latest_checkpoint`，不猜测 cycle 序号。应先审阅训练是否达到所选实验预算；checkpoint 文件的 `hgr_` 前缀不代表它训练了 HGR 算法，入口将核对其中原生 method 和 algorithm。命令没有 checkpoint 时会失败，不会退回 HGR 权重。

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/run_ch3_baseline_eval.sh --baseline B2_direct_mc --checkpoint "$(python -c 'import json; print(json.load(open("/home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B2_direct_mc/collision_terminal/train_seed2729_v1/summary.json"))["latest_checkpoint"])')" --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --manifest /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/final_manifests/ch3_validation_M20_seed12729_n100.json --episodes 100 --seed 12729 --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B2_direct_mc/collision_terminal/eval100_seed12729_v1
```

```bash
bash /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/scripts/linux/run_ch3_baseline_eval.sh --baseline B3_direct_boundary --checkpoint "$(python -c 'import json; print(json.load(open("/home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B3_direct_boundary/collision_terminal/train_seed2729_v1/summary.json"))["latest_checkpoint"])')" --reference-training-config /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/hgr/collision_terminal/hgr_seed2729_main1000_v1/config.json --manifest /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/final_manifests/ch3_validation_M20_seed12729_n100.json --episodes 100 --seed 12729 --output-dir /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B3_direct_boundary/collision_terminal/eval100_seed12729_v1
```

## 查看进度

B0/B1 每完成一场刷新 console log，示例 B1：

```bash
tail -F /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B1_bser_prior/collision_terminal/eval100_seed12729_v1/run_console.log
```

B2/B3 原生 evaluator 每完成一场更新 native progress，示例 B2：

```bash
watch -n 10 cat /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR/outputs/chapter3/baselines/B2_direct_mc/collision_terminal/eval100_seed12729_v1/native_evaluation/evaluation_progress.json
```

## 复验代码

该测试集有 B0/B1 两步对照和一场 B1 完整合成任务；不启动训练，不使用正式 validation 数据作单元测试。

```bash
cd /home/legion/AUV-Thesis/AUV-Thesis-CH3-HGR && python -B -m unittest tests.test_ch3_baseline_registry -v
```

Windows 对应入口为 `scripts/run_ch3_baseline_eval.bat`、`scripts/train_ch3_direct_mc.bat`、`scripts/train_ch3_direct_boundary.bat`，参数与 Linux 完全一致；在已有 AUV Python 环境中运行即可。
