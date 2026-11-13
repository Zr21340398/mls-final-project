# 模型压缩、虚拟设备分配与独立测试

本文针对已完成的 `hpc4_core_v1` 实验。通过真实 CPU 测量 FP32/INT8 大小、推理延迟和验证误差，再根据预设的虚拟设备资源模型选择变体。独立测试另设冻结步骤。执行前把本指南、`scripts/final_experiments.py`、`scripts/hpc4_final_cpu.sbatch` 和 `scripts/hpc4_final_test.sbatch` 同步到服务器；确认数据、run、checkpoint、`hpc/env.sh` 和 core 计划仍在服务器原位置。

以下在服务器 Bash 的项目根目录执行。账户 `mscaiehpc42026`、CPU 分区 `amd` 和 GPU 分区 `gpu-l20` 按实际授权调整。批处理脚本默认每个作业 4 CPU、16 GB 内存、2 小时；测试脚本另申请 1 张 GPU。可在 `sbatch` 命令中加 `--time=HH:MM:SS` 覆盖，具体时限依据节点配额与运行情况调整。

## 1. CPU 压缩与虚拟模型分配

示例选择 coverage 的三个种子，使用验证集最佳 checkpoint；正式使用哪一种已有方法由你们在查看测试集之前确定。此操作使用验证集，读取 run 记录中的数据目录，不运行独立测试。

登录节点：
       
```bash
cd ~/projects/mls-final-project
bash -n scripts/hpc4_final_cpu.sbatch
sbatch --account=mscaiehpc42026 --partition=amd \
  scripts/hpc4_final_cpu.sbatch compress \
  --jobs reports/plans/hpc4_core_v1/core_jobs.json \
  --out reports/compression_hpc4_v1 \
  --repeats 100 --tolerance 0.05 \
  --runs core_coverage_equal_s7 core_coverage_equal_s17 core_coverage_equal_s27
```

查看进度和结果，将 `JOBID` 换成提交时返回的编号：

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
cat slurm-compress-JOBID.out
cat reports/compression_hpc4_v1/variants.csv
cat reports/compression_hpc4_v1/assignments.csv
```

`variants.csv` 逐种子记录实际 CPU 上 FP32/INT8 的序列化大小、batch=1 推理中位数和 p95、验证 macro MAE；`assignments.csv` 记录每个资源档位选用的变体以及模型化 p95。每个子目录还有 `fp32.pt`、`int8.pt` 和 `compression.json`；`experiment.json` 保存种子、配置/数据指纹、checkpoint 哈希、重复次数和阈值。

选型逻辑：验证 MAE 不超过 FP32 的 1.05 倍、模型文件大小满足虚拟内存阈值、p95 延迟除以设备速度倍率后满足虚拟时延阈值；符合条件时选文件较小的版本。没有合格版本时输出空值。`0.05` 是本例允许的相对误差上升上限，应在看结果之前确定；可用 `--tolerance` 调整为其他预先确定的阈值。

分配结果来自资源模型；序列化文件大小只是内存下限，实际运行时内存和框架开销需额外测量。CPU 测时反映所分配计算节点，虚拟设备时延使用倍率推算。为比较稳定，记录节点型号、负载与线程数；不要把不同节点的测量直接混为同一次对照。当前程序输出的是选型建议，实际家庭设备上的在线切换尚未部署。

输出目录必须为新目录；失败时保留原目录检查日志，用新输出路径重新执行。已完成的训练 run 和 checkpoint 不会被覆盖。目录和 Slurm 日志默认被 Git 忽略，应按项目备份规则保存。

## 2. 冻结独立测试清单

先完成验证集方案选择：决定要比较哪些方法、种子及 checkpoint 规则。程序对非 DP run 固定使用验证集最佳 checkpoint，对 DP run 使用最后 checkpoint。最终测试不会自动执行。

下例将 core 计划中的全部 21 个 run 加入冻结清单。若只测试事先选择的 run，可在命令末尾添加 `--runs run名称1 run名称2 ...`；每个名称应与 `core_jobs.json` 中的 run 目录名一致。

```bash
sbatch --account=mscaiehpc42026 --partition=amd \
  scripts/hpc4_final_cpu.sbatch prepare-tests \
  --jobs reports/plans/hpc4_core_v1/core_jobs.json \
  --out reports/final_eval_hpc4_v1/test_plan.json
```

该步骤检查 run 状态、锁、数据指纹、配置和所选 checkpoint，并在清单中保存 SHA256；它不加载或评估测试样本。存在已有测试输出或未完成测试目录时会停止，需先检查现有结果。冻结文件一经创建不会被覆盖；重新规划请使用新的文件名。

作业完成后审阅完整清单，确认与正式评估协议一致：

```bash
cat slurm-compress-JOBID.out
cat reports/final_eval_hpc4_v1/test_plan.json
```

清单包含每个 run 的 checkpoint 轮次、策略、哈希、数据指纹以及实验分组。测试前调整了原 run 的配置或 checkpoint 时，需要停止并重新检查，保留旧清单，必要时生成新清单；不要使用测试结果调整方案。

## 3. 在冻结后运行测试

**下面的命令会实际访问测试集，并在每个 run 写入 `test/metrics.json` 与 `test/predictions.csv.gz`。** 只有检查并确认冻结清单之后执行。当前 core run 的配置使用 CUDA，因此通过 GPU 批处理测试：

```bash
bash -n scripts/hpc4_final_test.sbatch
sbatch --account=mscaiehpc42026 --partition=gpu-l20 \
  scripts/hpc4_final_test.sbatch run-tests \
  --plan reports/final_eval_hpc4_v1/test_plan.json --confirm-frozen
```

按返回的作业编号查看进度：

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -n 80 slurm-final-test-JOBID.out
```

每次测试前会再次校验冻结配置、最新 checkpoint 和选中 checkpoint 的哈希。若有已经完成且与清单一致的测试，程序会保留并跳过；因此同一作业超时后，可确认旧作业结束后重新提交相同命令，从未完成的 run 继续。若发现 `.test-pending`，先核查日志与目录；程序会停止并保留现场，避免静默覆盖。已完成结果与清单不一致时也会停止，不能借此更换 checkpoint 反复试测。

当数据复制到新路径时，确保 manifest 指纹与冻结清单一致，并在上述 `run-tests` 命令末尾添加 `--data 新路径`。压缩命令当前读取 run 原始 metadata 中的数据路径，需要在原路径访问数据。

## 4. 汇总与留存

全部测试成功后，提交 CPU 汇总作业：

```bash
sbatch --account=mscaiehpc42026 --partition=amd \
  scripts/hpc4_final_cpu.sbatch summarize-tests \
  --plan reports/final_eval_hpc4_v1/test_plan.json \
  --out reports/final_eval_hpc4_v1/summary_v1
```

`summary_v1/test_runs.csv` 包含每次测试的 MAE、RMSE、家庭 MAE p90、方法、种子及 checkpoint；`test_groups.csv` 按除种子和数据路径外的配置指纹分组，包含均值与样本标准差（只有一个种子时为空）。同一计划使用同一数据指纹；DP 参数等其他配置差异会形成不同分组。seasonal 三个种子的结果通常相同，不能把它们当作三份独立数据集。

保存训练 run、冻结清单、压缩输出、两类 Slurm 日志、数据 manifest、节点环境和源码版本。测试会给 run 增加文件：若此前已 seal，应在所有预计输出完成后重新 seal 并 verify。`test/predictions.csv.gz` 和逐家庭指标属于未单独做 DP 保护的研究输出，公开分享前先检查内容和用途。

## 5. 本流程覆盖的范围

本文完成模型压缩、虚拟设备分配和冻结后的模型测试。隐私对照、安全聚合设计说明、完整参与/掉线分析和最终报告需要按照项目整体计划另行完成。教师允许综合模拟；在报告中分别标注节点实测和虚拟设备推算。
