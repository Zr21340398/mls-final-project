# Federated Household Energy Forecasting

家庭能源预测项目的第一版可运行实现。训练、验证、测试、异构资源仿真、隐私预算记录和实验归档使用同一套配置。

**实现状态**：PyTorch GRU/MLP、FedAvg/等客户端聚合、三种调度、掉线/超时/资源预算、中央客户端级 DP 实验、掩码正确性模拟、CPU INT8、日志监控和报告汇总均有实现。`masking_sim` 用于掩码正确性模拟：目前尚未实现真实协议的密钥交换、秘密分享及掩码提交后的掉线恢复。仿真中的客户端没有物理隔离。不要将其报告为真实安全聚合系统。正式 SecAgg+ 集成应作为后续独立验证里程碑。

## 执行主线：现在该做什么

**不要把 README 中所有命令从头到尾执行一遍。** 正文按实验阶段组织；附录是新机器安装、故障恢复及后续阶段的参考。

| 阶段 | 当前状态 / 操作 | 阅读位置 |
| --- | --- | --- |
| 环境与数据 | 服务器环境和数据准备已完成；先检查 60 户数据路径 | 第 3.1 节 |
| 真实数据 pilot | 已完成 bootstrap 和报告；bootstrap 已包含 10 轮训练 | 主线第 1 节，无需补跑 train |
| 笔记本扩展验证 | **36 次训练与结果分析已完成；查看结论并备份归档** | 主线 2.5；归档命令见 2.4 |
| L20 正式实验 | 从 GPU 检查和节点校准开始；训练短测按需选择 | 第 3 节 |

第 1–2 节保留 Windows 预实验记录；正式实验在服务器 Bash 中按第 3 节执行。服务器环境与数据准备已完成，从第 3.1 节提交 GPU 检查和校准作业开始。

## 1. 真实数据 pilot（已完成，供回顾）

**当前状态：已完成，跳过本节，直接查看 2.5 的预实验结论。** 新环境尚未安装时先看附录 A。

以下命令自动完成官方 ZIP 下载、解压、12 户真实数据处理、CPU 性能测量、已校准配置生成，并运行 10 轮 pilot（只使用训练/验证集）：

```powershell
.venv\Scripts\python.exe -m energyfl bootstrap-pilot --run runs/lcl_pilot_v1
```

默认下载来自 [London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)，保留 UK Power Networks 来源与 Creative Commons Attribution 署名信息。下载文件约 796 MB（约 759 MiB），解压数据约 10 GB；程序会检查可用磁盘空间。默认网络超时 60 秒、最多 3 次重试，支持服务端允许时的断点续传。

命令的输出包括：

- `data/raw/lcl/lcl-168.zip` 和下载来源/哈希记录。
- `data/raw/lcl/csv/` 下的 CSV 及逐文件哈希清单。
- `data/processed/lcl_pilot/` 下的 12 户数据与时间划分记录。
- `reports/pilot_setup/profile.json`：3 次计时、每次 2 个预热 step + 20 个测量 step，采用每 step 耗时中位数。
- `configs/pilot_cpu.calibrated.json`：自动填入 `reference_step_s`、`calibration_source` 和校准文件 SHA256。原始 `pilot_cpu.json` 不会修改。
- `runs/lcl_pilot_v1/`：pilot 训练日志；不自动评估测试集。

重复执行会校验并复用已完成阶段。中断下载保留 `.part`；中断解压保留已完成 CSV；中断预处理保留独立 staging 目录用于诊断，重试重新执行预处理，不混用半成品。强制杀进程遗留锁时，错误信息会指出具体 lock；确认对应进程结束后才移除该锁。完成的数据/配置与请求不一致时会报错，不静默覆盖或更换 cohort。

**`bootstrap-pilot --run ...` 内部已经调用训练流程。** 即使没有单独输入 `train`，也已完成 10 轮训练，不需要补跑附录中的训练示例。该命令不自动生成 HTML 报告，可随后执行：

```powershell
.venv\Scripts\python.exe -m energyfl analyze --runs runs --out reports/lcl_pilot_v1
```

当前这一步也已完成。`analyze` 只读取已有结果生成报告，不会训练。当前直接查看 2.5 的预实验结论，不需要重新执行训练或依次执行附录。

## 2. 笔记本扩展验证（训练与分析已完成）

**当前状态：36 / 36 次运行均完成 50 轮，分析结论见 2.5；尚未生成各 run 的完整性清单。** 2.1–2.3 保留为实验设计与运行记录，不需要重跑。归档命令见 2.4。原 README 的 6.2 / 6.3 对应现在的 2.2 / 2.3。

本阶段扩大的是实验轮数和对照次数，仍使用已经处理好的 12 户真实 LCL 数据，不重新下载、不覆盖 `runs/lcl_pilot_v1`，也不重新创建虚拟环境。正式 60 户与隐私实验仍安排在后续 L20 阶段；这一步用于检查方案是否值得扩大；结论的观察范围为该 12 户家庭。

### 2.1 要验证的问题与固定条件

| 实验 | 组合 | 次数 | 验证目的 |
| --- | --- | --- | --- |
| 调度主对照 | random / fastest / coverage × 种子 7 / 17 / 27，超时 60 秒 | 9 | 50 轮是否比 10 轮学得更充分；调度差异能否跨种子复现 |
| 超时敏感性 | 三种调度 × 三个种子 × 超时 1 / 3 / 10 秒 | 27 | 调度结论是否依赖超时设定；短超时是否牺牲慢设备参与及预测效果 |

共 36 次，每次从头训练 50 轮。主对照的 60 秒结果也作为敏感性实验的宽松端点，无需重复运行。固定 GRU hidden=32、每轮 6 户、每户 10 个 minibatch 更新、batch=32、学习率 0.001、CPU 2 线程、等客户端聚合、裁剪开启、DP 关闭、plain 聚合、可用率 1.0、掉线率 0.1、最少成功客户端数 4。保持相同数据、时间划分、归一化与校准报告；同一随机种子下各调度使用相同的资源轨迹。

**超时依据**：当前校准下，三个资源档位的模拟完成时间约为 0.39 / 0.86 / 2.13 秒，并叠加 ±10% 波动。因此 1 秒会排除慢档设备，3 秒通常容纳全部正常设备，10 / 60 秒主要改变等待掉线客户端的成本。这些完成时间来自模拟模型。先固定这组阈值，再查看新实验结果，避免按效果挑选超时值。

已完成 pilot 的 10 轮训练循环约 28 秒。线性外推 36 × 50 轮约 84 分钟，仅作预算参考；实际时间受温度、后台负载、验证及文件写入影响，原计划预留 1–3 小时。本批实际记录累计 8.77 小时，单 run 中位数 150.38 秒，但存在小时级轮内计时异常；初始预算不能作为已验证的实耗承诺，详见 2.5。顺序执行，不要同时开启多个训练任务；接通电源并避免自动睡眠。模型很小，可在 CPU 上运行；正式实验可选择服务器 GPU。保留所有 checkpoint 会增加磁盘占用，应检查剩余空间。

### 2.2 一次生成全部配置（不启动训练）

以下 PowerShell 在项目目录运行。复用现有校准，只改变轮数、种子、调度及超时；不要改动校准耗时。现有 `experiment_matrix.py` 的 core 套件还包含其他基线；本阶段按下面的专用配置清单运行。

```powershell
cd D:\AIEproject\MAIE5532_MLS\final
$planDir = 'reports/plans/laptop_validation_v1'
$runRoot = 'runs/laptop_validation_v1'
if ((Test-Path $planDir) -or (Test-Path $runRoot)) {
    throw '该批次已存在。恢复训练请使用下一段；新批次请更换两个目录名。'
}
$baseJson = Get-Content 'configs/pilot_cpu.calibrated.json' -Raw -Encoding UTF8
New-Item -ItemType Directory -Path $planDir -Force | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding($false)
$jobs = @()
foreach ($deadline in @(60, 1, 3, 10)) {
    foreach ($scheduler in @('random', 'fastest', 'coverage')) {
        foreach ($seed in @(7, 17, 27)) {
            $cfg = $baseJson | ConvertFrom-Json
            $cfg.seed = $seed
            $cfg.train.rounds = 50
            $cfg.train.scheduler = $scheduler
            $cfg.simulation.deadline_s = $deadline
            $label = "${scheduler}_d${deadline}_s${seed}"
            $cfgPath = "$planDir/$label.json"
            [System.IO.File]::WriteAllText(
                (Join-Path $PWD $cfgPath),
                ($cfg | ConvertTo-Json -Depth 30), $utf8)
            $jobs += [pscustomobject]@{ config=$cfgPath; run="$runRoot/$label" }
        }
    }
}
[System.IO.File]::WriteAllText(
    (Join-Path $PWD "$planDir/jobs.json"),
    (ConvertTo-Json -InputObject $jobs -Depth 5), $utf8)
Write-Host "已生成 $($jobs.Count) 个实验配置，尚未启动训练。"
```

### 2.3 顺序执行与中断恢复

以下命令会实际启动训练，先执行 9 次 60 秒对照，再执行 27 次敏感性实验。可单独重新运行这一段：已完成的实验跳过，已有实验使用 `--resume`。如果修改了代码或配置，恢复校验可能拒绝继续；应保留旧记录，另建新批次。强制终止产生锁时，按附录 C.2 确认进程已退出后再解锁。

```powershell
cd D:\AIEproject\MAIE5532_MLS\final
$jobs = Get-Content 'reports/plans/laptop_validation_v1/jobs.json' -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($job in $jobs) {
    $statusPath = Join-Path $job.run 'status.json'
    if (Test-Path $statusPath) {
        $status = Get-Content $statusPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($status.status -eq 'complete') { continue }
    }
    $trainArgs = @('-m', 'energyfl', 'train', '--config', $job.config, '--run', $job.run)
    if (Test-Path (Join-Path $job.run 'config.json')) { $trainArgs += '--resume' }
    & .venv\Scripts\python.exe @trainArgs
    if ($LASTEXITCODE -ne 0) { throw "实验失败，已停止后续任务：$($job.run)" }
}
.venv\Scripts\python.exe -m energyfl analyze --runs runs/laptop_validation_v1 --out reports/laptop_validation_v1
```

**可选：** 在另一个终端查看进度；不打开监控也不会影响训练：

```powershell
.venv\Scripts\python.exe -m energyfl dashboard --runs runs/laptop_validation_v1 --port 8765
```

打开 `http://localhost:8765`。分析工具只扫描指定目录的直接子目录，因此这里必须传入 `runs/laptop_validation_v1`，不能只传 `runs`。

### 2.4 如何判断结果与保留证据

此阶段只查看训练/验证集，**不执行 `test`，不使用 `--test-after`**。先通过验证确定正式方案，随后再对冻结方案进行独立测试。

- 学习效果：比较第 10 / 50 轮及最佳验证轮的 macro MAE、RMSE、家庭 MAE 的 p90；观察最后 10 轮的趋势与波动。不要把每个种子的最低单点误差当作稳定收敛。
- 调度效果：同一种子配对比较三种调度，报告 3 个种子的均值和样本标准差，同时保留每个种子的结果。比较相同轮数，并从逐轮记录比较共同模拟时间预算内的误差；不要只比较最终虚拟耗时。
- 公平性：统计每户被选次数、成功贡献次数及各资源档位差异。100% coverage 只说明至少参与过一次，不能代表参与均衡。
- 超时影响：分别统计随机掉线、非掉线但超时、更新失败的轮次以及成功贡献数；失败轮也消耗时间。1 秒下慢档参与降低是预期压力条件，不应据此直接淘汰调度器。
- 预测基线：使用相同验证窗口的季节性基线作参考；原 pilot 的验证 macro MAE 约 0.18977 kWh，GRU 第 10 轮约 0.16536 kWh。这只是原 12 户结果，不能外推为正式数据集的结论。后续正式实验补齐 local、centralized 与 FedAvg 对照。

`reports/laptop_validation_v1/report.html` 和 `runs.csv` 提供逐次结果及曲线；当前 `seed_summary.json` 只汇总**测试**指标，因此本阶段为空是正常的。这些跨种子验证统计、逐户参与统计及共同模拟时间比较已从 `rounds/*/metrics.json`、`clients.json` 自动提取，见 2.5。现有 HTML 仍不自动完成这些分析；新增脚本可重新生成数值汇总。

保留整套 `reports/plans/laptop_validation_v1`、`runs/laptop_validation_v1`、原校准报告和处理后数据清单；每个 run 已包含配置、环境、源码快照、资源轨迹、逐轮指标和 checkpoint。完成训练后可逐一建立并检查完整性清单：

```powershell
$jobs = Get-Content 'reports/plans/laptop_validation_v1/jobs.json' -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($job in $jobs) {
    & .venv\Scripts\python.exe -m energyfl seal --run $job.run
    if ($LASTEXITCODE -ne 0) { throw "归档失败：$($job.run)" }
    & .venv\Scripts\python.exe -m energyfl verify --run $job.run
    if ($LASTEXITCODE -ne 0) { throw "完整性检查失败：$($job.run)" }
}
```

当前 Git 忽略规则已放行本批真实数据预实验：`runs/lcl_pilot_v1/`、`runs/laptop_validation_v1/` 及对应报告、计划和校准记录，包含小型模型 checkpoint。`data/raw/`、`data/processed/` 仍被忽略，数据集需独立备份；新实验目录默认仍被忽略。只保存 HTML 不足以支持最终报告复核。后续若添加测试结果，应重新生成完整性清单。

本阶段分析已完成，结论支持进入正式实验准备，但尚不能将 50 轮或某个调度器宣布为最优；仍需按第 3 节固定正式实验协议，再迁移到 L20 上的 60 户正式对照和隐私实验。更换设备需重新校准；不能直接把 CPU 的耗时报告当成 L20 实测值。若虚拟设备模型也随新校准改变，应明确记录并重新确定超时尺度，不能把两套模拟时间混为一组对照。

### 2.5 已完成的预实验分析与结论

分析日期：2026-10-29。本次只读取已有结果，没有重新训练或运行测试。审阅了 36 次运行、1,800 轮记录；各 run 保存配置与计划一致，数据指纹和保存的源码指纹一致，同种子下资源轨迹一致。扩大的是轮数和对照次数，家庭数仍为 12 户。

- [预实验结论文档](reports/preexperiment_analysis_v1/preexperiment_conclusions.docx)：完整分析、适用范围、耗时异常及正式实验准备。
- [跨种子汇总](reports/preexperiment_analysis_v1/groups.csv)、[逐次结果](reports/preexperiment_analysis_v1/runs.csv)与[结构化分析](reports/preexperiment_analysis_v1/analysis.json)。
- [同种子配对差值](reports/preexperiment_analysis_v1/paired.csv)、[共同模拟时间比较](reports/preexperiment_analysis_v1/same_time.csv)、[逐户参与](reports/preexperiment_analysis_v1/household_participation.csv)、[逐轮数据](reports/preexperiment_analysis_v1/rounds.csv)。
- [分析来源哈希](reports/preexperiment_analysis_v1/evidence_sha256.json)：本次读取文件的 SHA256 快照；完整 run 的归档仍需执行 seal/verify。

下表为 **3 秒超时、第 50 轮验证结果**。MAE 单位 kWh，“±”是 3 个种子的样本标准差；模拟时间和 Jain 指数为跨种子均值。Jain 基于全部 12 户的成功贡献次数，越接近 1 表示贡献次数越均衡；预测误差公平性需单独评估。

| 调度 | MAE 均值 ± SD | 累计模拟时间 秒 | 零贡献家庭数 | 贡献 Jain |
| --- | --- | --- | --- | --- |
| random | 0.15141 ± 0.00339 | 127.49 | 0 | 0.977 |
| coverage | 0.15254 ± 0.00327 | 107.66 | 0 | 0.669 |
| fastest | 0.15451 ± 0.00314 | 89.53 | 6 | 0.499 |

**学习与调度：** 50 轮比各自第 10 轮的平均 MAE 更低，但不能统一认定已收敛。coverage 相比 random 减少约 15.56% 模拟时间，MAE 增加约 0.75%；全部家庭均有贡献，但快档贡献约为慢档的 4.1 倍。fastest 在本批所有超时下均有 6 户从未贡献。coverage 预留的是较久未被选择者的机会，不保证成功贡献均衡。

**共同时间：** 每个种子用三种调度第 50 轮耗时的最小值作为共同模拟预算，取预算内最后完整轮次；3 秒下 random / coverage / fastest 的平均 MAE 为 0.15411 / 0.15455 / 0.15451。coverage 在 2 个种子优于 random，在另 1 个更差，尚无稳定精度优势。这是事后描述性分析，正式实验应预先固定比较预算。

**超时：** 同调度同种子的 3 / 10 / 60 秒完整验证曲线完全一致，不能当成额外独立重复。只要选中者有人掉线或超时，当前模拟器就等待整个 deadline；即使已满足最少人数也不提前结束。因此三档主要改变等待成本。1 秒是慢档无法成功返回的压力场景，random / coverage / fastest 平均失败轮数为 21.33 / 8.00 / 0.67。3 秒只可作为当前校准模型的参考值，不能直接宣称为真实硬件或 L20 的最优超时。

**算力与计时：** 36 次 run 累计记录 8.77 小时，单 run 中位数 150.38 秒，单轮中位数 2.81 秒；峰值进程 RSS 约 300.7–303.4 MB。`random_d60_s27` 第 9 轮单独记录 13,887.51 秒，异常主要落在轮内计时。现有日志不足以确认原因，原始记录保留；本批实测耗时不作为干净的调度速度对照，也不能据此直接预算 L20 卡时。

**结论范围：** 这些是 12 户、3 种子、固定家庭与资源映射的验证结果；尚无独立测试、正式 60 户基线、DP、压缩或真实安全聚合的性能结论。数据与硬件档位可能混杂，后续需改变资源映射并增加独立重复。

可选重算数值（只读原始 run；覆盖本分析目录的派生 CSV/JSON，不更新 Word 正文）：

```powershell
.venv\Scripts\python.exe scripts/analyze_preexperiment.py
```

**现在要做：** 阅读结论后，用 2.4 的 seal/verify 命令建立完整性清单，再备份完整 run、计划配置、处理后数据、校准记录、源码和分析材料。本次分析时 36 次 run 均没有 `integrity.json`，不能将“完成训练”写成“已完成完整归档”。

## 3. L20 正式实验

服务器环境和数据集已经准备完成。按本节依次执行 GPU 检查与节点校准 → 固定协议 → 生成并执行实验 → 验证分析 → 冻结后测试 → 保存归档。10 轮训练短测可按需选择。第 1–2 节的 12 户预实验保留供参考。

以下示例使用 `~/projects/mls-final-project` 和 `data/processed/lcl_60`；若你的实际目录不同，先替换路径。环境初始化以服务器已验证的方式为准；若已按照 [HPC4 指南](docs/HKUST_HPC4_L20_GUIDE.md) 建立 `hpc/env.sh`，在计算节点通过 `source hpc/env.sh` 加载。示例账户 `mscaiehpc42026` 与分区 `gpu-l20` 沿用现有指南，提交前确认账户授权与站点设置。

### 3.1 通过 sbatch 检查 GPU 和数据

在服务器登录节点、项目根目录执行。确认代码已包含 `scripts/hpc4_gpu_check.sbatch`，且 `hpc/env.sh` 能加载当前节点的 Python 模块和虚拟环境。示例账户和分区沿用现有服务器指南，提交前按实际授权调整：

```bash
cd ~/projects/mls-final-project
bash -n scripts/hpc4_gpu_check.sbatch
sbatch --account=mscaiehpc42026 --partition=gpu-l20 scripts/hpc4_gpu_check.sbatch check
```

记录返回的 JobID；作业排队后自动申请一张 GPU，完成检查或出错后释放资源。无需保持 SSH 连接：

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
cat slurm-check-JOBID.out
```

将 `JOBID` 替换为实际编号。脚本检查 `configs/l20.json` 指向的数据、全部家庭文件哈希、60 户真实数据、PyTorch CUDA 和一次 GPU 计算；日志包含数据指纹和 GPU 型号。数据路径不同时先将 `configs/l20.json` 中的 `data` 改为已准备的数据路径，并记录配置。`ENERGYFL_ENV` 可指定其他环境初始化脚本，例如 `sbatch --export=ALL,ENERGYFL_ENV=hpc/my_env.sh ...`。

### 3.2 节点校准与可选训练短测

跳过训练短测时，可将检查和校准放入一次作业，检查结束后自动校准并释放节点。若之前检查已通过，也可以直接提交此命令，它会再次检查 GPU 与数据：

```bash
sbatch --account=mscaiehpc42026 --partition=gpu-l20 scripts/hpc4_gpu_check.sbatch calibrate
```

作业成功后查看 `slurm-check-JOBID.out`、`reports/hpc4_setup_v1/profile.json` 和 `configs/l20.hpc4_v1.json`。后续计划使用新配置作为 base。脚本采用 `configs/l20.json` 中的模型、数据、batch、设备和线程；这些条件改变后，用新的配置和报告路径重新校准。已有同名结果时校准器会检查是否匹配，设置变化时先修改脚本中的输出路径，保留原记录。校准器也会执行少量训练 step 来测量耗时，完整正式训练从第 3.4 节的各实验独立开始。

完成校准后可直接进入第 3.3 节，固定正式轮数、调度、超时和评估协议，再生成 core 计划。当前模板为 100 轮、每轮 12 户、每户 20 steps、batch 64；正式执行前请确认这些设置和预计 GPU 预算。跳过 10 轮短测时，单次运行时间尚无完整流程实测，可通过第一个正式 run 的日志和 `sacct` 结果估算后续作业时限。

如需事先观察训练、验证和 checkpoint 的整体耗时，可另行提交 10 轮短测：

```bash
sbatch hpc/run.sbatch -m energyfl train --config configs/l20.hpc4_v1.json --run runs/hpc4_preflight_v1 --stop-after 10
```

使用独立短测目录；正式实验以新的计划和 run 目录启动。模拟时间、实际墙钟时间与模拟通信量分开记录。

### 3.3 在生成配置前固定正式协议

将下列选择写入实验记录，再生成矩阵。当前模板用于起步，正式预算和阈值由团队结合节点校准及已掌握的运行记录确定。

- **训练与预测：** 预测下一小时总用电量；固定数据版本、模型、batch、训练轮数、客户端数、学习率和选择 checkpoint 的规则。训练与验证阶段保留测试集。
- **主要对照：** core 包含 seasonal、local、centralized、样本加权 FedAvg，以及 random、fastest、coverage 等客户端加权对照；默认种子为 7、17、27，共 21 个配置。seasonal 使用一轮，其他配置按设定轮数训练。
- **系统评价：** 预先确定调度策略、超时、掉线与可用率、共同模拟时间或共同轮数比较方式；统计 MAE、RMSE、家庭误差 p90、成功贡献次数、失败轮、模拟时间和实际耗时。
- **模型假设：** 客户端、网络和设备资源在单机程序中模拟。当前家庭与资源档位按固定顺序映射；若正式协议要求随机化映射，需要先开发并验证，再用新源码版本和新批次执行。当前模拟器在选中客户端掉线或超时时等待至 deadline；如需提前聚合对照，需要先实现该规则。
- **隐私与压缩：** 先完成预测和调度对照，再决定 privacy 和 dropout 套件的具体范围；保留 plain 基线。`masking_sim` 为掩码正确性模拟；真实安全聚合协议尚待独立实现。DP 结果需要同时说明预算、机制和隐私边界。

### 3.4 生成 core 计划并核查

在项目根目录、已激活环境中执行。生成配置前先确认计划目录和运行目录均为空或不存在；重做计划使用新目录。矩阵继承校准配置中的训练参数、数据路径、截止时间和资源模型。

```bash
python scripts/experiment_matrix.py --base configs/l20.hpc4_v1.json --suite core --seeds 7 17 27 --plan reports/plans/hpc4_core_v1 --runs-root runs/hpc4_core_v1
python -c 'import json; p=json.load(open("reports/plans/hpc4_core_v1/core_jobs.json")); print("jobs",len(p)); [print(x["method"],x["config"],x["run"]) for x in p]'
```

检查 21 个配置的参数、数据指纹和目标目录；确认 GPU 检查、校准、实验协议及预算后提交。配置生成器的 `--execute` 会顺序运行作业，并使用当前计算节点资源；长任务推荐按下面的 Slurm 流程提交。预实验的分析脚本固定读取旧的 36 次 run；正式数据的逐户公平性和共同模拟时间汇总需要新增或调整分析代码。

### 3.5 顺序提交训练作业

使用服务器已经准备好的 `hpc/env.sh` 和 HPC4 指南中的 `hpc/run.sbatch`；该脚本需在作业中加载环境并执行 `srun python -u "$@"`。确保 `logs/` 存在，检查脚本中的账号、分区、单卡、内存、线程和时限，然后在登录节点执行：

```bash
cd ~/projects/mls-final-project
mkdir -p logs
bash -n hpc/env.sh
bash -n hpc/run.sbatch
python - <<'PY' > /tmp/energyfl_core_jobs.tsv
import json
for job in json.load(open('reports/plans/hpc4_core_v1/core_jobs.json')):
    print(job['config'] + '\t' + job['run'])
PY
previous=''
while IFS=$'\t' read -r config run; do
    if [ -n "$previous" ]; then
        jobid=$(sbatch --parsable --dependency="afterok:$previous" hpc/run.sbatch -m energyfl train --config "$config" --run "$run") || break
    else
        jobid=$(sbatch --parsable hpc/run.sbatch -m energyfl train --config "$config" --run "$run") || break
    fi
    jobid=${jobid%%;*}
    printf '%s %s %s\n' "$jobid" "$config" "$run"
    previous=$jobid
done < /tmp/energyfl_core_jobs.tsv
```

复制保存输出的作业编号及配置对应关系。后一个作业在前一个成功退出后开始；失败后检查队列中依赖作业的状态和失败原因，再决定恢复及后续提交。每次提交都要使用新的 run 目录；`--resume` 仅用于恢复已有的同一 run。

```bash
squeue -u "$USER"
scontrol show job JOBID
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
# 将 JOBID 换成实际值
```

作业中断后确认原作业已退出，再使用对应计划文件和 run 路径恢复：

```bash
sbatch hpc/run.sbatch -m energyfl train --config reports/plans/hpc4_core_v1/core_coverage_equal_s7.json --run runs/hpc4_core_v1/core_coverage_equal_s7 --resume
```

续训需要显式追加 `--resume`；重新执行原来的新建命令会因目录非空而停止。已完整提交的每一轮均有 checkpoint，程序从最后一个完整轮次继续；中断时正在执行的轮次重新计算。续训需重新申请 GPU，使用同一 config、run、数据指纹和源码；更改训练轮数也属于配置更改，需要建立新实验。Slurm 不会自动重提本示例作业，需在确认旧作业结束后手动提交恢复命令。

批量链中断时，先恢复失败的单个 run，再检查所有后续作业是否因依赖失败未启动；按计划重新提交尚未执行的作业，避免重复提交已完成或仍在排队的任务。

恢复要求配置、数据指纹及源码一致。遇到遗留 `RUNNING.lock`，先确认对应进程已经停止，再使用 `python -m energyfl unlock --run RUN路径 --confirmed-process-stopped`。检查 status、完整轮次和 Slurm 日志后再续跑；无需删除 checkpoint。

### 3.6 其他套件和压缩

core 结果核查后，根据已固定的研究问题分别生成计划：

```bash
python scripts/experiment_matrix.py --base configs/l20.hpc4_v1.json --suite privacy --seeds 7 17 27 --plan reports/plans/hpc4_privacy_v1 --runs-root runs/hpc4_privacy_v1
python scripts/experiment_matrix.py --base configs/l20.hpc4_v1.json --suite dropout --seeds 7 17 27 --plan reports/plans/hpc4_dropout_v1 --runs-root runs/hpc4_dropout_v1
```

privacy 默认 5 种设置 × 3 种子，dropout 默认 3 个掉线率 × 3 种调度 × 3 种子。检查各自 `privacy_jobs.json` / `dropout_jobs.json`、DP 预算及 GPU 时限，再参照 3.5 提交对应计划；将读取路径和临时清单名称改为该套件的文件。DP run 可能以 `privacy_budget_exhausted` 状态结束，检查该状态、已完成轮次和隐私预算。

量化需要共享神经模型，采用 CPU 推理测量，放在已加载环境的 CPU 计算节点，使用实际完成的 run 和新的输出目录：

```bash
python -m energyfl compress --run runs/hpc4_core_v1/core_coverage_equal_s7 --out reports/compression_hpc4_coverage_s7
```

记录压缩前后模型大小、验证 MAE 和 CPU 推理延迟。量化结果及设备适配规则以对应环境的测量为依据。

### 3.7 检查验证结果并冻结方案

作业完成后在计算节点运行：

```bash
python -m energyfl analyze --runs runs/hpc4_core_v1 --out reports/hpc4_core_v1
```

对 privacy、dropout 分别更换输入和输出目录。分析器读取指定目录的直接子 run；`runs.csv` 和 HTML 包含单次运行及验证曲线，`seed_summary.json` 汇总已有测试指标，测试前可能为空。结合逐轮 `metrics.json` 和 `clients.json` 检查学习趋势、各种子结果、每户被选择与成功贡献、掉线/超时、失败轮和资源使用。固定公平性、模拟时间和预算指标后补充跨种子统计及图表。资源映射固定、模拟时间、网络与能量假设等条件须在最终报告中说明。

根据验证集预先确定要测试的配置、种子和 checkpoint 规则，记录冻结版本、配置与数据指纹。DP 使用 last checkpoint；其他方案按预先固定的 best 或 last 规则选择。

### 3.8 独立测试与归档

测试在冻结协议之后进行。测试使用 run 记录的计算设备，CUDA run 需提交到 GPU 节点；以下命令只示范一个已完成的非 DP run：

```bash
sbatch hpc/run.sbatch -m energyfl test --run runs/hpc4_core_v1/core_coverage_equal_s7 --checkpoint best
```

每个 run 只执行一次测试；DP run 使用 `--checkpoint last`。测试后重新运行 `analyze` 更新测试汇总。若数据目录迁移，可传 `--data` 指向指纹一致的处理后数据。

对完成且计划测试已结束的每个 run 创建和检查完整性清单；以下为单次示例：

```bash
python -m energyfl seal --run runs/hpc4_core_v1/core_coverage_equal_s7
python -m energyfl verify --run runs/hpc4_core_v1/core_coverage_equal_s7
```

如归档后新增测试文件，应重新生成清单并核验。保存完整 runs、计划、配置、节点校准报告、数据 manifest、环境版本、源码提交号、Slurm 日志和汇总结果，另存一份备份。原始和处理后数据继续通过独立存储备份；新正式实验目录默认受 Git 忽略，检查大小与内容后再按需加入版本管理。下载示例和服务器脚本模板见 [HPC4 指南](docs/HKUST_HPC4_L20_GUIDE.md)。

## 附录 A：首次安装（仅新机器需要，当前跳过）

当前这台 Windows 电脑已配置并验证项目 .venv，可跳过本节。不要在已有环境上重复执行 python -m venv .venv，尤其不要使用不同版本的 Python 覆盖它。后续直接使用 .venv\Scripts\python.exe。以下命令仅适用于尚未创建环境的新机器。

Python 3.10+，建议 3.12。Windows CPU 首次安装流程：

```powershell
cd D:\AIEproject\MAIE5532_MLS\final
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt
```

Linux/L20 使用独立环境，在 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)选择与节点驱动兼容的 **2.8.0 CUDA wheel**，再安装 `requirements.txt`。CPU wheel 不能使用 L20。不要把 Windows `.venv` 复制到 Linux。无需安装 Flower/Ray 即可运行此版本。

后续命令中的 `python` 均指项目环境；Windows 可直接替换为 `.venv\Scripts\python.exe`。无需 `pip install -e .`，在项目根目录执行即可。

## 附录 B：合成数据与单元测试（按需工程检查）

已有真实数据 pilot 完成后，无需为了继续主线补跑本节。首次检查安装或修改代码后，可按需运行；合成测试不替代真实数据实验。

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
powershell -ExecutionPolicy Bypass -File scripts/smoke.ps1
```

脚本生成明确标为 synthetic 的数据、预处理 8 个家庭，训练 3 轮，单独评估测试集，输出 `reports/smoke/report.html`。已有同名实验或数据时会拒绝覆盖；更换输出路径，或直接查看已有结果。合成数据的数值只能用于工程验证，不能作为最终报告的 LCL 实验结果。

## 附录 C：独立命令参考（按需使用）

本附录替代旧版第 4 节。**当前笔记本批量验证已完成；主线 2.2 / 2.3 仅供回顾与恢复参考。** 以下各命令对应不同情形，不要整节依次复制运行。示例实验目录与主线不同，使用前必须替换为目标实验路径。

### C.1 拆开运行下载、预处理与校准

**按需参考。** 只在需要分别运行下载、预处理或校准时选择相应命令；已完成 bootstrap 的当前电脑无需执行。

```powershell
python -m energyfl download --out data/raw/lcl
python -m energyfl prepare --csv "data/raw/lcl/csv/*.csv" --out data/processed/lcl_pilot --start 2013-01-01 --end 2013-07-01 --clients 12
python -m energyfl calibrate --config configs/pilot_cpu.json --out-config configs/pilot_cpu.calibrated.json --report reports/pilot_setup/profile.json
```

`bootstrap-pilot` 需要自己生成的处理记录；手动 prepare 后直接执行 calibrate/train 即可，不必再跑 bootstrap。仅准备而不训练时省略 `--run`。

### C.2 恢复中断的单次训练或处理遗留锁

**仅训练中断时使用。** 主线 2.3 已自动跳过完成的任务，并恢复已有任务，通常无需手动运行此命令。下面以其中一个实验为例：

```powershell
.venv\Scripts\python.exe -m energyfl train --config reports/plans/laptop_validation_v1/random_d60_s7.json --run runs/laptop_validation_v1/random_d60_s7 --resume
```

每次实验目录必须唯一；配置、数据哈希、源码发生变化均不可恢复进原实验。`--stop-after 1` 可以主动验证恢复流程。恢复读取最后一个完整 round 事务；`.pending-*` 为未提交残留，会重新执行该轮。强制杀进程可能留下 `RUNNING.lock`，确认旧进程已终止后执行：

```powershell
.venv\Scripts\python.exe -m energyfl unlock --run runs/laptop_validation_v1/random_d60_s7 --confirmed-process-stopped
```



### C.3 独立测试与量化评估（当前验证阶段不执行）

**前提：正式实验方案冻结，目标 run 已完成训练。** 以下为将来的正式实验示例；`runs/coverage_s7` 是示例目录，请按实际 run 路径替换。

```powershell
# 独立测试：仅在决定使用测试集作最终评估时执行
python -m energyfl test --run runs/coverage_s7 --checkpoint best
```

测试默认选择最佳验证 checkpoint，每个 run 只允许测试一次，保存逐样本压缩预测。DP 实验必须 `--checkpoint last`，因为按照非私有验证指标选择模型需要单独考虑隐私计费。不要用测试结果修改配置后把重复评估伪装为独立测试。

```powershell
# 量化评估：仅在开展模型压缩实验时执行
python -m energyfl compress --run runs/coverage_s7 --out reports/compression_s7
```

### C.4 重新生成报告或查看监控（可选，不训练）

主线 2.3 的末尾已经生成报告。只想刷新报告时可单独执行第一条；想查看实时进度时可执行第二条。监控服务会占用当前终端，按 Ctrl+C 停止。

```powershell
.venv\Scripts\python.exe -m energyfl analyze --runs runs/laptop_validation_v1 --out reports/laptop_validation_v1
.venv\Scripts\python.exe -m energyfl dashboard --runs runs/laptop_validation_v1 --port 8765
```

## 附录 D：L20 正式实验命令示例（后续阶段，当前跳过）

**以下为按需参考的命令示例。** 服务器环境和数据已准备完成，正式实验步骤见第 3 节。以下 `python` 指节点环境中的 Python。Windows 环境不能复制到 Linux；迁移时还需检查数据与配置路径。

### D.1 节点校准与单次训练示例

```powershell
# 仅在 L20 环境和 60 户数据准备好后执行；生成节点专用校准
python -m energyfl calibrate --config configs/l20.json --out-config configs/l20.calibrated.json --report reports/profile_l20.json
# 校准成功后，如需启动这一单次实验，再执行
python -m energyfl train --config configs/l20.calibrated.json --run runs/coverage_s7
```

`calibrate` 自动完成性能测量和配置生成，不需要复制数值。换机器、模型、batch size、线程数或处理后数据，应生成新的校准配置/报告。训练会验证测量文件哈希与工作负载，并将它复制为 run 内的 `calibration_report.json`，防止原文件改变造成证据丢失。只有校准过的参考训练耗时是实测值；带宽、设备速度倍率、电池预算仍是明确声明的模拟假设。



### D.2 正式实验矩阵示例（与当前 36 次验证不同）

```powershell
python scripts/experiment_matrix.py --base configs/l20.calibrated.json --suite core --plan reports/plans/core_v1
```

默认只生成计划，不启动租用算力。`--execute` 才顺序执行所有任务；`--test-after` 应仅在实验方案冻结后使用。三种 suite 为 `core`、`privacy`、`dropout`，默认种子 7/17/27。重新生成时使用新的 `--plan` 目录和 `--runs-root` 输出目录；脚本不会覆盖已有实验。

`core` 含季节性、local、centralized、sample-weighted FedAvg，以及 equal-weight 的 random/fastest/coverage。公平比较调度器时比较后三者；不能把加权方式的变化也归因给调度器。默认所有神经联邦配置都启用更新裁剪，关闭 `privacy.clip_updates` 才是未裁剪 FedAvg；DP 始终强制裁剪。隐私 suite 让所有分支使用相同裁剪，以免混淆裁剪与掩码/噪声效果。

核心说明见 [算力与实现边界](docs/COMPUTE_AND_DESIGN.md)、[实验数据与报告流程](docs/EXPERIMENT_RECORDS.md)、[隐私说明](docs/PRIVACY.md)。

实验默认值与正式测试前的冻结规则见 [实验方案说明](docs/EXPERIMENT_PROTOCOL.md)。这些规则不要求先获得额外确认才能运行 pilot。

## 附录 E：数据处理与模型设计说明（无需执行）

### E.1 数据与时间划分

数据处理使用 `Std` 家庭，48 小时输入预测下一小时两段读数之和（kWh）。所有家庭共用 70/15/15 时间边界；归一化只使用训练集。短输入缺失最多向前填两格，缺失目标不补。按源文件无时区钟面处理重复时间，平均后记录计数，不能声称已恢复 DST 的真实物理时序。

处理采用两遍分块扫描，训练期数据决定 cohort，验证/测试缺失不参与选择。实际合格家庭数可能少于请求数；bootstrap 会检查它是否满足每轮客户端数要求。正式 60 户实验另建 `data/processed/lcl_60`，不会复用 12 户 pilot 数据冒充正式 cohort。

### E.2 GRU 在本项目中的角色

当前单层 GRU hidden=32 共 3,873 个参数，以 96 个半小时历史点和 6 个特征预测下一小时总用电量。它适合作为本项目的主预测模型：支持时间序列输入，模型小，便于反复开展联邦聚合、调度、裁剪/DP 与 CPU 动态 INT8 实验。项目的系统问题不要求使用大型模型。

但模型架构本身不保证准确率，也不能替代系统功能验证。需要通过季节性、local、centralized 等对照，以及 60 户、多种子的泛化结果证明适用性；量化后还需单独测量误差变化、体积和 CPU 推理耗时。当前预测目标为下一小时总量；未来 24 小时逐点预测和设备级负荷分解需要另行设计。真实安全聚合仍需独立实现与验证，不能因 GRU 能训练就视为整个项目已满足所有要求。

## 附录 F：HTTPS 证书错误排查（仅遇到下载错误时参考）

下载器通过 `truststore` 使用操作系统原生证书链验证，保留主机名和证书校验；Windows 使用系统证书存储及原生链构建能力。项目已安装 `truststore==0.10.5`，新机器安装 `requirements.txt` 会自动安装它。

如果仍报告 `CERTIFICATE_VERIFY_FAILED`，检查系统时间和机构代理的信任证书。如机构IT提供了可信的PEM CA文件，可设置 `ENERGYFL_CA_BUNDLE` 指向该文件后重试。不要使用关闭SSL验证的代码。证书错误会立即给出诊断，不会重复无效重试。
