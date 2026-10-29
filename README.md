# Federated Household Energy Forecasting

家庭能源预测项目的第一版可运行实现。训练、验证、测试、异构资源仿真、隐私预算记录和实验归档使用同一套配置。

**实现状态**：PyTorch GRU/MLP、FedAvg/等客户端聚合、三种调度、掉线/超时/资源预算、中央客户端级 DP 实验、掩码正确性模拟、CPU INT8、日志监控和报告汇总均有实现。`masking_sim` **不是 Flower SecAgg+**：目前尚未实现真实协议的密钥交换、秘密分享及掩码提交后的掉线恢复。仿真中的客户端没有物理隔离。不要将其报告为真实安全聚合系统。正式 SecAgg+ 集成应作为后续独立验证里程碑。

## 执行主线：现在该做什么

**不要把 README 中所有命令从头到尾执行一遍。** 正文按实验阶段组织；附录是新机器安装、故障恢复及后续阶段的参考。

| 阶段 | 当前状态 / 操作 | 阅读位置 |
| --- | --- | --- |
| 环境安装 | 当前电脑已有可用 `.venv`，跳过 | 附录 A，仅新机器需要 |
| 真实数据 pilot | 已完成 bootstrap 和报告；bootstrap 已包含 10 轮训练 | 主线第 1 节，无需补跑 train |
| 笔记本扩展验证 | **36 次训练与结果分析已完成；查看结论并备份归档** | 主线 2.5；归档命令见 2.4 |
| L20 正式实验 | 当前进入方案与环境准备；尚未启动 | 主线第 3 节；附录 D 仅作命令参考 |

当前所有 PowerShell 命令均在 `D:\AIEproject\MAIE5532_MLS\final` 执行，使用 `.venv\Scripts\python.exe`。本批配置、36 次训练和分析均已完成，不要重复执行 2.2 / 2.3；先阅读 2.5，再完成证据归档与第 3 节的正式实验准备。正文中的“说明”无需执行命令，“可选”不影响主线完成。

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

本阶段扩大的是实验轮数和对照次数，仍使用已经处理好的 12 户真实 LCL 数据，不重新下载、不覆盖 `runs/lcl_pilot_v1`，也不重新创建虚拟环境。正式 60 户与隐私实验仍安排在后续 L20 阶段；这一步用于检查方案是否值得扩大，而不是宣称 12 户已经代表全部家庭。

### 2.1 要验证的问题与固定条件

| 实验 | 组合 | 次数 | 验证目的 |
| --- | --- | --- | --- |
| 调度主对照 | random / fastest / coverage × 种子 7 / 17 / 27，超时 60 秒 | 9 | 50 轮是否比 10 轮学得更充分；调度差异能否跨种子复现 |
| 超时敏感性 | 三种调度 × 三个种子 × 超时 1 / 3 / 10 秒 | 27 | 调度结论是否依赖超时设定；短超时是否牺牲慢设备参与及预测效果 |

共 36 次，每次从头训练 50 轮。主对照的 60 秒结果也作为敏感性实验的宽松端点，无需重复运行。固定 GRU hidden=32、每轮 6 户、每户 10 个 minibatch 更新、batch=32、学习率 0.001、CPU 2 线程、等客户端聚合、裁剪开启、DP 关闭、plain 聚合、可用率 1.0、掉线率 0.1、最少成功客户端数 4。保持相同数据、时间划分、归一化与校准报告；同一随机种子下各调度使用相同的资源轨迹。

**超时依据**：当前校准下，三个资源档位的模拟完成时间约为 0.39 / 0.86 / 2.13 秒，并叠加 ±10% 波动。因此 1 秒会排除慢档设备，3 秒通常容纳全部正常设备，10 / 60 秒主要改变等待掉线客户端的成本。这些是模拟完成时间，不是三种真实硬件的测量。先固定这组阈值，再查看新实验结果，避免按效果挑选超时值。

已完成 pilot 的 10 轮训练循环约 28 秒。线性外推 36 × 50 轮约 84 分钟，仅作预算参考；实际时间受温度、后台负载、验证及文件写入影响，原计划预留 1–3 小时。本批实际记录累计 8.77 小时，单 run 中位数 150.38 秒，但存在小时级轮内计时异常；初始预算不能作为已验证的实耗承诺，详见 2.5。顺序执行，不要同时开启多个训练任务；接通电源并避免自动睡眠。模型很小，无需 GPU；这不表示正式实验必须继续使用笔记本。保留所有 checkpoint 会增加磁盘占用，应检查剩余空间。

### 2.2 一次生成全部配置（不启动训练）

以下 PowerShell 在项目目录运行。复用现有校准，只改变轮数、种子、调度及超时；不要改动校准耗时。现有 `experiment_matrix.py` 的 core 套件还包含其他基线，不等于这里的 36 次实验，因此本阶段按下面的专用配置清单运行。

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
- [分析来源哈希](reports/preexperiment_analysis_v1/evidence_sha256.json)：本次读取文件的 SHA256 快照；不是训练前防篡改证明，也不替代完整 run 的 seal/verify。

下表为 **3 秒超时、第 50 轮验证结果**。MAE 单位 kWh，“±”是 3 个种子的样本标准差；模拟时间和 Jain 指数为跨种子均值。Jain 基于全部 12 户的成功贡献次数，越接近 1 越均衡，不表示预测误差公平性。

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

## 3. 后续主线：准备 L20 正式实验

预实验分析已完成，可以开始准备正式实验；无需重跑本批 36 次训练。启动正式训练前，依次固定以下内容：

1. **数据与对照：** 准备独立版本的 60 户数据，保存家庭清单与划分；补齐相同验证窗口下的季节性、local、centralized 和联邦预测基线。
2. **评价协议：** 保留 random / fastest / coverage，预先确定训练上限或验证停止规则、主指标、共同模拟时间预算；增加种子并改变家庭与资源档位映射。50 轮目前只是预实验预算，不是已证明的最优轮数。
3. **模拟语义：** 明确保持当前等待截止时间的规则，还是新增满足条件后提前结束的对照；明确虚拟设备参数是否保持不变。修改模型后使用新目录，不与本批混为同一对照。
4. **算力预算：** 节点安装 CUDA 环境并重新校准，先完成一个受控短任务，分别记录训练、验证、保存时间和 GPU 利用率，再预算整批卡时。当前小 GRU 不保证 GPU 更快。
5. **后续阶段：** 先完成预测和调度对照，再增加 DP、压缩与安全聚合验证，最后对冻结方案进行独立测试。`masking_sim` 仍不是真实 SecAgg+。

附录 D 提供命令形式，不能替代节点环境和数据准备。无需在当前笔记本上提前执行这些 GPU 配置。当前准备阶段不运行独立测试或隐私实验。

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

## 附录 C：独立命令参考（按需使用，不是必做顺序）

本附录替代旧版第 4 节。**当前笔记本批量验证已完成；主线 2.2 / 2.3 仅供回顾与恢复参考。** 以下各命令对应不同情形，不要整节依次复制运行。示例实验目录与主线不同，使用前必须替换为目标实验路径。

### C.1 拆开运行下载、预处理与校准

**按需参考，不是 bootstrap 之后的追加步骤。** 只在需要分别运行下载、预处理或校准时选择相应命令；已完成 bootstrap 的当前电脑无需执行。

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

**前提：正式实验方案冻结，目标 run 已完成训练。** 以下为将来的正式实验示例；`runs/coverage_s7` 不是本阶段自动生成的目录。

```powershell
# 独立测试：仅在决定使用测试集作最终评估时执行
python -m energyfl test --run runs/coverage_s7 --checkpoint best
```

测试默认选择最佳验证 checkpoint，每个 run 只允许测试一次，保存逐样本压缩预测。DP 实验必须 `--checkpoint last`，因为使用非私有验证指标选择最佳模型本身不是已计费的 DP 机制。不要用测试结果修改配置后把重复评估伪装为独立测试。

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

**这不是可立即整段执行的部署流程。** 笔记本结果分析已完成；仍需确定正式方案，在 L20 节点配置 CUDA 版 PyTorch 环境，并准备好 `data/processed/lcl_60` 的 60 户数据。以下 `python` 指节点环境中的 Python。Windows 环境不能复制到 Linux；迁移时还需检查数据与配置路径。

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

但模型架构本身不保证准确率，也不能替代系统功能验证。需要通过季节性、local、centralized 等对照，以及 60 户、多种子的泛化结果证明适用性；量化后还需单独测量误差变化、体积和 CPU 推理耗时。当前预测目标仅为下一小时总量，不是未来 24 小时逐点预测或设备级负荷分解。真实安全聚合仍需独立实现与验证，不能因 GRU 能训练就视为整个项目已满足所有要求。

## 附录 F：HTTPS 证书错误排查（仅遇到下载错误时参考）

下载器通过 `truststore` 使用操作系统原生证书链验证，保留主机名和证书校验；Windows 使用系统证书存储及原生链构建能力。项目已安装 `truststore==0.10.5`，新机器安装 `requirements.txt` 会自动安装它。

如果仍报告 `CERTIFICATE_VERIFY_FAILED`，检查系统时间和机构代理的信任证书。如机构IT提供了可信的PEM CA文件，可设置 `ENERGYFL_CA_BUNDLE` 指向该文件后重试。不要使用关闭SSL验证的代码。证书错误会立即给出诊断，不会重复无效重试。
