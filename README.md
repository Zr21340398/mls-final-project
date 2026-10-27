# Federated Household Energy Forecasting

家庭能源预测项目的第一版可运行实现。训练、验证、测试、异构资源仿真、隐私预算记录和实验归档使用同一套配置。

**实现状态**：PyTorch GRU/MLP、FedAvg/等客户端聚合、三种调度、掉线/超时/资源预算、中央客户端级 DP 实验、掩码正确性模拟、CPU INT8、日志监控和报告汇总均有实现。`masking_sim` **不是 Flower SecAgg+**：目前尚未实现真实协议的密钥交换、秘密分享及掩码提交后的掉线恢复。仿真中的客户端没有物理隔离。不要将其报告为真实安全聚合系统。正式 SecAgg+ 集成应作为后续独立验证里程碑。

## 1. 首次安装（已有可用 .venv 时跳过）

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

## 2. 先在笔记本验证全流程

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
powershell -ExecutionPolicy Bypass -File scripts/smoke.ps1
```

脚本生成明确标为 synthetic 的数据、预处理 8 个家庭，训练 3 轮，单独评估测试集，输出 `reports/smoke/report.html`。已有同名实验或数据时会拒绝覆盖；更换输出路径，或直接查看已有结果。合成数据的数值只能用于工程验证，不能作为最终报告的 LCL 实验结果。

## 3. 自动下载、预处理与校准

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

如果只需要其中一个阶段：

```powershell
python -m energyfl download --out data/raw/lcl
python -m energyfl prepare --csv "data/raw/lcl/csv/*.csv" --out data/processed/lcl_pilot --start 2013-01-01 --end 2013-07-01 --clients 12
python -m energyfl calibrate --config configs/pilot_cpu.json --out-config configs/pilot_cpu.calibrated.json --report reports/pilot_setup/profile.json
```

`bootstrap-pilot` 需要自己生成的处理记录；手动 prepare 后直接执行 calibrate/train 即可，不必再跑 bootstrap。仅准备而不训练时省略 `--run`。

数据处理使用 `Std` 家庭，48 小时输入预测下一小时两段读数之和（kWh）。所有家庭共用 70/15/15 时间边界；归一化只使用训练集。短输入缺失最多向前填两格，缺失目标不补。按源文件无时区钟面处理重复时间，平均后记录计数，不能声称已恢复 DST 的真实物理时序。

处理采用两遍分块扫描，训练期数据决定 cohort，验证/测试缺失不参与选择。实际合格家庭数可能少于请求数；bootstrap 会检查它是否满足每轮客户端数要求。正式 60 户实验另建 `data/processed/lcl_60`，不会复用 12 户 pilot 数据冒充正式 cohort。

## 4. 训练、恢复和独立测试

```powershell
python -m energyfl calibrate --config configs/l20.json --out-config configs/l20.calibrated.json --report reports/profile_l20.json
python -m energyfl train --config configs/l20.calibrated.json --run runs/coverage_s7
python -m energyfl train --config configs/l20.calibrated.json --run runs/coverage_s7 --resume
python -m energyfl test --run runs/coverage_s7 --checkpoint best
python -m energyfl compress --run runs/coverage_s7 --out reports/compression_s7
python -m energyfl analyze --runs runs --out reports/comparison
python -m energyfl dashboard --runs runs --port 8765
```

`calibrate` 自动完成性能测量和配置生成，不需要复制数值。换机器、模型、batch size、线程数或处理后数据，应生成新的校准配置/报告。训练会验证测量文件哈希与工作负载，并将它复制为 run 内的 `calibration_report.json`，防止原文件改变造成证据丢失。只有校准过的参考训练耗时是实测值；带宽、设备速度倍率、电池预算仍是明确声明的模拟假设。

每次实验目录必须唯一；配置、数据哈希、源码发生变化均不可恢复进原实验。`--stop-after 1` 可以主动验证恢复流程。恢复读取最后一个完整 round 事务；`.pending-*` 为未提交残留，会重新执行该轮。强制杀进程可能留下 `RUNNING.lock`，确认旧进程已终止后执行：

```powershell
python -m energyfl unlock --run runs/coverage_s7 --confirmed-process-stopped
```

测试默认选择最佳验证 checkpoint，每个 run 只允许测试一次，保存逐样本压缩预测。DP 实验必须 `--checkpoint last`，因为使用非私有验证指标选择最佳模型本身不是已计费的 DP 机制。不要用测试结果修改配置后把重复评估伪装为独立测试。

## 5. 组织实验

```powershell
python scripts/experiment_matrix.py --base configs/l20.calibrated.json --suite core --plan reports/plans/core_v1
```

默认只生成计划，不启动租用算力。`--execute` 才顺序执行所有任务；`--test-after` 应仅在实验方案冻结后使用。三种 suite 为 `core`、`privacy`、`dropout`，默认种子 7/17/27。重新生成时使用新的 `--plan` 目录和 `--runs-root` 输出目录；脚本不会覆盖已有实验。

`core` 含季节性、local、centralized、sample-weighted FedAvg，以及 equal-weight 的 random/fastest/coverage。公平比较调度器时比较后三者；不能把加权方式的变化也归因给调度器。默认所有神经联邦配置都启用更新裁剪，关闭 `privacy.clip_updates` 才是未裁剪 FedAvg；DP 始终强制裁剪。隐私 suite 让所有分支使用相同裁剪，以免混淆裁剪与掩码/噪声效果。

核心说明见 [算力与实现边界](docs/COMPUTE_AND_DESIGN.md)、[实验数据与报告流程](docs/EXPERIMENT_RECORDS.md)、[隐私说明](docs/PRIVACY.md)。

实验默认值与正式测试前的冻结规则见 [实验方案说明](docs/EXPERIMENT_PROTOCOL.md)。这些规则不要求先获得额外确认才能运行 pilot。



## HTTPS 证书错误排查

下载器通过 `truststore` 使用操作系统原生证书链验证，保留主机名和证书校验；Windows 使用系统证书存储及原生链构建能力。项目已安装 `truststore==0.10.5`，新机器安装 `requirements.txt` 会自动安装它。

如果仍报告 `CERTIFICATE_VERIFY_FAILED`，检查系统时间和机构代理的信任证书。如机构IT提供了可信的PEM CA文件，可设置 `ENERGYFL_CA_BUNDLE` 指向该文件后重试。不要使用关闭SSL验证的代码。证书错误会立即给出诊断，不会重复无效重试。
