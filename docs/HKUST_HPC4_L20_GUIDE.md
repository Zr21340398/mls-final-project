# HKUST HPC4 L20 家庭能源预测项目运行指南

适配日期：2026-11-05。项目为 MAIE5532 家庭能源预测，服务器目录约定为 `~/projects/mls-final-project`。本指南依据另一个项目的 HPC4 使用经验和当前仓库实现改写；未连接服务器、未实际提交作业。SSH 别名 `hkust-hpc4`、账户 `mscaiehpc42026`、分区 `amd` / `gpu-l20`、Edge 初始化路径来自该经验，不代表本次已核实其可用性。先确认本项目可使用该账户，并以服务器模块、分区和配额查询结果为准。

## 1. 当前进度与操作顺序

已完成并归档到 Git 的是 12 户 pilot 和 36 次扩展验证，不需要在服务器重复跑。接下来按顺序执行：获取代码 → CPU 环境与 60 户数据准备 → L20 校准和两轮短测 → 固定正式实验方案 → 批处理训练 → 汇总与备份。

`configs/l20.json` 是起步模板，不是已经批准的最终实验协议：100 轮、每轮 12 户、每户 20 steps、batch 64、4 CPU 线程、单 GPU。三种调度、截止时间、资源映射和评价预算仍应依据预实验结论确定。现有代码未实现资源映射随机化；`masking_sim` 不是真实 SecAgg+。

所有 Linux 命令均在服务器执行；PowerShell 只用于连接、传输。以下下载、安装、处理数据和训练放在计算节点执行；登录节点用于 Git、提交作业和查看日志。不要整篇一次粘贴。

## 2. 获取代码与选择存储

本机 PowerShell：

```powershell
ssh hkust-hpc4
```

服务器登录节点，首次执行：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Zr21340398/mls-final-project.git
cd mls-final-project
mkdir -p hpc logs
```

已有 checkout 时先 `git status`，确认本地改动后再更新。训练运行期间不要更新其源码目录。仓库内已有预实验 checkpoint、配置和报告；没有原始或处理后数据，也没有服务器环境。

项目数据默认位于 `data/raw/lcl` 与 `data/processed/lcl_60`，约 796 MB 下载、约 10 GB 原始解压文件，另需处理结果空间。先用 `df -h .` 检查空间及账户配额。大数据可放站点允许的 scratch 并在项目中建立数据目录链接；重要 run 和报告保存在持久共享存储并另行备份。scratch 的清理策略和配额以学校当前规定为准，不作为唯一结果存储。

## 3. CPU 节点准备 Linux 环境

登录节点单独执行，等待资源分配：

```bash
srun --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
```

CPU 节点：

```bash
cd ~/projects/mls-final-project
source /opt/shared/.spack-edge/dist/bin/setup-env.sh
module spider python
```

选择实际可用的 Python 3.12 模块，下面的 `python/3.12.9` 仅为原经验中的例子。首次创建环境；已经建立的环境无需重建：

```bash
module load python/3.12.9
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

单独安装 **PyTorch 2.8.0 的 CUDA wheel**，再安装 `requirements.txt`（该文件不安装 torch）。例如，确认节点驱动支持 CUDA 12.6 wheel 后可执行：

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

驱动与 CUDA wheel 的兼容性需在 GPU 节点检查；这里不要求为普通 PyTorch 预编译包额外编译 CUDA 扩展。不要复制 Windows `.venv` 或安装 CPU wheel 后直接启动 GPU 训练。

保存以下内容为 `hpc/env.sh`，替换成刚才实际使用的 Python 模块，使用 UTF-8、LF 换行：

```bash
# 在项目根目录 source 本文件。
source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y || return 1
module load python/3.12.9 || return 1
source .venv/bin/activate || return 1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 4. 准备 60 户真实数据

仍在 CPU 节点、项目根目录执行：

```bash
source hpc/env.sh
python -m energyfl download --out data/raw/lcl
python -m energyfl prepare --csv 'data/raw/lcl/csv/*.csv' --out data/processed/lcl_60 --start 2013-01-01 --end 2013-07-01 --clients 60
```

程序在内部展开带引号的 CSV 通配符。准备完成后检查实际家庭数（请求 60 户不等于无需检查就一定得到 60 户）：

```bash
python -c 'import json; m=json.load(open("data/processed/lcl_60/manifest.json")); print("households:", len(m["clients"])); print("fingerprint:", m["fingerprint"]); print("synthetic:", m["synthetic"])'
```

此处使用原计划的半年窗口作为起步方案；改变家庭数或时间窗口需保存为新数据版本。已有完整处理目录时先检查并复用，不重复执行 prepare 覆盖；数据不一致时改用新目录，并同步修改训练模板中的 `data`。

不调用默认 `bootstrap-pilot`：它针对旧 12 户 pilot，而且仓库已有 Windows 校准文件，可能触发路径不匹配。60 户实验使用本节独立下载与预处理流程。若计算节点无法访问下载源，可上传本机原始数据目录（含下载/解压记录），再运行处理命令；不要关闭 HTTPS 证书验证。

完成后 `exit` 返回登录节点。

## 5. L20 校准与两轮完整短测

登录节点单独申请一张卡：

```bash
srun --account=mscaiehpc42026 --partition=gpu-l20 --nodes=1 --ntasks=1 --cpus-per-task=4 --gpus-per-node=1 --mem=32G --time=00:30:00 --pty bash
```

GPU 节点：

```bash
cd ~/projects/mls-final-project
source hpc/env.sh
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(512,512,device="cuda"); y=x@x; torch.cuda.synchronize(); print(y.shape)'
python -m energyfl calibrate --config configs/l20.json --out-config configs/l20.hpc4_v1.json --report reports/hpc4_setup_v1/profile.json
python -m pip freeze > reports/hpc4_setup_v1/environment.freeze.txt
python -m energyfl train --config configs/l20.hpc4_v1.json --run runs/hpc4_preflight_v1 --stop-after 2
```

短测调用真实训练、验证和保存流程，但只完成两轮，不是正式性能结果。查看 `runs/hpc4_preflight_v1/rounds/0002/metrics.json`、日志和 checkpoint。校准输出会包含服务器绝对路径，应保留在该节点共享文件系统；换数据、模型、batch、设备或线程后，用新的文件名重新校准。

不要把 `configs/pilot_cpu.calibrated.json` 当作 L20 校准。已有同名短测或校准文件时先检查状态；改变条件则创建 v2 路径，不覆盖证据。

记录短测耗时、进程内存和 GPU 使用情况，再决定正式时限。小 GRU 不保证 GPU 比 CPU 更快。校准只测参考 step，带宽、设备倍率、能量单位仍是模拟假设。CPU 阶段的 1 / 3 / 10 / 60 秒阈值不可不加说明直接解释为 L20 设备的合理截止时间。

完成后 `exit`，并在登录节点用 `squeue -u "$USER"` 确认交互作业结束。

## 6. 批处理运行

保存为 `hpc/run.sbatch`，UTF-8、LF。账户、分区和时限按实际授权修改：

```bash
#!/bin/bash
#SBATCH --job-name=energyfl
#SBATCH --account=mscaiehpc42026
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -e
cd "$SLURM_SUBMIT_DIR"
source hpc/env.sh
if [ "$#" -eq 0 ]; then
    echo 'Usage: sbatch hpc/run.sbatch -m energyfl train ...'
    exit 2
fi
hostname
date
nvidia-smi
git rev-parse HEAD
srun python -u "$@"
date
```

先确认正式配置、训练上限和模拟语义，以下为单次运行示例，不代表完整正式实验已经定稿。登录节点执行：

```bash
cd ~/projects/mls-final-project
mkdir -p logs
bash -n hpc/env.sh
bash -n hpc/run.sbatch
sbatch hpc/run.sbatch -m energyfl train --config configs/l20.hpc4_v1.json --run runs/hpc4_core_v1/coverage_s7
```

已有 `scripts/l20.slurm` 是另一种通用入口，需要 `ENERGYFL_PYTHON`，默认还指向未校准模板。采用本文 `hpc/run.sbatch` 即可，不需要两个入口都执行。申请多卡不会自动让当前程序并行。

批量方案可由 `scripts/experiment_matrix.py` 生成。使用节点校准配置作为 base，新的计划目录和 runs-root；先只生成并检查，不直接加 `--execute`。例如：

```bash
python scripts/experiment_matrix.py --base configs/l20.hpc4_v1.json --suite core --plan reports/plans/hpc4_core_v1 --runs-root runs/hpc4_core_v1
```

core 默认包含 7 种基线/调度 × 3 种子共 21 次。它不等同于已完成的 36 次笔记本对照。正式运行时以生成的 `core_jobs.json` 为准，避免与上面的单次示例重复运行。隐私及 dropout 套件在对应阶段另行生成；不要启用 `--test-after` 提前评估测试集。

## 7. 查看状态与恢复

登录节点将 123456 替换成实际 JobID：

```bash
squeue -u "$USER"
scontrol show job 123456
tail -n 80 logs/energyfl-123456.out
tail -n 80 logs/energyfl-123456.err
sacct -j 123456 --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

确认作业已终止后，使用相同源码、数据和配置恢复同一个 run：

```bash
sbatch hpc/run.sbatch -m energyfl train --config configs/l20.hpc4_v1.json --run runs/hpc4_core_v1/coverage_s7 --resume
```

恢复会校验配置、数据和源码哈希；更改轮数或其他参数需要新实验，不能直接修改旧配置再恢复。程序从最后完整 round 恢复，未提交轮会重算。若报 RUNNING.lock，先用队列及进程状态确认旧作业已停止，再按错误提示执行 unlock；不能在训练仍运行时解锁。

已归档的 Windows run 保留原样。重新处理数据可能因路径等来源信息使指纹不同；不能只改 manifest 强行通过旧 checkpoint 的校验。新实验使用服务器新数据版本和新 run。

## 8. 汇总与保存证据

在 CPU 计算节点或允许的轻量分析环境运行：

```bash
python -m energyfl analyze --runs runs/hpc4_core_v1 --out reports/hpc4_core_v1
```

分析器只扫描直接子目录。未做测试时 `seed_summary.json` 为空是正常现象；它汇总的是测试指标。`scripts/analyze_preexperiment.py` 固定针对旧 36 次实验，不能直接用于本批 60 户结果，需另行扩展正式结果汇总逻辑。

完成的每个 run 执行 seal/verify，并一起保存计划、校准报告、数据 manifest、依赖版本、源码提交号及 Slurm 日志。单次示例：

```bash
python -m energyfl seal --run runs/hpc4_core_v1/coverage_s7
python -m energyfl verify --run runs/hpc4_core_v1/coverage_s7
```

本机 PowerShell，把服务器结果下载到新的接收目录，避免覆盖本机预实验：

```powershell
New-Item -ItemType Directory -Force 'D:\AIEproject\MAIE5532_MLS\final\hpc4_downloads\v1'
scp -r hkust-hpc4:~/projects/mls-final-project/runs/hpc4_core_v1 'D:\AIEproject\MAIE5532_MLS\final\hpc4_downloads\v1\'
scp -r hkust-hpc4:~/projects/mls-final-project/reports 'D:\AIEproject\MAIE5532_MLS\final\hpc4_downloads\v1\'
scp -r hkust-hpc4:~/projects/mls-final-project/logs 'D:\AIEproject\MAIE5532_MLS\final\hpc4_downloads\v1\'
```

Git 当前仅放行旧 pilot/扩展验证目录。新的正式 run、报告、日志和节点校准配置应在检查大小与内容后明确加入版本管理；原始与处理后数据继续单独备份，不加入 Git。hpc4_downloads 只是本机接收副本，不要再重复提交它。

正式方案冻结之后才执行独立 test；DP 实验使用 last checkpoint，其他方案按事先指定规则选择。不要为确认服务器可运行而提前用测试集调参。

## 9. 来源与适用边界

本指南依据用户提供的另一项目 `HKUST_HPC4_L20_GUIDE.md`，并核对本项目 CLI、校准校验、恢复逻辑和实验矩阵脚本。学校的实时账户权限、软件模块、存储政策和 GPU 可用性尚未在本次任务中验证。

学校查询入口：[HPC4](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/hpc4)、[存储](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/hpc4/fileandstorage)、[Slurm](https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/hpc4/slurm)。PyTorch wheel 按 [历史版本安装说明](https://pytorch.org/get-started/previous-versions/) 和节点驱动匹配。
