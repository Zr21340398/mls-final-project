# 模型压缩、虚拟设备分配与独立测试

本文面向刚在 HPC4 clone 项目、准备接手实验的同学。项目已有 21 次 `hpc4_core_v1` 训练记录；本指南从检查仓库、建立自己的环境和准备相同数据开始，继续进行 CPU 模型压缩、虚拟设备分配及冻结后的独立测试。已完成的训练不需要重跑。

以下在服务器 Bash 的项目根目录执行。账户 `mscaiehpc42026`、CPU 分区 `amd` 和 GPU 分区 `gpu-l20` 按实际授权调整。批处理脚本默认每个作业 4 CPU、16 GB 内存、2 小时；测试脚本另申请 1 张 GPU。可在 `sbatch` 命令中加 `--time=HH:MM:SS` 覆盖，具体时限依据节点配额与运行情况调整。

## 0. 新同学首次准备

下面区分 **登录节点** 与 **CPU 计算节点**。请先和项目同学确认共享数据的位置、读取权限、最终测试方案以及你可以使用的 Slurm 账户。示例工作目录为 `~/projects/mls-final-project`，如果 clone 在其他位置，使用自己的实际路径。

### 0.1 检查 clone 版本和现有实验

登录节点：

```bash
cd ~/projects/mls-final-project
git status --short
git log -1 --oneline
ls scripts/final_experiments.py scripts/hpc4_final_cpu.sbatch scripts/hpc4_final_test.sbatch
ls reports/plans/hpc4_core_v1/core_jobs.json
ls runs/hpc4_core_v1/core_coverage_equal_s7/rounds/0100/checkpoint.pt
```

检查实验脚本是否可以运行并显示压缩操作参数：

```bash
python3 scripts/final_experiments.py compress --help
```

如果脚本或 core 记录缺失，联系项目同学先推送最新版，再在**本地改动已经备份且允许更新**时执行 `git pull --ff-only`。如果 pull 提示会覆盖改动或无法快进，先保存现场并和项目同学协调；不要用强制重置。仓库中保存 core 的 run、配置、模型 checkpoint 和日志；原始和处理后数据位于 Git 忽略目录，需要按 0.3 获取。

这次交接更新调整了压缩和测试的数据路径解析。已经完成的 core 模型可按本指南评估；旧训练 run 如果需要续训，须使用其原始源码版本并通过原有源码哈希校验。后续新训练使用单独的 run 目录。

### 0.2 申请 CPU 节点并配置自己的环境

登录节点单独执行，等待 CPU 计算节点分配：

```bash
srun --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
```

在分配到的 CPU 节点建立独立 Linux 环境。参考原实验环境中的模块和 CUDA 版 PyTorch；模块名应与该服务器当前可用版本一致：

```bash
cd ~/projects/mls-final-project
source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y
module spider python
module load python/3.12.9-3lxwd5b
python --version
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

如果 `module spider` 显示的模块名不同，请选兼容 Python 3.12 的实际模块，并把后续环境脚本中的模块名同步改好。已有可用 `.venv` 时直接使用它，避免覆盖。原环境记录可在 `reports/hpc4_setup_v1/environment.freeze.txt` 查看；选择不同版本应记录原因和 `pip freeze`。网络访问或 CUDA wheel 安装失败时，先按学校网络及模块说明排查。

仓库的 `hpc/env.sh` 已作为作业环境入口。先检查：

```bash
cat hpc/env.sh
source hpc/env.sh
which python
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

`hpc/env.sh` 中加载的模块应与你刚安装时一致，并以项目根目录为基准激活自己的 `.venv`；如需修改，先保留原内容并只在自己的工作副本调整。CPU 节点上看到 CUDA 不可用是正常的；GPU 测试需要 GPU 作业中再次检查。每次重新连接后，作业脚本会各自在计算节点加载该文件，无需重建环境。

### 0.3 获取与 21 次 core 相同的处理后数据

**重要：**压缩与最终测试必须使用原 run 保存的数据指纹对应的同一份处理后数据。原数据未提交 Git；只重新下载公开原始数据并不保证重新生成的 manifest 指纹一致。优先从原实验同学处获得完整的 `data/processed/lcl_60` 目录，包含 `manifest.json` 和所有家庭数据文件；通过双方有权访问的共享目录或经过约定的文件传输复制。无需上传原始数据到 Git，也无需修改原 run 的 metadata。

若你已经获得原目录的只读权限，可在**自己的 CPU 计算节点终端**参考以下方式复制。先把 `DATA_SOURCE` 改成同学提供且可读取的真实目录；该路径只是示例，访问权限需要事先确认：

```bash
cd ~/projects/mls-final-project
DATA_SOURCE=/home/rzhoubf/projects/mls-final-project/data/processed/lcl_60
# 目标目录已有内容时先停下检查，避免覆盖。
if [ -e data/processed/lcl_60 ]; then echo '目标目录已存在，请检查后再操作'; else
  test -r "$DATA_SOURCE/manifest.json" && mkdir -p data/processed/lcl_60 && \
  rsync -a -- "$DATA_SOURCE/" data/processed/lcl_60/
fi
```

如果无法读取 `DATA_SOURCE`，请项目同学通过双方有权访问的存储提供副本，再放到自己的 `data/processed/lcl_60/`。不要修改别人主目录的权限或覆盖原数据。复制完成后，仍在 CPU 节点加载环境并校验全部家庭文件及 21 次 run 的数据指纹：

```bash
cd ~/projects/mls-final-project
source hpc/env.sh
python - <<'PY'
import json
from pathlib import Path
from energyfl.data import load_dataset

manifest, households = load_dataset('data/processed/lcl_60')
jobs = json.loads(Path('reports/plans/hpc4_core_v1/core_jobs.json').read_text())
fingerprints = {json.loads((Path(j['run']) / 'metadata.json').read_text())['dataset_fingerprint'] for j in jobs}
print('已检查家庭数:', len(households), '数据指纹:', manifest['fingerprint'])
print('run 数:', len(jobs), 'run 数据指纹:', fingerprints)
assert len(households) == 60 and not manifest['synthetic']
assert len(jobs) == 21 and fingerprints == {manifest['fingerprint']}, '数据版本不一致，请联系原实验同学'
print('数据和 run 指纹一致')
PY
```

这会读取处理后数组并验证文件哈希，不会运行测试集评估。如果校验失败，停止后续步骤并取得正确的数据副本。压缩和测试默认读取 run 配置中的 `data/processed/lcl_60`，从自己的项目根目录解析；每次实际加载仍会复核指纹和家庭文件哈希。无需重新在 GPU 上校准旧模型；校准报告是原训练的历史记录。

### 0.4 返回登录节点并确认作业脚本

CPU 节点准备完成后执行 `exit`，回到登录节点；再进入项目目录：

```bash
cd ~/projects/mls-final-project
bash -n scripts/hpc4_final_cpu.sbatch
bash -n scripts/hpc4_final_test.sbatch
```

随后从第 1 节提交压缩作业。创建冻结测试清单和最终测试的命令分开执行；在团队确认最终方案前，先进行压缩与验证结果分析。每次提交 `sbatch` 都会分配自己的计算节点并在结束后释放。退出或断开 SSH 后已提交的作业继续排队或运行。

## 1. CPU 压缩与虚拟模型分配

示例选择 coverage 的三个种子，使用验证集最佳 checkpoint；正式使用哪一种已有方法由你们在查看测试集之前确定。此操作使用验证集，自动读取项目目录下已校验的数据，不运行独立测试。

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

压缩和测试从项目根目录读取 run 配置中的相对数据路径 `data/processed/lcl_60`，无需指定额外路径参数；程序会检查 manifest 与原 run 的数据指纹。

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
