# 模型压缩、虚拟设备分配与独立测试

本文给已经在 HPC4 clone 仓库、接手剩余实验的同学使用。操作顺序：检查已有结果 → 配置自己的环境 → 校验已上传的数据 → 查看压缩结果 → 团队确定测试方案 → 冻结测试清单 → 执行测试 → 汇总和保存。

## 当前项目状态

截至 2026-11-14，对照 GitHub `main` 的 `cf49aab` 检查：

- 已有 21 次 `hpc4_core_v1` 训练，记录、配置和 checkpoint 位于 `runs/hpc4_core_v1/`；计划位于 `reports/plans/hpc4_core_v1/`。
- 已有 `data/processed/lcl_60/`：1 个 manifest 和 60 个家庭数据文件，目录约 7 MB。manifest 指纹与抽查的 coverage run 一致。原始下载数据另行保存。
- 已有三个 coverage 种子的压缩与虚拟模型分配结果，位于 `reports/compression_hpc4_v1/`，包含 `variants.csv`、`assignments.csv` 和逐种子结果。
- 检查时 21 个 core run 根目录均未发现 `test/`；同学开始测试前仍须检查自己的工作副本，防止重复测试。

**因此，clone 完整后优先校验已上传的数据并查看现有压缩结果。** 如果该目录或文件缺失，再检查 clone 版本、下载是否完整以及 Git 文件状态；原实验同学提供相同数据副本可作为恢复方式。

项目采用真实笔记本或服务器运行联邦学习模拟；设备和网络参数按设定建模。本指南涉及模型压缩、虚拟设备分配和独立测试。隐私对照、系统分析、安全聚合设计及最终报告仍按团队整体计划完成。

## 0. 刚 clone 后的准备

以下操作区分服务器**登录节点**与**计算节点**。示例项目目录是 `~/projects/mls-final-project`，若 clone 的实际目录不同，请替换 `cd` 路径。账户 `mscaiehpc42026`、CPU 分区 `amd` 和 GPU 分区 `gpu-l20` 来自当前项目经验；请使用自己有权使用的账户，并以服务器实际配置为准。

### 0.1 登录并检查仓库

从自己的电脑连接服务器，使用自己已有的 SSH 命令或别名。进入登录节点后：

```bash
cd ~/projects/mls-final-project
pwd
git status -sb
git log -1 --oneline
ls scripts/final_experiments.py scripts/hpc4_final_cpu.sbatch scripts/hpc4_final_test.sbatch
ls reports/plans/hpc4_core_v1/core_jobs.json
ls runs/hpc4_core_v1/core_coverage_equal_s7/rounds/0100/checkpoint.pt
ls data/processed/lcl_60/manifest.json
ls reports/compression_hpc4_v1/variants.csv reports/compression_hpc4_v1/assignments.csv
```

如果当前副本的最新提交早于团队推送的版本，先确认 `git status` 的本地改动已经保存，再执行 `git pull --ff-only`。更新失败时保留提示并与项目同学协调。若检查到文件缺失，请先确认 clone 分支、拉取情况和文件路径；如果数据不完整，可以从原实验同学处取得完整 `data/processed/lcl_60/` 副本，再执行 0.3 的指纹检查。无需修改原 run 的 metadata。

### 0.2 在 CPU 计算节点配置环境

环境尚未建立时，从登录节点申请 CPU 计算节点并等待分配：

```bash
srun --account=mscaiehpc42026 --partition=amd --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
```

以下在分配到的 CPU 节点执行。原实验使用 Python 3.12.9 与 PyTorch 2.8.0 CUDA 12.6 wheel。示例模块名来自原环境，实际以节点可用模块为准：

```bash
cd ~/projects/mls-final-project
source /opt/shared/.spack-edge/dist/bin/setup-env.sh -y
module spider python
module load python/3.12.9-3lxwd5b
python --version
```

首次建立自己的环境：确认当前项目目录还没有 `.venv`，然后执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip check
```

已有可用 `.venv` 时激活已有环境，避免覆盖。安装所选模块与 CUDA wheel 版本应相互匹配；若示例模块不可用，使用 `module spider python` 查询，并将自己的 `hpc/env.sh` 中模块名调整为实际版本。旧环境信息可查看 `reports/hpc4_setup_v1/environment.freeze.txt`。

项目作业脚本执行时会加载 `hpc/env.sh`。在 CPU 节点检查：

```bash
cat hpc/env.sh
source hpc/env.sh
which python
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)'
```

文件应加载实际安装环境所用的 Python 模块并激活当前项目 `.venv`。CPU 节点无需显示可用 GPU；最终 CUDA 测试在已分配的 GPU 节点进行。后续重新连接无需重复安装环境。

### 0.3 校验数据与全部 run 的对应关系

仍在 CPU 计算节点、项目根目录执行。此命令读取并校验 manifest 和每个家庭文件的哈希，对照 21 个 run 的数据指纹；它不会计算测试集结果。

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
print('家庭数:', len(households), 'synthetic:', manifest['synthetic'])
print('数据指纹:', manifest['fingerprint'])
print('run 数:', len(jobs), 'run 数据指纹:', fingerprints)
assert len(households) == 60 and not manifest['synthetic']
assert len(jobs) == 21 and fingerprints == {manifest['fingerprint']}, '数据版本不一致，停止并联系原实验同学'
print('家庭文件与 run 数据指纹检查通过')
PY
```

如果数据文件缺失、损坏或指纹不一致，先修复 clone 或取得完整副本。保留原 manifest，避免修改指纹。程序从项目根目录解析 run 配置中的 `data/processed/lcl_60`，不需要额外指定路径。重新下载原始数据并预处理可能产生不同 manifest 指纹；后续评估应使用与原 run 匹配的数据版本。

完成后执行 `exit` 返回登录节点，再进入项目目录：

```bash
cd ~/projects/mls-final-project
bash -n scripts/hpc4_final_cpu.sbatch
bash -n scripts/hpc4_final_test.sbatch
```

批处理作业在计算节点自动执行 `source hpc/env.sh`，完成后释放节点。SSH 断开后已提交的作业继续由 Slurm 管理。

## 1. 查看已完成的压缩与虚拟模型分配

当前仓库已有 coverage 的 3 个种子结果，使用各 run 的验证集最佳 checkpoint、100 次重复和 5% 相对验证误差阈值。先直接查看：

```bash
cd ~/projects/mls-final-project
cat reports/compression_hpc4_v1/variants.csv
cat reports/compression_hpc4_v1/assignments.csv
cat reports/compression_hpc4_v1/experiment.json
```

`variants.csv` 保存实际 CPU 节点上的 FP32/INT8 文件大小、batch=1 推理中位数和 p95、验证 macro MAE；`assignments.csv` 保存三个虚拟资源档位的选择。逐种子目录还保存模型变体及 `compression.json`。

本批记录中，FP32 文件为 18,161 字节，INT8 为 8,227 字节；INT8 模型更小，测得的 CPU 中位推理延迟高于 FP32。三个资源档位均按既定规则选择 INT8。请结合逐种子实际数据呈现结果，**不要预设压缩一定加速**。

当前选型以验证误差阈值、序列化大小和模拟 p95 延迟判断；序列化大小是内存下限，CPU 计时只反映实际测试节点，虚拟设备延迟由倍率推算。该结果提供模拟选型依据；当前没有真实家庭设备上的在线切换。

### 如需重测

本节只在团队决定重测其他模型或节点时执行。`reports/compression_hpc4_v1` 已存在；输出使用新目录以保留原始结果。示例仍选 coverage 的三个种子：

```bash
sbatch --account=mscaiehpc42026 --partition=amd \
  scripts/hpc4_final_cpu.sbatch compress \
  --jobs reports/plans/hpc4_core_v1/core_jobs.json \
  --out reports/compression_hpc4_v2 \
  --repeats 100 --tolerance 0.05 \
  --runs core_coverage_equal_s7 core_coverage_equal_s17 core_coverage_equal_s27
```

`--tolerance 0.05` 为相对验证 MAE 的最大允许上升幅度，需在查看重测结果前确定。记录节点、线程和负载条件，使用新目录保存输出。执行后查看 `slurm-compress-JOBID.out` 和对应目录中的 CSV。失败时保留日志及输出目录，检查后用新的目录重试。

## 2. 冻结独立测试清单

与团队先确定比较的方法、种子、数据和 checkpoint 规则。程序对非 DP run 选验证集最佳 checkpoint，DP run 选最后 checkpoint。下例将 core 的 21 个 run 写入清单；只选择部分已确定的 run 时，可添加 `--runs run名称1 run名称2 ...`。

在登录节点执行；**本步骤不会评估测试集**：

```bash
cd ~/projects/mls-final-project
sbatch --account=mscaiehpc42026 --partition=amd \
  scripts/hpc4_final_cpu.sbatch prepare-tests \
  --jobs reports/plans/hpc4_core_v1/core_jobs.json \
  --out reports/final_eval_hpc4_v1/test_plan.json
```

记录返回的 JobID，确认成功后阅读日志及清单：

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
cat slurm-compress-JOBID.out
cat reports/final_eval_hpc4_v1/test_plan.json
```

清单记录每个 run 的策略、选择轮次、配置和 checkpoint 哈希、数据指纹和分组。检查团队是否认可所有项目与选择规则，再进入第 3 节。已有测试、未完成测试目录或未结束的训练会阻止生成；已存在的清单不会被覆盖。需要重新规划时保存旧文件并使用新路径。

## 3. 团队确认后执行独立测试

**此命令会访问测试集，在每个 run 中写入 `test/metrics.json` 和 `test/predictions.csv.gz`。** 先确认第 2 节的方案已冻结，再从登录节点提交 GPU 作业：

```bash
sbatch --account=mscaiehpc42026 --partition=gpu-l20 \
  scripts/hpc4_final_test.sbatch run-tests \
  --plan reports/final_eval_hpc4_v1/test_plan.json --confirm-frozen
```

可按分区上限在 `sbatch` 后添加 `--time=HH:MM:SS` 调整本次作业时限。作业脚本会加载自己的环境，按清单顺序评估。查看状态时把 `JOBID` 换成实际编号：

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
tail -n 80 slurm-final-test-JOBID.out
```

已经完成且与清单匹配的测试会被跳过；GPU 作业超时后，先确认旧作业已结束，再提交同一命令继续。若有 `.test-pending` 或结果与清单不匹配，先检查日志和目录，程序会停止以保留现场。测试使用 run 配置中的项目相对数据路径，并校验数据指纹。

## 4. 汇总、备份和交接

所有选定测试成功后，在登录节点提交 CPU 汇总作业：

```bash
sbatch --account=mscaiehpc42026 --partition=amd \
  scripts/hpc4_final_cpu.sbatch summarize-tests \
  --plan reports/final_eval_hpc4_v1/test_plan.json \
  --out reports/final_eval_hpc4_v1/summary_v1
```

查看 `slurm-compress-JOBID.out`、`summary_v1/test_runs.csv`、`summary_v1/test_groups.csv` 和 `provenance.json`。逐 run 指标包含 MAE、RMSE、家庭 MAE p90、checkpoint；分组表列出种子数、均值和样本标准差。分组基于除种子和数据路径外的配置指纹；单个种子的标准差为空。seasonal 的三个种子通常结果相同。

保留冻结清单、各 run 的测试结果、压缩输出、Slurm 日志、数据 manifest、校准与软件环境记录及源码版本。若 run 曾经 seal，新增测试文件后重新 seal 和 verify。`test/predictions.csv.gz` 与逐家庭指标未单独做 DP 保护；公开分享前检查内容与用途。

当前 `data/processed/lcl_60/` 已由团队上传并由 Git 跟踪。后续避免用 `git add .` 随意加入大文件或新实验输出；检查待提交文件、文件大小和远端权限，并保留独立备份。

## 5. 常见问题

- **SSH 断开**：重新连接后进入项目目录，用 `squeue`、`sacct` 和作业日志查看；已提交的作业由 Slurm 管理。
- **`source hpc/env.sh` 找不到 `.venv`**：确认在项目根目录，环境已在该 clone 中创建，模块版本与安装时一致。
- **CUDA 不可用**：检查测试作业是否实际分配 GPU、使用 CUDA 版 PyTorch 和正确的环境脚本；CPU 压缩本身使用 CPU。
- **数据指纹不同**：用 0.3 校验全部家庭文件，确认所有 run 引用同一版本，必要时向原实验同学取得完整副本。
- **输出目录已存在**：先看已有结果和日志；压缩或汇总重试使用新目录，冻结清单也使用新文件名。
- **原实验需要续训**：旧训练 run 按其记录的源码哈希校验；使用原源码版本恢复。当前更新后的代码用于完成本指南的压缩与测试。
