# 实验过程留存与最终分析

## 目录契约

```text
runs/<唯一实验ID>/
  config.json                 完整配置，包含种子、模型、资源与隐私参数
  metadata.json               数据哈希、源码哈希、环境、命令、模拟/安全声明
  dataset_manifest.json       数据来源、输入文件哈希、时间划分、家庭清单及质量统计
  environment.freeze.txt      实际运行环境包版本
  source.zip                  运行开始时源码、配置、测试、说明快照
  trace.json                  所有轮次的公开模拟条件，调度公平比较可核对
  status.json                 当前状态；监控可重复读取
  rounds/0001/
    checkpoint.pt             全局/本地模型、轮次、参与次数、DP预算状态
    metrics.json              验证误差、真实时间、模拟时间、字节、覆盖率等
    clients.json              每客户端是否可用/被选/幸存、资源与训练统计
  test/
    metrics.json              固定checkpoint的汇总与逐家庭测试结果
    predictions.csv.gz        每个家庭、目标时间、真实kWh与预测kWh
```

`rounds` 是权威记录；只有完整事务目录才算完成，`status.json` 是可重建的显示信息。中断后不会把同一轮追加两遍。恢复要求配置/处理后数据/源码一致；硬件和库升级可能影响数值，环境迁移应新开实验。非 DP 模式的连续训练与断点恢复有模型权重完全一致的回归测试。启用安全随机噪声的 DP 运行只保证已提交轮状态延续，不承诺重新执行未提交轮能产生相同随机噪声。

每一轮保留 checkpoint：小模型成本低，后期能重新评估/检查收敛。较大模型或 local 基线可能增大目录体积，运行后查看实际大小再归档；不要在实验进行中删除 checkpoint。最终报告不需要保留每个训练 batch 的输入或梯度。本版不保存逐 batch 梯度、真实家庭原始窗口副本或模型更新向量到日志；公开数据路径与哈希足够追溯。

## 建议的冻结流程

1. 数据审计后冻结 `manifest.json`；保存下载来源与许可。不要将不同 cohort 的结果直接放在同一比较组。
2. 在训练/验证集确定模型、超参数和模拟时间阈值，生成实验矩阵；将配置目录纳入版本控制。
3. 至少 3 个种子运行主要比较，使用同一个 processed 数据与资源 trace 生成规则。
4. 冻结方案后显式执行 `test`，每次 run 仅保存一次，不能用测试结果调参。
5. 运行 `analyze`，得到逐 run CSV、同配置不同种子的均值/样本标准差 JSON、可直接浏览的 HTML 收敛曲线。分组指纹包含除种子/路径外的完整配置及数据指纹，防止把不同实验误平均。单种子标准差记 null。
6. 最终表格分别呈现：macro MAE、macro RMSE、家庭 MAE 的90分位、实际 wall time、虚拟时间、模型化字节、失败轮数、参与覆盖率、epsilon，以及量化 size/median/P95。

当前报告展示验证 MAE vs round。若需要 time-to-target，先在 pilot 后声明固定验证阈值，再从逐轮 `virtual_elapsed_s` 找第一次达到它的轮次；未达到的实验必须记为未达到，不能删除。分析脚本不会自动挑一个有利阈值。

## 备份和防误用

- 每次任务结束，把整个 run 目录和 processed manifest 归档到持久存储；超算临时盘/作业 scratch 不可作为唯一副本。将原始数据放独立数据目录，避免为每个种子复制 10 GB 文件。
- 建议保留至少两份：节点工作副本 + 本地或机构持久存储副本。传输后核对清单哈希，或自行对归档包计算 SHA256。本代码不会自动向外部服务上传数据。
- 保留 `source.zip` 和 `environment.freeze.txt`，环境文件只有版本记录，并不代表 Linux/Windows可直接互换。
- 原始CSV、processed、runs、reports默认被 `.gitignore` 排除；**git不是实验数据备份**。代码与配置用 Git，实验大文件另行归档。
- `clients.json`、验证/测试指标和预测属于研究日志，未做 DP 保护。不能把整个 runs 文件夹当成经过 DP 的公开发布物；正式发布需单独挑选并审计模型与统计。


## 归档校验命令

```powershell
python -m energyfl seal --run runs/coverage_s7
# 将整个目录复制到持久存储后，在目标位置校验：
python -m energyfl verify --run runs/coverage_s7
```

`integrity.json` 记录完成实验的每个文件 SHA256；修改或丢失文件会让校验失败。seal 只允许已结束且未持锁的实验，建议在测试完成后执行；后续新增输出后重新 seal。它检查数据完整性，不是针对恶意篡改的签名。

复制 processed 数据到另一机器后，可用 `test --data 新路径` 覆盖原节点路径；manifest 指纹仍必须一致。
