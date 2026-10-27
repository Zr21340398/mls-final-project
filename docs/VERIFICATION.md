# 当前版本验证记录

本次在 Windows / Python 3.12 / PyTorch 2.8.0 CPU 环境完成验证。

- 9 项自动测试通过，详细输出见 `reports/verification/unittest.log`。
- 测试覆盖时间分区/未来数据独立性、缺失窗口、模型参数内存独立性、掩码抵消与会计、独立调度trace、掉线不更新、所有基础方法、DP预算停止、CPU量化、断点恢复和归档完整性。
- `scripts/smoke.ps1` 已执行成功：8 个合成家庭、3 轮训练、独立测试、HTML/CSV报告。示例已保留在 `runs/smoke`，不要直接重复创建同名run。
- CPU profiling、FP32/INT8导出与100次推理测量已完成。此次INT8文件更小，但推理更慢；结果见 `reports/compression_smoke/compression.json`。
- `seal` 和 `verify` 检查18个实验文件通过；监控端口的HTTP响应与SVG收敛曲线已验证，验证后已关闭临时服务。
- 隐私实验矩阵只做配置生成验证，没有提交正式GPU实验。

**尚未验证：**真实LCL文件的下载与数据质量、L20/CUDA运行、超算队列、真实Flower SecAgg+协议。合成数据的精度与此CPU的耗时不能当成正式实验结论。

完整机器可读记录：`reports/verification/verification.json`。后续更换源代码后应重新执行相应测试，并创建新的实验目录。


## 自动下载与校准功能验证

新增功能后共有15项自动测试通过，记录在 `reports/verification/automation-tests.log`。测试覆盖HTTP续传、服务端忽略Range时完整重下、缓存/CSV损坏检测、路径越界拒绝、空间不足、校准配置与证据一致性、pilot流水线重复执行，以及原有训练测试。

已对官方ZIP地址进行真实小范围请求，返回HTTP206并验证ZIP文件头；没有在这次验证中下载完整795,722,689字节归档。完整bootstrap使用小型本地HTTP/CSV fixture测试，不冒充真实LCL结果。

已实际运行自动校准CLI，生成 `configs/smoke.calibrated.json` 和 `reports/calibration_smoke/profile.json`，再次运行成功复用并校验缓存。机器可读验证记录见 `reports/verification/automation-verification.json`。
