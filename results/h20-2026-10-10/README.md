# H20 MNIST CUDA 推理复现 — 2026-10-10

重启开发环境并重新上传原 A100 项目后，在 H20 上重新编译相同 CUDA 源码，四份公开权重各执行 10 次，共 40 次全部通过。每次正确数均与 A100 记录一致，最高准确率仍为 99.52%。本次没有重新训练或修改推理实现。

## 环境与核验

- 源码版本：`f5eb36a053f84ea48d1f193e53e4b9870fc31226`，运行前 Git 工作区干净。
- 正式测量：北京时间 2026-10-10 10:26:31–10:27:38；元数据保留 UTC 时间。
- 上传包的源码、Makefile、四份权重与 GitHub 版本一致；测试数据哈希与 10 月 8 日测量一致。
- GPU 由 PyTorch CUDA runtime 检测为 NVIDIA H20，compute capability 9.0，78 个 SM；平台报告可用显存 9,185,201,428 字节，约 8.55 GiB。
- 本次 nvidia-smi 返回空 GPU 表格及 Driver/CUDA Version N/A，不能据此确定驱动版本或同机负载；原始输出已保留。CUDA 推理实际成功。
- CUDA Toolkit 12.6（V12.6.85），PyTorch 2.8.0+cu126 仅用于设备信息探测。
- Batch size 256；每轮交错执行四份权重，各 10 次；每次独立进程内部预热一个 batch。
- 编译参数与 Makefile 一致：`-O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90`。
- 按权重独立复算 CSV 的中位数、均值、范围和总体标准差，均与 40 条原始记录一致；不剔除样本。

## 实测结果

时间单位为 ms。kernel 包含归一化与网络 GPU 算子；端到端还含逐批数据传输与主机端正确数统计，不含模型读取、显存分配或 Python 进程启动。

| 权重 | 准确率 | kernel 中位数 | kernel 均值 ± 总体标准差 | kernel 范围 | 端到端中位数 | 端到端范围 |
|---|---:|---:|---:|---:|---:|---:|
| `lenet_96.90.bin` | 96.90% | 78.728 | 80.318 ± 3.245 | 78.552–86.812 | 80.888 | 80.553–88.779 |
| `lenet_97.07.bin` | 97.07% | 78.612 | 78.603 ± 0.055 | 78.475–78.678 | 80.647 | 80.519–80.915 |
| `deep_97.45.bin` | 97.45% | 118.054 | 121.526 ± 12.954 | 107.790–141.532 | 120.159 | 109.871–143.588 |
| `deep_99.52.bin` | 99.52% | 117.191 | 122.545 ± 11.093 | 107.709–141.380 | 119.275 | 109.799–143.464 |

三层 CNN 的 kernel 耗时约为 108–142 ms，仍有波动。最高精度模型第 10 轮 kernel 117.880 ms、端到端 142.948 ms，该差异保留在原始记录中，当前资料不足以确定原因。

## A100 与此前 H20 对照

A100 是 README 中保留的共享设备单次记录，本次未重跑 A100。两次 H20 均为各权重 10 次测量的中位数。

| 权重 | A100 kernel 单次 | H20 10/8 kernel 中位数 | H20 10/10 kernel 中位数 | A100 端到端单次 | H20 10/10 端到端中位数 |
|---|---:|---:|---:|---:|---:|
| `lenet_96.90.bin` | 101.020 | 78.767 | 78.728 | 106.066 | 80.888 |
| `lenet_97.07.bin` | 101.372 | 78.762 | 78.612 | 106.479 | 80.647 |
| `deep_97.45.bin` | 106.158 | 124.060 | 118.054 | 111.278 | 120.159 |
| `deep_99.52.bin` | 106.393 | 128.435 | 117.191 | 111.354 | 119.275 |

两次 H20 的准确率均与 A100 一致。本次最高精度模型的 kernel 中位数由 128.435 ms 变为 117.191 ms；平台资源分配、共享负载、时钟和功率未受控，不能将两次变化解释为代码优化，也不能以历史 A100 单次和 H20 中位数给出公平的整卡性能排名。

## 持久化位置与复现

项目保存在 `/gemini/code/ucas-gpu-mnist-cuda-inference`；上传包及 MNIST IDX 数据保留于 `/gemini/code/gpu_`。本次结果位于项目的 `results/h20-2026-10-10`。不再依赖旧 `/root/gpu_mnist` 临时目录。

```bash
cd /gemini/code/ucas-gpu-mnist-cuda-inference
export PATH=/usr/local/cuda/bin:/root/miniconda3/bin:$PATH
make -B
python benchmarks/run_benchmark.py \
  --executable ./mnist_infer \
  --data-dir /gemini/code/gpu_/data \
  --batch 256 --repeats 10 \
  --build-command 'nvcc -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -o mnist_infer src/mnist_infer.cu' \
  --output results/h20-new-run
```

输出目录必须为新目录，避免覆盖已有记录。正式测量前的一次最高精度连通性测试（kernel 198.333 ms、端到端 206.284 ms、准确率 99.52%）不纳入预定的 40 次统计。

## 原始资料

- [runs.json](runs.json)：全部 40 次推理的指标、时间戳、退出码和标准输出/错误。
- [summary.csv](summary.csv)：逐权重统计。
- [metadata.json](metadata.json)：版本、文件 SHA-256、设备信息、编译命令及起止时间。
- [build.log](build.log)：实际重新编译的输出。
- [测量脚本](../../benchmarks/run_benchmark.py)：复算和复现入口。
