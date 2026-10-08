# CUDA CNN MNIST Inference

本项目源于中国科学院大学《GPU 架构和编程》课程大作业（一）的基础题，实现 MNIST 手写数字分类的 CUDA CNN 推理。

卷积、最大池化、ReLU 和全连接均由项目中的 CUDA kernel 实现；不调用 cuDNN、cuBLAS、Thrust 等现成计算库。显存分配、数据传输、kernel 启动和计时使用 CUDA Runtime API。

## 网络与公开权重

| 权重文件 | 网络 | A100 正确数 | A100 准确率 | kernel 耗时 | 端到端耗时 |
| --- | --- | ---: | ---: | ---: | ---: |
| `checkpoints/lenet_96.90.bin` | 两层卷积 LeNet | 9,690 / 10,000 | 96.90% | 101.020 ms | 106.066 ms |
| `checkpoints/lenet_97.07.bin` | 两层卷积 LeNet | 9,707 / 10,000 | 97.07% | 101.372 ms | 106.479 ms |
| `checkpoints/deep_97.45.bin` | 三层卷积 CNN | 9,745 / 10,000 | 97.45% | 106.158 ms | 111.278 ms |

以上数值为 NVIDIA A100-SXM4-80GB 上、batch size 256 的单次 CUDA 推理结果；三个正确数均与 CPU 参考前向一致。耗时会受同机其他任务影响。不同权重使用独立的 8 字节格式标识，程序会自动选择对应网络。

两层卷积网络：`Conv5×5(6) → MaxPool2×2 → Conv5×5(16) → MaxPool2×2 → FC120 → FC84 → FC10`。

三层卷积网络：`Conv5×5(12) → MaxPool2×2 → Conv3×3(24) → Conv3×3(32) → MaxPool2×2 → FC120 → FC84 → FC10`。

卷积和前两层全连接之后使用 ReLU。输入为 28×28 灰度图，像素值除以 255。

## 构建与运行

需要 CUDA Toolkit（`nvcc`，建议 12.x）和 Python 3。下载标准 MNIST IDX 文件并编译：

```bash
python scripts/download_mnist.py
make
```

在 A100 上运行三个公开版本：

```bash
./mnist_infer checkpoints/lenet_96.90.bin data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte 256
./mnist_infer checkpoints/lenet_97.07.bin data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte 256
./mnist_infer checkpoints/deep_97.45.bin data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte 256
```

下载脚本使用标准 MNIST IDX 镜像，其数据与课程给出的 [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist) 相同。若已下载数据，可把四个未压缩的 IDX 文件放入 `data/`，无需重复下载。

输出中的 `accuracy` 是测试集正确率；`kernel_ms` 是预热后 GPU 上的归一化和网络算子耗时，不含主机与设备之间的数据传输；`end_to_end_ms` 包含每批数据传输、GPU 推理和主机端准确率统计，不含模型文件读取与显存分配。

## 文件

- `src/mnist_infer.cu`：两个网络的 CUDA 推理、IDX/权重读取和指标统计。
- `scripts/download_mnist.py`：下载并解压 MNIST。
- `checkpoints/`：三个公开权重。训练所需代码和其他实验权重不包含在此发布目录中。

代码按 [MIT License](LICENSE) 发布。
