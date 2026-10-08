NVCC ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90

.PHONY: all clean
all: mnist_infer

mnist_infer: src/mnist_infer.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f mnist_infer
