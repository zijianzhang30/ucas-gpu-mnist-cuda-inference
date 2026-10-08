#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) \
  throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e)); } while (0)

// All activations are NHWC. Convolution weights are [KH, KW, Cin, Cout].
__global__ void pixels_to_float(const unsigned char* src, float* dst, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = float(src[i]) * (1.0f / 255.0f);
}

template<int H, int W, int CI, int CO, int K>
__global__ void conv_relu(const float* x, const float* w, const float* bias,
                          float* y, int batch) {
  constexpr int OH = H - K + 1, OW = W - K + 1;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch * OH * OW * CO) return;
  int oc = t % CO, col = (t / CO) % OW, row = (t / (CO * OW)) % OH;
  int image = t / (CO * OW * OH);
  float sum = bias[oc];
  for (int kh = 0; kh < K; ++kh)
    for (int kw = 0; kw < K; ++kw)
      for (int ci = 0; ci < CI; ++ci)
        sum += x[((image * H + row + kh) * W + col + kw) * CI + ci] *
               w[((kh * K + kw) * CI + ci) * CO + oc];
  y[t] = sum > 0.0f ? sum : 0.0f;
}

template<int H, int W, int C>
__global__ void max_pool_2x2(const float* x, float* y, int batch) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch * (H / 2) * (W / 2) * C) return;
  int c = t % C, col = (t / C) % (W / 2), row = (t / (C * (W / 2))) % (H / 2);
  int image = t / (C * (W / 2) * (H / 2));
  int p = ((image * H + row * 2) * W + col * 2) * C + c;
  float top = x[p] > x[p + C] ? x[p] : x[p + C];
  float bottom = x[p + W * C] > x[p + (W + 1) * C] ? x[p + W * C] : x[p + (W + 1) * C];
  y[t] = top > bottom ? top : bottom;
}

template<int IN, int OUT, bool RELU>
__global__ void dense(const float* x, const float* w, const float* bias,
                      float* y, int batch) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= batch * OUT) return;
  int image = t / OUT, o = t % OUT;
  float sum = bias[o];
  for (int i = 0; i < IN; ++i) sum += x[image * IN + i] * w[i * OUT + o];
  y[t] = RELU && sum < 0.0f ? 0.0f : sum;
}

struct DeviceBuffer {
  void* p = nullptr;
  explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
  ~DeviceBuffer() { if (p) cudaFree(p); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  template<typename T = float> T* as() const { return static_cast<T*>(p); }
};

static uint32_t read_be32(std::ifstream& f) {
  unsigned char b[4];
  if (!f.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("Truncated IDX header");
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) | (uint32_t(b[2]) << 8) | b[3];
}

static std::vector<unsigned char> read_images(const std::string& path, uint32_t& count) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open images: " + path);
  if (read_be32(f) != 2051) throw std::runtime_error("Invalid IDX image magic");
  count = read_be32(f);
  if (read_be32(f) != 28 || read_be32(f) != 28) throw std::runtime_error("Images must be 28x28");
  std::vector<unsigned char> data(size_t(count) * 784);
  if (!f.read(reinterpret_cast<char*>(data.data()), data.size())) throw std::runtime_error("Truncated image file");
  return data;
}

static std::vector<unsigned char> read_labels(const std::string& path, uint32_t expected) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open labels: " + path);
  if (read_be32(f) != 2049 || read_be32(f) != expected) throw std::runtime_error("Invalid IDX labels");
  std::vector<unsigned char> data(expected);
  if (!f.read(reinterpret_cast<char*>(data.data()), data.size())) throw std::runtime_error("Truncated label file");
  return data;
}

struct WeightFile {
  bool deep;
  std::vector<float> values;
};

static WeightFile read_weights(const std::string& path) {
  // Eight-byte magic distinguishes the LeNet and three-convolution networks.
  constexpr size_t lenet_count = 44426;
  constexpr size_t deep_count = 82446;
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot open weights: " + path);
  auto bytes = f.tellg();
  f.seekg(0);
  char magic[8];
  if (!f.read(magic, 8)) throw std::runtime_error("Truncated weight file");
  const std::string tag(magic, 8);
  if (tag != "MNISTCNN" && tag != "MNISTD3!") throw std::runtime_error("Wrong weight file magic");
  bool deep = tag == "MNISTD3!";
  size_t count = deep ? deep_count : lenet_count;
  if (bytes != std::streampos(8 + count * sizeof(float))) throw std::runtime_error("Wrong weight file size");
  std::vector<float> weights(count);
  if (!f.read(reinterpret_cast<char*>(weights.data()), count * sizeof(float))) throw std::runtime_error("Truncated weights");
  return {deep, std::move(weights)};
}

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 5) {
      std::fprintf(stderr, "Usage: %s weights.bin t10k-images-idx3-ubyte t10k-labels-idx1-ubyte [batch=256]\n", argv[0]);
      return 2;
    }
    int batch_size = argc == 5 ? std::stoi(argv[4]) : 256;
    if (batch_size < 1 || batch_size > 4096) throw std::runtime_error("Batch must be in [1,4096]");
    uint32_t count;
    auto images = read_images(argv[2], count);
    auto labels = read_labels(argv[3], count);
    if (count == 0) throw std::runtime_error("Image file is empty");
    auto weight_file = read_weights(argv[1]);
    bool deep = weight_file.deep;
    const auto& weights = weight_file.values;
    DeviceBuffer d_weights(weights.size() * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_weights.p, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    const float* w[12]; size_t off = 0;
    const int lenet_sizes[10] = {150, 6, 2400, 16, 30720, 120, 10080, 84, 840, 10};
    const int deep_sizes[12] = {300, 12, 2592, 24, 6912, 32, 61440, 120, 10080, 84, 840, 10};
    const int* sizes = deep ? deep_sizes : lenet_sizes;
    for (int i = 0; i < (deep ? 12 : 10); ++i) { w[i] = d_weights.as<float>() + off; off += sizes[i]; }
    DeviceBuffer input(size_t(batch_size) * 784);
    DeviceBuffer a(size_t(batch_size) * 24 * 24 * 12 * sizeof(float));
    DeviceBuffer b(size_t(batch_size) * 12 * 12 * 12 * sizeof(float));
    DeviceBuffer c(size_t(batch_size) * 10 * 10 * 24 * sizeof(float));
    DeviceBuffer d(size_t(batch_size) * 8 * 8 * 32 * sizeof(float));
    DeviceBuffer g(size_t(batch_size) * 4 * 4 * 32 * sizeof(float));
    DeviceBuffer e(size_t(batch_size) * 120 * sizeof(float));
    DeviceBuffer f(size_t(batch_size) * 84 * sizeof(float));
    DeviceBuffer out(size_t(batch_size) * 10 * sizeof(float));
    DeviceBuffer x(size_t(batch_size) * 784 * sizeof(float));
    std::vector<float> logits(size_t(batch_size) * 10);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
    double kernel_ms = 0;
    uint32_t correct = 0;
    constexpr int threads = 256;
    auto grid = [](int n) { return (n + threads - 1) / threads; };
    auto infer = [&](int n) {
      pixels_to_float<<<grid(n * 784), threads>>>(input.as<unsigned char>(), x.as<float>(), n * 784);
      if (deep) {
        conv_relu<28,28,1,12,5><<<grid(n*24*24*12),threads>>>(x.as<float>(),w[0],w[1],a.as<float>(),n);
        max_pool_2x2<24,24,12><<<grid(n*12*12*12),threads>>>(a.as<float>(),b.as<float>(),n);
        conv_relu<12,12,12,24,3><<<grid(n*10*10*24),threads>>>(b.as<float>(),w[2],w[3],c.as<float>(),n);
        conv_relu<10,10,24,32,3><<<grid(n*8*8*32),threads>>>(c.as<float>(),w[4],w[5],d.as<float>(),n);
        max_pool_2x2<8,8,32><<<grid(n*4*4*32),threads>>>(d.as<float>(),g.as<float>(),n);
        dense<512,120,true><<<grid(n*120),threads>>>(g.as<float>(),w[6],w[7],e.as<float>(),n);
        dense<120,84,true><<<grid(n*84),threads>>>(e.as<float>(),w[8],w[9],f.as<float>(),n);
        dense<84,10,false><<<grid(n*10),threads>>>(f.as<float>(),w[10],w[11],out.as<float>(),n);
      } else {
        conv_relu<28,28,1,6,5><<<grid(n*24*24*6),threads>>>(x.as<float>(),w[0],w[1],a.as<float>(),n);
        max_pool_2x2<24,24,6><<<grid(n*12*12*6),threads>>>(a.as<float>(),b.as<float>(),n);
        conv_relu<12,12,6,16,5><<<grid(n*8*8*16),threads>>>(b.as<float>(),w[2],w[3],c.as<float>(),n);
        max_pool_2x2<8,8,16><<<grid(n*4*4*16),threads>>>(c.as<float>(),d.as<float>(),n);
        dense<256,120,true><<<grid(n*120),threads>>>(d.as<float>(),w[4],w[5],e.as<float>(),n);
        dense<120,84,true><<<grid(n*84),threads>>>(e.as<float>(),w[6],w[7],f.as<float>(),n);
        dense<84,10,false><<<grid(n*10),threads>>>(f.as<float>(),w[8],w[9],out.as<float>(),n);
      }
      CUDA_CHECK(cudaGetLastError());
    };
    int warmup_count = std::min<uint32_t>(batch_size, count);
    CUDA_CHECK(cudaMemcpy(input.p, images.data(), size_t(warmup_count) * 784, cudaMemcpyHostToDevice));
    infer(warmup_count);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto wall_start = std::chrono::steady_clock::now();
    for (uint32_t base = 0; base < count; base += batch_size) {
      int n = std::min<uint32_t>(batch_size, count - base);
      CUDA_CHECK(cudaMemcpy(input.p, images.data() + size_t(base) * 784, size_t(n) * 784, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaEventRecord(start));
      infer(n);
      CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed; CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop)); kernel_ms += elapsed;
      CUDA_CHECK(cudaMemcpy(logits.data(), out.p, size_t(n) * 10 * sizeof(float), cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i) {
        int pred = int(std::max_element(logits.begin() + i*10, logits.begin() + (i+1)*10) - (logits.begin() + i*10));
        correct += pred == labels[base+i];
      }
    }
    double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
    std::printf("model=%s images=%u correct=%u accuracy=%.4f%% kernel_ms=%.3f end_to_end_ms=%.3f kernel_us_per_image=%.3f\n",
                deep ? "deep" : "lenet", count, correct, 100.0*correct/count, kernel_ms, wall_ms, kernel_ms*1000/count);
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what()); return 1;
  }
}
