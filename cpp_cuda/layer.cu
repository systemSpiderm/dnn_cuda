#include "layer.h"
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include "utils.h"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

// -----------------------------------------------------------------------------
//                                Conv2d 部分
// -----------------------------------------------------------------------------
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,   // [batch_size, in_c, H, W]
    const float* __restrict__ weights, // [out_c, in_c, k, k]
    const float* __restrict__ biases,  // [out_c]
    float* __restrict__ output,        // [batch_size, out_c, OH, OW]
    int batch_size, 
    int in_c, int H, int W,
    int out_c, int KH, int KW,
    int stride, int padding,
    int OH, int OW)
{
    int oc = blockIdx.x; // output channel
    int oh = blockIdx.y * blockDim.y + threadIdx.y; // output height
    int ow = threadIdx.x + (blockIdx.z % ((OW + blockDim.x - 1) / blockDim.x)) * blockDim.x;
    int b  = blockIdx.z / ((OW + blockDim.x - 1) / blockDim.x); // batch index

    if (b < batch_size && oc < out_c && oh < OH && ow < OW) {
        float sum = biases[oc];
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        int in_idx = ((b * in_c + ic) * H + ih) * W + iw;
                        int w_idx  = ((oc * in_c + ic) * KH + kh) * KW + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
        int out_idx = ((b * out_c + oc) * OH + oh) * OW + ow;
        output[out_idx] = sum;
    }
}

__global__ void conv2d_backward_kernel(
    const float* __restrict__ input,    // [B, in_c, H, W]
    const float* __restrict__ weights,  // [out_c, in_c, KH, KW]
    const float* __restrict__ doutput,  // [B, out_c, OH, OW]
    float* __restrict__ dinput,         // [B, in_c, H, W]
    float* __restrict__ dweights,       // [out_c, in_c, KH, KW]
    float* __restrict__ dbiases,        // [out_c]
    int batch_size,
    int in_c, int H, int W,
    int out_c, int KH, int KW,
    int stride, int padding,
    int OH, int OW)
{
    int oc = blockIdx.x; // output channel
    int oh = blockIdx.y * blockDim.y + threadIdx.y; // output height
    int ow = threadIdx.x + (blockIdx.z % ((OW + blockDim.x - 1) / blockDim.x)) * blockDim.x;
    int b  = blockIdx.z / ((OW + blockDim.x - 1) / blockDim.x); // batch index

    if (b < batch_size && oc < out_c && oh < OH && ow < OW) {
        int out_idx = ((b * out_c + oc) * OH + oh) * OW + ow;
        float grad_out = doutput[out_idx];

        // 累加偏置
        atomicAdd(&dbiases[oc], grad_out);

        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;

                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        int in_idx = ((b * in_c + ic) * H + ih) * W + iw;
                        int w_idx  = ((oc * in_c + ic) * KH + kh) * KW + kw;

                        // 累加权重梯度（所有 batch 累加）
                        atomicAdd(&dweights[w_idx], input[in_idx] * grad_out);

                        // 累加输入梯度（当前 batch）
                        atomicAdd(&dinput[in_idx], weights[w_idx] * grad_out);
                    }
                }
            }
        }
    }
}


Conv2d::Conv2d(int batch_size, int in_c, int out_c, int k, int s, int p)
    : batch_size(batch_size), in_channels(in_c), out_channels(out_c),
      kernel_size(k), stride(s), padding(p),
      input_cache(nullptr),
      H_cache(0), W_cache(0), OH_cache(0), OW_cache(0),
      weights(nullptr), dweights(nullptr),
      biases(nullptr), dbiases(nullptr),
      weights_d(nullptr), biases_d(nullptr),
      dweights_d(nullptr), dbiases_d(nullptr),
      input_cache_d(nullptr),
      is_gpu(false)
{
    int wsize = out_c * in_c * k * k;
    weights  = new float[wsize];
    dweights = new float[wsize];
    biases   = new float[out_c];
    dbiases  = new float[out_c];
    H_cache = 0;
    W_cache = 0;
    C_cache = in_c;
    OC_cache = out_c;
    // 初始化
    for (int i = 0; i < wsize; ++i) {
        weights[i]  = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        dweights[i] = 0.0f;
    }
    for (int i = 0; i < out_c; ++i) {
        biases[i]  = 0.0f;
        dbiases[i] = 0.0f;
    }
}

Conv2d::~Conv2d() {
    delete[] weights;
    delete[] dweights;
    delete[] biases;
    delete[] dbiases;
    if (input_cache) delete[] input_cache;

    // 释放 GPU 端
    if (weights_d)      cudaFree(weights_d);
    if (dweights_d)     cudaFree(dweights_d);
    if (biases_d)       cudaFree(biases_d);
    if (dbiases_d)      cudaFree(dbiases_d);
    if (input_cache_d)  cudaFree(input_cache_d);
}

// CPU 版 forward
float* Conv2d::forward_cpu(float* input, int H, int W) { // 输入是 [batch_size, in_c, H, W]
    int input_size = in_channels * H * W;
    H_cache = H;
    W_cache = W;
    if (input_cache != nullptr) delete[] input_cache;
    
    input_cache = new float[batch_size * input_size]();
    memcpy(input_cache, input, sizeof(float) * batch_size * input_size);
    int OH = (H + 2 * padding - kernel_size) / stride + 1;
    int OW = (W + 2 * padding - kernel_size) / stride + 1;
    OH_cache = OH;
    OW_cache = OW;
    return conv2d_multi(
        input_cache, weights, biases, batch_size, 
        in_channels, H_cache, W_cache, out_channels, 
        kernel_size, kernel_size, stride, padding);
}

// CPU 版 backward
float* Conv2d::backward_cpu(float* doutput) {
    int OH = OH_cache;
    int OW = OW_cache;
    int H = H_cache;
    int W = W_cache;
    int N = batch_size;  // 新增：保存批量大小

    // 初始化 dinput，形状: [N, in_channels, H, W]
    float* dinput = new float[N * in_channels * H * W]();
    
    // 遍历每一个样本
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    int out_idx = ((n * out_channels + oc) * OH + oh) * OW + ow;

                    // 更新偏置
                    dbiases[oc] += doutput[out_idx];

                    for (int ic = 0; ic < in_channels; ++ic) {
                        for (int kh = 0; kh < kernel_size; ++kh) {
                            for (int kw = 0; kw < kernel_size; ++kw) {
                                int ih = oh * stride + kh - padding;
                                int iw = ow * stride + kw - padding;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int input_idx = ((n * in_channels + ic) * H + ih) * W + iw;
                                    int weight_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;

                                    dweights[weight_idx] += input_cache[input_idx] * doutput[out_idx];
                                    dinput[input_idx] += weights[weight_idx] * doutput[out_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }



    

    return dinput;  // [N, in_channels, H, W]
}


// GPU 版 forward
float* Conv2d::forward_gpu(float* input_d, int H, int W) {
    H_cache = H;
    W_cache = W;
    int OH = (H + 2 * padding - kernel_size) / stride + 1;
    int OW = (W + 2 * padding - kernel_size) / stride + 1;
    OH_cache = OH; OW_cache = OW;

    // 先把 input_d 拷贝到 input_cache_d
    if (input_cache_d) {cudaFree(input_cache_d);input_cache_d = nullptr;}
    int in_size = batch_size * in_channels * H * W;
    CUDA_CHECK(cudaMalloc(&input_cache_d, sizeof(float) * in_size));
    CUDA_CHECK(cudaMemcpy(input_cache_d, input_d, sizeof(float) * in_size, cudaMemcpyDeviceToDevice));
    // 分配输出
    float* output_d;
    int out_size = batch_size * out_channels * OH * OW;
    CUDA_CHECK(cudaMalloc(&output_d, sizeof(float) * out_size));
    // 配置 kernel launch
    dim3 blockDim(16, 16);
    dim3 gridDim(out_channels,
                 (OH + blockDim.y - 1) / blockDim.y,
                 (OW + blockDim.x - 1) / blockDim.x * batch_size);
    conv2d_forward_kernel<<<gridDim, blockDim>>>(
        input_cache_d, weights_d, biases_d, output_d,
        batch_size, 
        in_channels, H, W,
        out_channels, kernel_size, kernel_size,
        stride, padding, OH, OW
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    return output_d;
}

float* Conv2d::backward_gpu(float* doutput_d) {
    int H = H_cache, W = W_cache;
    int OH = OH_cache, OW = OW_cache;
    int in_c = in_channels, out_c = out_channels, k = kernel_size;

    // 分配 dinput [batch_size, in_c, H, W]
    float* dinput_d;
    int input_size = batch_size * in_c * H * W;
    CUDA_CHECK(cudaMalloc(&dinput_d, sizeof(float) * input_size));
    CUDA_CHECK(cudaMemset(dinput_d, 0, sizeof(float) * input_size));

    // 清零 dweights 和 dbiases（所有 batch 共享累加）
    int wsize = out_c * in_c * k * k;
    CUDA_CHECK(cudaMemset(dweights_d, 0, sizeof(float) * wsize));
    CUDA_CHECK(cudaMemset(dbiases_d, 0, sizeof(float) * out_c));

    // 配置 kernel 启动参数
    dim3 blockDim(16, 16);
    dim3 gridDim(
        out_c,
        (OH + blockDim.y - 1) / blockDim.y,
        batch_size * ((OW + blockDim.x - 1) / blockDim.x)
    );

    // 启动 backward kernel
    conv2d_backward_kernel<<<gridDim, blockDim>>>(
        input_cache_d, weights_d, doutput_d,
        dinput_d, dweights_d, dbiases_d,
        batch_size,
        in_c, H, W,
        out_c, k, k,
        stride, padding,
        OH, OW
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // 返回 dinput（调用者负责 cudaFree）
    return dinput_d;
}


// 通用接口
float* Conv2d::forward(float* input, int H, int W) {
    if (is_gpu) {
        // 假设 input 已经是 device pointer
        return forward_gpu(input, H, W);
    } else {
        return forward_cpu(input, H, W);
    }
}

float* Conv2d::backward(float* doutput) {
    if (is_gpu) {
        // 假设 doutput 已经在 device 上
        return backward_gpu(doutput);
    } else {
        return backward_cpu(doutput);
    }
}

__global__ void conv2d_update_params_kernel(
    float* weights, const float* dweights,
    float* biases,  const float* dbiases,
    float lr, int batch_size,
    int wsize, int out_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < wsize) {
        weights[idx] -= lr * dweights[idx] / batch_size;
        // 清零梯度
        ((float*)dweights)[idx] = 0.0f;
    }

    if (idx < out_channels) {
        biases[idx] -= lr * dbiases[idx] / batch_size;
        ((float*)dbiases)[idx] = 0.0f;
    }
}


// step：CPU/GPU 都要更新参数
void Conv2d::step(float lr) {
    int wsize = out_channels * in_channels * kernel_size * kernel_size;
    if (!is_gpu) {
        // CPU 更新
        for (int i = 0; i < wsize; ++i) {
            weights[i]  -= lr * dweights[i] / batch_size;
            dweights[i] = 0;
        }
        for (int i = 0; i < out_channels; ++i) {
            biases[i]  -= lr * dbiases[i] / batch_size;
            dbiases[i]  = 0;
        }




    } else {

        int threads = 256;
        int blocks = (max(wsize, out_channels) + threads - 1) / threads;

        conv2d_update_params_kernel<<<blocks, threads>>>(
            weights_d, dweights_d,
            biases_d, dbiases_d,
            lr, batch_size,
            wsize, out_channels
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// 将参数和梯度数组都搬到 GPU
void Conv2d::cuda() {
    is_gpu = true;
    int wsize = out_channels * in_channels * kernel_size * kernel_size;
    // 分配并拷贝 weights, dweights, biases, dbiases
    CUDA_CHECK(cudaMalloc(&weights_d,  sizeof(float) * wsize));
    CUDA_CHECK(cudaMemcpy(weights_d, weights, sizeof(float) * wsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dweights_d, sizeof(float) * wsize));
    CUDA_CHECK(cudaMemset(dweights_d, 0, sizeof(float) * wsize));

    CUDA_CHECK(cudaMalloc(&biases_d,  sizeof(float) * out_channels));
    CUDA_CHECK(cudaMemcpy(biases_d, biases, sizeof(float) * out_channels, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dbiases_d, sizeof(float) * out_channels));
    CUDA_CHECK(cudaMemset(dbiases_d, 0, sizeof(float) * out_channels));
    // input_cache_d 延迟到 forward 时分配
}

// -----------------------------------------------------------------------------
//                            AvgPool2d 部分
// -----------------------------------------------------------------------------
__global__ void avgpool_forward_kernel(
    const float* __restrict__ input,  // [B, C, H, W]
    float* __restrict__ output,       // [B, C, OH, OW]
    int batch_size,
    int C, int H, int W,
    int k, int s,
    int OH, int OW)
{
    int c = blockIdx.x;  // channel
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = threadIdx.x + (blockIdx.z % ((OW + blockDim.x - 1) / blockDim.x)) * blockDim.x;
    int b  = blockIdx.z / ((OW + blockDim.x - 1) / blockDim.x);  // batch index

    if (b >= batch_size || c >= C || oh >= OH || ow >= OW) return;

    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int ih = oh * s + i;
            int iw = ow * s + j;
            if (ih < H && iw < W) {  // 加边界判断更安全
                int input_idx = ((b * C + c) * H + ih) * W + iw;
                sum += input[input_idx];
            }
        }
    }

    int output_idx = ((b * C + c) * OH + oh) * OW + ow;
    output[output_idx] = sum / (k * k);
}


__global__ void avgpool_backward_kernel(
    const float* __restrict__ doutput, // [B, C, OH, OW]
    float* __restrict__ dinput,        // [B, C, H, W]
    int batch_size,
    int C, int H, int W,
    int k, int s,
    int OH, int OW)
{
    int c = blockIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = threadIdx.x + (blockIdx.z % ((OW + blockDim.x - 1) / blockDim.x)) * blockDim.x;
    int b  = blockIdx.z / ((OW + blockDim.x - 1) / blockDim.x);  // batch index

    if (b >= batch_size || c >= C || oh >= OH || ow >= OW) return;

    // 平均池化每个像素的梯度
    float grad = doutput[((b * C + c) * OH + oh) * OW + ow] / (k * k);

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            int ih = oh * s + i;
            int iw = ow * s + j;
            if (ih < H && iw < W) {
                int idx = ((b * C + c) * H + ih) * W + iw;
                atomicAdd(&dinput[idx], grad);
            }
        }
    }
}


AvgPool2d::AvgPool2d(int batch_size, int k, int s)
    : batch_size(batch_size), kernel_size(k), stride(s),
      input_cache(nullptr), C_cache(0), H_cache(0), W_cache(0),
      OH_cache(0), OW_cache(0),
      input_cache_d(nullptr),
      is_gpu(false) {}

AvgPool2d::~AvgPool2d() {
    if (input_cache) delete[] input_cache;
    if (input_cache_d) cudaFree(input_cache_d);
}

float* AvgPool2d::forward_cpu(float* input, int C, int H, int W) {
    C_cache = C;
    H_cache = H;
    W_cache = W;
    
    if (input_cache != nullptr) delete[] input_cache;
    input_cache = new float[batch_size * C * H * W]();
    memcpy(input_cache, input, sizeof(float) * batch_size * C * H * W);
    
    int OH = (H - kernel_size) / stride + 1;
    int OW = (W - kernel_size) / stride + 1;
    OH_cache = OH;
    OW_cache = OW;
    
    float* output = new float[batch_size * C * OH * OW]();
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < OH; ++i) {
                for (int j = 0; j < OW; ++j) {
                    float sum = 0.0f;
                    for (int ki = 0; ki < kernel_size; ++ki) {
                        for (int kj = 0; kj < kernel_size; ++kj) {
                            int ih = i * stride + ki;
                            int iw = j * stride + kj;
                            sum += input[b * C * H * W + c * H * W + ih * W + iw];
                        }
                    }
                    output[b * C * OH * OW + c * OH * OW + i * OW + j] = sum / (kernel_size * kernel_size);
                }
            }
        }
    }

    return output;
}


float* AvgPool2d::backward_cpu(float* doutput) {
    int C = C_cache, H = H_cache, W = W_cache;
    int OH = OH_cache, OW = OW_cache;

    float* dinput = new float[batch_size * C * H * W]();

    for (int b = 0; b < batch_size; ++b)
    for (int c = 0; c < C; ++c)
    for (int oh = 0; oh < OH; ++oh)
    for (int ow = 0; ow < OW; ++ow) {
        int dout_idx = ((b * C + c) * OH + oh) * OW + ow;
        float grad = doutput[dout_idx] / (kernel_size * kernel_size);

        for (int kh = 0; kh < kernel_size; ++kh)
        for (int kw = 0; kw < kernel_size; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;

            if (ih < H && iw < W) {  // 无 padding，左上角肯定合法
                int din_idx = ((b * C + c) * H + ih) * W + iw;
                dinput[din_idx] += grad;
            }
        }
    }


    return dinput;
}



float* AvgPool2d::forward_gpu(float* input_d, int C, int H, int W) {
    C_cache = C; H_cache = H; W_cache = W;
    int OH = (H - kernel_size) / stride + 1;
    int OW = (W - kernel_size) / stride + 1;
    OH_cache = OH;
    OW_cache = OW;

    // 缓存输入
    int in_size = batch_size * C * H * W;
    if (input_cache_d) cudaFree(input_cache_d);
    CUDA_CHECK(cudaMalloc(&input_cache_d, sizeof(float) * in_size));
    CUDA_CHECK(cudaMemcpy(input_cache_d, input_d, sizeof(float) * in_size, cudaMemcpyDeviceToDevice));

    // 分配输出
    float* output_d;
    int out_size = batch_size * C * OH * OW;
    CUDA_CHECK(cudaMalloc(&output_d, sizeof(float) * out_size));

    // 配置 kernel 启动参数
    dim3 blockDim(16, 16);
    dim3 gridDim(
        C,
        (OH + blockDim.y - 1) / blockDim.y,
        batch_size * ((OW + blockDim.x - 1) / blockDim.x)
    );

    avgpool_forward_kernel<<<gridDim, blockDim>>>(
        input_cache_d, output_d,
        batch_size, C, H, W,
        kernel_size, stride,
        OH, OW
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    return output_d;
}


float* AvgPool2d::backward_gpu(float* doutput_d) {
    int C = C_cache, H = H_cache, W = W_cache;
    int OH = OH_cache, OW = OW_cache;

    // 分配并清空 dinput
    float* dinput_d;
    CUDA_CHECK(cudaMalloc(&dinput_d, sizeof(float) * batch_size * C * H * W));
    CUDA_CHECK(cudaMemset(dinput_d, 0, sizeof(float) * batch_size * C * H * W));

    dim3 blockDim(16, 16);
    dim3 gridDim(
        C,
        (OH + blockDim.y - 1) / blockDim.y,
        batch_size * ((OW + blockDim.x - 1) / blockDim.x)
    );

    avgpool_backward_kernel<<<gridDim, blockDim>>>(
        doutput_d, dinput_d,
        batch_size, C, H, W,
        kernel_size, stride, OH, OW
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    return dinput_d;
}


float* AvgPool2d::forward(float* input, int C, int H, int W) {
    if (is_gpu) return forward_gpu(input, C, H, W);
    else return forward_cpu(input, C, H, W);
}

float* AvgPool2d::backward(float* doutput) {
    if (is_gpu) return backward_gpu(doutput);
    else return backward_cpu(doutput);
}

void AvgPool2d::cuda() {
    is_gpu = true;
    // input_cache_d 在 forward 时分配即可
}

// -----------------------------------------------------------------------------
//                                  ReLU 部分
// -----------------------------------------------------------------------------
__global__ void relu_forward_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = input[idx];
        output[idx] = (v > 0.0f) ? v : 0.0f;
    }
}


__global__ void relu_backward_kernel(const float* __restrict__ input,
                                     const float* __restrict__ doutput,
                                     float* __restrict__ dinput,
                                     int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dinput[idx] = (input[idx] > 0.0f) ? doutput[idx] : 0.0f;
    }
}

ReLU::ReLU(int batch_size) : batch_size(batch_size), input_cache(nullptr), size_cache(0), input_cache_d(nullptr), is_gpu(false) {}

ReLU::~ReLU() {
    if (input_cache) delete[] input_cache;
    if (input_cache_d) cudaFree(input_cache_d);
}

float* ReLU::forward_cpu(float* input, int C, int H, int W) {   // 输入是 [batch_size, C, H, W]
    this -> C = C;
    this -> H = H;
    this -> W = W;
    size_cache = C * H * W;
    if (input_cache != nullptr) delete[] input_cache;
    input_cache = new float[batch_size * size_cache]();
    memcpy(input_cache, input, sizeof(float) * batch_size * size_cache);
    float* output = new float[batch_size * size_cache]();

    for (int i = 0; i < batch_size * size_cache; ++i)
        output[i] = std::max(0.0f, input[i]);
    return output;
}

float* ReLU::backward_cpu(float* doutput) {
    float* dinput = new float[batch_size * size_cache];
    for (int i = 0; i < batch_size * size_cache; ++i) {
        dinput[i] = (input_cache[i] > 0.0f) ? doutput[i] : 0.0f;
    }

 
    return dinput;
}

float* ReLU::forward_gpu(float* input_d, int C, int H, int W) {
    this->C = C;
    this->H = H;
    this->W = W;
    size_cache = batch_size * C * H * W;

    // 缓存输入
    if (input_cache_d) cudaFree(input_cache_d);
    CUDA_CHECK(cudaMalloc(&input_cache_d, sizeof(float) * size_cache));
    CUDA_CHECK(cudaMemcpy(input_cache_d, input_d, sizeof(float) * size_cache, cudaMemcpyDeviceToDevice));

    // 分配输出
    float* output_d;
    CUDA_CHECK(cudaMalloc(&output_d, sizeof(float) * size_cache));

    // 启动 Kernel
    int threads = 256;
    int blocks = (size_cache + threads - 1) / threads;
    relu_forward_kernel<<<blocks, threads>>>(input_cache_d, output_d, size_cache);
    CUDA_CHECK(cudaDeviceSynchronize());

    return output_d;
}



float* ReLU::backward_gpu(float* doutput_d) {
    // 注意：假设 size_cache 已在 forward 中保存过 (包含 batch)
    int total_size = batch_size * C * H * W;
    if (size_cache != total_size) {
        std::cerr << "ReLU backward_gpu: size_cache mismatch!" << std::endl;
        return nullptr;
    }

    float* dinput_d;
    CUDA_CHECK(cudaMalloc(&dinput_d, sizeof(float) * size_cache));

    int threads = 256;
    int blocks  = (size_cache + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(
        input_cache_d, doutput_d, dinput_d, size_cache
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    return dinput_d;
}


float* ReLU::forward(float* input, int C, int H, int W) {
    if (is_gpu) return forward_gpu(input, C, H, W);
    else return forward_cpu(input, C, H, W);
}

float* ReLU::backward(float* doutput) {
    if (is_gpu) return backward_gpu(doutput);
    else return backward_cpu(doutput);
}

void ReLU::cuda() {
    is_gpu = true;
}

// -----------------------------------------------------------------------------
//                               Linear（全连接）部分
// -----------------------------------------------------------------------------
Linear::Linear(int batch_size, int in_f, int out_f)
    : batch_size(batch_size), in_features(in_f), out_features(out_f),
      weights(nullptr), biases(nullptr),
      dweights(nullptr), dbiases(nullptr),
      input_cache(nullptr),
      weights_d(nullptr), biases_d(nullptr),
      dweights_d(nullptr), dbiases_d(nullptr),
      input_cache_d(nullptr),
      is_gpu(false)
{
    weights  = new float[out_f * in_f];
    biases   = new float[out_f];
    dweights = new float[out_f * in_f];
    dbiases  = new float[out_f];
    for (int i = 0; i < out_f * in_f; ++i) {
        weights[i]  = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        dweights[i] = 0.0f;
    }
    for (int i = 0; i < out_f; ++i) {
        biases[i]  = 0.0f;
        dbiases[i] = 0.0f;
    }
}

Linear::~Linear() {
    delete[] weights;
    delete[] biases;
    delete[] dweights;
    delete[] dbiases;
    if (input_cache) delete[] input_cache;
    if (weights_d)      cudaFree(weights_d);
    if (biases_d)       cudaFree(biases_d);
    if (dweights_d)     cudaFree(dweights_d);
    if (dbiases_d)      cudaFree(dbiases_d);
    if (input_cache_d)  cudaFree(input_cache_d);
}

float* Linear::forward_cpu(float* input) {  // 输入是 [batch_size, in_features]
    if (input_cache != nullptr) delete[] input_cache;
    input_cache = new float[batch_size * in_features]();
    memcpy(input_cache, input, sizeof(float) * batch_size * in_features);
    float* output = new float[batch_size * out_features]();

    for (int b = 0; b < batch_size; b++) {
        float* input_b = input + b * in_features;
        float* output_b = output + b * out_features;
        //memcpy(output_b, biases, sizeof(float) * out_features);
        matmul(weights, input_b, output_b, out_features, in_features, 1);
        for (int i = 0; i < out_features; i++) {
            output_b[i] += biases[i];
    }
    }
    return output;
}


float* Linear::backward_cpu(float* doutput) {
    float* dinput = new float[batch_size * in_features]();
    
    for (int b = 0; b < batch_size; ++b) {
        float* doutput_b = doutput + b * out_features;
        float* input_b = input_cache + b * in_features;
        float* dinput_b = dinput + b * in_features;

        for (int i = 0; i < out_features; ++i) {
            dbiases[i] += doutput_b[i];  // 累加偏置梯度
            for (int j = 0; j < in_features; ++j) {
                dweights[i * in_features + j] += doutput_b[i] * input_b[j];   // 权重梯度累加
                dinput_b[j] += doutput_b[i] * weights[i * in_features + j];   // 当前样本 dinput
            }
        }
    }

   

    return dinput;  // [batch_size, in_features]
}

__global__ void linear_forward_kernel(
    const float* __restrict__ weights, // [out_features, in_features]
    const float* __restrict__ biases,  // [out_features]
    const float* __restrict__ input,   // [batch_size, in_features]
    float* __restrict__ output,        // [batch_size, out_features]
    int batch_size,
    int in_f, int out_f)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x; // out_feature index
    int batch_idx = blockIdx.y;                          // batch index

    if (out_idx >= out_f || batch_idx >= batch_size) return;

    float sum = biases[out_idx];
    const float* input_row = input + batch_idx * in_f;
    for (int j = 0; j < in_f; ++j) {
        sum += weights[out_idx * in_f + j] * input_row[j];
    }
    output[batch_idx * out_f + out_idx] = sum;
}


__global__ void linear_backward_kernel(
    const float* __restrict__ weights,   // [out_f, in_f]
    const float* __restrict__ doutput,   // [B, out_f]
    float* __restrict__ dinput,          // [B, in_f]
    float* __restrict__ dweights,        // [out_f, in_f]
    float* __restrict__ dbiases,         // [out_f]
    const float* __restrict__ input,     // [B, in_f]
    int batch_size,
    int in_f, int out_f)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 输出特征维度
    int b = blockIdx.y;  // batch index

    if (out_idx >= out_f || b >= batch_size) return;

    float grad_out = doutput[b * out_f + out_idx];

    // dbias
    atomicAdd(&dbiases[out_idx], grad_out);

    for (int j = 0; j < in_f; ++j) {
        float x = input[b * in_f + j];
        float w = weights[out_idx * in_f + j];

        // dweights 累加
        atomicAdd(&dweights[out_idx * in_f + j], grad_out * x);

        // dinput[b, j] 累加
        atomicAdd(&dinput[b * in_f + j], w * grad_out);
    }
}


float* Linear::forward_gpu(float* input_d) {
    // 缓存输入
    if (input_cache_d) cudaFree(input_cache_d);
    CUDA_CHECK(cudaMalloc(&input_cache_d, sizeof(float) * batch_size * in_features));
    CUDA_CHECK(cudaMemcpy(input_cache_d, input_d, sizeof(float) * batch_size * in_features, cudaMemcpyDeviceToDevice));

    // 分配输出
    float* output_d;
    CUDA_CHECK(cudaMalloc(&output_d, sizeof(float) * batch_size * out_features));

    // 配置 kernel 启动参数
    int threads = 256;
    int blocks_x = (out_features + threads - 1) / threads;
    dim3 blockDim(threads);
    dim3 gridDim(blocks_x, batch_size);  // 每个 batch 一组 block

    linear_forward_kernel<<<gridDim, blockDim>>>(
        weights_d, biases_d,
        input_cache_d, output_d,
        batch_size, in_features, out_features
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    return output_d;
}


float* Linear::backward_gpu(float* doutput_d) {
    int din_size = batch_size * in_features;
    float* dinput_d;
    CUDA_CHECK(cudaMalloc(&dinput_d, sizeof(float) * din_size));
    CUDA_CHECK(cudaMemset(dinput_d, 0, sizeof(float) * din_size));

    // 清零 dweights 和 dbiases
    CUDA_CHECK(cudaMemset(dweights_d, 0, sizeof(float) * out_features * in_features));
    CUDA_CHECK(cudaMemset(dbiases_d, 0, sizeof(float) * out_features));

    // 启动 kernel
    int threads = 256;
    int blocks_x = (out_features + threads - 1) / threads;
    dim3 blockDim(threads);
    dim3 gridDim(blocks_x, batch_size);  // 每个样本分一组 block

    linear_backward_kernel<<<gridDim, blockDim>>>(
        weights_d, doutput_d,
        dinput_d, dweights_d, dbiases_d,
        input_cache_d,
        batch_size, in_features, out_features
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    return dinput_d;
}

__global__ void linear_update_params_kernel(
    float* weights, const float* dweights,
    float* biases,  const float* dbiases,
    float lr, int batch_size,
    int wsize, int out_features)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 更新 weights 和清零梯度
    if (idx < wsize) {
        weights[idx] -= lr * dweights[idx] / batch_size;
        ((float*)dweights)[idx] = 0.0f;
    }
    // 更新 biases 和清零梯度
    if (idx < out_features) {
        biases[idx] -= lr * dbiases[idx] / batch_size;
        ((float*)dbiases)[idx] = 0.0f;
    }
}


void Linear::step(float lr) {
    int wsize = out_features * in_features;
    if (is_gpu) {
        int threads = 256;
        int blocks = (max(wsize, out_features) + threads - 1) / threads;

        linear_update_params_kernel<<<blocks, threads>>>(
            weights_d, dweights_d,
            biases_d, dbiases_d,
            lr, batch_size,
            wsize, out_features
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        for (int i = 0; i < wsize; ++i) {
            weights[i]  -= lr * dweights[i] / batch_size;
            dweights[i] = 0.0f;
        }
        for (int i = 0; i < out_features; ++i) {
            biases[i]  -= lr * dbiases[i] / batch_size;
            dbiases[i]  = 0.0f;
        }

     
    }
    
}

float* Linear::forward(float* input) {
    if (is_gpu) return forward_gpu(input);
    else return forward_cpu(input);
}

float* Linear::backward(float* doutput) {
    if (is_gpu) return backward_gpu(doutput);
    else return backward_cpu(doutput);
}

void Linear::cuda() {
    is_gpu = true;
    int wsize = out_features * in_features;
    // 分配并拷贝 weights, biases
    CUDA_CHECK(cudaMalloc(&weights_d, sizeof(float) * wsize));
    CUDA_CHECK(cudaMemcpy(weights_d, weights, sizeof(float) * wsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dweights_d, sizeof(float) * wsize));
    CUDA_CHECK(cudaMemset(dweights_d, 0, sizeof(float) * wsize));

    CUDA_CHECK(cudaMalloc(&biases_d, sizeof(float) * out_features));
    CUDA_CHECK(cudaMemcpy(biases_d, biases, sizeof(float) * out_features, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dbiases_d, sizeof(float) * out_features));
    CUDA_CHECK(cudaMemset(dbiases_d, 0, sizeof(float) * out_features));
    // input_cache_d 延迟到 forward
}

// -----------------------------------------------------------------------------

std::vector<float> Conv2d::get_parameters() const {
    size_t W = out_channels * in_channels * kernel_size * kernel_size;
    size_t B = out_channels;
    std::vector<float> params;
    params.reserve(W + B);

    // 先把 weights 拷回 host
    std::vector<float> host_w(W);
    if (is_gpu) {
        cudaMemcpy(host_w.data(), weights_d, W * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_w.data(), weights, W * sizeof(float));
    }
    params.insert(params.end(), host_w.begin(), host_w.end());

    // 再把 biases 拷回 host
    std::vector<float> host_b(B);
    if (is_gpu) {
        cudaMemcpy(host_b.data(), biases_d, B * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_b.data(), biases, B * sizeof(float));
    }
    params.insert(params.end(), host_b.begin(), host_b.end());

    return params;
}

void Conv2d::set_parameters(const std::vector<float>& params) {
    size_t W = out_channels * in_channels * kernel_size * kernel_size;
    size_t B = out_channels;
    // 把前 W 个写入 weights
    std::memcpy(weights, params.data(), W * sizeof(float));
    // 再把后 B 个写入 biases
    std::memcpy(biases, params.data() + W, B * sizeof(float));
    // 如果在 GPU 上，还要同步到 device
    if (is_gpu) {
        cudaMemcpy(weights_d, weights, W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(biases_d, biases, B * sizeof(float), cudaMemcpyHostToDevice);
    }
}

std::vector<float> Linear::get_parameters() const {
    size_t W = out_features * in_features;
    size_t B = out_features;
    std::vector<float> params;
    params.reserve(W + B);

    // weights
    std::vector<float> host_w(W);
    if (is_gpu) {
        cudaMemcpy(host_w.data(), weights_d, W * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_w.data(), weights, W * sizeof(float));
    }
    params.insert(params.end(), host_w.begin(), host_w.end());

    // biases
    std::vector<float> host_b(B);
    if (is_gpu) {
        cudaMemcpy(host_b.data(), biases_d, B * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_b.data(), biases, B * sizeof(float));
    }
    params.insert(params.end(), host_b.begin(), host_b.end());

    return params;
}

void Linear::set_parameters(const std::vector<float>& params) {
    size_t W = out_features * in_features;
    size_t B = out_features;
    std::memcpy(weights, params.data(), W * sizeof(float));
    std::memcpy(biases,  params.data() + W, B * sizeof(float));
    if (is_gpu) {
        cudaMemcpy(weights_d, weights, W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(biases_d,  biases,  B * sizeof(float), cudaMemcpyHostToDevice);
    }
}


std::vector<float> Conv2d::get_gradients() const {
    size_t W = out_channels * in_channels * kernel_size * kernel_size;
    size_t B = out_channels;
    std::vector<float> grads;
    grads.reserve(W + B);

    // dweights
    std::vector<float> host_dw(W);
    if (is_gpu) {
        cudaMemcpy(host_dw.data(), dweights_d, W * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_dw.data(), dweights, W * sizeof(float));
    }
    grads.insert(grads.end(), host_dw.begin(), host_dw.end());

    // dbiases
    std::vector<float> host_db(B);
    if (is_gpu) {
        cudaMemcpy(host_db.data(), dbiases_d, B * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_db.data(), dbiases, B * sizeof(float));
    }
    grads.insert(grads.end(), host_db.begin(), host_db.end());

    return grads;
}

void Conv2d::set_gradients(const std::vector<float>& grads) {
    size_t W = out_channels * in_channels * kernel_size * kernel_size;
    size_t B = out_channels;

    std::memcpy(dweights, grads.data(), W * sizeof(float));
    std::memcpy(dbiases, grads.data() + W, B * sizeof(float));

    if (is_gpu) {
        cudaMemcpy(dweights_d, dweights, W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dbiases_d,  dbiases,  B * sizeof(float), cudaMemcpyHostToDevice);
    }
}


std::vector<float> Linear::get_gradients() const {
    size_t W = out_features * in_features;
    size_t B = out_features;
    std::vector<float> grads;
    grads.reserve(W + B);

    // dweights
    std::vector<float> host_dw(W);
    if (is_gpu) {
        cudaMemcpy(host_dw.data(), dweights_d, W * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_dw.data(), dweights, W * sizeof(float));
    }
    grads.insert(grads.end(), host_dw.begin(), host_dw.end());

    // dbiases
    std::vector<float> host_db(B);
    if (is_gpu) {
        cudaMemcpy(host_db.data(), dbiases_d, B * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        std::memcpy(host_db.data(), dbiases, B * sizeof(float));
    }
    grads.insert(grads.end(), host_db.begin(), host_db.end());

    return grads;
}

void Linear::set_gradients(const std::vector<float>& grads) {
    size_t W = out_features * in_features;
    size_t B = out_features;

    std::memcpy(dweights, grads.data(), W * sizeof(float));
    std::memcpy(dbiases, grads.data() + W, B * sizeof(float));

    if (is_gpu) {
        cudaMemcpy(dweights_d, dweights, W * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dbiases_d,  dbiases,  B * sizeof(float), cudaMemcpyHostToDevice);
    }
}
