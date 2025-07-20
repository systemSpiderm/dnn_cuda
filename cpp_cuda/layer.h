#ifndef LAYER_H
#define LAYER_H
#pragma once
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>
class Conv2d {
public:
    Conv2d(int batch_size, int in_channels, int out_channels, int kernel_size, int stride=1, int padding=0);
    ~Conv2d();
    // 原 CPU 版 forward/backward
    float* forward_cpu(float* input, int H, int W);
    float* backward_cpu(float* doutput);
    // GPU 版 forward/backward
    float* forward_gpu(float* input_d, int H, int W);
    float* backward_gpu(float* doutput_d);
    // 通用接口：根据 is_gpu 决定调用 CPU/GPU
    float* forward(float* input, int H, int W);
    float* backward(float* doutput);
    void cuda();
    void step(float lr);

    int get_C_in() const { return C_cache; }
    int get_C_out() const { return OC_cache; }
    int get_H_out() const { return OH_cache; }
    int get_W_out() const { return OW_cache; }
    int get_H_in() const { return H_cache; }
    int get_W_in() const { return W_cache; }

    /// 扁平化返回 weights + biases
    std::vector<float> get_parameters() const;
    /// 从扁平化向量里恢复 weights + biases
    void set_parameters(const std::vector<float>& params);


    std::vector<float> get_gradients() const;
    void set_gradients(const std::vector<float>& grads);

private:
    bool is_gpu = false;  // 是否使用 GPU   
    float* weights;   // [out_channels * in_channels * KH * KW]
    float* dweights;  // same size as weights
    float* biases;    // [out_channels]
    float* dbiases;   // [out_channels]
    float* input_cache; // [in_channels * H * W] (cached input)
    // GPU 端参数
    float* weights_d;   // 设备端权重
    float* biases_d;    // 设备端偏置
    float* dweights_d;  // 设备端梯度
    float* dbiases_d;
    float* input_cache_d; // 设备端输入 cache
    int batch_size, in_channels, out_channels, kernel_size, stride, padding;
    int H_cache, W_cache, OH_cache, OW_cache, C_cache, OC_cache;  // input/output dims cached for backward
};

class AvgPool2d {
public:
    AvgPool2d(int batch_size, int kernel_size, int stride);
    ~AvgPool2d();
    float* forward_cpu(float* input, int C, int H, int W);
    float* backward_cpu(float* doutput);
    float* forward_gpu(float* input_d, int C, int H, int W);
    float* backward_gpu(float* doutput_d);
    float* forward(float* input, int C, int H, int W);
    float* backward(float* doutput);
    void cuda();
    int get_C_in() const { return C_cache; }
    int get_C_out() const { return C_cache; }
    int get_H_out() const { return OH_cache; }
    int get_W_out() const { return OW_cache; }
    int get_H_in() const { return H_cache; }
    int get_W_in() const { return W_cache; }
    
private:
    bool is_gpu = false;  // 是否使用 GPU
    int batch_size, kernel_size, stride;
    float* input_cache;  // [batch_size * C * H * W]
    int C_cache_d, H_cache_d, W_cache_d, OH_cache_d, OW_cache_d;
    float* input_cache_d; // 设备端输入 cache
    int H_cache, W_cache, C_cache, OH_cache, OW_cache; // cached input dimensions for backward
};

class ReLU {
public:
    ReLU(int batch_size);
    ~ReLU();
    float* forward_cpu(float* input, int C, int H, int W);
    float* backward_cpu(float* doutput);
    float* forward_gpu(float* input_d, int C, int H, int W);
    float* backward_gpu(float* doutput_d);
    float* forward(float* input, int C, int H, int W);
    float* backward(float* doutput);
    void cuda();

    int get_C_in() const { return C; }
    int get_C_out() const { return C; }
    int get_H_out() const { return H; }
    int get_W_out() const { return W; }
    int get_H_in() const { return H; }
    int get_W_in() const { return W; }


private:
    bool is_gpu = false;  // 是否使用 GPU
    float* input_cache;  // [C * H * W]
    float* input_cache_d; // 设备端输入 cache
    int C, H, W;
    int batch_size, size_cache;
};

class Linear {
public:
    Linear(int batch_size, int in_features, int out_features);
    ~Linear();
    float* forward_cpu(float* input);
    float* backward_cpu(float* doutput);
    float* forward_gpu(float* input_d);
    float* backward_gpu(float* doutput_d);
    float* forward(float* input);
    float* backward(float* doutput);
    void cuda();
    void step(float lr);

    std::vector<float> get_parameters() const;
    void set_parameters(const std::vector<float>& params);

    
    std::vector<float> get_gradients() const;
    void set_gradients(const std::vector<float>& grads);

private:
    bool is_gpu = false;  // 是否使用 GPU
    // CPU 端
    float* weights;   // [out_features * in_features]
    float* dweights;  // same size
    float* biases;    // [out_features]
    float* dbiases;   // [out_features]
    float* input_cache; // [in_features]
    // GPU 端
    float* weights_d;
    float* biases_d;
    float* dweights_d;
    float* dbiases_d;
    float* input_cache_d;

    int batch_size, in_features, out_features;
};

#endif