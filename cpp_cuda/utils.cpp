//utils.cpp
#include "utils.h"
#include <fstream>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cstring>

// 生成 one-hot 批处理矩阵 [batch_size, num_classes]
float* one_hot_batch(int* labels, int batch_size, int num_classes) {
    float* targets = new float[batch_size * num_classes]();
    for (int b = 0; b < batch_size; ++b) {
        targets[b * num_classes + labels[b]] = 1.0f;
    }
    return targets;
}

// softmax 批处理：输入 logits[b * num_classes]，输出 probs[b * num_classes]
float* softmax_batch(float* logits, int batch_size, int num_classes) {
    float* probs = new float[batch_size * num_classes]();
    for (int b = 0; b < batch_size; ++b) {
        float* logit = logits + b * num_classes;
        float* prob = probs + b * num_classes;

        // 1. 找最大值做数值稳定
        float max_val = logit[0];
        for (int i = 1; i < num_classes; ++i) {
            if (logit[i] > max_val) max_val = logit[i];
        }

        // 2. 计算 exp 和总和
        float sum = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            prob[i] = std::exp(logit[i] - max_val);
            sum += prob[i];
        }

        // 3. 归一化
        for (int i = 0; i < num_classes; ++i) {
            prob[i] /= sum;
        }
    }
    return probs;
}

// 计算批量交叉熵损失（返回平均值）
float cross_entropy_loss_batch(float* predictions, float* targets, int batch_size, int num_classes) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        float* pred = predictions + b * num_classes;
        float* target = targets + b * num_classes;
        for (int i = 0; i < num_classes; ++i) {
            total_loss -= target[i] * std::log(pred[i] + 1e-8f);
        }
    }
    return total_loss / batch_size;
}

// 计算 softmax + cross entropy 的梯度：[batch_size, num_classes]
float* softmax_cross_entropy_derivative_batch(float* predictions, float* targets, int batch_size, int num_classes) {
    float* grad = new float[batch_size * num_classes]();
    for (int b = 0; b < batch_size; ++b) {
        float* pred = predictions + b * num_classes;
        float* target = targets + b * num_classes;
        float* grad_b = grad + b * num_classes;

        for (int i = 0; i < num_classes; ++i) {
            grad_b[i] = (pred[i] - target[i]);  // 平均化梯度
        }
    }
    return grad;
}


void matmul(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i * K + j] = 0.0f;
            for (int k = 0; k < N; ++k) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
}

float* conv2d_multi(
    const float* input,          // [batch_size * in_channels * H * W]
    const float* weights,        // [out_channels * in_channels * KH * KW]
    const float* biases,         // [out_channels]
    int batch_size, 
    int in_channels,
    int H,
    int W,
    int out_channels,
    int KH,
    int KW,
    int stride,
    int padding
) {
    int OH = (H + 2 * padding - KH) / stride + 1;
    int OW = (W + 2 * padding - KW) / stride + 1;
    float* output = new float[batch_size * out_channels * OH * OW]();

    for (int b = 0; b < batch_size; ++b) {
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float sum = biases[oc];
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int input_idx = (b * in_channels * H * W + (ic * H + ih) * W + iw);
                                int weight_idx = (((oc * in_channels + ic) * KH + kh) * KW + kw);
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
                int output_idx = ((oc * OH + oh) * OW + ow);
                output[b*out_channels*OH*OW + output_idx] = sum;
            }
        }
    }
    }
    return output;
}

