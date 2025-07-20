#ifndef UTILS_H
#define UTILS_H
#include <string>

float* one_hot_batch(int* labels, int batch_size, int num_classes);
float* softmax_batch(float* logits, int batch_size, int num_classes);
float cross_entropy_loss_batch(float* predictions, float* targets, int batch_size, int num_classes);
float* softmax_cross_entropy_derivative_batch(float* predictions, float* targets, int batch_size, int num_classes);

// 前提：vec的大小为num_classes，且vec的初始值为0
float* one_hot(int label, int num_classes = 10);

float* softmax(float* logits, int size);

float cross_entropy_loss(float* prediction, float* target, int size);

float* softmax_cross_entropy_derivative(float* prediction, float* target, int size);

void matmul(float* A, float* B, float* C, int M, int N, int K);


float* conv2d_multi(
    const float* input,          // [in_channels * H * W]
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
);



#endif
