#ifndef MODEL_H
#define MODEL_H
#pragma once
#include "layer.h"
#include <vector>
#include <mpi.h>

class Model {
public:
    Model(int batch_size=32);
    float* forward(float* input, int C, int H, int W);
    void backward(float* dloss);
    void step(float lr);
    void cuda();
    std::vector<float> get_parameters() const;
    void set_parameters(const std::vector<float>& params);
    void sync_weights();

    std::vector<float> get_gradients() const;
    void set_gradients(const std::vector<float>& grads);
    void allreduce_gradients();  // 使用 MPI_Allreduce 同步平均梯度

private:
    bool is_gpu = false;  // 是否使用 GPU
    Conv2d conv1;
    ReLU relu1;
    AvgPool2d pool1;
    Conv2d conv2;
    ReLU relu2;
    AvgPool2d pool2;
    Linear fc1;
    //ReLU relu3;
    Linear fc2;

};

#endif

