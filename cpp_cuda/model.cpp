//model.cpp
#include "model.h"
#include <cmath>
#include <iostream>
#include <iomanip>  // for std::setw, std::setprecision
#include <mpi.h>


Model::Model(int batch_size)
    : conv1(batch_size, 1, 8, 3, 1, 1),
      relu1(batch_size),
      pool1(batch_size, 2, 2),
      relu2(batch_size),
      conv2(batch_size, 8, 16, 3, 1, 1),
      pool2(batch_size, 2, 2),
      fc1(batch_size, 7 * 7 * 16, 128),
      fc2(batch_size, 128, 10) {}


float* Model::forward(float* input, int C, int H, int W) {
    float* out_conv1 = conv1.forward(input, H, W);
    //printf("conv1 success\n");
    float* out_relu1 = relu1.forward(out_conv1, conv1.get_C_out(), conv1.get_H_out(), conv1.get_W_out());
    //printf("relu1 success\n");
    if (is_gpu) cudaFree(out_conv1);
    else        delete[] out_conv1;

    float* out_pool1 = pool1.forward(out_relu1, relu1.get_C_out(), relu1.get_H_out(), relu1.get_W_out());
    //printf("pool1 success\n");
    if (is_gpu) cudaFree(out_relu1);
    else        delete[] out_relu1;

    float* out_conv2 = conv2.forward(out_pool1, pool1.get_H_out(), pool1.get_W_out());
    //printf("conv2 success\n");
    if (is_gpu) cudaFree(out_pool1);
    else        delete[] out_pool1;

    float* out_relu2 = relu2.forward(out_conv2, conv2.get_C_out(), conv2.get_H_out(), conv2.get_W_out());
    //printf("relu2 success\n");
    if (is_gpu) cudaFree(out_conv2);
    else        delete[] out_conv2;

    float* out_pool2 = pool2.forward(out_relu2, relu2.get_C_out(), relu2.get_H_out(), relu2.get_W_out());
    if (is_gpu) cudaFree(out_relu2);
    else        delete[] out_relu2;

    int flatten_size = pool2.get_C_out() * pool2.get_H_out() * pool2.get_W_out();
    float* out_fc1 = fc1.forward(out_pool2);
    if (is_gpu) cudaFree(out_pool2);
    else        delete[] out_pool2;

    float* out_fc2 = fc2.forward(out_fc1);
    if (is_gpu) cudaFree(out_fc1);
    else        delete[] out_fc1;

    return out_fc2;
}

// 修改 backward 里的释放方式
void Model::backward(float* dloss) {
    float* dfc2 = fc2.backward(dloss);
    //printf("fc2 backward success\n");

    float* dfc1 = fc1.backward(dfc2);
    //printf("fc1 backward success\n");
    if (is_gpu) cudaFree(dfc2);
    else        delete[] dfc2;

    float* dpool2 = pool2.backward(dfc1);
    //printf("pool2 backward success\n");
    if (is_gpu) cudaFree(dfc1);
    else        delete[] dfc1;

    float* drelu2 = relu2.backward(dpool2);
    //printf("relu2 backward success\n");
    if (is_gpu) cudaFree(dpool2);
    else        delete[] dpool2;

    float* dconv2 = conv2.backward(drelu2);
    //printf("conv2 backward success\n");
    if (is_gpu) cudaFree(drelu2);
    else        delete[] drelu2;

    float* dpool1 = pool1.backward(dconv2);
   // printf("pool1 backward success\n");
    if (is_gpu) cudaFree(dconv2);
    else        delete[] dconv2;

    float* drelu1 = relu1.backward(dpool1);
    //printf("relu1 backward success\n");
    if (is_gpu) cudaFree(dpool1);
    else        delete[] dpool1;

    float* dconv1 = conv1.backward(drelu1);
    //printf("conv1 backward success\n");
    if (is_gpu) cudaFree(drelu1);
    else        delete[] drelu1;

    // 最后释放 dconv1
    if (is_gpu) cudaFree(dconv1);
    else        delete[] dconv1;
}

void Model::step(float lr) {
    conv1.step(lr);
    conv2.step(lr);
    fc1.step(lr);
    fc2.step(lr);
}

void Model::cuda() {
    is_gpu = true;
    conv1.cuda();
    pool1.cuda();
    conv2.cuda();
    pool2.cuda();
    fc1.cuda();
    fc2.cuda();
    relu1.cuda();
    relu2.cuda();
}

// --- 新增：参数导出/导入接口 ---
std::vector<float> Model::get_parameters() const {
    std::vector<float> params;
    // 各层参数依次拼接：conv1, conv2, fc1, fc2
    auto p1 = conv1.get_parameters();
    auto p2 = conv2.get_parameters();
    auto p3 = fc1.get_parameters();
    auto p4 = fc2.get_parameters();
    params.reserve(p1.size() + p2.size() + p3.size() + p4.size());
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    params.insert(params.end(), p3.begin(), p3.end());
    params.insert(params.end(), p4.begin(), p4.end());
    return params;
}

void Model::set_parameters(const std::vector<float>& params) {
    // 确定各层参数长度
    size_t s1 = conv1.get_parameters().size();
    size_t s2 = conv2.get_parameters().size();
    size_t s3 = fc1.get_parameters().size();
    size_t s4 = fc2.get_parameters().size();
    // 拆分并分配
    auto it = params.begin();
    conv1.set_parameters(std::vector<float>(it, it + s1));
    it += s1;
    conv2.set_parameters(std::vector<float>(it, it + s2));
    it += s2;
    fc1.set_parameters(std::vector<float>(it, it + s3));
    it += s3;
    fc2.set_parameters(std::vector<float>(it, it + s4));
}



void Model::sync_weights() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<float> params;

    if (rank == 0) {
        // 从 GPU 拉取模型参数到 CPU（只在 rank 0 执行）
        params = get_parameters();
    }

    // 广播参数长度
    int param_size = 0;
    if (rank == 0) {
        param_size = static_cast<int>(params.size());
    }
    MPI_Bcast(&param_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // 所有进程准备接收 buffer（包括 rank 0 自己）
    if (rank != 0) {
        params.resize(param_size);  // rank 0 已有，无需 resize
    }

    // 广播参数数据（float 类型）
    MPI_Bcast(params.data(), param_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 将广播回来的参数设置回 GPU（所有进程都执行）
    set_parameters(params);
}




std::vector<float> Model::get_gradients() const {
    std::vector<float> grads;

    auto g1 = conv1.get_gradients();
    auto g2 = conv2.get_gradients();
    auto g3 = fc1.get_gradients();
    auto g4 = fc2.get_gradients();

    grads.reserve(g1.size() + g2.size() + g3.size() + g4.size());
    grads.insert(grads.end(), g1.begin(), g1.end());
    grads.insert(grads.end(), g2.begin(), g2.end());
    grads.insert(grads.end(), g3.begin(), g3.end());
    grads.insert(grads.end(), g4.begin(), g4.end());

    return grads;
}

void Model::set_gradients(const std::vector<float>& grads) {
    // 先获取各层梯度长度
    size_t s1 = conv1.get_gradients().size();
    size_t s2 = conv2.get_gradients().size();
    size_t s3 = fc1.get_gradients().size();
    size_t s4 = fc2.get_gradients().size();

    auto it = grads.begin();
    conv1.set_gradients(std::vector<float>(it, it + s1));
    it += s1;
    conv2.set_gradients(std::vector<float>(it, it + s2));
    it += s2;
    fc1.set_gradients(std::vector<float>(it, it + s3));
    it += s3;
    fc2.set_gradients(std::vector<float>(it, it + s4));
}

#include <mpi.h>

void Model::allreduce_gradients() {
    std::vector<float> grads = get_gradients();
    std::vector<float> grads_avg(grads.size());

    // MPI_Allreduce 计算平均梯度
    MPI_Allreduce(grads.data(), grads_avg.data(), grads.size(), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (float& g : grads_avg) {
        g /= size;
    }

    // 将平均梯度设置回模型
    set_gradients(grads_avg);
}
