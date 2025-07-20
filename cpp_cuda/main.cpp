#include <mpi.h>
#include "model.h"
#include "utils.h"
#include "dataloader.h"
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    }

int max_element_host(const float* prob, int size) {
    int max_index = 0;
    float max_value = prob[0];
    for (int i = 1; i < size; ++i) {
        if (prob[i] > max_value) {
            max_value = prob[i];
            max_index = i;
        }
    }
    return max_index;
}

// ————— 全局参数 ————— //
const int global_batch = 256;
const int epochs = 30;
const float lr = 0.01f;
const bool is_gpu = 1;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 若 CPU 模式，仅使用 rank 0
    if (!is_gpu && rank != 0) {
        MPI_Finalize();
        return 0;
    }

    const int local_batch = is_gpu ? (global_batch / size) : global_batch;

    if (rank == 0) {
        std::cout << "Running MNIST training with batch size: " << global_batch
                  << ", epochs: " << epochs
                  << ", learning rate: " << lr
                  << ", using GPU: " << (is_gpu ? "Yes" : "No") << "\n";
    }

    if (is_gpu) {
        CUDA_CHECK(cudaSetDevice(rank));
    }

    DataLoader train_loader(
        "../mnist_data/MNIST/raw/train-images-idx3-ubyte",
        "../mnist_data/MNIST/raw/train-labels-idx1-ubyte"
    );
    DataLoader test_loader(
        "../mnist_data/MNIST/raw/t10k-images-idx3-ubyte",
        "../mnist_data/MNIST/raw/t10k-labels-idx1-ubyte"
    );

    const int H = train_loader.image_rows();
    const int W = train_loader.image_cols();
    const int num_classes = 10;
    const int num_train = train_loader.total_samples();
    const int steps_per_epoch = num_train / global_batch;

    Model model(local_batch);
    if (is_gpu) {
        model.cuda();
        model.sync_weights();  // 将 rank0 的模型广播到其他进程
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int epoch_correct = 0;

        if (rank == 0 || is_gpu) {
            train_loader.reset();
        }

        for (int step = 0; step < steps_per_epoch; ++step) {
            int offset = step * global_batch + (is_gpu ? rank * local_batch : 0);
            train_loader.seek(offset);

            float* image_host = new float[local_batch * H * W]();
            int* label_host = new int[local_batch]();
            train_loader.next_batch(local_batch, image_host, label_host);

            float* image_dev = image_host;
            if (is_gpu) {
                CUDA_CHECK(cudaMalloc(&image_dev, sizeof(float) * local_batch * H * W));
                CUDA_CHECK(cudaMemcpy(image_dev, image_host, sizeof(float) * local_batch * H * W, cudaMemcpyHostToDevice));
            }

            float* output = model.forward(image_dev, 1, H, W);

            float* probs_host_arr = new float[local_batch * num_classes]();
            if (is_gpu) {
                CUDA_CHECK(cudaMemcpy(probs_host_arr, output, sizeof(float) * local_batch * num_classes, cudaMemcpyDeviceToHost));
            } else {
                std::memcpy(probs_host_arr, output, sizeof(float) * local_batch * num_classes);
            }

            float* probs = softmax_batch(probs_host_arr, local_batch, num_classes);
            float* targets = one_hot_batch(label_host, local_batch, num_classes);
            float loss = cross_entropy_loss_batch(probs, targets, local_batch, num_classes);
            epoch_loss += loss * global_batch;

            int correct = 0;
            for (int b = 0; b < local_batch; ++b) {
                int pred = max_element_host(probs + b * num_classes, num_classes);
                if (pred == label_host[b]) correct++;
            }
            epoch_correct += correct;

            float* grad_host = softmax_cross_entropy_derivative_batch(probs, targets, local_batch, num_classes);
            float* grad_dev = grad_host;
            if (is_gpu) {
                CUDA_CHECK(cudaMalloc(&grad_dev, sizeof(float) * local_batch * num_classes));
                CUDA_CHECK(cudaMemcpy(grad_dev, grad_host, sizeof(float) * local_batch * num_classes, cudaMemcpyHostToDevice));
            }

            model.backward(grad_dev);
            if (is_gpu) model.allreduce_gradients();
            model.step(lr);

            // 清理
            if (is_gpu) {
                CUDA_CHECK(cudaFree(image_dev));
                CUDA_CHECK(cudaFree(output));
                CUDA_CHECK(cudaFree(grad_dev));
            } else {
                delete[] output;
            }

            delete[] image_host;
            delete[] label_host;
            delete[] probs_host_arr;
            delete[] probs;
            delete[] targets;
            delete[] grad_host;
        }

        int global_correct = 0;
        if (is_gpu) {
            MPI_Allreduce(&epoch_correct, &global_correct, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        } else {
            global_correct = epoch_correct;
        }

        if (rank == 0) {
            float acc = 100.0f * global_correct / num_train;
            float avg_loss = epoch_loss / num_train;
            std::cout << "[Epoch " << epoch + 1 << "] Loss=" << avg_loss << ", Acc=" << acc << "%\n";
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        double secs = std::chrono::duration<double>(end_time - start_time).count();
        std::cout << "Total training time: " << secs << " seconds\n";
    }

    // ————— TEST阶段 ————— //
    int test_batch_size = is_gpu ? (global_batch / size) : global_batch;
    int steps_test = test_loader.total_samples() / (test_batch_size * (is_gpu ? size : 1));

    if (steps_test == 0) {
        if (rank == 0) {
            std::cerr << "Warning: Not enough test data for the given batch size.\n";
        }
        MPI_Finalize();
        return 0;
    }

    int correct_local = 0;

    for (int s = 0; s < steps_test; ++s) {
        int offset = s * test_batch_size * (is_gpu ? size : 1) + (is_gpu ? rank * test_batch_size : 0);
        test_loader.seek(offset);

        float* image_host = new float[test_batch_size * H * W]();
        int* label_host = new int[test_batch_size]();
        test_loader.next_batch(test_batch_size, image_host, label_host);

        float* image_dev = image_host;
        if (is_gpu) {
            CUDA_CHECK(cudaMalloc(&image_dev, sizeof(float) * test_batch_size * H * W));
            CUDA_CHECK(cudaMemcpy(image_dev, image_host, sizeof(float) * test_batch_size * H * W, cudaMemcpyHostToDevice));
        }

        float* output = model.forward(image_dev, 1, H, W);
        float* probs_host_arr = new float[test_batch_size * num_classes]();
        if (is_gpu) {
            CUDA_CHECK(cudaMemcpy(probs_host_arr, output, sizeof(float) * test_batch_size * num_classes, cudaMemcpyDeviceToHost));
        } else {
            std::memcpy(probs_host_arr, output, sizeof(float) * test_batch_size * num_classes);
        }

        for (int b = 0; b < test_batch_size; ++b) {
            int pred = max_element_host(probs_host_arr + b * num_classes, num_classes);
            if (pred == label_host[b]) correct_local++;
        }

        if (is_gpu) {
            CUDA_CHECK(cudaFree(image_dev));
            CUDA_CHECK(cudaFree(output));
        } else {
            delete[] output;
        }

        delete[] image_host;
        delete[] label_host;
        delete[] probs_host_arr;
    }

    int correct_total = 0;
    MPI_Reduce(&correct_local, &correct_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        int total_test = steps_test * test_batch_size * (is_gpu ? size : 1);
        float acc = correct_total * 100.0f / total_test;
        std::cout << "[Test Accuracy] " << correct_total << "/" << total_test << " = " << acc << "%\n";
    }

    MPI_Finalize();
    return 0;
}
