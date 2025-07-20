#!/usr/bin/env bash
set -e
CUDA_HOME=/usr/local/cuda
MPI_CC=mpic++  # 或 g++ 用于串行测试

# 1) 编译 CUDA 代码
nvcc -std=c++17 -c layer.cu -o layer.o

# 2) 分别编译每个 cpp 文件
$MPI_CC -std=c++17 -I$CUDA_HOME/include -c main.cpp -o main.o
$MPI_CC -std=c++17 -I$CUDA_HOME/include -c model.cpp -o model.o
$MPI_CC -std=c++17 -I$CUDA_HOME/include -c dataloader.cpp -o dataloader.o
$MPI_CC -std=c++17 -I$CUDA_HOME/include -c utils.cpp -o utils.o

# 3) 链接所有目标文件
$MPI_CC main.o model.o dataloader.o utils.o layer.o \
    -L$CUDA_HOME/lib64 -lcudart -o main

echo "Built multi-GPU MPI+CUDA executable: ./main"
