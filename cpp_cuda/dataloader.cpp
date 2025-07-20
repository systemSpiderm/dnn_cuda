#include "dataloader.h"
#include <fstream>
#include <iostream>
#include <cstring>

DataLoader::DataLoader(const std::string& image_path, const std::string& label_path)
    : images(nullptr), labels(nullptr), num_images(0), rows(0), cols(0), current_index(0) {
    if (!load_images(image_path) || !load_labels(label_path)) {
        std::cerr << "Failed to load MNIST dataset.\n";
    }
}

DataLoader::~DataLoader() {
    delete[] images;
    delete[] labels;
}

int DataLoader::reverse_int(int i) {
    unsigned char c1 = i & 0xFF;
    unsigned char c2 = (i >> 8) & 0xFF;
    unsigned char c3 = (i >> 16) & 0xFF;
    unsigned char c4 = (i >> 24) & 0xFF;
    return ((int)c1 << 24) | ((int)c2 << 16) | ((int)c3 << 8) | c4;
}

bool DataLoader::load_images(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    int magic = 0;
    file.read((char*)&magic, 4); magic = reverse_int(magic);
    file.read((char*)&num_images, 4); num_images = reverse_int(num_images);
    file.read((char*)&rows, 4); rows = reverse_int(rows);
    file.read((char*)&cols, 4); cols = reverse_int(cols);

    images = new float[num_images * rows * cols];

    for (int i = 0; i < num_images; ++i)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                file.read((char*)&pixel, 1);
                images[i * rows * cols + r * cols + c] = pixel / 255.0f;
            }

    return true;
}

bool DataLoader::load_labels(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) return false;

    int magic = 0;
    file.read((char*)&magic, 4); magic = reverse_int(magic);
    file.read((char*)&num_images, 4); num_images = reverse_int(num_images);

    labels = new int[num_images];
    for (int i = 0; i < num_images; ++i) {
        unsigned char label = 0;
        file.read((char*)&label, 1);
        labels[i] = label;
    }

    return true;
}

bool DataLoader::next_batch(int batch_size, float* image_batch, int* label_batch) {
    if (current_index >= num_images) return false;

    int remaining = num_images - current_index;
    int actual_batch = (batch_size < remaining) ? batch_size : remaining;

    int image_size = rows * cols;

    for (int i = 0; i < actual_batch; ++i) {
        std::memcpy(
            &image_batch[i * image_size],
            &images[(current_index + i) * image_size],
            sizeof(float) * image_size
        );
        label_batch[i] = labels[current_index + i];
    }

    current_index += actual_batch;
    return true;
}

// 从任意位置重置
void DataLoader::reset(int start_index) {
    if (start_index >= 0 && start_index < num_images)
        current_index = start_index;
    else
        current_index = 0;
}

// 设置读取位置（语义等同于 reset）
void DataLoader::seek(int index) {
    reset(index);
}
