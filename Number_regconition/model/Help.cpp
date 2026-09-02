#include "Help.h"

#include <fstream>
#include <iostream>

std::vector<float> Softmax(const std::vector<float>& input) {
    int n = input.size();
    std::vector<float> output(n);

    float maxVal = *std::max_element(input.begin(), input.end());

    float sum = 0.0f;
    for (auto& i : input) {
        sum += std::exp(i - maxVal);
    }

    for (int i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - maxVal) / sum;
    }

    return output;
}


int32_t readInt(std::ifstream& f) {
    unsigned char bytes[4];
    f.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

std::vector<std::vector<float>> readMNISTImages(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    int magic = readInt(f);
    int numImages = readInt(f);
    int rows = readInt(f);
    int cols = readInt(f);

    std::cout << "Magic=" << magic
        << " NumImages=" << numImages
        << " Rows=" << rows
        << " Cols=" << cols << std::endl;

    std::vector<std::vector<uint8_t>> images(numImages, std::vector<uint8_t>(rows * cols));
    for (int i = 0; i < numImages; i++) {
        f.read((char*)images[i].data(), rows * cols);
    }

    std::vector<std::vector<float>> imgFloat(images.size());

    for (int i = 0; i < images.size(); i++) {
        imgFloat[i].resize(28 * 28);
        for (int j = 0; j < images[i].size(); j++) {
            imgFloat[i][j] = static_cast<float>(images[i][j]) / 255.0f;
        }
    }

    return imgFloat;
}

std::vector<std::vector<float>> readMNISTLabels(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    int magic = readInt(f);
    int numLabels = readInt(f);

    std::vector<uint8_t> labels(numLabels);
    f.read((char*)labels.data(), numLabels);

    std::vector<std::vector<float>> labelFloat(labels.size());

    for (size_t i = 0; i < labels.size(); i++) {
        labelFloat[i].resize(10, 0.0f);
        labelFloat[i][labels[i]] = 1.0f;
    }

    return labelFloat;
}

std::vector<uint8_t> readMNISTLabelsOrigin(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    int magic = readInt(f);
    int numLabels = readInt(f);

    std::vector<uint8_t> labels(numLabels);
    f.read((char*)labels.data(), numLabels);

    return labels;
}

void drawImage(const std::vector<float>& image, int rows, int cols) {
    const std::string shades = " .:-=+*#%@";
    // from light (space) to dark (@)

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            float val = image[r * cols + c]; // pixel value
            int idx = static_cast<int>(val * (shades.size() - 1));
            std::cout << shades[idx];
        }
        std::cout << "\n";
    }
}