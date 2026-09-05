#include "Help.h"

#include <fstream>
#include <iostream>
#include <random>

std::vector<float> Softmax(const std::vector<float>& input) {
    size_t n = input.size();
    std::vector<float> output(n);

    float maxVal = *std::max_element(input.begin(), input.end());

    float sum = 0.0f;
    for (auto& i : input) {
        sum += std::exp(i - maxVal);
    }

    for (size_t i = 0; i < n; i++) {
        output[i] = std::exp(input[i] - maxVal) / sum;
    }

    return output;
}

std::vector<float> ReLU(const std::vector<float>& z){
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); i++) {
        out[i] = ReLU(z[i]);
    }

    return out;
}

std::vector<float> OneHot(size_t num) {
    std::vector<float> out(10, 0.0f);
    out[num] = 1.0f;
    return out;
}

float randomFloat(float a, float b) {
    static std::random_device rd;   // seed
    static std::mt19937 gen(rd());  // Mersenne Twister engine
    std::uniform_real_distribution<float> dist(a, b);
    return dist(gen);
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

void SaveDatasetBinary(const std::string& filename,
    const std::vector<std::vector<float>>& images,
    const std::vector<int>& labels) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    uint32_t numSamples = static_cast<uint32_t>(images.size());
    uint32_t imgWidth = 28;
    uint32_t imgHeight = 28;

    // 1. Write Header metadata
    outFile.write(reinterpret_cast<char*>(&numSamples), sizeof(numSamples));
    outFile.write(reinterpret_cast<char*>(&imgWidth), sizeof(imgWidth));
    outFile.write(reinterpret_cast<char*>(&imgHeight), sizeof(imgHeight));

    // 2. Write Data Loop
    for (size_t i = 0; i < numSamples; ++i) {
        // Write the label (int)
        outFile.write(reinterpret_cast<const char*>(&labels[i]), sizeof(labels[i]));

        // Write the 784 float pixels (28 * 28)
        outFile.write(reinterpret_cast<const char*>(images[i].data()), images[i].size() * sizeof(float));
    }

    outFile.close();
    std::cout << "Successfully saved " << numSamples << " samples to " << filename << std::endl;
}

void LoadDatasetBinary(const std::string& filename,
    std::vector<std::vector<float>>& images,
    std::vector<int>& labels) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return;
    }

    uint32_t numSamples = 0;
    uint32_t imgWidth = 0;
    uint32_t imgHeight = 0;

    // Read Header
    inFile.read(reinterpret_cast<char*>(&numSamples), sizeof(numSamples));
    inFile.read(reinterpret_cast<char*>(&imgWidth), sizeof(imgWidth));
    inFile.read(reinterpret_cast<char*>(&imgHeight), sizeof(imgHeight));

    images.resize(numSamples);
    labels.resize(numSamples);

    size_t imgSize = imgWidth * imgHeight;

    // Read Data Loop
    for (size_t i = 0; i < numSamples; ++i) {
        inFile.read(reinterpret_cast<char*>(&labels[i]), sizeof(labels[i]));

        images[i].resize(imgSize);
        inFile.read(reinterpret_cast<char*>(images[i].data()), imgSize * sizeof(float));
    }

    inFile.close();
    std::cout << "Successfully loaded " << numSamples << " samples from " << filename << std::endl;
}