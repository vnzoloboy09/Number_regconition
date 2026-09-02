#pragma once

#include <vector>
#include <cmath>
#include <string>
#include <cstdint>

inline float ReLU(float x) {
	return std::max(0.0f, x);
}

inline float ReLUDerivertive(float x) {
	return (x > 0 ? 1.0f : 0.0f);
}

std::vector<float> ReLU(const std::vector<float>& z) {
	std::vector<float> out(z.size());
	for (size_t i = 0; i < z.size(); i++) {
		out[i] = ReLU(z[i]);
	}

	return out;
}

std::vector<float> Softmax(const std::vector<float>& input);

int32_t readInt(std::ifstream& f);

std::vector<std::vector<float>> readMNISTImages(const std::string& filename);

std::vector<std::vector<float>> readMNISTLabels(const std::string& filename);

std::vector<uint8_t> readMNISTLabelsOrigin(const std::string& filename);

void drawImage(const std::vector<float>& image, int rows = 28, int cols = 28);