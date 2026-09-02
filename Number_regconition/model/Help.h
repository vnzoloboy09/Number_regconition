#pragma once

#include <vector>
#include <cmath>

inline float ReLU(float x) {
	return std::max(0.0f, x);
}

inline float ReLUDerivertive(float x) {
	return (x > 0 ? 1.0f : 0.0f);
}

std::vector<float> Softmax(const std::vector<float>& input);