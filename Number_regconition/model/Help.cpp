#include "Help.h"

std::vector<float> Softmax(const std::vector<float>& input) {
	size_t n = input.size();
	std::vector<float> output(n);

	float sum = 0.0f;
	for (auto& i : input) {
		sum += std::exp(i);
	}

	for (size_t i = 0; i < n; i++) {
		output[i] = std::exp(input[i]) / sum;
	}

	return output;
}