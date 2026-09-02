#pragma once

#include <vector>

class Layer {
public:
	size_t inputCnt, outputCnt;
	std::vector<float> weights; // actually 2D
	std::vector<float> biases;
	std::vector<float> delta;
	std::vector<float> inputs;
	std::vector<float> outputs; // aka: activations

public:
	Layer(size_t inputCnt, size_t outputCnt);

	void Forward(const std::vector<float>& inputs, const bool& applyReLU);
};
