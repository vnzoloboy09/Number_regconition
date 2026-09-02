#include "Layer.h"

#include <random>
#include "Help.h"

Layer::Layer(size_t inCnt, size_t outCnt)
	: inputCnt(inCnt), outputCnt(outCnt), delta(outCnt), weights(inCnt * outCnt), 
	biases(outCnt, 0.0f), inputs(inCnt), z(outCnt, 0.0f), outputs(outCnt)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	float stddev = std::sqrt(2.0f / static_cast<float>(inputCnt));
	std::normal_distribution<float> d(0.0f, stddev);

	for (auto& w : weights) {
		w = d(gen);
	}
}

void Layer::Forward(const std::vector<float>& _inputs, bool isOutLayer) {
	this->inputs = _inputs;
	for (size_t i = 0; i < outputCnt; i++) {
		z[i] = biases[i];
		for (size_t j = 0; j < inputCnt; j++) {
			z[i] += inputs[j] * weights[i * inputCnt + j];
		}

		if (!isOutLayer) {
			outputs = ReLU(z);
		}
		else {
			outputs = Softmax(z);
		}
	}
}