#include "Layer.h"

#include <random>
#include "Help.h"

Layer::Layer(size_t inCnt, size_t outCnt)
	: inputCnt(inCnt), outputCnt(outCnt), delta(outCnt), weights(inCnt * outCnt), 
	biases(outCnt, 0.0f), inputs(inCnt), z(outCnt, 0.0f), outputs(outCnt)
{
	for (auto& w : weights) {
		w = static_cast<float>(rand()) / RAND_MAX - 0.5f;
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