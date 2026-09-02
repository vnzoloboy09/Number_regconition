#include "Layer.h"

#include <random>
#include "Help.h"

Layer::Layer(size_t inCnt, size_t outCnt)
	: inputCnt(inCnt), outputCnt(outCnt), delta(outCnt),
	weights(inCnt * outCnt), biases(outCnt, 0.0f), inputs(inCnt), outputs(outCnt)
{
	for (auto& w : weights) {
		w = static_cast<float>(rand()) / RAND_MAX - 0.5f;
	}
}

void Layer::Forward(const std::vector<float>& _inputs, const bool& applyReLU) {
	this->inputs = _inputs;
	for (size_t i = 0; i < outputCnt; i++) {
		float z = biases[i];
		for (size_t j = 0; j < inputCnt; j++) {
			z += inputs[j] * weights[i * inputCnt + j];
		}

		if (applyReLU) {
			outputs[i] = ReLU(z);
		}
		else {
			outputs[i] = z;
		}
	}
}