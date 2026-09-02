#include "Network.h"
#include "Help.h"

static constexpr bool APPLY_RELU = true;

Network::Network(const std::vector<size_t>& topology) {
	m_Layers.reserve(topology.size() - 1);
	for (size_t i = 1; i < topology.size(); i++) {
		m_Layers.emplace_back(topology[i - 1], topology[i]);
	}
}

void Network::Forward(const std::vector<float>& inputs) {
	m_Layers[0].Forward(inputs, APPLY_RELU);
	for (size_t i = 1; i < m_Layers.size() - 1; i++) {
		m_Layers[i].Forward(m_Layers[i - 1].outputs, APPLY_RELU);
	}

	m_Layers.back().Forward(m_Layers[m_Layers.size() - 2].outputs, !APPLY_RELU);
	m_Layers.back().outputs = Softmax(m_Layers.back().outputs);
}

void Network::BackProp(const std::vector<float>& target, float lr) {
	for (size_t i = 0; i < m_Layers.back().outputCnt; i++) {
		float a = m_Layers.back().outputs[i];
		float error = a - target[i];
		m_Layers.back().delta[i] = error * ReLUDerivertive(a);
	}

	for (int l = m_Layers.size() - 2; l >= 0; l--) {
		Layer& current = m_Layers[l];
		Layer& next = m_Layers[l + 1];

		for (size_t i = 0; i < next.inputCnt; i++) {
			float error = 0.0f;

			for (size_t j = 0; j < next.outputCnt; j++) {
				error += next.weights[j * next.inputCnt + i] * next.delta[j] * ReLUDerivertive(current.outputs[i]);
			}
		}
	}

	for (auto& layer : m_Layers) {
		for (size_t i = 0; i < layer.outputCnt; i++) {
			layer.biases[i] -= lr * layer.delta[i];
			for (size_t j = 0; j < layer.inputCnt; j++) {
				layer.weights[i * layer.inputCnt + j] -= lr * layer.delta[i] * layer.inputs[j];
			}
		}
	}
}

void Network::Train(const std::vector<std::vector<float>>& inputs,
	const std::vector<std::vector<float>>& targets, float lr, size_t epochs)
{
	for (size_t e = 0; e < epochs; e++) {
		for (size_t i = 0; i < inputs.size(); i++) {
			Forward(inputs[i]);
			BackProp(targets[i], 0.1f);
		}
	}
}
