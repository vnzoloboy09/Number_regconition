#pragma once

#include "Layer.h"

class Network {
public:
	Network(const std::vector<size_t>& topology);
	Network(const char* filename);

	void Forward(const std::vector<float>& inptus);

	void BackProp(const std::vector<float>& target, float lr);
	void Train(const std::vector<std::vector<float>>& inputs, 
		const std::vector<std::vector<float>>& targets, float lr, size_t epochs);

	void Save(const char* filename) const;

	const std::vector<float>& GetOutput() const { return m_Layers.back().outputs; }

private:
	std::vector<Layer> m_Layers;
};

