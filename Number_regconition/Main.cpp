#include <iostream>

#include "model/Network.h"

int main()
{
    Network net({ 2, 4, 2 });

    std::vector<std::vector<float>> X = {
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.0f, 0.0f}
    };
    std::vector<std::vector<float>> Y = { 
        {0.0f, 1.0f}, 
        {1.0f, 0.0f}, 
        {1.0f, 0.0f}, 
        {0.0f, 1.0f} 
    };

    net.Train(X, Y, 0.1f, 10000);
    
    net.Forward({ 0.0f, 1.0f });
    std::vector<float> out = net.GetOutput();
    for (auto& x : out) {
        std::cout << x << ' ';
    }
}

