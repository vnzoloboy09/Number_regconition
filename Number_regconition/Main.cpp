#include <iostream>

#include "model/Network.h"

int main()
{
    std::vector<std::vector<float>> X = {
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {0.0f, 0.0f}
    };
    
    std::vector<float> out(2);
    Network net("model/Model.bin");
    for (auto& i : X) {
        net.Forward(i);
         out = net.GetOutput();

        for(auto& x : out) {
            std::cout << x << ' ';
        }
        std::cout << '\n';
    }

    return 0;
}

