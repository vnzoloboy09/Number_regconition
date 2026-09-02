#include <iostream>

#include "model/Network.h"
#include "model/Help.h"

int main()
{
    srand(42);

    //Network net({ 28 * 28, 128, 128, 10 });
    Network net("model/Model.bin");

    //auto trainImg = readMNISTImages("train/images.idx3-ubyte");
    //auto trainLabel = readMNISTLabels("train/labels.idx1-ubyte");
    //net.Train(trainImg, trainLabel, 0.01f, 5);

    //net.Save("model/Model.bin");

    auto testImg = readMNISTImages("test/images.idx3-ubyte");
    auto testlabel = readMNISTLabelsOrigin("test/labels.idx1-ubyte");

    std::vector<float> out;
    int cnt = 0;
    for (size_t i = 0; i < testImg.size(); i++) {
        net.Forward(testImg[i]);
        out = net.GetOutput();
        float m = 0.0f;
        uint8_t ans = 0;
        for (uint8_t x = 0; x < out.size(); x++) {
            if (out[x] > m) {
                m = out[x];
                ans = x;
            }
        }
        std::cout << static_cast<int>(ans) << " - " << static_cast<int>(testlabel[i]) << '\n';
        if (ans == testlabel[i]) {
            cnt++;
        }
    }

    std::cout << cnt << "/" << testImg.size() << '\n';

    return 0;
}

