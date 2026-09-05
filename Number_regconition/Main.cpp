#include <iostream>

#include "model/Network.h"
#include "model/Help.h"
#include "Application.h"

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

    Application app;
    app.Run();

    return 0;
}

