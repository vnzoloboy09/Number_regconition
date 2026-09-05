#include <GLFW/glfw3.h>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "model/Network.h"

class Application {
public:
    Application();

    ~Application();

    void GetModelPredic(const std::vector<float>& inputs);
    void RenderDrawingCanvasWindow();
    void RenderDatasetAugmentationWindow();

    void Run();

private:
    GLFWwindow* window = nullptr;
    Network m_Network;
    std::vector<float> m_CanvasData;

    size_t m_PredictNumber = 0;
    std::vector<float> m_Confidence;
};
