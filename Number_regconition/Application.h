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

    void SetInput(const std::vector<float>& in) {
        m_Inputs = in;
    }

    void Run();

private:
    GLFWwindow* window = nullptr;
    Network m_Network;
    std::vector<float> m_Inputs;
    
    int m_UserLabel = 0;
    size_t m_PredictNumber = 0;
    float m_Confidence = 0.0f;
};
