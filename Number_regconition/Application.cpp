#include "Application.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

#include "model/Network.h"

void Application::GetModelPredic(const std::vector<float>& inputs) {
    m_Confidence = 0.0f;
    m_Network.Forward(inputs);
    auto& out = m_Network.GetOutput();
    for (size_t i = 0; i < out.size(); i++) {
        if (out[i] > m_Confidence) {
            m_Confidence = out[i];
            m_PredictNumber = i;
        }
    }
}

void Application::RenderDrawingCanvasWindow() {
    ImGui::SetNextWindowSize(ImVec2(510, 340), ImGuiCond_FirstUseEver);
    ImGui::Begin("28x28 Digit Recognition");

    // Persistent variables
    static std::vector<float> canvasData(28 * 28, 0.0f);
    static float penRadius = 1.6f;

    // ==================== LEFT SIDE: CANVAS ====================
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize(280.0f, 280.0f);

    ImGui::InvisibleButton("canvas_click_area", canvasSize);
    bool isActive = ImGui::IsItemActive();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    // Mouse drawing logic with smooth outer fade
    if (isActive && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 mousePos = ImGui::GetIO().MousePos;
        float relX = (mousePos.x - canvasPos.x) / (canvasSize.x / 28.0f);
        float relY = (mousePos.y - canvasPos.y) / (canvasSize.y / 28.0f);
        int cx = static_cast<int>(relX);
        int cy = static_cast<int>(relY);
        int intRadius = static_cast<int>(std::ceil(penRadius));

        for (int dy = -intRadius; dy <= intRadius; ++dy) {
            for (int dx = -intRadius; dx <= intRadius; ++dx) {
                int nx = cx + dx;
                int ny = cy + dy;
                if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
                    float cellCenterX = static_cast<float>(nx) + 0.5f;
                    float cellCenterY = static_cast<float>(ny) + 0.5f;
                    float dist = std::sqrt((cellCenterX - relX) * (cellCenterX - relX) +
                        (cellCenterY - relY) * (cellCenterY - relY));
                    if (dist <= penRadius) {
                        float normalizedDist = dist / penRadius;
                        // Exponential falloff with dim outer edges
                        float factor = std::exp(-4.5f * normalizedDist * normalizedDist);
                        float brightness = std::clamp(factor, 0.0f, 1.0f);

                        int index = ny * 28 + nx;
                        canvasData[index] = std::clamp(canvasData[index] + (brightness * 0.3f), 0.0f, 1.0f);
                    }
                }
            }
        }
    }

    // Render Canvas Background
    drawList->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(0, 0, 0, 255));

    // Render Pixels
    float cellSize = canvasSize.x / 28.0f;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float val = canvasData[y * 28 + x];
            if (val > 0.0f) {
                ImVec2 pMin = ImVec2(canvasPos.x + (x * cellSize), canvasPos.y + (y * cellSize));
                ImVec2 pMax = ImVec2(pMin.x + cellSize, pMin.y + cellSize);
                ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(val, val, val, 1.0f));
                drawList->AddRectFilled(pMin, pMax, col);
            }
        }
    }
    drawList->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), ImGui::GetColorU32(ImGuiCol_Border));


    // ==================== RIGHT SIDE: CONTROLS & RESULTS ====================
    ImGui::SameLine(); // Move cursor to the right of the canvas

    ImGui::BeginGroup();
    {
        ImGui::Text("Settings");
        ImGui::SetNextItemWidth(160.0f);
        ImGui::SliderFloat("Blur", &penRadius, 1.0f, 3.0f, "%.1f");

        ImGui::Spacing();

        if (ImGui::Button("Clear Canvas", ImVec2(160, 0))) {
            std::fill(canvasData.begin(), canvasData.end(), 0.0f);
            m_PredictNumber = -1; // Reset output on clear
        }

        if (ImGui::Button("Detect", ImVec2(160, 0))) {
            GetModelPredic(canvasData);
            if (m_PredictNumber == m_UserLabel) {
                std::cout << "Fuck yeahh \n";
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Output Result Section
        ImGui::Text("Prediction:");
        if (m_PredictNumber >= 0) {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(50, 255, 50, 255));
            ImGui::Text("Digit: %d", m_PredictNumber);
            ImGui::Text("Conf:  %.1f%%", m_Confidence * 100.0f);
            ImGui::PopStyleColor();
        }
        else {
            ImGui::TextDisabled("Awaiting input...");
        }

        ImGui::Text("Actual Label:");
        ImGui::SetNextItemWidth(160.0f);
        ImGui::InputInt("##UserLabelInput", &m_UserLabel);

        if (m_UserLabel < 0 || m_UserLabel > 9) {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 60, 60, 255));
            ImGui::Text("Error: Must be 0 to 9");
            ImGui::PopStyleColor();
        }
        else {
            ImGui::TextDisabled("Valid label range");
        }
    }
    ImGui::EndGroup();

    ImGui::End();
}

Application::Application() 
    : m_Network("model/Model.bin")
{
    if (!glfwInit()) assert("Failed to init GLFW");

    // Create window with OpenGL context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1280, 720, "ImGui + GLFW Example", nullptr, nullptr);
    if (!window) assert("Failed to create window");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

Application::~Application() {
    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void Application::Run() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        RenderDrawingCanvasWindow();

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }
}