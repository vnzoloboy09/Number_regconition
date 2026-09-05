#include "Application.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

#include "model/Help.h"
#include "model/Network.h"

Application::Application()
    : m_Network("model/Model.bin"), m_CanvasData(28 * 28, 0.0f), m_Confidence(10, 0.0f)
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

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

Application::~Application() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}

void Application::GetModelPredic(const std::vector<float>& inputs) {
    m_Network.Forward(inputs);
    m_Confidence = m_Network.GetOutput();
    float cof = 0.0f;
    for (size_t i = 0; i < m_Confidence.size(); i++) {
        if (cof < m_Confidence[i]) {
            cof = m_Confidence[i];
            m_PredictNumber = i;
        }
    }
}

void Application::RenderDrawingCanvasWindow() {
    ImGui::SetNextWindowSize(ImVec2(720, 500), ImGuiCond_FirstUseEver);
    ImGui::Begin("28x28 Digit Recognition");

    static float penRadius = 1.8f;

    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        if (ImGui::IsKeyPressed(ImGuiKey_C, false)) {
            std::fill(m_CanvasData.begin(), m_CanvasData.end(), 0.0f);
            m_PredictNumber = -1;
        }
    }

    // ==================== LEFT SIDE: CANVAS ====================
    ImVec2 canvasPos = ImGui::GetCursorScreenPos();
    ImVec2 canvasSize(420.0f, 420.0f);

    ImGui::InvisibleButton("canvas_click_area", canvasSize);
    bool isHovered = ImGui::IsItemHovered();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    bool canvasChanged = false;

    // Mouse drawing
    if (isHovered) {
        bool isDrawing = ImGui::IsMouseDown(ImGuiMouseButton_Right);
        bool isErasing = ImGui::IsMouseDown(ImGuiMouseButton_Left);

        if (isDrawing || isErasing) {
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
                            int index = ny * 28 + nx;
                            if (isDrawing) {
                                float normalizedDist = dist / penRadius;
                                float factor = std::exp(-4.5f * normalizedDist * normalizedDist);
                                float brightness = std::clamp(factor, 0.0f, 1.0f);
                                m_CanvasData[index] = std::clamp(m_CanvasData[index] + (brightness * 0.6f), 0.0f, 1.0f);
                            }
                            else if (isErasing) {
                                m_CanvasData[index] = 0.0f; // Erase
                            }
                            canvasChanged = true;
                        }
                    }
                }
            }
        }
    }

    drawList->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(0, 0, 0, 255));

    // Render Pixels
    float cellSize = canvasSize.x / 28.0f;
    float totalSum = 0.0f;
    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float val = m_CanvasData[y * 28 + x];
            totalSum += val;
            if (val > 0.0f) {
                ImVec2 pMin = ImVec2(canvasPos.x + (x * cellSize), canvasPos.y + (y * cellSize));
                ImVec2 pMax = ImVec2(pMin.x + cellSize, pMin.y + cellSize);
                ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(val, val, val, 1.0f));
                drawList->AddRectFilled(pMin, pMax, col);
            }
        }
    }
    drawList->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), ImGui::GetColorU32(ImGuiCol_Border));

    if (canvasChanged) {
        if (totalSum > 0.0f) {
            GetModelPredic(m_CanvasData);
        }
        else {
            m_PredictNumber = -1;
            m_Confidence.assign(m_Confidence.size(), 0.0f);
        }
    }


    // ==================== RIGHT SIDE: CONTROLS & 0-9 RESULTS ====================
    ImGui::SameLine();
    ImGui::BeginGroup();
    {
        ImGui::Text("Settings");
        ImGui::SetNextItemWidth(200.0f);
        ImGui::SliderFloat("Brush Size", &penRadius, 1.0f, 3.0f, "%.1f");

        ImGui::Spacing();
        ImGui::TextDisabled("Controls: Right-Click to Draw");
        ImGui::TextDisabled("          Left-Click to Erase");
        ImGui::TextDisabled("          Press 'C' to Clear");

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Digit Confidence Breakdown:");
        ImGui::Spacing();

        for (int i = 0; i <= 9; ++i) {
            float conf = m_Confidence[i];

            if (i == m_PredictNumber) {
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(98, 97, 189, 255));
            }

            ImGui::Text(" %d: ", i);
            ImGui::SameLine(40);

            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.529f, 0.808f, 0.922f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));

            char buf[32];
            sprintf_s(buf, "%.1f%%", conf * 100.0f);
            ImGui::ProgressBar(conf, ImVec2(160, 16), buf);
            ImGui::PopStyleColor(2);
            if (i == m_PredictNumber) {
                ImGui::PopStyleColor();
            }
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }
    ImGui::EndGroup();

    ImGui::End();
}

void Application::RenderDatasetAugmentationWindow() {
    ImGui::SetNextWindowSize(ImVec2(650, 480), ImGuiCond_FirstUseEver);
    ImGui::Begin("Dataset Augmentation Tool");

    // Persistent augmentation parameters
    static float augRotation = 0.0f;       // Degrees (-45 to 45)
    static float augTranslationX = 0.0f;   // Pixels (-6 to 6)
    static float augTranslationY = 0.0f;   // Pixels (-6 to 6)
    static float augScale = 1.0f;          // Zoom (0.7 to 1.3)
    static float augNoise = 0.0f;          // Noise intensity (0.0 to 0.3)
    static int   samplesPerImage = 3;      // Variations to generate per image

    // Buffer for the augmented preview
    static std::vector<float> previewImage(28 * 28, 0.0f);

    // Helper lambda to apply transformations from a source 28x28 buffer to a destination buffer
    auto transformImage = [&](const std::vector<float>& src, std::vector<float>& dst,
        float rot, float tx, float ty, float scale, float noise) {
            std::fill(dst.begin(), dst.end(), 0.0f);
            float cx = 13.5f, cy = 13.5f; // Center of 28x28 image
            float rad = rot * (3.14159265f / 180.0f);
            float cosA = std::cos(rad);
            float sinA = std::sin(rad);

            for (int y = 0; y < 28; ++y) {
                for (int x = 0; x < 28; ++x) {
                    // Reverse mapping for affine transformation (Scale -> Rotate -> Translate)
                    float px = (static_cast<float>(x) - cx - tx) / scale;
                    float py = (static_cast<float>(y) - cy - ty) / scale;

                    float srcX = cosA * px + sinA * py + cx;
                    float srcY = -sinA * px + cosA * py + cy;

                    // Nearest neighbor sampling with boundary check
                    int isrcX = static_cast<int>(std::round(srcX));
                    int isrcY = static_cast<int>(std::round(srcY));

                    if (isrcX >= 0 && isrcX < 28 && isrcY >= 0 && isrcY < 28) {
                        float val = src[isrcY * 28 + isrcX];

                        // Add slight random noise if enabled
                        if (noise > 0.0f) {
                            float randomVal = (static_cast<float>(rand()) / RAND_MAX) * noise;
                            val = std::clamp(val + randomVal, 0.0f, 1.0f);
                        }
                        dst[y * 28 + x] = val;
                    }
                }
            }
        };

    // Always update the preview using the current m_CanvasData
    transformImage(m_CanvasData, previewImage, augRotation, augTranslationX, augTranslationY, augScale, augNoise);

    // ==================== LEFT SIDE: PREVIEWS ====================
    ImGui::BeginGroup();
    {
        ImGui::Text("Current Canvas (Origin)");
        ImVec2 p1 = ImGui::GetCursorScreenPos();
        ImVec2 previewSize(140.0f, 140.0f);
        ImGui::InvisibleButton("orig_canvas", previewSize);
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        // Draw background for original canvas
        drawList->AddRectFilled(p1, ImVec2(p1.x + previewSize.x, p1.y + previewSize.y), IM_COL32(0, 0, 0, 255));

        float cellSize = previewSize.x / 28.0f;
        for (int y = 0; y < 28; ++y) {
            for (int x = 0; x < 28; ++x) {
                float val = m_CanvasData[y * 28 + x];
                if (val > 0.0f) {
                    ImVec2 minPos(p1.x + x * cellSize, p1.y + y * cellSize);
                    ImVec2 maxPos(minPos.x + cellSize, minPos.y + cellSize);
                    drawList->AddRectFilled(minPos, maxPos, ImGui::ColorConvertFloat4ToU32(ImVec4(val, val, val, 1.0f)));
                }
            }
        }
        drawList->AddRect(p1, ImVec2(p1.x + previewSize.x, p1.y + previewSize.y), ImGui::GetColorU32(ImGuiCol_Border));

        ImGui::Spacing();
        ImGui::Text("Augmented Preview");
        ImVec2 p2 = ImGui::GetCursorScreenPos();
        ImGui::InvisibleButton("aug_canvas", previewSize);

        // Draw background for augmented preview
        drawList->AddRectFilled(p2, ImVec2(p2.x + previewSize.x, p2.y + previewSize.y), IM_COL32(0, 0, 0, 255));

        for (int y = 0; y < 28; ++y) {
            for (int x = 0; x < 28; ++x) {
                float val = previewImage[y * 28 + x];
                if (val > 0.0f) {
                    ImVec2 minPos(p2.x + x * cellSize, p2.y + y * cellSize);
                    ImVec2 maxPos(minPos.x + cellSize, minPos.y + cellSize);
                    drawList->AddRectFilled(minPos, maxPos, ImGui::ColorConvertFloat4ToU32(ImVec4(val, val, val, 1.0f)));
                }
            }
        }
        drawList->AddRect(p2, ImVec2(p2.x + previewSize.x, p2.y + previewSize.y), ImGui::GetColorU32(ImGuiCol_Border));

        ImGui::Spacing();
        if (m_PredictNumber != -1) {
            ImGui::Text("Detected Label: %d", m_PredictNumber);
        }
        else {
            ImGui::Text("Draw something on canvas!");
        }
    }
    ImGui::EndGroup();

    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();

    // ==================== RIGHT SIDE: CONTROLS & BATCH PROCESS ====================
    ImGui::BeginGroup();
    {
        ImGui::Text("Transformation Controls");
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::SliderFloat("Rotation (deg)", &augRotation, -45.0f, 45.0f, "%.1f°");
        ImGui::SliderFloat("Shift X", &augTranslationX, -5.0f, 5.0f, "%.1f px");
        ImGui::SliderFloat("Shift Y", &augTranslationY, -5.0f, 5.0f, "%.1f px");
        ImGui::SliderFloat("Scale / Zoom", &augScale, 0.7f, 1.3f, "%.2f");
        ImGui::SliderFloat("Random Noise", &augNoise, 0.0f, 0.2f, "%.2f");

        if (ImGui::Button("Random", ImVec2(150, 25))) {
            augRotation = RandomFloat(-45.0f, 45.0f);
            augTranslationX = RandomFloat(-5.0f, 5.0f);
            augTranslationY = RandomFloat(-5.0f, 5.0f);
            augScale = RandomFloat(0.7f, 1.3f);
            augNoise = RandomFloat(0.0f, 0.2f);
        }

        if (ImGui::Button("Reset Parameters", ImVec2(150, 25))) {
            augRotation = 0.0f;
            augTranslationX = 0.0f;
            augTranslationY = 0.0f;
            augScale = 1.0f;
            augNoise = 0.0f;
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        ImGui::Text("Batch Dataset Augmentation");
        ImGui::SliderInt("Variations / Image", &samplesPerImage, 1, 10);

        if (ImGui::Button("Genarate", ImVec2(220, 40))) {
            // TODO: Loop through your 60,000 dataset arrays here. 
            // Apply randomized versions of these slider limits to each image and save them.
        }
        ImGui::TextDisabled("Applies random variations to all 60k items.");
    }
    ImGui::EndGroup();

    ImGui::End();
}

void Application::Run() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        RenderDrawingCanvasWindow();
        RenderDatasetAugmentationWindow();

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