#include <bgfx/bgfx.h>
#include <bx/math.h>

#include <GLFW/glfw3.h>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#elif defined(__APPLE__)
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint16_t kMainViewId = 0;

constexpr uint32_t kMessageDestroy = 0x0002;
constexpr uint32_t kMessageSize = 0x0005;
constexpr uint32_t kMessageFrame = 0x000f;
constexpr uint32_t kMessageClose = 0x0010;
constexpr uint32_t kMessageKeyDown = 0x0100;

typedef uint32_t (*MethodPtr)(void*, uint32_t, uint64_t, uint64_t);

struct CallbackFunction {
    MethodPtr method = nullptr;
    void* thisVal = nullptr;
};

GLFWwindow* g_window = nullptr;
CallbackFunction g_callback{};
uint32_t g_width = 0;
uint32_t g_height = 0;
uint32_t g_frameCounter = 0;
bool g_bgfxInitialized = false;
bool g_glfwInitialized = false;

void dispatchMessage(uint32_t uMsg, uint64_t wParam, uint64_t lParam)
{
    if (g_callback.method != nullptr) {
        g_callback.method(g_callback.thisVal, uMsg, wParam, lParam);
    }
}

void setPlatformData(bgfx::PlatformData& platformData, GLFWwindow* window)
{
#if defined(_WIN32)
    platformData.nwh = glfwGetWin32Window(window);
#elif defined(__linux__)
    platformData.ndt = glfwGetX11Display();
    platformData.nwh = reinterpret_cast<void*>(static_cast<uintptr_t>(glfwGetX11Window(window)));
#elif defined(__APPLE__)
    platformData.nwh = glfwGetCocoaWindow(window);
#else
    (void)window;
    platformData.ndt = nullptr;
    platformData.nwh = nullptr;
#endif
}

uint32_t mapGlfwKeyToVirtualKey(int key)
{
    switch (key) {
    case GLFW_KEY_ESCAPE:
        return 0x1b;
    case GLFW_KEY_SPACE:
        return 0x20;
    default:
        return static_cast<uint32_t>(key);
    }
}

uint32_t colorChannel(uint32_t frame, uint32_t channelOffset)
{
    const float phase = static_cast<float>((frame + channelOffset) % 256) / 255.0f;
    return static_cast<uint32_t>(phase * 255.0f);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    (void)window;
    (void)scancode;
    (void)mods;

    if (action == GLFW_PRESS) {
        dispatchMessage(kMessageKeyDown, mapGlfwKeyToVirtualKey(key), 0);
    }
}

void closeCallback(GLFWwindow* window)
{
    (void)window;
    dispatchMessage(kMessageClose, 0, 0);
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    (void)window;
    dispatchMessage(kMessageSize, static_cast<uint64_t>(width), static_cast<uint64_t>(height));
}

} // namespace

extern "C" {

intptr_t create_window(const char* title, uint32_t width, uint32_t height, MethodPtr method, void* thisVal)
{
    if (!g_glfwInitialized) {
        if (!glfwInit()) {
            return 0;
        }
        g_glfwInitialized = true;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int>(width),
        static_cast<int>(height),
        title,
        nullptr,
        nullptr);

    if (window == nullptr) {
        return 0;
    }

    g_window = window;
    g_width = width;
    g_height = height;
    g_callback.method = method;
    g_callback.thisVal = thisVal;

    glfwSetWindowUserPointer(window, &g_callback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowCloseCallback(window, closeCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    return reinterpret_cast<intptr_t>(window);
}

void close_window(uint32_t exitCode)
{
    (void)exitCode;
    if (g_window != nullptr) {
        glfwSetWindowShouldClose(g_window, GLFW_TRUE);
    }
}

void destroy_window(intptr_t hwnd)
{
    GLFWwindow* window = reinterpret_cast<GLFWwindow*>(hwnd);
    if (window != nullptr) {
        glfwDestroyWindow(window);
    }

    if (g_window == window) {
        g_window = nullptr;
        g_callback.method = nullptr;
        g_callback.thisVal = nullptr;
    }

    if (g_glfwInitialized) {
        glfwTerminate();
        g_glfwInitialized = false;
    }
}

void create_bgfx(intptr_t hwnd, uint32_t width, uint32_t height)
{
    GLFWwindow* window = reinterpret_cast<GLFWwindow*>(hwnd);
    if (window == nullptr) {
        return;
    }

    g_window = window;
    g_width = width;
    g_height = height;
    g_frameCounter = 0;

    bgfx::PlatformData platformData{};
    setPlatformData(platformData, window);

    bgfx::Init init{};
    init.type = bgfx::RendererType::Count;
    init.resolution.width = width;
    init.resolution.height = height;
    init.resolution.reset = BGFX_RESET_VSYNC;
    init.platformData = platformData;

    if (!bgfx::init(init)) {
        return;
    }

    bgfx::setViewClear(kMainViewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, 0x303030ff, 1.0f, 0);
    bgfx::setViewRect(kMainViewId, 0, 0, static_cast<uint16_t>(width), static_cast<uint16_t>(height));
    g_bgfxInitialized = true;
}

void run_bgfx_frame()
{
    if (!g_bgfxInitialized || g_window == nullptr) {
        return;
    }

    int framebufferWidth = 0;
    int framebufferHeight = 0;
    glfwGetFramebufferSize(g_window, &framebufferWidth, &framebufferHeight);

    const uint32_t width = static_cast<uint32_t>(framebufferWidth);
    const uint32_t height = static_cast<uint32_t>(framebufferHeight);

    if (width != g_width || height != g_height) {
        g_width = width;
        g_height = height;
        bgfx::reset(g_width, g_height, BGFX_RESET_VSYNC);
        bgfx::setViewRect(kMainViewId, 0, 0, static_cast<uint16_t>(g_width), static_cast<uint16_t>(g_height));
    }

    const uint32_t red = colorChannel(g_frameCounter, 0);
    const uint32_t green = colorChannel(g_frameCounter, 85);
    const uint32_t blue = colorChannel(g_frameCounter, 170);
    const uint32_t clearColor = 0xff000000u | (red << 16) | (green << 8) | blue;

    bgfx::setViewClear(kMainViewId, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH, clearColor, 1.0f, 0);
    bgfx::touch(kMainViewId);

    bgfx::dbgTextClear();
    bgfx::dbgTextPrintf(0, 1, 0x0f, "Hello from tslang");
    bgfx::dbgTextPrintf(0, 2, 0x0a, "bgfx + GLFW cross-platform sample");
    bgfx::dbgTextPrintf(0, 3, 0x0c, "Press Escape to quit");

    bgfx::frame();
    ++g_frameCounter;
}

void cleanup_bgfx()
{
    if (g_bgfxInitialized) {
        bgfx::shutdown();
        g_bgfxInitialized = false;
    }
}

int run_loop()
{
    if (g_window == nullptr) {
        return 1;
    }

    while (!glfwWindowShouldClose(g_window)) {
        glfwPollEvents();
        dispatchMessage(kMessageFrame, 0, 0);
    }

    dispatchMessage(kMessageDestroy, 0, 0);
    destroy_window(reinterpret_cast<intptr_t>(g_window));
    return 0;
}

} // extern "C"
