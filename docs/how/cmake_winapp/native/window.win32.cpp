#pragma comment(linker, "/subsystem:windows")

#include "window.win32.h"

const TCHAR* CLASS_NAME = TEXT("Application Window");

HINSTANCE instance;
int cmdShow;
HWND main_hwnd;

typedef uint32_t(*method_ptr)(void*, uint32_t, uint64_t, uint64_t);
typedef struct { method_ptr method; void* thisVal; } callback_function;

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

void register_window_class() {
	// Register the window class.
	WNDCLASSEX win_class = {};
	win_class.cbSize = sizeof(WNDCLASSEX);
	win_class.style = CS_HREDRAW | CS_VREDRAW;
	win_class.lpfnWndProc = WindowProc;
	win_class.hInstance = instance;
	win_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
	win_class.lpszClassName = CLASS_NAME;
	RegisterClassEx(&win_class);
}

extern "C"
{
	uint32_t show_window(intptr_t hwnd, uint32_t cmdShow) {
		return ShowWindow(reinterpret_cast<HWND>(hwnd), cmdShow);
	}

	uint32_t destroy_window(intptr_t hwnd) {
		return DestroyWindow(reinterpret_cast<HWND>(hwnd));
	}

	void close_window(uint32_t exitCode) {
		PostQuitMessage(exitCode);
	}

	intptr_t default_window_procedure(intptr_t hwnd, uint32_t msg, uint64_t wparam, uint64_t lparam) {
		return DefWindowProc((HWND)hwnd, msg, wparam, lparam);
	}

	intptr_t create_window(char* title, intptr_t parent_hwnd, method_ptr method, void* thisVal) {

		const DWORD win_style = WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_VISIBLE | WS_OVERLAPPEDWINDOW;

		// Create the window.
		auto hwnd = CreateWindowEx(
			WS_EX_APPWINDOW,                // Optional window styles.
			CLASS_NAME,                     // Window class
			title,                          // Window text
			win_style,                      // Window style

			// Size and position
			CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

			(HWND)parent_hwnd,              // Parent window    
			nullptr,                        // Menu
			instance,                       // Instance handle
			nullptr                         // Additional application data
		);

		SetForegroundWindow(hwnd);

		auto method_ptr = new callback_function();
		method_ptr->method = method;		
		method_ptr->thisVal = thisVal;
		SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR)method_ptr);

		show_window(reinterpret_cast<intptr_t>(hwnd), cmdShow);

		if (!main_hwnd) {
			main_hwnd = hwnd;
		}

		return reinterpret_cast<intptr_t>(hwnd);
	}

	int messages_loop() {
		MSG msg = { };
		while (GetMessage(&msg, NULL, 0, 0))
		{
			if (msg.message == WM_QUIT) {
				return (int)msg.wParam;
			}

			TranslateMessage(&msg);
			DispatchMessage(&msg);

			RedrawWindow(main_hwnd, nullptr, nullptr, RDW_INTERNALPAINT);
		}

		DestroyWindow(main_hwnd);

		return 0;
	}

	extern void Main(void);

} // extern "C"

int WINAPI _tWinMain(HINSTANCE hInstance, HINSTANCE, LPTSTR pCmdLine, int nCmdShow)
{
	instance = hInstance;
	cmdShow = nCmdShow;

	register_window_class();

	// main inject
	Main();

	return messages_loop();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	bool result = false;
	auto callback_function_ptr = reinterpret_cast<callback_function*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
	if (callback_function_ptr) {
		//auto result = callback_function_ptr->operator()(uMsg, wParam, lParam);
		auto result = callback_function_ptr->method(callback_function_ptr->thisVal, uMsg, wParam, lParam);

		switch (uMsg) {
		case WM_DESTROY:
			SetWindowLongPtr(hwnd, GWLP_USERDATA, (LONG_PTR) nullptr);
			delete callback_function_ptr;
		}

		return result;
	}

	return DefWindowProc(hwnd, uMsg, wParam, lParam);
}