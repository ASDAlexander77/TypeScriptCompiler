// all declarations here will be ignored

type uint32_t = TypeOf<1>;
type uint64_t = TypeOf<4294967297>;
type intptr_t = TypeOf<4294967297>;

type callback_function = (uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t) => uint32_t;
declare function create_window(title: string, parent_hwnd: intptr_t, handler: callback_function, val: uint32_t, handler2: callback_function, val2: uint32_t): intptr_t;
declare function close_window(exitCode: uint32_t): void;
declare function destroy_window(hwnd: intptr_t): uint32_t;
declare function default_window_procedure(hwnd: intptr_t, msg: uint32_t, wparam: uint64_t, lparam: uint64_t): intptr_t;

declare function create_vulkan(hwnd: intptr_t): void;
declare function run_vulkan(): void;
declare function cleanup_vulkan(): void;

enum Messages {
    Destroy = 0x0002,
    Size = 0x0005,
    Paint = 0x000f,
    Close = 0x0010,
    KeyDown = 0x0100,
    Erasebkgnd = 0x0014
}

enum Keys {
    Escape = 0x1b,
    Space = 0x20
}
