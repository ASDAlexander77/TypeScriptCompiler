// all declarations here will be ignored

type uint32_t = TypeOf<1>;
type uint64_t = TypeOf<4294967297>;
type intptr_t = TypeOf<4294967297>;

type callback_function = (uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t) => uint32_t;

declare function create_window(title: string, width: uint32_t, height: uint32_t, handler: callback_function): intptr_t;
declare function close_window(exitCode: uint32_t): void;
declare function destroy_window(hwnd: intptr_t): void;
declare function create_bgfx(hwnd: intptr_t, width: uint32_t, height: uint32_t): void;
declare function run_bgfx_frame(): void;
declare function cleanup_bgfx(): void;

enum Messages {
    Destroy = 0x0002,
    Size = 0x0005,
    Frame = 0x000f,
    Close = 0x0010,
    KeyDown = 0x0100
}

enum Keys {
    Escape = 0x1b,
    Space = 0x20
}
