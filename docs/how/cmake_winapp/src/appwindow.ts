/// <reference path="window.win32.d.ts" />

type uint32_t = TypeOf<1>;
type uint64_t = TypeOf<4294967297>;
type intptr_t = TypeOf<4294967297>;

type callback_function = (uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t) => uint32_t;
declare function create_window(title: string, parent_hwnd: intptr_t, handler: callback_function, val: uint32_t, handler2: callback_function, val2: uint32_t): intptr_t;
declare function close_window(exitCode: uint32_t): void;
declare function destroy_window(hwnd: intptr_t): uint32_t;
declare function default_window_procedure(hwnd: intptr_t, msg: uint32_t, wparam: uint64_t, lparam: uint64_t): intptr_t;

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

export class AppWindow {

    private handler_window: intptr_t;

    constructor(parent_handler_window?: intptr_t) {
        this.handler_window = create_window('Hello World!', parent_handler_window, this.onMessage);
    }

    protected onMessage(uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t): intptr_t {
        switch (uMsg) {
            case Messages.Close:
                close_window(0);
                break;
            case Messages.Erasebkgnd:
                return 1;
            case Messages.Paint:
                break;
            case Messages.Destroy:
                break;
            case Messages.KeyDown: // key down
                switch (wParam) {
                    case Keys.Escape: // key escape
                        close_window(0);
                        break;
                    case Keys.Space: // key space
                        // open new window
                        break;
                }

                return 0;
        }

        return default_window_procedure(this.handler_window, uMsg, wParam, lParam);
    }
}

export class Application {
    static appWindow: AppWindow;

    static run() {
        this.appWindow = new AppWindow();
    }
}

export function Main()
{
    Application.run();
}