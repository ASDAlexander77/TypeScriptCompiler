/// <reference path="window.win32.d.ts" />

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
        create_vulkan(this.handler_window);
    }

    protected onMessage(uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t): intptr_t {
        switch (uMsg) {
            case Messages.Close:
                close_window(0);
                break;
            case Messages.Erasebkgnd:
                return 1;
            case Messages.Paint:
                run_vulkan();
                break;
            case Messages.Destroy:
                cleanup_vulkan();
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