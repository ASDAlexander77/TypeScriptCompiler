/// <reference path="window.win32.d.ts" />

export class AppWindow {

    private handler_window: intptr_t;

    export constructor(parent_handler_window?: intptr_t) {
        this.handler_window = create_window('Hello World!', parent_handler_window, this.onMessage);
    }

    protected onMessage(uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t): uint32_t {
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
