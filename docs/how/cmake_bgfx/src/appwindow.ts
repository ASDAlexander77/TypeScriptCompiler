/// <reference path="bgfx_glfw.d.ts" />

const WINDOW_WIDTH: uint32_t = 800;
const WINDOW_HEIGHT: uint32_t = 600;

export class AppWindow {
    private handler_window: intptr_t;

    export constructor() {
        this.handler_window = create_window('tslang bgfx sample', WINDOW_WIDTH, WINDOW_HEIGHT, this.onMessage);
        create_bgfx(this.handler_window, WINDOW_WIDTH, WINDOW_HEIGHT);
    }

    protected onMessage(uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t): uint32_t {
        switch (uMsg) {
            case Messages.Close:
                close_window(0);
                break;
            case Messages.Frame:
                run_bgfx_frame();
                break;
            case Messages.Destroy:
                cleanup_bgfx();
                break;
            case Messages.KeyDown:
                switch (wParam) {
                    case Keys.Escape:
                        close_window(0);
                        break;
                }
                return 0;
        }

        return 0;
    }
}
