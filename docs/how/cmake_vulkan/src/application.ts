/// <reference path="window.win32.d.ts" />

declare class AppWindow {
    declare constructor(parent_handler_window?: intptr_t);
    declare protected onMessage(uMsg: uint32_t, wParam: uint64_t, lParam: uint64_t): intptr_t;
}

export class Application {
    static appWindow: AppWindow;

    export static run() {
        this.appWindow = new AppWindow();
    }
}
