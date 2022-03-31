/// <reference path="window.win32.d.ts" />

import "./appwindow";

export class Application {
    static appWindow: AppWindow;

    export static run() {
        this.appWindow = new AppWindow();
    }
}
