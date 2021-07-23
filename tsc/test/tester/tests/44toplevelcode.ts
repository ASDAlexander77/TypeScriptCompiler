let s2: string;
let xyz = 12;

function pause(ms: number): void {}

namespace control {
    export function runInBackground(a: () => void): void {
        a();
    }
}

function main() {
    print("test top level code");
    let xsum = 0;
    let forclean = () => {};
    for (let i = 0; i < 11; ++i) {
        xsum = xsum + i;
        // TODO: this code breaks execution, i think because of variable i in stack which is cleared after exiting
        forclean = () => {
            i = 0;
        };
    }
    forclean();
    forclean = null;
    assert(xsum == 55, "mainfor");

    control.runInBackground(() => {
        xsum = xsum + 10;
    });

    pause(20);
    assert(xsum == 65, "mainforBg");
    xsum = 0;

    assert(xyz == 12, "init");

    function incrXyz() {
        xyz++;
        return 0;
    }
    let unusedInit = incrXyz();

    assert(xyz == 13, "init2");
    xyz = 0;

    for (let e of [""]) {
    }

    s2 = "";
    for (let i = 0; i < 3; i++) {
        // TODO: this code breaks execution, i think because of variable i or copy in stack which is cleared after exiting
        let copy = i;
        control.runInBackground(() => {
            pause(10 * copy + 1);
            s2 = s2 + copy;
        });
    }
    pause(200);
    assert(s2 == "012");
    print("done.");
}
