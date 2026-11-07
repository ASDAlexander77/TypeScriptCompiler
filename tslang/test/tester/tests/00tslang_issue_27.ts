// TSLANG_ISSUES 27: a template literal passed straight to a call is a new string the caller holds
// until the call returns. Under rc it arrived with no reference behind it, so a callee handing its
// parameter to a call first (console.log, whose rest array retains and releases it) dropped the
// count to zero and freed the string before it stored it.

const NAMES = ["down", "off", "point", "tilt"];

// stands for console.log: the rest array holds what it is given and lets go of it on return
function log(...parts: string[]) {
    print(parts[0]);
}

class Hud {
    message: string = "";
    show(text: string) {
        log(text);
        this.message = text;
    }
}

let kept = "";
function keep(text: string) {
    log(text);
    kept = text;
}

function main() {
    const hud = new Hud();
    for (let i = 0; i < 8; i++) {
        hud.show(`Gravity: ${NAMES[i % 4]}`);
        // a fresh string made in the block reuses what was freed
        const other = `XXXXXXXX: ${NAMES[(i + 1) % 4]}`;
        assert(hud.message == "Gravity: " + NAMES[i % 4], "the method kept its template argument");
        assert(other == "XXXXXXXX: " + NAMES[(i + 1) % 4], "the next template");

        keep(`Keep ${i}: ${NAMES[i % 4]}`);
        const again = `YYYYYYYY: ${NAMES[(i + 2) % 4]}`;
        assert(kept == "Keep " + i + ": " + NAMES[i % 4], "the function kept its template argument");
        assert(again.length == 10 + NAMES[(i + 2) % 4].length, "the template after it");

        // an object or a literal array printed into a string is a new string too
        keep(<string>{ a: i });
        const third = `ZZZZZZZZ: ${NAMES[(i + 3) % 4]}`;
        assert(kept == "{ a: " + i + " }", "the function kept an object printed into a string");
        assert(third.length == 10 + NAMES[(i + 3) % 4].length, "the template after the object");

        hud.show(<string>[1, 2]);
        const fourth = `WWWWWWWW: ${NAMES[i % 4]}`;
        assert(hud.message == "[ 1, 2 ]", "the method kept a literal array printed into a string");
        assert(fourth.length == 10 + NAMES[i % 4].length, "the template after the array");
    }

    print("done.");
}
