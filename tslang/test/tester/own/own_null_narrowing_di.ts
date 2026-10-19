// @strict-null false
// -mm=own with debug info: in `if (r == null)` the narrowed `r` is recorded for the debugger as a
// cast of the class value to `null`, which holds no block and takes nothing. It was read as a
// second reference to the call's fresh result (the default library's String.match, built --di).
export class R {
    constructor(public v: int) {}
}

function mk(i: int): R | null {
    return i % 2 == 0 ? new R(i) : null;
}

export function f(i: int): int {
    const r = mk(i);
    if (r == null) {
        return 0;
    }

    return r.v;
}

function main() {
    let t = 0;
    for (let i = 0; i < 10; i++) {
        t += f(i);
    }

    assert(t == 0 + 2 + 4 + 6 + 8);
    print("done.");
}
