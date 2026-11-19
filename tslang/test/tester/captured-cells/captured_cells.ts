// A captured variable's storage is a cell, which each scope that declares one gives back. Two paths
// lost it under rc (read by captured-cells.cmake; it also runs in every model):
//  - a `for (let i ...)` body that captures `i` has an `i` of its own each iteration, declared before
//    the body's scope existed and listed nowhere: every iteration leaked its cell;
//  - a `return` written before the closure that captures a parameter released the parameter's value
//    and not its cell, which the variable only became once the closure was generated.

class N {
    constructor(public x: number) {}
}

@noinline function mk(i: number): N {
    return new N(i);
}

function loopCapture(): string {
    let s = "";
    const kept: (() => string)[] = [];
    for (let i = 0; i < 4; i++) {
        const f = (): string => "c" + i;
        s += f();
        if (i == 1) continue;
        if (i == 3) break;
        kept.push((): string => "k" + i);
    }

    for (const k of kept) {
        s += k();
    }

    return s;
}

@noinline function earlyReturn(p: N): N {
    p = mk(8);
    if (p.x > 7) {
        return mk(5);
    }

    const c = (): number => {
        p = mk(7);
        return p.x;
    };
    c();
    return p;
}

function main() {
    for (let r = 0; r < 100; r++) {
        assert(loopCapture() == "c0c1c2c3k0k2", "loop captures");
        assert(earlyReturn(mk(r)).x == 5, "early return");
    }

    print("done.");
}
