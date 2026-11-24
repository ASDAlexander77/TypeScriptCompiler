// TSLANG 176: a `for (let i ...)` body that assigns `i` still gives each closure the `i` of its iteration (it printed
// 3 3 3), its value carried to the next iteration at the end of the body and at a `continue`; a closure in a default
// value of a parameter that reads an earlier parameter (`(a = 5, f = () => a)`) crashed the program

function g(a = 5, f = () => a) {
    return f();
}

function h(a: number, f = () => () => a) {
    return f()();
}

function main() {
    const fs: (() => number)[] = [];
    for (let i = 0; i < 3; i++) {
        fs.push(() => i);
        if (i > 10) i = 0;
    }

    assert(fs[0]() == 0 && fs[1]() == 1 && fs[2]() == 2);

    const gs: (() => number)[] = [];
    for (let i = 0; i < 6; i++) {
        if (i == 1) {
            i = 2;
            continue;
        }

        gs.push(() => i);
        i++;
    }

    assert(gs.length == 3);
    assert(gs[0]() == 1 && gs[1]() == 3 && gs[2]() == 5);

    assert(g() == 5);
    assert(g(7) == 7);
    assert(h(4) == 4);

    print("done.");
}
