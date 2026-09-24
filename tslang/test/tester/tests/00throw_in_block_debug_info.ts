// A class thrown from inside a braced block, with debug info on (the suite runs every test with
// --di). The block opens a lexical debug scope, and the catch copy thunk built for the throw
// used to inherit that scope: LLVM then rejected the module ("!dbg attachment points at wrong
// subprogram for function"). The thunk has to be a function of its own to the debugger too.
class Err {
    constructor(public m: string) {}
}

function f(n: number) {
    if (n < 0) {
        throw new Err("negative");
    }

    return n;
}

function g(n: number) {
    let caught = "";
    try {
        f(n);
    } catch (e: Err) {
        caught = e.m;
    }

    return caught;
}

function main() {
    assert(f(1) == 1, "f(1)");
    assert(g(-1) == "negative", "g(-1)");
    assert(g(2) == "", "g(2)");
    print("done.");
}
