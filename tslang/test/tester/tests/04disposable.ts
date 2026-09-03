// 03disposable.ts covers the shape that first exposed the gap: a `using` at a function's own
// top level, unwinding with no enclosing try. This file covers the scopes that were left out
// then and are handled now - a `using` in a nested block, in a loop body, and one that shares
// its function with a `return`.

let disposed = 0;

class Res {
    [Symbol.dispose]() {
        disposed = disposed + 1;
    }
}

// a `using` inside an if-block, not the function's own body
function inNestedBlock(f: boolean) {
    if (f) {
        using r = new Res();
        throw 1;
    }
}

// a `using` inside a loop body
function inLoopBody() {
    for (let i = 0; i < 1; i++) {
        using r = new Res();
        throw 1;
    }
}

// two scopes deep, to show it is not just one level
function inDeepBlock() {
    for (let i = 0; i < 2; i++) {
        if (i == 1) {
            using r = new Res();
            throw 1;
        }
    }
}

// a function that both throws past a `using` and returns normally past one: the unwind path
// and the ordinary path each owe exactly one dispose, not two and not none
function throwsOrReturns(f: boolean) {
    using r = new Res();
    if (f) {
        throw 1;
    }

    return;
}

// the synthesized cleanup nests inside a hand-written try without disturbing it
function insideHandWrittenTry() {
    try {
        using r = new Res();
        throw 1;
    }
    catch (e: TypeOf<1>) {
        print("caught inner");
    }
}

// A `using` inside a catch clause. Synthesizing the unwind cleanup here crashes the compiler,
// so this block must be left on the plain dispose path - blockIsInsideCatchOrFinally. Covered
// because dropping the old root-body condition briefly re-enabled the wrapping here and no
// test noticed.
function insideCatchClause() {
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        using r = new Res();
        print("in catch");
    }
}

// the same for a finally clause
function insideFinallyClause() {
    try {
        print("try body");
    }
    finally {
        using r = new Res();
        print("in finally");
    }
}

function expectThrow(f: () => void) {
    try {
        f();
    }
    catch (e: TypeOf<1>) {
        return true;
    }

    return false;
}

function main() {
    disposed = 0;
    assert(expectThrow(() => inNestedBlock(true)), "nested block must throw");
    assert(disposed == 1, "using in a nested block must dispose while unwinding");

    disposed = 0;
    assert(expectThrow(() => inLoopBody()), "loop body must throw");
    assert(disposed == 1, "using in a loop body must dispose while unwinding");

    disposed = 0;
    assert(expectThrow(() => inDeepBlock()), "deep block must throw");
    assert(disposed == 1, "using two scopes deep must dispose while unwinding");

    disposed = 0;
    assert(expectThrow(() => throwsOrReturns(true)), "throwing path must throw");
    assert(disposed == 1, "a using sharing its function with a return must still dispose on throw");

    disposed = 0;
    throwsOrReturns(false);
    assert(disposed == 1, "the ordinary return path must dispose exactly once");

    disposed = 0;
    insideHandWrittenTry();
    assert(disposed == 1, "a using inside a hand-written try must dispose exactly once");

    disposed = 0;
    insideCatchClause();
    assert(disposed == 1, "a using inside a catch clause must dispose exactly once");

    disposed = 0;
    insideFinallyClause();
    assert(disposed == 1, "a using inside a finally clause must dispose exactly once");

    print("done.");
}
