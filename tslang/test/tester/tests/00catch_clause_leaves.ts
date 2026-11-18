// A catch clause left by a throw, with more inside it than a plain call: a `using` and locals of
// its own, a `return` on another path, a throw the inliner brought in.
//
// - A block in a catch or finally clause got no cleanup of its own (blockIsInsideCatchOrFinally),
//   so a `using` there was never disposed when the clause threw, and its heap locals leaked.
// - With one, a throw laid out after a `return`'s catchret had no funclet bundle on Windows, and
//   the program crashed (Win32ExceptionPass, findOtherCatchExits).
// - A throw the inliner brought into one branch of a catch clause, `catch (e) { f(i + 2); }` with
//   `f` inlined, had no funclet bundle either: codegen dropped the call, and the funclet ran on
//   into the next function. -O3 only, so the JIT's tests at -O3 found it.

let disposed = 0;

class Res {
    [Symbol.dispose]() {
        disposed++;
    }
}

function f(i: number) {
    if (i >= 0) throw i;
    return 1;
}

// the call in the catch throws out of the function (inlined at -O3)
function callOut(i: number) {
    try { f(i); } catch (e) { f(i + 2); }
    return -1;
}

// a using in the clause, which throws out of the function
function usingOut(i: number) {
    try { f(i); } catch (e) {
        using r = new Res();
        f(i + 6);
    }
    return -1;
}

// a local, a return on one path and a throw on the other, a try around takes the throw
function localReturnOrThrow(i: number) {
    try {
        try { f(i); } catch (e) {
            const n = <number>e;
            if (n < 0) return 2;
            throw n + 5;
        }
    } catch (e2) { return <number>e2; }
    return 0;
}

// the same with a using: disposed on both paths
function usingReturnOrThrow(i: number, k: number) {
    try {
        try { f(i); } catch (e) {
            using r = new Res();
            if (k > 0) return 2;
            throw 5;
        }
    } catch (e2) { return <number>e2; }
    return 0;
}

// a heap local in the clause, which throws out of the function
function heapLocalOut(i: number) {
    try { f(i); } catch (e) {
        const s = "x" + i;
        if (s.length > 0) f(i + 7);
    }
    return -1;
}

function main() {
    const n = 1000;
    for (let i = 0; i < n; i++) {
        let t: number = 0;
        try { callOut(i); t += 1; } catch (e) { t += <number>e - (i + 2); }
        try { usingOut(i); t += 1; } catch (e) { t += <number>e - (i + 6); }
        t += localReturnOrThrow(i) - (i + 5);
        t += usingReturnOrThrow(i, 1) - 2;
        t += usingReturnOrThrow(i, 0) - 5;
        try { heapLocalOut(i); t += 1; } catch (e) { t += <number>e - (i + 7); }
        if (t != 0) {
            print("wrong at", i, t);
            assert(false);
        }
    }

    assert(disposed == 3 * n);
    print("done.");
}
