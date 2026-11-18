// A `finally` that reads what the frame holds keeps it: its unwind copy then let the exception go
// on out of the function with nothing giving the value back, and a call in it that threw - the
// exception taking over from the one unwinding - lost it the same way (-mm=rc and own,
// UnwindReleasePass). `const s = "x" + i; try { f(i); } finally { if (s.length > 0) g(i); }`,
// 2M throws: 88 MB under rc, 42 MB now.
//
// What a test can see is that the finally still reads the right value, the exception that wins is
// the right one, and nothing is freed twice; the memory is in the PR.

let finallies = 0;
let disposed = 0;

class Res {
    [Symbol.dispose]() {
        disposed++;
    }
}

function f(i: number): number {
    if (i >= 0) throw i;
    return 1;
}

function g(i: number): number {
    if (i >= 0) throw i + 1000;
    return 1;
}

// the finally reads the value, and the exception goes on
function finallyReads(i: number) {
    const s = "x" + i;
    try {
        f(i);
    } finally {
        if (s.length > 0) finallies++;
    }
    return -1;
}

// the finally reads the value, and a call in it throws: its exception wins
function finallyThrows(i: number) {
    const s = "y" + i;
    try {
        f(i);
    } finally {
        if (s.length > 0) g(i);
    }
    return -1;
}

// a local, read in the finally, which then does not throw this time
function finallyNoThrow(i: number) {
    let s = "z" + i;
    try {
        f(-1);
    } finally {
        if (s.length > 0) g(-1);
    }
    return s.length;
}

// in a loop: the value made each round is given back each round
function inLoop(i: number) {
    let t: number = 0;
    for (let k = 0; k < 3; k++) {
        const s = "w" + (i + k);
        try {
            try {
                f(k - 1);
            } finally {
                t += s.length;
            }
        } catch (e) {
            t -= s.length;
        }
    }
    // only round 0 does not throw: its finally's count is the one left
    return t - ("w" + i).length;
}

// a `using` disposed on the way, and a finally that reads a value, in the caller
function caller(i: number) {
    const s = "v" + i;
    try {
        using r = new Res();
        return finallyThrows(i);
    } catch (e) {
        return <number>e - 1000 + (s.length > 0 ? 0 : 1);
    } finally {
        if (s.length == 0) finallies--;
    }
}

function main() {
    const n = 20000;
    for (let i = 0; i < n; i++) {
        let t: number = 0;
        try { finallyReads(i); t += 1; } catch (e) { t += <number>e - i; }
        try { finallyThrows(i); t += 1; } catch (e) { t += <number>e - (i + 1000); }
        t += finallyNoThrow(i) - ("z" + i).length;
        t += inLoop(i);
        t += caller(i) - i;
        if (t != 0) {
            print("wrong at", i, t);
            assert(false);
        }
    }

    assert(finallies == n);
    assert(disposed == n);
    print("done.");
}
