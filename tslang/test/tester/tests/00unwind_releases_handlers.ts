// An exception gives back what the frame holds wherever it starts (-mm=rc and own): a call in a
// catch clause, a call in a `try` whose handler does not take the exception for good (a `finally`,
// a catch that throws), and a `throw` statement itself (UnwindReleasePass, OwnedReturnConsumptionPass).
// Each of these leaked a temporary, a folded `const` or the caught value per throw - 2M throws
// through `catch (e) { const s = "x" + i; f(i); }`: 239 MB under rc, 41 MB now.
//
// What a test can see is that the values stay right - a handler that still reads a value must find
// it there, and nothing is freed twice - and that a `finally` and a `using` still run.

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

// a call in a catch clause that throws out of the function: the clause's local and the caught value
function inCatch(i: number) {
    try {
        f(i);
    } catch (e) {
        const s = "x" + i;
        if (s.length > 0) f(i + 7);
    }
    return -1;
}

// a value held across a `try` whose `finally` lets the exception go on
function throughFinally(i: number) {
    const s = "y" + i;
    try {
        f(i);
    } finally {
        finallies++;
    }
    return s.length;
}

// a clause's local, caught by a `try` around in the same function
function nested(i: number) {
    let r: number = 0;
    try {
        try {
            f(i);
        } catch (e) {
            const s = "z" + i;
            if (s.length > 0) f(i + 1);
        }
    } catch (e2) {
        r = <number>e2 - i;
    }
    return r;
}

// the handler still reads the value: it is not given back on the way in
function handlerReads(i: number) {
    const s = "w" + i;
    try {
        const q = "q" + i;
        f(i);
        return q.length;
    } catch (e) {
        if (s.length > 0) throw e;
    }
    return -1;
}

// a `throw` statement with a folded const and a temporary held
function throwHolding(i: number) {
    const s = "v" + i;
    if (s.length > 0) throw i + 2;
    return -1;
}

// a `throw` in a catch clause, a value from outside the `try` held
function throwInCatch(i: number) {
    const s = "u" + i;
    try {
        f(i);
    } catch (e) {
        if (s.length > 0) throw 5;
    }
    return -1;
}

// a `try` in a loop: the value lives on after the catch and is given back once, at its end
function inLoop(i: number) {
    let t: number = 0;
    for (let k = 0; k < 3; k++) {
        const s = "t" + (i + k);
        let caught = false;
        try {
            f(k - 1);
        } catch (e) {
            caught = true;
            t += s.length;
        }
        if (caught) t -= s.length;
    }
    return t;
}

// a `using` and a `catch` in the caller around a call in a catch clause
function caller(i: number) {
    try {
        using r = new Res();
        return inCatch(i);
    } catch (e) {
        return <number>e - 7;
    }
}

function main() {
    const n = 20000;
    for (let i = 0; i < n; i++) {
        let t: number = 0;
        try { inCatch(i); t += 1; } catch (e) { t += <number>e - (i + 7); }
        try { throughFinally(i); t += 1; } catch (e) { t += <number>e - i; }
        t += nested(i) - 1;
        try { handlerReads(i); t += 1; } catch (e) { t += <number>e - i; }
        try { throwHolding(i); t += 1; } catch (e) { t += <number>e - (i + 2); }
        try { throwInCatch(i); t += 1; } catch (e) { t += <number>e - 5; }
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
