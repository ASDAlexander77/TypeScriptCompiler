// Every way out of a catch ends it, the unwind paths included (Itanium: ItaniumEndCatchPass).
//
// A call in a catch clause that threw, or the rethrow a finally, a `using` or a typed catch that
// does not match ends with, left the catch it unwound out of begun: the exception it held was
// never freed, and the caught-exceptions stack grew by one each time. 2.7M of any of these shapes
// took 480 MB, in every memory model. What a test can see of it is that the values, the order and
// the finally counts stay right after many rounds, with no terminate; the memory is in the PR.

function f(i: number) {
    if (i >= 0) throw i;
    return 1;
}

function fs(i: number) {
    if (i >= 0) throw "s" + i;
    return 1;
}

let fin = 0;
let disposed = 0;

class Res {
    [Symbol.dispose]() {
        disposed++;
    }
}

// a call in the catch clause throws, a try around takes it
function callInCatch(i: number) {
    try {
        try { f(i); } catch (e) { f(i + 1); return 0; }
    } catch (e2) { return <number>e2; }
    return -1;
}

// a finally's rethrow
function finallyRethrow(i: number) {
    try {
        try { f(i); } finally { fin++; }
    } catch (e2) { return <number>e2; }
    return -1;
}

// a typed catch that does not match rethrows
function mismatch(i: number) {
    try {
        try { f(i); } catch (e: string) { return -2; }
    } catch (e2) { return <number>e2; }
    return -1;
}

// a call in the catch clause throws out of the function
function callInCatchOut(i: number) {
    try { f(i); } catch (e) { f(i + 2); }
    return -1;
}

// the mismatch rethrows out of the function
function mismatchOut(i: number) {
    try { f(i); } catch (e: string) { return -2; }
    return -1;
}

// the catch clause throws into its own finally, a try around takes it
function callInCatchFinally(i: number) {
    try {
        try { f(i); } catch (e) { f(i + 3); } finally { fin++; }
    } catch (e2) { return <number>e2; }
    return -1;
}

// the same, out of the function
function callInCatchFinallyOut(i: number) {
    try { f(i); } catch (e) { f(i + 4); } finally { fin++; }
    return -1;
}

// a try in a catch clause whose own catch throws
function nestedInCatch(i: number) {
    try {
        try { f(i); } catch (e) {
            try { f(i); } catch (e3) { f(i + 5); }
        }
    } catch (e2) { return <number>e2; }
    return -1;
}

// a throw a try nested in the catch clause takes: the outer catch carries on, then ends once
function throwTakenInCatch(i: number) {
    let r: number = 0;
    try { f(i); } catch (e) {
        try { throw 7; } catch (x) { r = <number>x; }
        r += <number>e;
    }
    return r;
}

// `throw e` out of the catch clause
function rethrowCaught(i: number) {
    try {
        try { fs(i); } catch (e) { throw e; }
    } catch (e2) { return (<string>e2).length; }
    return -1;
}

// a using's disposal on the way out, a try around takes it
function usingOut(i: number) {
    try {
        using r = new Res();
        f(i);
    } catch (e2) { return <number>e2; }
    return -1;
}

// a using in a catch clause whose clause throws: disposed on the way out
function usingInCatch(i: number) {
    try {
        try { f(i); } catch (e) {
            using r = new Res();
            f(i + 6);
        }
    } catch (e2) { return <number>e2; }
    return -1;
}

function main() {
    const n = 20000;
    let s: number = 0;
    for (let i = 0; i < n; i++) {
        let t: number = 0;
        t += callInCatch(i) - (i + 1);
        t += finallyRethrow(i) - i;
        t += mismatch(i) - i;
        try { callInCatchOut(i); t += 1; } catch (e) { t += <number>e - (i + 2); }
        try { mismatchOut(i); t += 1; } catch (e) { t += <number>e - i; }
        t += callInCatchFinally(i) - (i + 3);
        try { callInCatchFinallyOut(i); t += 1; } catch (e) { t += <number>e - (i + 4); }
        t += nestedInCatch(i) - (i + 5);
        t += throwTakenInCatch(i) - (7 + i);
        t += rethrowCaught(i) - ("s" + i).length;
        const before = disposed;
        t += usingOut(i) - i;
        t += usingInCatch(i) - (i + 6);
        t += disposed - before - 2;
        if (t != 0) {
            print("wrong at", i, t);
            assert(false);
        }

        s += t;
    }

    assert(s == 0);
    assert(fin == 3 * n);
    assert(disposed == 2 * n);
    print("done.");
}
