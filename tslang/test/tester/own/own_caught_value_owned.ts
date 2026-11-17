// The catch clause owns what was thrown (USER RULING 2026-11-17): under rc and own the value a
// `throw` hands the exception is released where the clause ends - on a `return`, a `break` or a
// `throw` out of it too - or moved where the clause gives it on, and a `catch {}` releases it as
// well. Before, it was never released at all. Each value is read after churn() has reused anything
// freed too early. Also a number thrown and caught in the same function, read in the clause: under
// the Windows JIT its box thunk could be the image's first function, at RVA 0 - "no copy function" -
// and the variable got the number's bytes instead of a box.
class E {
    constructor(public n: number, public msg: string) {}
}

class H {
    constructor(public x: number) {}
}

function churn() {
    const keep: H[] = [];
    for (let i = 0; i < 20; i++) keep.push(new H(77777));
    return keep.length;
}

function throwString(i: number) {
    if (i >= 0) throw "x" + i;
    return 1;
}

function throwClass(i: number) {
    if (i >= 0) throw new E(i, "m" + i);
    return 1;
}

function caughtString(i: number) {
    try {
        throwString(i);
    } catch (e) {
        churn();
        return (<string>e).length;
    }

    return 0;
}

function caughtClass(i: number) {
    try {
        throwClass(i);
    } catch (e) {
        churn();
        return (<E>e).n + (<E>e).msg.length;
    }

    return 0;
}

function noVariable(i: number) {
    try {
        throwClass(i);
    } catch {
        return 1;
    }

    return 0;
}

function inLoop(n: number) {
    let t = 0;
    for (let i = 0; i < n; i++) {
        try {
            throwString(i);
        } catch (e) {
            if (i == 2) break;
            t += (<string>e).length;
        }
    }

    return t;
}

function rethrow(i: number) {
    try {
        throwString(i);
    } catch (e) {
        throw e;
    }
}

function returned(i: number): any {
    try {
        throwString(i);
    } catch (e) {
        return e;
    }

    return null;
}

const kept: any[] = [];

function localNumber() {
    try {
        throw 42;
    } catch (e) {
        return typeof e == "s32" || typeof e == "number" ? 1 : 0;
    }
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 100; i++) {
        s += caughtString(i) + caughtClass(i) + noVariable(i) + inLoop(5);
        try {
            rethrow(i);
        } catch (e2) {
            churn();
            s += (<string>e2).length;
        }

        const r = returned(i);
        churn();
        s += (<string>r).length;

        try {
            throwString(i);
        } catch (e3) {
            kept.push(e3);
        }

        s += localNumber();
    }

    churn();
    assert(<string>kept[99] == "x99", "a caught value kept past its clause");
    // lengths of "x" + i: 10 of 2 and 90 of 3 = 290
    assert(s == 290 + (4950 + 290) + 100 + 100 * 4 + 290 + 290 + 100, "every value read after churn");
    print("done.");
}
