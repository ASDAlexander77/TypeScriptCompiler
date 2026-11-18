// The caught value of a catch clause with a `finally`, or inside a `try` of the same function: an
// exception out of a call in the clause leaves it there, and the finally's unwind copy, or the
// cleanup of the `try` around, gives it back (its slot emptied after every release, so it is given
// back once). Before, such a clause did not own its value at all, and every throw leaked it. Each
// value is read after churn() has reused anything freed too early.
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

let finallies = 0;

function withFinally(i: number) {
    try {
        throwString(i);
    } catch (e) {
        churn();
        return (<string>e).length;
    } finally {
        finallies++;
    }

    return 0;
}

function rethrowWithFinally(i: number) {
    try {
        throwString(i);
    } catch (e) {
        throw e;
    } finally {
        finallies++;
    }
}

function callThrowsWithFinally(i: number) {
    try {
        throwString(i);
    } catch (e) {
        const k = (<string>e).length;
        throwClass(i);
        return k;
    } finally {
        finallies++;
    }

    return 0;
}

function returnedWithFinally(i: number): any {
    try {
        throwClass(i);
    } catch (e) {
        return e;
    } finally {
        finallies++;
    }

    return null;
}

function nested(i: number) {
    try {
        try {
            throwString(i);
        } catch (e) {
            return (<string>e).length;
        }
    } catch (e2) {
        return -1;
    }

    return 0;
}

function nestedCallThrows(i: number) {
    try {
        try {
            throwString(i);
        } catch (e) {
            throwClass(i);
            return 0;
        }
    } catch (e2) {
        churn();
        return (<E>e2).n;
    }

    return 0;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 100; i++) {
        s += withFinally(i);
        try {
            rethrowWithFinally(i);
        } catch (e2) {
            churn();
            s += (<string>e2).length;
        }

        try {
            s += callThrowsWithFinally(i);
        } catch (e3) {
            churn();
            s += (<E>e3).n;
        }

        const r = returnedWithFinally(i);
        churn();
        s += (<E>r).msg.length;
        s += nested(i) + nestedCallThrows(i);
    }

    // lengths of "x" + i and "m" + i: 10 of 2 and 90 of 3 = 290; 0 + ... + 99 = 4950
    assert(s == 290 + 290 + 4950 + 290 + 290 + 4950, "every value read after churn");
    assert(finallies == 400, "every finally ran");
    print("done.");
}
