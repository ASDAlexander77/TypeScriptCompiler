// A `return`, `break` or `continue` out of a catch clause through a `try` with a `finally` nested in
// it, when the clause's `try` has a `finally` too. The catch used to be ended at the jump itself, so
// the nested finally's copy for the jump - and the release of a block of the clause wrapped for its
// unwind cleanup (an owning local, a `using`) - ran after the catch had ended, while its calls still
// unwound to pads nested in the catch: on Windows the module did not verify ("A single unwind edge
// may only enter one EH pad"), with a `using` in every memory model. Under rc such a clause owned
// nothing and leaked its locals and caught value. Now the catches around the nested try end where
// its finally's copy ends.
class R { constructor(public v: number, public s: string) {} }
@noinline function mk(i: number): R { return new R(i, "r" + i); }
@noinline function thr(i: number) { if (i >= 0) throw mk(i); }

let trail = "";
let disposed = 0;

class D {
    constructor(public tag: string) {}
    [Symbol.dispose]() { disposed++; trail += "d" + this.tag + ";"; }
}

// the shape spec 99 left unowned: a return through the nested finally
function returnThroughNested(i: number): number {
    try {
        thr(i);
    } catch {
        let b = mk(i);
        try {
            return b.v;
        } finally {
            trail += "i;";
        }
    } finally {
        trail += "o;";
    }
    return -1;
}

// the same with a `using` in the clause
function returnWithUsing(i: number): number {
    try {
        thr(i);
    } catch (e) {
        using d = new D("u");
        try {
            return (e as R).v + 1;
        } finally {
            trail += "i;";
        }
    } finally {
        trail += "o;";
    }
    return -1;
}

// two nested tries with finallys in the clause
function returnThroughTwo(i: number): number {
    try {
        thr(i);
    } catch (e) {
        const a = mk(i + 1);
        try {
            const b = mk(i + 2);
            try {
                return a.v + b.v + (e as R).v;
            } finally {
                trail += "1;";
            }
        } finally {
            trail += "2;";
        }
    } finally {
        trail += "o;";
    }
    return -1;
}

// a catch nested in a catch, both of tries with a finally
function returnFromInnerCatch(i: number): number {
    try {
        thr(i);
    } catch (e) {
        const a = mk(i);
        try {
            thr(i + 1);
        } catch (e2) {
            const b = mk(i + 2);
            try {
                return a.v + b.v + (e2 as R).v;
            } finally {
                trail += "f;";
            }
        } finally {
            trail += "m;";
        }
    } finally {
        trail += "o;";
    }
    return -1;
}

// break and continue out of the clause through the nested finally
function jumpsThroughNested(n: number): number {
    let s: number = 0;
    for (let i = 0; i < n; i++) {
        try {
            thr(i);
        } catch {
            const b = mk(i);
            try {
                if (i == 1) {
                    continue;
                }

                if (i == 3) {
                    break;
                }

                s += b.v;
            } finally {
                trail += "j" + i + ";";
            }
        } finally {
            trail += "o" + i + ";";
        }
    }

    return s;
}

// an exception out of the nested finally on the way back from the `return`: the outer finally runs
// and the caller catches it
function throwFromNestedFinally(i: number): number {
    try {
        thr(i);
    } catch {
        const b = mk(i);
        try {
            return b.v;
        } finally {
            trail += "i;";
            thr(i + 50);
        }
    } finally {
        trail += "o;";
    }
    return -1;
}

function main() {
    for (let i = 0; i < 200; i++) {
        trail = "";
        assert(returnThroughNested(i) == i && trail == "i;o;", "returnThroughNested");

        trail = "";
        disposed = 0;
        assert(returnWithUsing(i) == i + 1 && trail == "i;du;o;" && disposed == 1, "returnWithUsing");

        trail = "";
        assert(returnThroughTwo(i) == 3 * i + 3 && trail == "1;2;o;", "returnThroughTwo");

        trail = "";
        assert(returnFromInnerCatch(i) == 3 * i + 3 && trail == "f;m;o;", "returnFromInnerCatch");

        trail = "";
        assert(jumpsThroughNested(10) == 0 + 2 && trail == "j0;o0;j1;o1;j2;o2;j3;o3;", "jumpsThroughNested");

        trail = "";
        let caught: number = -1;
        try {
            throwFromNestedFinally(i);
        } catch (e) {
            caught = (e as R).v;
        }
        assert(caught == i + 50 && trail == "i;o;", "throwFromNestedFinally");
    }

    print("done.");
}
