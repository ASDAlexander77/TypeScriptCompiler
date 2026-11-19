// A catch or finally clause that holds a `try` of its own: its locals and its caught value own what
// they hold, as in any other clause - they used to own nothing there, and leaked under -mm=rc on
// every pass (`finally { try {} finally { const a = [mk(1)]; } }`).
//
// Such a clause's block is wrapped for its unwind cleanup, and inside a `try` that also has a
// `finally` that wrapping left the Windows catch funclet with an edge out of it to the finally and a
// catchswitch that unwound to the caller: the module did not verify ("Unwind edges out of a catch
// must have the same unwind dest as the parent catchswitch"). A `using` in such a clause did the
// same before. The exceptions below go through those clauses to where they belong, and each
// `finally` runs once.
class R { constructor(public v: number, public s: string) {} }
@noinline function mk(i: number): R { return new R(i, "r" + i); }
@noinline function thr(i: number) { if (i >= 0) throw mk(i); }

let trail = "";

// locals of a finally that holds a try: an array literal, a reassigned local
function finallyLocals(i: number): number {
    let r: number = 0;
    try {
        r = 1;
    } finally {
        try {
            r += 1;
        } finally {
            const a: R[] = [mk(i), mk(i + 1)];
            let k = mk(i);
            k = mk(i + 2);
            r += a[1].v + k.v;
        }
    }
    return r;
}

// the caught value and a local of a catch that holds a try, which catches again
function caughtInCatchWithTry(i: number): number {
    try {
        thr(i);
    } catch (e) {
        const k = mk(i);
        try {
            thr(i + 1);
        } catch (e2) {
            return k.v + (e2 as R).v + (e as R).v;
        }
    }
    return -1;
}

// a return through the nested try's finally, then the outer finally
function returnThroughFinallys(i: number): number {
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

// an exception out of the nested try in the catch: the outer finally runs, the caller catches it
function throwThroughFinally(i: number): number {
    try {
        thr(i);
    } catch {
        const b = mk(i);
        try {
            thr(b.v + 10);
        } finally {
            trail += "i;";
        }
    } finally {
        trail += "o;";
    }
    return -1;
}

// the same with a nested catch that throws again (a new object: under -mm=own `throw e2` in a try
// with a finally is refused at run time, nested or not - it copies the caught `any`)
function rethrowFromNestedCatch(i: number): number {
    try {
        thr(i);
    } catch (e) {
        const b = mk(i);
        try {
            thr(b.v + 20);
        } catch (e2) {
            trail += "c;";
            throw mk((e2 as R).v);
        }
    } finally {
        trail += "o;";
    }
    return -1;
}

// a finally with a try in it, inside a catch, inside a try with a finally
function finallyInCatch(i: number): number {
    let r: number = 0;
    try {
        thr(i);
    } catch {
        try {
            r = 1;
        } finally {
            const k = mk(i);
            try {
                r += k.v;
            } finally {
                trail += "f;";
            }
        }
    } finally {
        trail += "o;";
    }
    return r;
}

function main() {
    for (let i = 0; i < 200; i++) {
        assert(finallyLocals(i) == 2 + (i + 1) + (i + 2), "finallyLocals");
        assert(caughtInCatchWithTry(i) == 3 * i + 1, "caughtInCatchWithTry");

        trail = "";
        assert(returnThroughFinallys(i) == i && trail == "i;o;", "returnThroughFinallys");

        trail = "";
        let caught: number = -1;
        try {
            throwThroughFinally(i);
        } catch (e) {
            caught = (e as R).v;
        }
        assert(caught == i + 10 && trail == "i;o;", "throwThroughFinally");

        trail = "";
        caught = -1;
        try {
            rethrowFromNestedCatch(i);
        } catch (e) {
            caught = (e as R).v;
        }
        assert(caught == i + 20 && trail == "c;o;", "rethrowFromNestedCatch");

        trail = "";
        assert(finallyInCatch(i) == 1 + i && trail == "f;o;", "finallyInCatch");
    }

    print("done.");
}
