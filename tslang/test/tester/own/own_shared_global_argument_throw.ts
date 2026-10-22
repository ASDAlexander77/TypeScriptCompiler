// A Shared<T> global given to a call that may assign it is held by the caller across the call
// (own_shared_global_argument.ts). When the call throws instead of returning, the hold is given
// back on the way to the handler: before, only the normal path released it, and every throw left
// the global's block one count it never lost. Counts are checked where there is one (rc, own);
// gc and none report -1.
class Node {
    v = 0;
    tag = "";
}

function countIs(s: Shared<Node>, expected: number) {
    const c = Shared.count(s);
    return c < 0 || c == expected;
}

function mk(tag: string, v: number) {
    const n = new Node();
    n.tag = tag;
    n.v = v;
    return n;
}

let g = new Shared(mk("g", 1));

// may assign the global, so a caller holds what it gives it; throws before it does
function maybeAssign(h: Shared<Node>, assign: boolean) {
    if (assign) {
        g = new Shared(mk("replaced", 2));
    }

    if (h.value.v > 0) {
        throw "thrown";
    }

    return h.value.v;
}

function inTryCatch() {
    try {
        maybeAssign(g, false);
    } catch (e) {
        assert(countIs(g, 1), "the hold given back before the catch runs");
    }

    assert(countIs(g, 1), "try/catch");
}

function inTryFinally() {
    try {
        maybeAssign(g, false);
    } finally {
        assert(countIs(g, 1), "the hold given back before the finally runs");
    }
}

function throughFinally() {
    try {
        inTryFinally();
    } catch (e) {
    }

    assert(countIs(g, 1), "try/finally");
}

function inBlockWithLocal() {
    try {
        {
            const local = mk("local", 3);
            maybeAssign(g, false);
            print(local.v);
        }
    } catch (e) {
    }

    assert(countIs(g, 1), "a block with a local in a try body");
}

function inCatchWithFinally() {
    let caught = 0;
    try {
        try {
            throw "first";
        } catch (e) {
            maybeAssign(g, false);
        } finally {
            caught++;
        }
    } catch (e) {
        caught++;
    }

    assert(caught == 2 && countIs(g, 1), "a call in a catch clause with a finally");
}

function inLoop() {
    for (let i = 0; i < 3; i++) {
        try {
            maybeAssign(g, false);
        } catch (e) {
        }
    }

    assert(countIs(g, 1), "a try in a loop");
}

function main() {
    inTryCatch();
    throughFinally();
    inBlockWithLocal();
    inCatchWithFinally();
    inLoop();

    // and the normal path still holds across an assignment
    let ok = false;
    try {
        g = new Shared(mk("neg", -1));
        ok = maybeAssign(g, true) == -1;
    } catch (e) {
    }

    assert(ok && countIs(g, 1) && g.value.tag == "replaced", "the normal path");
    print("done.");
}
