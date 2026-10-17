// An empty catch clause catches. An untyped `catch { }` emits no ops into the try's catches region,
// and the lowering took a region holding only its terminator for "no catch clause" - the form of the
// try a scope's cleanup is wrapped in - so it made no landing pad and the exception went uncaught.
class Node {
    v = 0;
}

function h() {
    throw 1;
}

function emptyCatch() {
    try {
        h();
    } catch {
    }
}

function emptyCatchWithVariable() {
    try {
        h();
    } catch (e) {
    }
}

function emptyStatementInCatch() {
    try {
        h();
    } catch {
        ;
    }
}

function emptyCatchThenFinally() {
    let finallyRan = false;
    try {
        h();
    } catch {
    } finally {
        finallyRan = true;
    }

    return finallyRan;
}

function emptyCatchWithOwnedLocal() {
    try {
        let n = new Node();
        n.v = 1;
        h();
    } catch {
    }
}

function nestedEmptyCatches() {
    let reached = false;
    try {
        try {
            h();
        } catch {
        }

        reached = true;
        h();
    } catch {
    }

    return reached;
}

function emptyCatchInLoop() {
    let count = 0;
    for (let i = 0; i < 5; i++) {
        try {
            h();
        } catch {
        }

        count++;
    }

    return count;
}

function noThrowEmptyCatch() {
    let value = 0;
    try {
        value = 7;
    } catch {
    }

    return value;
}

class C {
    run() {
        try {
            h();
        } catch {
        }

        return true;
    }
}

function main() {
    emptyCatch();
    emptyCatchWithVariable();
    emptyStatementInCatch();
    assert(emptyCatchThenFinally(), "the finally after an empty catch runs");
    emptyCatchWithOwnedLocal();
    assert(nestedEmptyCatches(), "an inner empty catch lets the outer try go on");
    assert(emptyCatchInLoop() == 5, "an empty catch in a loop");
    assert(noThrowEmptyCatch() == 7, "a try that does not throw");
    assert(new C().run(), "an empty catch in a method");

    print("done.");
}
