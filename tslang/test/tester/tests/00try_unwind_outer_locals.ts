// An exception that leaves a scope gives back what that scope and the scopes around it took, on
// the paths 00using_unwind_outer_locals.ts does not cover.
//
// - out of a `try` with a `finally`: the copy of the finally that runs on the unwind leg releases
//   the scopes around the `try` once it is done, and only that copy - the finally still reads them,
//   and normal completion, a `return` and a `break` through it leave them live;
// - out of a block nested in a `try` body: such a block is wrapped in a catch-less TryOp of its own,
//   since the `try`'s cleanup releases its body's scope and no other;
// - from a `throw` statement, in a `catch` clause or anywhere else: the throw releases the scopes it
//   leaves, up to the nearest `try` body, after taking a reference for the thrown value.
//
// Under -mm=rc each of these leaked; the ownership verifier (--verify-ownership, swept over this
// corpus in ctest) reported them, and is quiet on this file now. The values are read again after
// churn() wherever a release one too many would free them early.
//
// Not covered, and still leaking: a call that throws out of a `catch` clause of a `try` with no
// `finally`, a typed catch that does not match in a `try` with no `finally`, and a call that throws
// where no `try` encloses it - none of those has a landing pad to release from.

let trail = "";

function h() {
    throw 1;
}

// allocates over whatever was freed last, so that a string released one time too many no longer
// reads as itself
function churn() {
    let junk = "";
    for (let i = 0; i < 64; i++) {
        junk = junk + "#";
    }

    return junk.length;
}

function expectThrow(f: () => void) {
    try {
        f();
    }
    catch (e: TypeOf<1>) {
        return true;
    }

    return false;
}

// the finally reads `s` after the exception: the release has to come after it
function throwThroughFinally(pre: string) {
    let s = pre + "1";
    try {
        h();
    }
    finally {
        churn();
        trail += s + ";";
    }

    return s;
}

function throwThroughEmptyFinally(pre: string) {
    let s = pre + "2";
    try {
        h();
    }
    finally {
    }

    return s;
}

// the same finally, completed normally: `s` is still the function's
function completeThroughFinally(pre: string) {
    let s = pre + "3";
    try {
        trail += "body;";
    }
    finally {
        trail += s + ";";
    }

    churn();
    return s;
}

// no unwind-only release in the copy a `return` runs (00try_return_through_finally.ts has more on
// a `return` through a finally)
function returnThroughFinally(pre: string) {
    let s = pre + "4";
    try {
        return s + "r";
    }
    finally {
        churn();
        trail += s + ";";
    }
}

function breakThroughFinally(pre: string) {
    let s = pre + "5";
    for (let i = 0; i < 3; i++) {
        try {
            if (i == 1) {
                break;
            }
        }
        finally {
            trail += s + i + ";";
        }
    }

    churn();
    return s;
}

function throwFromCatch(pre: string) {
    let s = pre + "6";
    try {
        h();
    }
    catch {
        trail += s + ";";
        throw 2;
    }

    return s;
}

function throwFromCatchWithFinally(pre: string) {
    let s = pre + "7";
    try {
        h();
    }
    catch {
        throw 2;
    }
    finally {
        churn();
        trail += s + ";";
    }

    return s;
}

// a try in a try body: the outer one's cleanup releases its own body, so the inner finally leaves it
function finallyInTryBody(pre: string) {
    let s = pre + "8";
    try {
        let t = s + "t";
        try {
            h();
        }
        finally {
            churn();
            trail += t + ";";
        }
    }
    catch {
        trail += "caught;";
    }

    churn();
    return s;
}

function throwThroughNestedScope(pre: string) {
    let s = pre + "9";
    try {
        {
            let t = s + "t";
            {
                let u = t + "u";
                trail += u + ";";
                h();
            }
        }
    }
    catch {
        trail += "caught;";
    }

    churn();
    return s;
}

// the nested scope completes and releases before the exception: nothing may release it again
function throwAfterNestedScope(pre: string) {
    let s = pre + "a";
    try {
        {
            let t = s + "t";
            trail += t + ";";
        }
        churn();
        h();
    }
    catch {
        trail += "caught;";
    }

    churn();
    return s;
}

function throwFromNestedLoopBody(pre: string) {
    let s = pre + "b";
    try {
        for (let i = 0; i < 3; i++) {
            let t = s + i;
            churn();
            trail += t + ";";
            if (i == 1) {
                h();
            }
        }
    }
    catch {
        trail += "caught;";
    }

    churn();
    return s;
}

function throwFromFunctionBody(pre: string) {
    let s = pre + "c";
    {
        let t = s + "t";
        trail += t + ";";
        if (t.length > 0) {
            throw 1;
        }
    }

    return s;
}

// the throw releases `s`, and must not take the thrown value with it
function throwLocal(pre: string) {
    let s = pre + "d";
    trail += s + ";";
    throw s;
}

function check(f: () => void, expected: string, what: string) {
    trail = "";
    f();
    churn();
    print(what + ": " + trail);
    assert(trail == expected, what);
}

function main() {
    let pre = "p";

    check(() => assert(expectThrow(() => { throwThroughFinally(pre); }), "must throw"), "p1;", "throw through a finally");
    check(() => assert(expectThrow(() => { throwThroughEmptyFinally(pre); }), "must throw"), "", "throw through an empty finally");

    trail = "";
    let c = completeThroughFinally(pre);
    print("complete through a finally: " + c + " " + trail);
    assert(c == "p3" && trail == "body;p3;", "complete through a finally");

    trail = "";
    let r = returnThroughFinally(pre);
    churn();
    print("return through a finally: " + r + " " + trail);
    assert(r == "p4r" && trail == "p4;", "return through a finally");

    trail = "";
    let b = breakThroughFinally(pre);
    print("break through a finally: " + b + " " + trail);
    assert(b == "p5" && trail == "p50;p51;", "break through a finally");

    check(() => assert(expectThrow(() => { throwFromCatch(pre); }), "must throw"), "p6;", "throw from a catch");
    check(() => assert(expectThrow(() => { throwFromCatchWithFinally(pre); }), "must throw"), "p7;", "throw from a catch with a finally");

    trail = "";
    let f = finallyInTryBody(pre);
    print("finally in a try body: " + f + " " + trail);
    assert(f == "p8" && trail == "p8t;caught;", "finally in a try body");

    trail = "";
    let n = throwThroughNestedScope(pre);
    print("throw through a nested scope: " + n + " " + trail);
    assert(n == "p9" && trail == "p9tu;caught;", "throw through a nested scope");

    trail = "";
    let a = throwAfterNestedScope(pre);
    print("throw after a nested scope: " + a + " " + trail);
    assert(a == "pa" && trail == "pat;caught;", "throw after a nested scope");

    trail = "";
    let l = throwFromNestedLoopBody(pre);
    print("throw from a nested loop body: " + l + " " + trail);
    assert(l == "pb" && trail == "pb0;pb1;caught;", "throw from a nested loop body");

    check(() => assert(expectThrow(() => { throwFromFunctionBody(pre); }), "must throw"), "pct;", "throw from a function body");

    trail = "";
    let thrown = false;
    try {
        throwLocal(pre);
    }
    catch {
        thrown = true;
    }
    churn();
    print("throw a local: " + trail);
    assert(thrown && trail == "pd;", "throw a local");

    print("done.");
}
