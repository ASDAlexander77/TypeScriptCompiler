// A `return` from a `try` body or catch clause runs the `finally` before the function returns, and
// the finally can read every scope around the `try`. So the `return` releases only the scopes
// inside the `try`; the copy of the finally that a `return` runs releases the scopes around it once
// it is done. Under -mm=rc the `return` used to release the whole frame and then run the finally,
// which read freed strings.
//
// And a `return` through two nested `finally`s runs both, inner first: the inner one's copy used to
// return straight to the caller and skip the outer one, in every memory model.
//
// The values are read again after churn() wherever a release one too many would free them early.

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

function returnFromBody(pre: string) {
    let s = pre + "1";
    try {
        return s + "r";
    }
    finally {
        churn();
        trail += s + ";";
    }
}

// a bare `return`, and the parameter read in the finally
function returnVoid(pre: string) {
    let s = pre + "2";
    try {
        trail += "body;";
        return;
    }
    finally {
        churn();
        trail += s + pre + ";";
    }
}

// the catch clause's own scope goes with the `return`, the function's after the finally
function returnFromCatch(pre: string) {
    let s = pre + "3";
    try {
        h();
    }
    catch {
        let k = s + "k";
        return k;
    }
    finally {
        churn();
        trail += s + ";";
    }

    return "";
}

// the body's own scope and a block nested in it go with the `return`
function returnFromNestedBlock(pre: string) {
    let s = pre + "4";
    try {
        let t = s + "t";
        {
            let u = t + "u";
            return u;
        }
    }
    finally {
        churn();
        trail += s + ";";
    }
}

// each finally reads the scope around its own `try`; the inner one's copy goes on to the outer one's
function returnThroughTwoFinallys(pre: string) {
    let a = pre + "5";
    try {
        let b = a + "b";
        try {
            let c = b + "c";
            return c;
        }
        finally {
            churn();
            trail += "i:" + b + ";";
        }
    }
    finally {
        churn();
        trail += "o:" + a + ";";
    }
}

// the scope around the `try` is a loop body's, and the `return` leaves the loop too
function returnInLoop(pre: string) {
    let s = pre + "6";
    for (let i = 0; i < 3; i++) {
        let t = s + i;
        try {
            if (i == 1) {
                return t;
            }
        }
        finally {
            churn();
            trail += t + ";";
        }
    }

    return "";
}

// the finally returns instead: its own `return` releases the scopes around the `try`, and the
// releases left for after the finally do not run as well
function returnFromFinally(pre: string, abort: boolean) {
    let s = pre + "7";
    try {
        return s + "r";
    }
    finally {
        churn();
        trail += s + ";";
        if (abort) {
            return s + "f";
        }
    }
}

// normal completion and an exception still release once, after the finally
function completeAndThrow(pre: string, fail: boolean) {
    let s = pre + "8";
    try {
        if (fail) {
            h();
        }
    }
    finally {
        churn();
        trail += s + ";";
    }

    return s;
}

function main() {
    let pre = "p";

    trail = "";
    let r1 = returnFromBody(pre);
    churn();
    print("return from a try body: " + r1 + " " + trail);
    assert(r1 == "p1r" && trail == "p1;", "return from a try body");

    trail = "";
    returnVoid(pre);
    churn();
    print("bare return: " + trail);
    assert(trail == "body;p2p;", "bare return");

    trail = "";
    let r3 = returnFromCatch(pre);
    churn();
    print("return from a catch: " + r3 + " " + trail);
    assert(r3 == "p3k" && trail == "p3;", "return from a catch");

    trail = "";
    let r4 = returnFromNestedBlock(pre);
    churn();
    print("return from a nested block: " + r4 + " " + trail);
    assert(r4 == "p4tu" && trail == "p4;", "return from a nested block");

    trail = "";
    let r5 = returnThroughTwoFinallys(pre);
    churn();
    print("return through two finallys: " + r5 + " " + trail);
    assert(r5 == "p5bc" && trail == "i:p5b;o:p5;", "return through two finallys");

    trail = "";
    let r6 = returnInLoop(pre);
    churn();
    print("return from a loop: " + r6 + " " + trail);
    assert(r6 == "p61" && trail == "p60;p61;", "return from a loop");

    trail = "";
    let r7 = returnFromFinally(pre, false);
    churn();
    let r7f = returnFromFinally(pre, true);
    churn();
    print("return from a finally: " + r7 + " " + r7f + " " + trail);
    assert(r7 == "p7r" && r7f == "p7f" && trail == "p7;p7;", "return from a finally");

    trail = "";
    let r8 = completeAndThrow(pre, false);
    churn();
    let thrown = false;
    try {
        completeAndThrow(pre, true);
    }
    catch {
        thrown = true;
    }
    churn();
    print("complete and throw: " + r8 + " " + trail);
    assert(thrown && r8 == "p8" && trail == "p8;p8;", "complete and throw");

    print("done.");
}
