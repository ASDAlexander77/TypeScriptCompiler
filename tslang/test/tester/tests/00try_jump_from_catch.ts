// A `break`, `continue` or `return` out of a catch clause leaves the catch: it ends the catch
// first, with or without a `finally`, and the clause's other way out - falling off its end - still
// ends it as well. On Windows every such exit is a catchret out of the catch funclet.
//
// A `break` or `continue` out of a catch with no finally used to leave without ending the catch,
// and on Windows any catch with two ways out - one of them a return, break or continue - kept only
// the first in layout: the other left the funclet with a bare branch, and its calls had no funclet
// bundle. Both crashed with 0x80000003.
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

// --- no finally ---

function breakFromCatch() {
    for (let i = 0; i < 3; i++) {
        try {
            h();
        }
        catch {
            if (i == 1) {
                break;
            }

            trail += "c" + i + ";";
        }

        trail += "a" + i + ";";
    }
}

function continueFromCatch() {
    for (let i = 0; i < 3; i++) {
        try {
            h();
        }
        catch {
            if (i == 1) {
                continue;
            }

            trail += "c" + i + ";";
        }

        trail += "a" + i + ";";
    }
}

function returnFromCatch(i: number) {
    try {
        h();
    }
    catch {
        if (i == 1) {
            return;
        }

        trail += "c" + i + ";";
    }

    trail += "a" + i + ";";
}

// a typed catch clause
function breakFromTypedCatch() {
    for (let i = 0; i < 3; i++) {
        try {
            h();
        }
        catch (e: number) {
            if (i == 1) {
                break;
            }

            trail += "c" + e + i + ";";
        }

        trail += "a" + i + ";";
    }
}

// --- with a finally ---

function breakFromCatchFinally() {
    for (let i = 0; i < 3; i++) {
        try {
            h();
        }
        catch {
            if (i == 1) {
                break;
            }

            trail += "c" + i + ";";
        }
        finally {
            trail += "f" + i + ";";
        }

        trail += "a" + i + ";";
    }
}

function continueFromCatchFinally() {
    for (let i = 0; i < 3; i++) {
        try {
            h();
        }
        catch {
            if (i == 1) {
                continue;
            }

            trail += "c" + i + ";";
        }
        finally {
            trail += "f" + i + ";";
        }

        trail += "a" + i + ";";
    }
}

// the finally reads a local of the block around the `try`
function breakFinallyReadsBlockLocal(pre: string) {
    let out = "";
    for (let i = 0; i < 3; i++) {
        let t = pre + "b" + i;
        {
            let s = t + "s";
            try {
                h();
            }
            catch {
                if (i == 1) {
                    break;
                }
            }
            finally {
                churn();
                trail += s + ";";
            }

            out += s + ",";
        }
    }

    churn();
    return out;
}

function continueFinallyReadsBlockLocal(pre: string) {
    let out = "";
    for (let i = 0; i < 3; i++) {
        let t = pre + "c" + i;
        {
            let s = t + "s";
            try {
                h();
            }
            catch {
                if (i == 1) {
                    continue;
                }
            }
            finally {
                churn();
                trail += s + t + ";";
            }

            out += s + ",";
        }
    }

    churn();
    return out;
}

// --- nesting ---

// a `break` out of a catch nested in another catch leaves both
function breakFromNestedCatch() {
    for (let i = 0; i < 3; i++) {
        try {
            h();
        }
        catch {
            try {
                h();
            }
            catch {
                if (i == 1) {
                    break;
                }

                trail += "n" + i + ";";
            }

            trail += "o" + i + ";";
        }

        trail += "a" + i + ";";
    }
}

// a `break` out of a loop inside the catch clause stays in the catch
function breakLoopInsideCatch() {
    try {
        h();
    }
    catch {
        for (let j = 0; j < 3; j++) {
            if (j == 1) {
                break;
            }

            trail += "j" + j + ";";
        }

        trail += "c;";
    }

    trail += "a;";
}

// a labelled `continue` out of a catch in the inner loop leaves the inner loop too
function labelledContinueFromCatch() {
    outer: for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
            try {
                h();
            }
            catch {
                if (j == 1) {
                    continue outer;
                }

                trail += "c" + i + j + ";";
            }
        }

        trail += "never;";
    }
}

// --- a finally that leaves on some paths only ---
//
// On the exceptional path such a finally runs inside a catch-all pad: its `return` or `break` ends
// that catch and swallows the exception, and running to its end hands the exception on.

function returnFromFinallySometimes(c: boolean) {
    try {
        h();
    }
    finally {
        if (c) {
            return "r;";
        }

        trail += "f;";
    }

    return "n;";
}

// the same with a try around it in the function, which catches what the finally hands on
function returnFromFinallySometimesInTry(c: boolean) {
    try {
        try {
            h();
        }
        finally {
            if (c) {
                return "r;";
            }

            trail += "f;";
        }
    }
    catch {
        trail += "c;";
    }

    return "n;";
}

function breakFromFinallySometimes() {
    for (let i = 0; i < 3; i++) {
        try {
            try {
                h();
            }
            finally {
                if (i == 1) {
                    break;
                }

                trail += "f" + i + ";";
            }
        }
        catch {
            trail += "c" + i + ";";
        }
    }
}

function main() {
    trail = "";
    breakFromCatch();
    print("break from a catch: " + trail);
    assert(trail == "c0;a0;", "break from a catch");

    trail = "";
    continueFromCatch();
    print("continue from a catch: " + trail);
    assert(trail == "c0;a0;c2;a2;", "continue from a catch");

    trail = "";
    for (let i = 0; i < 3; i++) {
        returnFromCatch(i);
    }

    print("return from a catch: " + trail);
    assert(trail == "c0;a0;c2;a2;", "return from a catch");

    trail = "";
    breakFromTypedCatch();
    print("break from a typed catch: " + trail);
    assert(trail == "c10;a0;", "break from a typed catch");

    trail = "";
    breakFromCatchFinally();
    print("break from a catch, finally: " + trail);
    assert(trail == "c0;f0;a0;f1;", "break from a catch, finally");

    trail = "";
    continueFromCatchFinally();
    print("continue from a catch, finally: " + trail);
    assert(trail == "c0;f0;a0;f1;c2;f2;a2;", "continue from a catch, finally");

    let pre = "p";

    trail = "";
    let r1 = breakFinallyReadsBlockLocal(pre);
    churn();
    print("break from a catch, finally reads a block local: " + r1 + " " + trail);
    assert(r1 == "pb0s," && trail == "pb0s;pb1s;", "break from a catch, finally reads a block local");

    trail = "";
    let r2 = continueFinallyReadsBlockLocal(pre);
    churn();
    print("continue from a catch, finally reads a block local: " + r2 + " " + trail);
    assert(r2 == "pc0s,pc2s," && trail == "pc0spc0;pc1spc1;pc2spc2;", "continue from a catch, finally reads a block local");

    trail = "";
    breakFromNestedCatch();
    print("break from a nested catch: " + trail);
    assert(trail == "n0;o0;a0;", "break from a nested catch");

    trail = "";
    breakLoopInsideCatch();
    print("break a loop inside a catch: " + trail);
    assert(trail == "j0;c;a;", "break a loop inside a catch");

    trail = "";
    labelledContinueFromCatch();
    print("labelled continue from a catch: " + trail);
    assert(trail == "c00;c10;", "labelled continue from a catch");

    trail = "";
    let f1 = returnFromFinallySometimes(true);
    trail += f1;
    try {
        let f2 = returnFromFinallySometimes(false);
        trail += f2;
    }
    catch {
        trail += "caught;";
    }

    print("return from a finally on some paths: " + trail);
    assert(trail == "r;f;caught;", "return from a finally on some paths");

    trail = "";
    let t1 = returnFromFinallySometimesInTry(true);
    trail += t1;
    let t2 = returnFromFinallySometimesInTry(false);
    trail += t2;
    print("return from a finally on some paths, in a try: " + trail);
    assert(trail == "r;f;c;n;", "return from a finally on some paths, in a try");

    trail = "";
    breakFromFinallySometimes();
    print("break from a finally on some paths: " + trail);
    assert(trail == "f0;c0;", "break from a finally on some paths");

    print("done.");
}
