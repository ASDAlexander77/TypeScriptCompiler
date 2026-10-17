// A `finally` that holds a try, or that returns or breaks, on the exceptional path (Windows).
//
// There the copy of the finally that runs while an exception unwinds was a cleanup funclet,
// and a cleanup funclet can neither return to normal flow nor hold pads of its own. So
// `finally { return s; }` left the region with no end, and a try in the finally had its pads
// placed outside the funclet they sit in; Win32ExceptionPass crashed on both. Such a finally is
// now entered through a catch-all, which its `return` ends, and which rethrows when the finally
// runs to its end.

let trail = "";

function thrower() {
    throw 1;
}

// the return swallows the exception
function returnInFinally(pre: string) {
    let s = pre + "1";
    try {
        thrower();
    } finally {
        return s;
    }
}

// and with nothing thrown, it is just the return
function returnInFinallyNoThrow(pre: string) {
    let s = pre + "n";
    try {
        trail += "body;";
    } finally {
        return s;
    }
}

// a try in the finally catches its own exception, and the first one carries on out
function tryInFinally(pre: string) {
    try {
        thrower();
    } finally {
        try {
            thrower();
        } catch {
            trail += "in " + pre + ";";
        } finally {
            trail += "inner " + pre + ";";
        }
    }

    trail += "unreached;";
    return pre;
}

// the same with a try around it in the function, which takes the first exception
function tryInFinallyCaughtFurtherOut(pre: string) {
    let s = pre + "4";
    try {
        try {
            thrower();
        } finally {
            try {
                thrower();
            } catch {
                trail += "in " + s + ";";
            }

            trail += "fin " + s + ";";
        }
    } catch {
        trail += "outer " + s + ";";
    }

    trail += "after " + s + ";";
}

// a return in a finally inside a try that catches: the catch never sees the exception
function returnInFinallyCaughtFurtherOut(pre: string): string {
    let s = pre + "5";
    try {
        try {
            thrower();
        } finally {
            return s;
        }
    } catch {
        trail += "never;";
    }

    return "no";
}

// a break out of the finally swallows the exception as well
function breakInFinally(pre: string) {
    let s = pre + "6";
    for (let i = 0; i < 3; i++) {
        try {
            thrower();
        } finally {
            trail += "loop " + s + ";";
            break;
        }
    }

    trail += "done " + s + ";";
}

function main() {
    assert(returnInFinally("p") == "p1", "return in finally");

    trail = "";
    assert(returnInFinallyNoThrow("p") == "pn", "return in finally, nothing thrown");
    assert(trail == "body;", "return in finally, nothing thrown: trail");

    trail = "";
    try {
        tryInFinally("p2");
    } catch {
        trail += "c;";
    }

    assert(trail == "in p2;inner p2;c;", "try in finally");

    trail = "";
    tryInFinallyCaughtFurtherOut("p");
    assert(trail == "in p4;fin p4;outer p4;after p4;", "try in finally, caught further out");

    trail = "";
    assert(returnInFinallyCaughtFurtherOut("p") == "p5", "return in finally, caught further out");
    assert(trail == "", "return in finally, caught further out: trail");

    trail = "";
    breakInFinally("p");
    assert(trail == "loop p6;done p6;", "break in finally");

    print("done.");
}
