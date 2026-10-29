// A finally that throws while an exception unwinds replaces it, as in JavaScript. On Windows the finally run on
// the way out of an exception was a cleanup funclet, which may not throw: a second exception there terminated the
// process. A finally that may throw is now entered through a catch-all that rethrows at its end, and its calls
// unwind to the enclosing handler (the copy of the finally on that path had lost them, which -O3 turned into a
// crash).
function thrower(n: number) {
    if (n > 0) {
        throw n;
    }
}

function rethrowsOriginal() {
    let ran = false;
    let caught = 0.0;
    try {
        try { thrower(1); } finally { ran = true; thrower(0); }
    } catch (e: number) {
        caught = e;
    }

    return ran && caught == 1;
}

function main() {
    let caught = 0.0;
    try {
        try { thrower(1); } finally { thrower(2); }
    } catch (e: number) {
        caught = e;
    }
    assert(caught == 2, "a call in the finally replaces the exception");

    caught = 0;
    try {
        try { thrower(1); } finally { throw 3; }
    } catch (e: number) {
        caught = e;
    }
    assert(caught == 3, "a throw in the finally replaces it");

    caught = 0;
    try {
        try { print("body"); } finally { thrower(4); }
    } catch (e: number) {
        caught = e;
    }
    assert(caught == 4, "a finally that throws on the normal path");

    assert(rethrowsOriginal(), "a finally whose call does not throw rethrows the original");

    let order = "";
    caught = 0;
    try {
        try {
            try { thrower(1); } finally { order += "a"; thrower(5); }
        } finally {
            order += "b";
        }
    } catch (e: number) {
        caught = e;
    }
    assert(caught == 5 && order == "ab", "an outer finally runs on the new exception's way out");

    print("done.");
}
