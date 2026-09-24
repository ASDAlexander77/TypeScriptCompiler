// A try inside another try's catch clause (or finally) whose typed catch does not match. Its
// exception belongs to whatever handles the *parent's* exceptions - a try around the parent -
// but the lowering sent it to the parent's own landing pad, the handler already running. On
// Windows that caught it again forever (a hang), or it left the function uncaught; on Linux it
// reached std::terminate. A try in a parent's finally with a `using` crashed the compiler on
// Windows. With a `using` in the inner try, Linux also rethrew with nothing caught - that shape
// is in 00try_mismatch_in_catch_clause_using.ts.

let disposed = 0;
let finals = 0;
let wrong = 0;
let outer = 0;

class Res {
    [Symbol.dispose]() {
        disposed++;
    }
}

function reset() {
    disposed = 0;
    finals = 0;
    wrong = 0;
    outer = 0;
}

function inUntypedCatch() {
    try {
        try {
            throw 1;
        } catch (o) {
            try {
                throw "x";
            } catch (e: number) {
                wrong++;
            }
        }
    } catch (e2) {
        outer++;
    }
}

function inTypedCatch() {
    try {
        try {
            throw 1;
        } catch (o: number) {
            try {
                throw "x";
            } catch (e: number) {
                wrong++;
            }
        }
    } catch (e2) {
        outer++;
    }
}

function withFinally() {
    try {
        try {
            throw 1;
        } catch (o) {
            try {
                throw "x";
            } catch (e: number) {
                wrong++;
            } finally {
                finals++;
            }
        }
    } catch (e2) {
        outer++;
    }
}

function inParentFinally() {
    try {
        try {
            finals += 10;
        } finally {
            try {
                using r = new Res();
                throw "x";
            } catch (e: number) {
                wrong++;
            } finally {
                finals++;
            }
        }
    } catch (e2) {
        outer++;
    }
}

// two levels up: the parent is itself in a catch clause, whose try has a finally
function twoLevelsUp() {
    try {
        try {
            throw 1;
        } catch (a) {
            try {
                throw 2;
            } catch (b) {
                try {
                    throw "x";
                } catch (e: number) {
                    wrong++;
                }
            }
        } finally {
            finals++;
        }
    } catch (e2) {
        outer++;
    }
}

function main() {
    reset();
    inUntypedCatch();
    assert(wrong == 0 && outer == 1, "in an untyped catch clause");

    reset();
    inTypedCatch();
    assert(wrong == 0 && outer == 1, "in a typed catch clause");

    reset();
    withFinally();
    assert(wrong == 0 && finals == 1 && outer == 1, "with its own finally");

    reset();
    inParentFinally();
    assert(disposed == 1 && finals == 11 && wrong == 0 && outer == 1, "in the parent's finally");

    reset();
    twoLevelsUp();
    assert(wrong == 0 && finals == 1 && outer == 1, "two levels up, through a finally");

    print("done.");
}
