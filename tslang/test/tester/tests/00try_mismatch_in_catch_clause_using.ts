// 00try_mismatch_in_catch_clause.ts, with a `using` in the inner try. On Linux the mismatch
// rethrow after the cleanup ran with nothing caught: "terminate called without an active
// exception". On Windows a `using` inside a catch clause did not work at all: its cleanup pad was
// not nested in the catch funclet - LLVM's verifier rejected the edge out of the catch ahead of
// time - and without a try around it Win32ExceptionPass crashed the compiler, having taken the
// cleanup pad for the end of the catch.

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

function withUsing() {
    try {
        try {
            throw 1;
        } catch (o) {
            try {
                using r = new Res();
                throw "x";
            } catch (e: number) {
                wrong++;
            }
        }
    } catch (e2) {
        outer++;
    }
}

function withUsingAndFinally() {
    try {
        try {
            throw 1;
        } catch (o) {
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

// the shape that crashed the compiler on Windows: no try around the catch, and nothing thrown in
// the inner try, so its typed catch is not taken
function noTryAround() {
    try {
        throw 1;
    } catch (o) {
        try {
            using r = new Res();
            finals++;
        } catch (e: number) {
            wrong++;
        }
    }
}

function main() {
    reset();
    withUsing();
    assert(disposed == 1 && wrong == 0 && outer == 1, "with a using");

    reset();
    withUsingAndFinally();
    assert(disposed == 1 && finals == 1 && wrong == 0 && outer == 1, "with a using and a finally");

    reset();
    noTryAround();
    assert(disposed == 1 && finals == 1 && wrong == 0, "no try around the catch");

    print("done.");
}
