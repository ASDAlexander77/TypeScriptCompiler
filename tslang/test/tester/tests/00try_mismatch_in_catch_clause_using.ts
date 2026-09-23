// 00try_mismatch_in_catch_clause.ts, with a `using` in the inner try. Linux only: on Windows a
// `using` inside a catch clause is broken on main already - its cleanup pad is not nested in the
// catch funclet, which fails LLVM's verifier ahead of time, and with a typed catch and no try
// around it the JIT crashes in Win32ExceptionPass (see 00try_using_catch.ts, section 9.11).
// On Linux the mismatch rethrow after the cleanup ran with nothing caught: "terminate called
// without an active exception".

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

function main() {
    reset();
    withUsing();
    assert(disposed == 1 && wrong == 0 && outer == 1, "with a using");

    reset();
    withUsingAndFinally();
    assert(disposed == 1 && finals == 1 && wrong == 0 && outer == 1, "with a using and a finally");

    print("done.");
}
