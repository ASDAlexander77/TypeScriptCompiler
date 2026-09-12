// A `using` in a try body whose catch clause makes more than one call. The cleanup funclet and
// the catch funclet then coexist in one function, and the catch funclet needs a callee-saved
// register to hold the call target.
//
// This shape used to corrupt a callee-saved register across the call under `--emit=jit --opt`,
// in every memory model: `main` kept a pointer in `rsi` and got it back with the low 32 bits
// zeroed. The cause was our own C++ EH metadata - CatchableType::sizeOrOffset said a caught
// `int` was 8 bytes, so the CRT copied 8 bytes into a 4-byte frame slot and overwrote what sat
// above it. Ahead of time nothing lived there; in the JIT's large code model a saved register
// did. Fixed by giving each catchable type its real size; see
// docs/reference-counting-evaluation.md section 9.15.
//
// Not covered here, and a different bug: putting the `using` one scope deeper - inside an `if`
// within the try body - crashes the *compiler*, in every memory model. That is the synthesized
// cleanup TryOp nesting inside a real TryOp's body, recorded in section 9.11 as already broken.

let disposed = 0;

class Res {
    [Symbol.dispose]() {
        disposed = disposed + 1;
    }
}

function twoCallsInCatch() {
    try {
        using r = new Res();
        throw 1;
    }
    catch (e: TypeOf<1>) {
        print("a");
        print("b");
    }
}

// the caller keeps a value live across the call, which is what made the clobber observable
function callerHoldsValueAcrossCall() {
    disposed = 0;
    twoCallsInCatch();
    return disposed;
}

// the same overflow, one type up: a caught number is genuinely 8 bytes, so this shape has to
// keep working after narrowing `int` to 4
function catchesANumber() {
    try {
        using r = new Res();
        throw 1.5;
    }
    catch (e: number) {
        print("num");
        print("caught");
    }
}

function main() {
    assert(callerHoldsValueAcrossCall() == 1, "a using in a try body disposes once when its catch runs");

    disposed = 0;
    catchesANumber();
    assert(disposed == 1, "the same with a number-typed catch");

    print("done.");
}
