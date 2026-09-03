// A `using` in a try body whose catch clause makes more than one call. The cleanup funclet and
// the catch funclet then coexist in one function, and the catch funclet needs a callee-saved
// register to hold the call target.
//
// AOT only, deliberately. Under `--emit=jit --opt` this exact shape corrupts a callee-saved
// register across the call - `main` keeps a pointer in `rsi`, and `f` gives it back with its low
// 32 bits zeroed - so the caller faults after the catch has already run. It fails that way in
// every memory model and predates the ownership work; see docs/reference-counting-evaluation.md
// section 9.15 for the dump analysis. Compiled ahead of time, from the same IR, it is correct,
// which is what this file locks in. Add the JIT variants when the unwind defect is fixed.

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

// Not covered here, and not a new bug: putting the `using` one scope deeper - inside an `if`
// within the try body - crashes the *compiler*, in every memory model. That is the synthesized
// cleanup TryOp nesting inside a real TryOp's body, recorded in section 9.11 of the same
// document as already broken before any of this.

function main() {
    disposed = 0;
    twoCallsInCatch();
    assert(disposed == 1, "a using in a try body disposes once when its catch runs");

    print("done.");
}
