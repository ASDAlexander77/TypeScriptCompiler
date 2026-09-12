// A `using` scope nested inside another scope that also has to dispose on unwind. Two shapes,
// both of which used to crash the compiler and were avoided by guards in MLIRGenImpl.h rather
// than fixed:
//
//  - a `using` one scope deeper than a hand-written try's body (an `if`, or a bare `{ }`).
//  - an outer `using` scope that contains an inner one. That one had its own guard,
//    blockHasNestedUsing, whose cost was that the *outer* using did not dispose on unwind at
//    all: it stood down from being wrapped so that the inner one could be.
//
// Both were the same bug, in Win32ExceptionPass::ToInvoke. Given an operation that was already
// an invoke, it split the block at it to make room for a new one - but an invoke already ends
// its block, so it ended up alone in the new continuation block, which every caller then erased
// it from. That left an empty block with no terminator, and the real continuation with no
// predecessors, and the empty block crashed the inliner. An invoke needs its unwind edge
// redirected, not a block. See docs/reference-counting-evaluation.md section 9.17.
//
// Still guarded, and still genuinely broken: a `using` in a catch or finally *clause*
// (blockIsInsideCatchOrFinally). Re-checked against this fix - a different cause.

let disposed = "";

class Res {
    name: string;

    constructor(n: string) {
        this.name = n;
    }

    [Symbol.dispose]() {
        disposed = disposed + this.name;
    }
}

// one scope deeper than the try body, and the exception unwinds through it
function nestedInIf(flag: boolean) {
    try {
        if (flag) {
            using r = new Res("r");
            throw 1;
        }
    }
    catch (e: TypeOf<1>) {
        disposed = disposed + "!";
    }
}

// the same, in a bare block rather than an `if`
function nestedInBlock() {
    try {
        {
            using b = new Res("b");
            throw 1;
        }
    }
    catch (e: TypeOf<1>) {
        disposed = disposed + "!";
    }
}

// an outer using scope containing an inner one, with the throw after the inner scope has
// already closed: the inner disposes at its own scope exit, the outer on the unwind
function outerAndInner() {
    using a = new Res("a");
    {
        using c = new Res("c");
    }
    throw 1;
}

// the same pair, but the throw happens while both are still live
function outerAndInnerBothLive() {
    using a = new Res("a");
    {
        using c = new Res("c");
        throw 1;
    }
}

function caught(f: () => void) {
    try {
        f();
    }
    catch (e: TypeOf<1>) {
        return true;
    }

    return false;
}

function main() {
    disposed = "";
    nestedInIf(true);
    assert(disposed == "r!", "a using nested in an if inside a try body disposes on unwind");

    disposed = "";
    nestedInIf(false);
    assert(disposed == "", "and nothing runs when that branch is not taken");

    disposed = "";
    nestedInBlock();
    assert(disposed == "b!", "the same for a bare nested block");

    disposed = "";
    assert(caught(() => outerAndInner()), "the throw must reach the caller");
    assert(disposed == "ca", "inner disposes at its scope exit, outer on the unwind");

    disposed = "";
    assert(caught(() => outerAndInnerBothLive()), "the throw must reach the caller");
    assert(disposed == "ca", "both dispose on the unwind, innermost first");

    print("done.");
}
