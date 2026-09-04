// Two bugs, both reached by inlining a function whose body ends in a throw. Both hit AOT and
// JIT alike, and only under --opt, because that is what turns the MLIR inliner on.
//
// 1. MLIR's inliner has a fast path for a single-block callee: it offers the block's
//    terminator to the dialect's handleTerminator hook and then erases it outright, on the
//    assumption that a terminator is return-like and its operands are all it had left to say.
//    `ts.ThrowCall` is a terminator too, and ours only knew what to do with a return, so the
//    throw was erased and the caller carried on as if the callee had returned. `callsIt` below
//    compiled down to a function that does nothing but return. Fixed by declining that fast
//    path for any terminator that is not a return - the multi-block path leaves it in place.
//
// 2. With the throw no longer deleted, one inlined into a catch clause crashed the backend.
//    Win32ExceptionPass ends a catch region at a _CxxThrowException call by splitting the block
//    ahead of it and emitting the catchret there, which puts the throw outside the funclet -
//    but it also collected that same call for a "funclet" bundle, leaving it naming a pad it
//    had already returned from. An end-of-catch marker is what normally keeps those two apart,
//    and a throw the inliner brought in arrives without one.
//
// See docs/reference-counting-evaluation.md section 9.16.

let steps = 0;

function thrower() {
    throw 5;
}

// (1) a plain call with no try in sight, and the throwing call is all `callsIt` ends with
function callsIt() {
    steps = steps + 1;
    thrower();
}

// (2) the same helper called from a catch clause, so the inlined throw lands inside a funclet
function throwsFromCatch() {
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        thrower();
    }
}

// a callee that also has a returning path, so its throw is not the only terminator and the
// inliner takes it down the multi-block path instead - the one that was always correct
function throwsOnlyWhenAsked(doThrow: boolean) {
    if (doThrow) {
        throw 3;
    }

    steps = steps + 10;
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
    steps = 0;
    assert(caught(() => callsIt()), "an inlined helper's throw must survive the inliner");
    assert(steps == 1, "and the rest of the inlined body must still run");

    assert(caught(() => throwsFromCatch()), "an inlined throw inside a catch clause must escape it");

    steps = 0;
    assert(caught(() => throwsOnlyWhenAsked(true)), "a conditional throw must survive too");
    assert(!caught(() => throwsOnlyWhenAsked(false)), "and the returning path must still return");
    assert(steps == 10, "the returning path must have run its body");

    print("done.");
}
