// A `using` declaration's cleanup region has to know how far the block it stands for actually
// got. It did not, and was wrong about it in both directions - the two defects read off the IR
// in docs/reference-counting-evaluation.md section 9.62 and left open there.
//
// The block is wrapped in a synthesized catch-less TryOp so an exception unwinding through it
// still disposes (mlirGenBlockWithUnwindCleanup). The declaration's storage is hoisted out in
// front of that TryOp, and everything else - the initializer, the store, the scope exit's own
// disposal - stays in the body. The cleanup region is reached by the unwind edge from any of
// them, and disposed unconditionally:
//
//   - the body's own disposal is a call in the try body, so it unwinds to that same cleanup.
//     A `[Symbol.dispose]()` that threw was therefore followed by the cleanup disposing the
//     same variable a second time, and the second throw - arriving while the first was still
//     unwinding - terminated the process.
//   - the `using` initializer's `new` is in the try body too, so a constructor that threw left
//     the cleanup disposing a slot nothing had ever been stored into, reading a vtable out of
//     whatever the frame happened to hold. An access violation, every time.
//
// Both are fixed by a boolean beside the slot: false until the initializing store has run, and
// cleared before each disposal rather than after it. See mlirGenDisposeOne.

let disposed = 0;
let steps = 0;

class Res {
    [Symbol.dispose]() {
        disposed = disposed + 1;
    }
}

class Boom {
    constructor(fail: boolean) {
        if (fail) {
            throw 1;
        }
    }

    [Symbol.dispose]() {
        disposed = disposed + 1;
    }
}

class ThrowsOnDispose {
    [Symbol.dispose]() {
        disposed = disposed + 1;
        throw 1;
    }
}

// the control: nothing throws, so the body disposes and the cleanup is never reached
function normalExit() {
    using r = new Res();
    steps = steps + 1;
}

// the control for the unwind leg: the throw is after the store, so the cleanup is the only
// thing that can dispose, and it must
function throwsAfterDeclaration() {
    using r = new Res();
    steps = steps + 1;
    throw 1;
}

// the constructor throws, so nothing was ever stored: the cleanup must dispose nothing
function constructorThrows() {
    using r = new Boom(true);
    steps = steps + 1;
}

// the same, one declaration in: the first is live and owes a dispose, the second never
// existed. Skipping the whole cleanup would pass the case above and fail this one.
function secondConstructorThrows() {
    using first = new Res();
    using second = new Boom(true);
    steps = steps + 1;
}

// the body's own disposal throws. It is disposed once, and the exception carries on out
// instead of the cleanup disposing the same variable again on top of it.
function disposalThrows() {
    using r = new ThrowsOnDispose();
    steps = steps + 1;
}

// the same constructor, in a hand-written try body. A different generation path -
// mlirGen(TryStatement) rather than the synthesized mlirGenBlockWithUnwindCleanup - with the
// same hoisting and the same cleanup region, so it has the same two ways to be wrong.
function constructorThrowsInTryBody() {
    try {
        using r = new Boom(true);
        steps = steps + 1;
    }
    catch (e5: TypeOf<1>) {
        steps = steps + 10;
    }
}

function main() {
    disposed = 0;
    steps = 0;
    normalExit();
    assert(disposed == 1, "the control: a normal exit disposes once");
    assert(steps == 1, "the control: the body ran");

    disposed = 0;
    steps = 0;
    try {
        throwsAfterDeclaration();
        assert(false, "unreachable: the function throws");
    }
    catch (e1: TypeOf<1>) {
        steps = steps + 1;
    }
    assert(disposed == 1, "the cleanup disposes what the body stored");
    assert(steps == 2, "the body ran and the throw was caught");

    disposed = 0;
    steps = 0;
    try {
        constructorThrows();
        assert(false, "unreachable: the constructor throws");
    }
    catch (e2: TypeOf<1>) {
        steps = steps + 1;
    }
    assert(disposed == 0, "a constructor that throws leaves nothing to dispose");
    assert(steps == 1, "the body never ran, and the throw was caught");

    disposed = 0;
    steps = 0;
    try {
        secondConstructorThrows();
        assert(false, "unreachable: the second constructor throws");
    }
    catch (e3: TypeOf<1>) {
        steps = steps + 1;
    }
    assert(disposed == 1, "the declaration that completed is disposed, the one that threw is not");
    assert(steps == 1, "the body never ran, and the throw was caught");

    disposed = 0;
    steps = 0;
    try {
        disposalThrows();
        assert(false, "unreachable: the disposal throws");
    }
    catch (e4: TypeOf<1>) {
        steps = steps + 1;
    }
    assert(disposed == 1, "a disposal that throws is not repeated by the cleanup");
    assert(steps == 2, "the body ran, and the disposal's own throw was caught");

    disposed = 0;
    steps = 0;
    constructorThrowsInTryBody();
    assert(disposed == 0, "the same in a hand-written try body");
    assert(steps == 10, "the body never ran, and the try's own catch took it");

    print("done.");
}
