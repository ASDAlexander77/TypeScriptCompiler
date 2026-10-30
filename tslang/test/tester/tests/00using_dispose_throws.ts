// A `[Symbol.dispose]()` that throws while another exception is already unwinding through its
// `using`. It terminated the process, in every model, in both tiers, on both back ends.
//
// - Windows: the cleanup region ran in a cleanup funclet, and an exception out of a cleanup
//   funclet is one thrown during unwinding to the MSVC personality, which terminates (0x80000003).
//   A `finally` had the same problem and #733 entered it through a catch-all pad instead; the
//   cleanup region is now entered the same way when it may throw (windowsCleanupCatches).
// - Linux: the cleanup shared its landing pad with the try's catch and ran before it, with
//   nothing caught, and its dispose call had no unwind edge - so the new exception left the
//   function, and with nothing outside to catch it the process terminated
//   (linuxCleanupCatches).
//
// A disposal's throw sends the cleanup's calls to a copy of the cleanup (cleanupRetries), which
// runs the rest of it - the resources not yet disposed, and the releases, which the ownership
// verifier reported as skipped on that path - and then hands the new exception on.
//
// What the disposal throws REPLACES the exception it interrupted, as a throw in a `finally`
// does. That is not quite JavaScript: TC39's explicit resource management throws a
// `SuppressedError` carrying both. tslang has no `SuppressedError`; the remaining resources are
// disposed, as JavaScript does (`twoUsings`).
//
// A second disposal that throws, from the copy, is a nested exception there: it is not caught
// again in the copy, and the releases after it are skipped. Where it goes differs by back end -
// this try's catch on Windows, where the funclet's calls unwind with it; the caller on Linux - so
// `bothThrow` checks only what both do: nothing terminates, every resource is disposed, and the
// last exception thrown is the one that arrives.

class Thrower {
    [Symbol.dispose]() {
        throw "from dispose";
    }
}

let trail = "";

class Named {
    constructor(private n: string) {}
    [Symbol.dispose]() {
        trail = trail + this.n;
    }
}

class NamedThrower {
    constructor(private n: string) {}
    [Symbol.dispose]() {
        trail = trail + this.n;
        throw "from " + this.n;
    }
}

// the plain shape: this try's own catch takes what the disposal throws
function inTryCatch() {
    let s = "";
    try {
        using d = new Thrower();
        throw "first";
    } catch (e: string) {
        s = e;
    }

    return s;
}

// no catch on the try: its finally still runs, and the disposal's exception goes on outwards
function withFinallyOnly() {
    let s = "";
    let fin = 0;
    try {
        try {
            using d = new Thrower();
            throw "first";
        } finally {
            fin = 1;
        }
    } catch (e: string) {
        s = e;
    }

    return s + fin;
}

// a `using` in a block of its own inside the try: the block's cleanup hands on to the try around it
function inNestedBlock() {
    let s = "";
    try {
        {
            using d = new Thrower();
            throw "first";
        }
    } catch (e: string) {
        s = e;
    }

    return s;
}

// nothing in the function catches: the caller does
function cleanupOnly() {
    using d = new Thrower();
    throw "first";
}

function callerCatches() {
    let s = "";
    try {
        cleanupOnly();
    } catch (e: string) {
        s = e;
    }

    return s;
}

// the inner catch takes a number, so the disposal's string goes on to the outer one
function innerCatchMismatch() {
    let s = "";
    try {
        try {
            using d = new Thrower();
            throw 1;
        } catch (e: number) {
            s = "inner";
        }
    } catch (e: string) {
        s = e;
    }

    return s;
}

function inLoop() {
    let s = "";
    for (let i = 0; i < 3; i++) {
        try {
            using d = new Thrower();
            if (i == 1) throw "first";
        } catch (e: string) {
            s = s + i;
        }
    }

    return s;
}

// on the way out of an exception, the cleanup goes on past a disposal that throws: `a` is disposed
function twoUsings() {
    let s = "";
    trail = "";
    try {
        using a = new Named("a");
        using b = new NamedThrower("B");
        throw "first";
    } catch (e: string) {
        s = e;
    }

    return s + "/" + trail;
}

// both disposals throw: the second is thrown from the cleanup's copy (see the header)
function bothThrowInner() {
    let s = "";
    try {
        using a = new NamedThrower("A");
        using b = new NamedThrower("B");
        throw "first";
    } catch (e: string) {
        s = e;
    }

    return s;
}

function bothThrow() {
    let s = "";
    trail = "";
    try {
        s = bothThrowInner();
    } catch (e: string) {
        s = e;
    }

    return s + "/" + trail;
}

// on the normal path the same throw is from the body, so the cleanup still disposes `a`
function twoUsingsNormalPath() {
    let s = "";
    trail = "";
    try {
        using a = new Named("a");
        using b = new NamedThrower("B");
    } catch (e: string) {
        s = e;
    }

    return s + "/" + trail;
}

// and a disposal that does not throw still lets the first exception through
function disposalDoesNotThrow() {
    let s = "";
    trail = "";
    try {
        using a = new Named("a");
        throw "first";
    } catch (e: string) {
        s = e;
    }

    return s + "/" + trail;
}

function main() {
    assert(inTryCatch() == "from dispose", "this try's catch takes the disposal's exception");
    assert(withFinallyOnly() == "from dispose1", "the finally runs and the disposal's exception goes on");
    assert(inNestedBlock() == "from dispose", "a nested block's cleanup hands on to the try around it");
    assert(callerCatches() == "from dispose", "the caller catches the disposal's exception");
    assert(innerCatchMismatch() == "from dispose", "a catch of another type passes it on");
    assert(inLoop() == "012", "every iteration's disposal is caught");
    assert(twoUsings() == "from B/Ba", "the disposal's exception replaces the first, and the rest is disposed");
    assert(bothThrow() == "from A/BA", "a second disposal that throws arrives, with everything disposed");
    assert(twoUsingsNormalPath() == "from B/Ba", "on the normal path the cleanup disposes the rest");
    assert(disposalDoesNotThrow() == "first/a", "a disposal that does not throw keeps the first exception");

    print("done.");
}
