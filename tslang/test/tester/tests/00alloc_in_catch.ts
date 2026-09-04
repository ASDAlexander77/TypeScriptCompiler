// Allocating inside a catch or finally clause, on Win64, where the handler is a separate funclet
// and every call in it has to carry a `funclet` bundle naming its pad.
//
// The compiler used to ask for a zeroed block as `malloc` followed by a zero-fill, and LLVM
// recognises that pair and rewrites it into `calloc` - building the replacement call without
// carrying the original's operand bundles over. Inside a handler that dropped the bundle, and the
// funclet was then emitted as a bare prologue with no body and no catchret, so it faulted the
// moment it ran. Only `-mm=gc` was unaffected, and by accident: GCPass deletes the zero-fill, so
// the pattern the fold looks for never reached LLVM. That is why this needs the `-mm=none` and
// `-mm=rc` variants to be worth anything - under the default model it passes either way.
//
// It also only shows with optimisation on, and only when the allocation survives to be used
// inside the handler; a value the optimiser can drop takes the bug with it.
//
// See docs/reference-counting-evaluation.md section 9.28.

class Leaf {
    n: number;

    constructor(n: number) {
        this.n = n;
    }

    get(): number {
        return this.n;
    }
}

// the plainest shape: allocate in a catch clause and call through the result
function allocInCatch() {
    let total = 0;
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        const leaf = new Leaf(7);
        total = leaf.get();
    }

    return total;
}

// the same in a finally clause, which is a funclet of its own
function allocInFinally() {
    let total = 0;
    try {
        total = 1;
    }
    finally {
        const leaf = new Leaf(4);
        total = total + leaf.get();
    }

    return total;
}

// an array literal, so the allocation is not a class instance
function arrayInCatch() {
    let total = 0;
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        const xs = [3, 5, 9];
        total = xs[0] + xs[2];
    }

    return total;
}

// a second allocation in the same handler, used after the first
function twoAllocsInCatch() {
    let total = 0;
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        const a = new Leaf(2);
        const b = new Leaf(3);
        total = a.get() * b.get();
    }

    return total;
}

// NOT covered here: an allocation inside a try/catch nested within a catch clause. A nested
// try/catch inside a catch crashes on its own, with no allocation in it at all, in every memory
// model and at every optimisation level - a separate bug from this one, and one that would make
// this file fail for a reason it is not about.

// the handler allocates and then throws on, so the funclet is left by unwinding rather than by
// falling off its end
function allocInCatchThenThrow() {
    try {
        throw 1;
    }
    catch (e: TypeOf<1>) {
        const leaf = new Leaf(6);
        if (leaf.get() > 0) {
            throw 2;
        }
    }

    return 0;
}

function main() {
    assert(allocInCatch() == 7, "a value allocated in a catch clause is usable there");
    assert(allocInFinally() == 5, "and in a finally clause");
    assert(arrayInCatch() == 12, "an array literal allocated in a catch clause is usable there");
    assert(twoAllocsInCatch() == 6, "two allocations in one handler both survive");

    let rethrown = false;
    try {
        allocInCatchThenThrow();
    }
    catch (e: TypeOf<2>) {
        rethrown = true;
    }

    assert(rethrown, "a handler that allocates can still throw on");

    print("done.");
}
