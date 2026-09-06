// A value nothing receives is released at the end of the block that produced it (section 9.30).
// The end of the block is not the only way out of it: a `return` written inside an `if` - which is
// where returns are usually written - leaves from a nested region and never reaches that release,
// so the value was simply lost on that path. `raytrace.ts` spent most of what it held on exactly
// this shape, a ray built for a shadow test in a function that returns early when the light is
// blocked: 43.5 MB against `gc`'s 2.8 before, 6.2 MB after. See section 9.60.
//
// A leak is not something a test can assert, so these cases are here for the other direction: the
// release now emitted before each early return must not be a SECOND one. Every case builds over
// the memory it might have freed and then reads what it built, so a value given back twice is a
// wrong answer or a fault rather than nothing at all.
//
// `break` and `continue` are the cases that say why only returns are covered. Neither leaves the
// block holding the temporary - control comes back and runs the end-of-block release - so
// releasing at one as well would give the same reference back twice.

class Holder {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

function make(x: number): Holder {
    return new Holder(x);
}

// Allocate over whatever has just been freed, so a double release shows as a wrong answer.
function churn(): number {
    let total = 0.0;
    for (let i = 0; i < 32; i++) {
        let filler = new Holder(777.0);
        total = total + filler.x * 0.0;
    }

    return total;
}

// The shape itself: a temporary nothing receives, then a return out of an `if`.
function earlyReturnFromIf(takeIt: boolean): number {
    make(1.0);
    if (takeIt) {
        return 10.0;
    }

    return 20.0;
}

// Two of them, and a return under each.
function twoTemporariesTwoExits(takeIt: boolean): number {
    make(2.0);
    make(3.0);
    if (takeIt) {
        return 11.0;
    }

    return 21.0;
}

// The return is deeper than one block down.
function earlyReturnFromNestedIf(a: boolean, b: boolean): number {
    make(4.0);
    if (a) {
        if (b) {
            return 12.0;
        }
    }

    return 22.0;
}

// The return is inside a loop, which is inside the block that owns the temporary.
function earlyReturnFromLoop(stopAt: number): number {
    make(5.0);
    for (let i = 0; i < 8; i++) {
        if (i == stopAt) {
            return 13.0;
        }
    }

    return 23.0;
}

// A `break` does not leave this block - control reaches the end of it either way - so the value is
// given back exactly once. Releasing at the break as well would free it while the end-of-block
// release still had it, and `churn` below would then hand the same block out twice.
function breakDoesNotLeaveTheBlock(stopAt: number): number {
    let kept = make(6.0);
    for (let i = 0; i < 8; i++) {
        if (i == stopAt) {
            break;
        }
    }

    churn();

    return kept.x;
}

function continueDoesNotLeaveTheBlock(skipAt: number): number {
    let kept = make(7.0);
    let seen = 0.0;
    for (let i = 0; i < 8; i++) {
        if (i == skipAt) {
            continue;
        }

        seen = seen + 1.0;
    }

    churn();

    return kept.x + seen;
}

// The value the early return hands back is the caller's, not a temporary: the return retains it,
// so nothing here may give it away. Read after a churn, in another function, for the same reason.
function returnsWhatItBuilt(takeIt: boolean): Holder {
    let h = make(8.0);
    if (takeIt) {
        return h;
    }

    return make(9.0);
}

function readAfterEarlyReturn(): number {
    let h = returnsWhatItBuilt(true);
    churn();

    return h.x;
}

function readAfterTailReturn(): number {
    let h = returnsWhatItBuilt(false);
    churn();

    return h.x;
}

function main() {
    assert(earlyReturnFromIf(true) == 10.0, "early return out of an `if`");
    assert(earlyReturnFromIf(false) == 20.0, "the same function's tail return");
    assert(twoTemporariesTwoExits(true) == 11.0, "two temporaries, early exit");
    assert(twoTemporariesTwoExits(false) == 21.0, "two temporaries, tail exit");
    assert(earlyReturnFromNestedIf(true, true) == 12.0, "return from two blocks down");
    assert(earlyReturnFromNestedIf(true, false) == 22.0, "the path that falls through");
    assert(earlyReturnFromLoop(3) == 13.0, "return out of a loop");
    assert(earlyReturnFromLoop(99) == 23.0, "the loop that finishes");

    assert(breakDoesNotLeaveTheBlock(3) == 6.0, "`break` leaves the loop, not the block");
    assert(continueDoesNotLeaveTheBlock(3) == 14.0, "`continue` leaves the iteration, not the block");

    assert(readAfterEarlyReturn() == 8.0, "what an early return hands back survives");
    assert(readAfterTailReturn() == 9.0, "what a tail return hands back survives");

    print("done.");
}
