// Every function retains its result on the way out, so a call hands back a reference whether or
// not the caller does anything with it. Where a receiver takes that over the pair is balanced
// (00owned_call_results.ts); where nothing does - a result passed straight as an argument and
// then dropped, which is what expression-shaped code is made of - the reference stands with no
// owner. Those are given back at the end of the block that produced them (section 9.30).
//
// What these guard is the direction that corrupts. Under-releasing a temporary only leaks, which
// is invisible from inside the program; releasing one that was still owned frees a value while
// something is still using it. Every case therefore calls `churn()` between the release point and
// the read, so a freed block is claimed by something else and a use-after-free shows up as a
// wrong answer rather than as the value that used to be there.
//
// See docs/reference-counting-evaluation.md section 9.30.

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

function times(k: number, v: Vec): Vec {
    return new Vec(k * v.x);
}

function plus(a: Vec, b: Vec): Vec {
    return new Vec(a.x + b.x);
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

// the raytrace shape: every intermediate is a call result used as an argument and never bound
function nestedCallArguments() {
    let v = plus(times(2, new Vec(3)), new Vec(1));
    churn();

    return v.x;
}

// two levels of nesting, so an intermediate is itself built from intermediates
function deeperNesting() {
    let v = plus(plus(times(2, new Vec(1)), times(3, new Vec(1))), new Vec(4));
    churn();

    return v.x;
}

// the same expression evaluated repeatedly: each iteration's temporaries are given back at the
// end of that iteration, and the value carried out of the loop is untouched
function temporariesInALoop() {
    let total = 0;
    let last = new Vec(0);
    for (let i = 0; i < 8; i++) {
        last = plus(times(2, new Vec(i)), new Vec(1));
        total = total + last.x;
    }

    churn();

    return total + last.x;
}

// a temporary the callee keeps: push retains what it stores, so the release of the call's own
// reference must not take the element with it
function temporaryKeptByCallee() {
    let arr: Vec[] = [];
    arr.push(times(2, new Vec(5)));
    arr.push(new Vec(7));
    churn();

    return arr[0].x + arr[1].x;
}

// a result used as an argument AND bound to a local: the local's own reference has to outlive
// the argument use
function usedAsArgumentAndBound() {
    let a = new Vec(6);
    let b = plus(a, new Vec(1));
    churn();

    return a.x + b.x;
}

// The callee allocates before it reads its arguments, so a temporary released too early is not
// merely freed but overwritten before the read that needs it. Without this, `plus(a, new Vec(1))`
// reads a freed block that still happens to hold its old value and the case passes for the wrong
// reason - which is what the release-before-use probe showed about the case above.
function plusAfterChurn(a: Vec, b: Vec): Vec {
    churn();

    return new Vec(a.x + b.x);
}

function argumentReadAfterCalleeAllocates() {
    let v = plusAfterChurn(new Vec(20), new Vec(3));
    churn();

    return v.x;
}

// a discarded result - nothing reads it at all
function discardedResult() {
    let keep = new Vec(12);
    times(2, new Vec(4));
    plus(new Vec(1), new Vec(2));
    churn();

    return keep.x;
}

// a temporary handed to a callee that reads it through a field
function temporaryReadByCallee() {
    let v = times(3, plus(new Vec(2), new Vec(2)));
    churn();

    return v.x;
}

function main() {
    assert(nestedCallArguments() == 7, "nested call arguments survive their statement");
    assert(deeperNesting() == 9, "an intermediate built from intermediates survives");
    assert(temporariesInALoop() == 79, "a loop body's temporaries do not disturb what it carries out");
    assert(temporaryKeptByCallee() == 17, "a temporary the callee keeps is not freed under it");
    assert(usedAsArgumentAndBound() == 13, "a value used as an argument is still owned by its local");
    assert(argumentReadAfterCalleeAllocates() == 23, "a temporary argument survives a callee that allocates before reading it");
    assert(discardedResult() == 12, "discarding a result does not disturb anything live");
    assert(temporaryReadByCallee() == 12, "a temporary read by a callee survives the call");

    print("done.");
}
