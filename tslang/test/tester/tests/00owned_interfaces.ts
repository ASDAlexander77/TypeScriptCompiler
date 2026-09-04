// An interface value is a reference to whatever it was made from, and until section 9.31 it
// owned nothing: the type carries only a name, so the layout behind its `this` is not
// recoverable from it. It does not have to be - the value now carries the runtime type tag of
// its `this` beside the pointer, and release and retain go through that, the same way an `any`
// box has always worked.
//
// Making an interface an owner turned two dormant omissions into live over-releases, and both
// are guarded here. A block filled from a literal has to take a reference to what it holds
// (section 9.21) - `castTupleToInterface` allocates such a block and did not; and a value pushed
// into an array already owned has to be marked consumed - `retainInsertedElements` skipped its
// retain without saying so, and section 9.30 then released it at the end of the pushing block.
//
// Both are invisible while the block that builds is also the block that reads, which is why
// section 9.30's own tests missed them: the release goes at the END of the producing block, so a
// read in that same block still happens first. Every case here therefore builds in one block and
// reads in another, with `churn()` in between so a freed block is claimed by something else and
// a use-after-free shows up as a wrong answer rather than as the value that used to be there.
//
// See docs/reference-counting-evaluation.md section 9.31.

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

interface Holder {
    v: Vec;
}

interface Point {
    x: number;
}

interface Nested {
    inner: Holder;
    tag: number;
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

// the shape section 9.31 exists for: a literal boxed as an interface, handed back, and read by
// somebody else
function makeHolder(n: number): Holder {
    return { v: new Vec(n) };
}

function returnedThroughInterface() {
    let h = makeHolder(7);
    churn();

    return h.v.x;
}

// an interface temporary passed straight as an argument and never bound - what raytrace is made
// of, and what nothing gave back before 9.31
function readHolder(h: Holder): number {
    churn();

    return h.v.x;
}

function argumentNeverBound() {
    return readHolder({ v: new Vec(11) });
}

// The boxed block outlives the block that built it, so the reference it holds to `v` has to be
// one it took. Without the retain at the boxing site, `new Vec(13)` is the only owner, it is
// released at the end of makeAndKeep, and the read below finds whatever churn put there.
let kept: Holder[] = [];

function makeAndKeep() {
    kept.push({ v: new Vec(13) });
}

function boxedLiteralOutlivesItsBlock() {
    makeAndKeep();
    churn();

    return kept[0].v.x;
}

// The same question for the element itself rather than for what it holds: push takes over the
// reference `new Vec(17)` arrives with, so nothing may release it afterwards.
let vecs: Vec[] = [];

function pushOwned() {
    vecs.push(new Vec(17));
}

function pushedResultIsNotReleased() {
    pushOwned();
    churn();

    return vecs[0].x;
}

// an interface field inside another boxed literal: the outer block owns the inner interface,
// which owns the Vec
function makeNested(n: number): Nested {
    return { inner: { v: new Vec(n) }, tag: 3 };
}

function interfaceHeldByInterface() {
    let outer = makeNested(19);
    churn();

    return outer.inner.v.x + outer.tag;
}

// a class instance behind an interface, where the class keeps its own owner as well
class Counter implements Point {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

function classThroughInterface() {
    let c = new Counter(23);
    let p: Point = c;
    churn();

    return c.x + p.x;
}

// an interface with no `this` at all - the tag is null and both directions have to do nothing
// rather than read through it
function nullInterfaceIsInert() {
    let h: Holder = undefined;
    churn();

    return h == undefined ? 29 : 0;
}

// many interfaces built and dropped in a loop, each iteration's given back at the end of that
// iteration, with the one carried out untouched
function interfacesInALoop() {
    let last = makeHolder(0);
    let total = 0;
    for (let i = 1; i <= 8; i++) {
        last = makeHolder(i);
        total = total + readHolder({ v: new Vec(i) });
    }

    churn();

    return total + last.v.x;
}

function main() {
    assert(returnedThroughInterface() == 7, "a literal boxed as an interface survives its maker");
    assert(argumentNeverBound() == 11, "an interface argument nothing bound survives the call");
    assert(boxedLiteralOutlivesItsBlock() == 13, "a boxed literal owns what it holds");
    assert(pushedResultIsNotReleased() == 17, "a pushed owned result is not released under the array");
    assert(interfaceHeldByInterface() == 22, "an interface held by an interface stays alive");
    assert(classThroughInterface() == 46, "a class behind an interface keeps its own owner");
    assert(nullInterfaceIsInert() == 29, "a null interface releases nothing");
    assert(interfacesInALoop() == 44, "a loop's interface temporaries do not disturb what it carries out");

    print("done.");
}
