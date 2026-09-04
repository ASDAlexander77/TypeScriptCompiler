// A closure's `this` is its capture box - heap-allocated, and named nowhere in the function
// type, so until section 9.33 nothing ever gave it back. It carries a type tag beside the
// pointer now, the same arrangement an interface got in section 9.31, and releases through the
// box's own routine.
//
// The tag is what separates the two kinds of function value that share one representation. A
// closure owns its capture box; a bound method's `this` is an object that belongs to whoever
// holds it, and `obj.m` must not take a reference to `obj` or hand one back. Only a closure
// built over captured variables gets a tag - `boundMethodKeepsItsObject` below is the case that
// fails loudly if that ever stops being true.
//
// What every case guards is the over-release direction. A capture box freed while a closure
// still refers to it reads whatever was allocated over it, so each case calls `churn()` between
// the point a release could happen and the point the captured values are read. Cases where the
// closure escapes - returned, stored, kept in an array - are the ones that would break if the
// box were given back at the end of the block that built it.
//
// See docs/reference-counting-evaluation.md section 9.33.

class Vec {
    x: number;

    constructor(x: number) {
        this.x = x;
    }
}

// Allocate over whatever has just been freed, so a use-after-free reads something else.
function churn() {
    for (let i = 0; i < 64; i++) {
        let filler = new Vec(999);
    }
}

function apply(f: (v: Vec) => number, v: Vec): number {
    return f(v);
}

// The callee allocates before it calls, so a box released too early is not merely freed but
// overwritten before the closure reads it.
function applyAfterChurn(f: (v: Vec) => number, v: Vec): number {
    churn();

    return f(v);
}

// the shape section 9.33 is about: a closure built, used as an argument, and never bound
function closureAsArgument() {
    let base = new Vec(5);

    return apply((v: Vec) => v.x + base.x, base);
}

// the same, with the box's release point crossed by an allocation before the call happens
function closureReadAfterCalleeAllocates() {
    let base = new Vec(6);

    return applyAfterChurn((v: Vec) => v.x + base.x, base);
}

// A closure that outlives the block that made it: the box must not be given back at the end of
// `makeAdder`, which is exactly where a discarded one would be.
//
// These two capture a number rather than an object on purpose. A captured *object* carried out
// of its frame is freed by that frame's own scope exit - a separate, older bug that has nothing
// to do with the capture box (section 9.33), and one that would mask what these are here to
// check.
function makeAdder(k: number): (v: Vec) => number {
    let bump = k + 1;

    return (v: Vec) => v.x + bump;
}

function closureEscapesItsBlock() {
    let add = makeAdder(9);
    churn();

    return add(new Vec(1));
}

// a closure kept in an array, called long after the block that built it is gone
let handlers: ((v: Vec) => number)[] = [];

function registerHandler(k: number) {
    let bump = k + 1;
    handlers.push((v: Vec) => v.x + bump);
}

function closureKeptInArray() {
    registerHandler(20);
    churn();

    return handlers[0](new Vec(2));
}

// many closures built and dropped, each iteration's box given back at the end of that iteration
function closuresInALoop() {
    let total = 0;
    for (let i = 1; i <= 8; i++) {
        let base = new Vec(i);
        total = total + apply((v: Vec) => v.x + base.x, base);
    }

    churn();

    return total;
}

// A bound method is not a closure: its `this` is the object, owned by whoever holds it. If a
// bound method ever took ownership of its receiver, this frees `holder` and reads churn's
// filler instead of 30.
class Holder {
    v: Vec;

    constructor(v: Vec) {
        this.v = v;
    }

    read(): number {
        return this.v.x;
    }
}

function boundMethodKeepsItsObject() {
    let holder = new Holder(new Vec(30));
    let f = holder.read;
    churn();

    return f() + holder.v.x - holder.read();
}

function main() {
    assert(closureAsArgument() == 10, "a closure used as an argument survives the call");
    assert(closureReadAfterCalleeAllocates() == 12, "a capture box survives a callee that allocates first");
    assert(closureEscapesItsBlock() == 11, "a returned closure keeps its capture box");
    assert(closureKeptInArray() == 23, "a stored closure keeps its capture box");
    assert(closuresInALoop() == 72, "a loop's capture boxes do not disturb one another");
    assert(boundMethodKeepsItsObject() == 30, "a bound method does not take ownership of its object");

    print("done.");
}
