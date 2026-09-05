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
// The cases from `capturedObjectEscapes` down are about the other half: who owns the *cell* a
// captured variable lives in, which is section 9.34, and the ones from
// `capturedParameterEscapes` are about a cell that starts out holding something the frame does
// not own - a captured parameter - which is section 9.35.
//
// See docs/reference-counting-evaluation.md sections 9.33 to 9.35.

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
// These two capture a number, so that what they check is the box's own lifetime and nothing
// else. What a captured *object* needs on top of that is section 9.34's question, and has cases
// of its own further down.
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

// A variable a closure captures by reference does not live in the frame: its storage is a heap
// block of its own - a cell - so that the frame and the closure read and write the same
// variable. Section 9.34 is about who owns that cell, and the cases below are the four shapes
// that answer differently.
//
// Until then the frame released the *value* at scope exit and left the cell to leak, so a
// captured object carried out of its frame was read through a pointer to freed memory - the
// oldest bug in this file's neighbourhood, and older than any of the ownership work.
function makeObjectAdder(k: number): (v: Vec) => number {
    let bump = new Vec(k);

    return (v: Vec) => v.x + bump.x;
}

function capturedObjectEscapes() {
    let add = makeObjectAdder(40);
    churn();

    return add(new Vec(2));
}

// Captured by value rather than through a cell: a `const` goes into the box as a copy of the
// reference, and a copy of a reference is a further owner of what it points at.
function makeReader(): () => number {
    const held = new Vec(15);

    return () => held.x;
}

function constCaptureIsOwned() {
    let read = makeReader();
    churn();

    return read();
}

// One variable, two closures. While both live the cell has three owners - the frame and each
// box - and nothing may free it until the last of them lets go.
function makePair(k: number): ((v: Vec) => number)[] {
    let shared = new Vec(k);

    return [(v: Vec) => v.x + shared.x, (v: Vec) => v.x - shared.x];
}

function capturedCellSharedByTwoClosures() {
    let fns = makePair(10);
    churn();

    return fns[0](new Vec(5)) + fns[1](new Vec(5));
}

// The other direction: the frame outlives every closure over the variable. Dropping the last
// box must not take the variable with it, which is what the frame's own reference to the cell
// is for.
function frameOutlivesTheClosure() {
    let kept = new Vec(3);
    {
        let f = (v: Vec) => v.x + kept.x;
        apply(f, new Vec(1));
    }

    churn();

    return kept.x;
}

// Assigning to a captured variable is still an ordinary assignment - the value in the cell is
// replaced, the cell is not - and the frame and the closure see the one variable.
function mutateThroughCapture() {
    let cur = new Vec(1);
    let step = () => { cur = new Vec(cur.x + 1); };

    step();
    step();
    churn();

    return cur.x;
}

// A captured parameter lives in a cell too, and the difference is what the cell starts out
// holding: an argument, which belongs to the caller. So the cell takes a reference to it, which
// is what makes a cell the owner of its contents whoever put them there - section 9.35.
//
// The escape is two frames deep on purpose. `new Vec(50)` is a discarded temporary of
// `holdParamReader`, released at the end of that block, so if the cell had not taken a reference
// the value is gone before `capturedParameterEscapes` ever reads it.
function makeParamReader(v: Vec): () => number {
    return () => v.x;
}

function holdParamReader(): () => number {
    return makeParamReader(new Vec(50));
}

function capturedParameterEscapes() {
    let read = holdParamReader();
    churn();

    return read();
}

// Assigning to a captured parameter from inside the closure gives up what the cell held - which
// is the caller's argument. The caller reads it afterwards, so if the cell were giving up a
// reference it never took, `held` is freed here while `mutateCapturedParameter` still holds it.
function bumpThroughCapture(v: Vec): number {
    let step = () => { v = new Vec(v.x + 1); };

    step();

    return v.x;
}

function mutateCapturedParameter() {
    let held = new Vec(1);
    let bumped = bumpThroughCapture(held);
    churn();

    return bumped + held.x;
}

// The same assignment seen from the other side: written in the frame that declared the
// parameter, after the closure over it exists. The value stored has to be taken by the cell, or
// nothing holds it and the end of the block gives it back as a discarded temporary.
function reassignCapturedParameter(v: Vec): () => number {
    let read = () => v.x;
    v = new Vec(v.x + 10);

    return read;
}

function capturedParameterReassignedInFrame() {
    let read = reassignCapturedParameter(new Vec(5));
    churn();

    return read();
}

function main() {
    assert(closureAsArgument() == 10, "a closure used as an argument survives the call");
    assert(closureReadAfterCalleeAllocates() == 12, "a capture box survives a callee that allocates first");
    assert(closureEscapesItsBlock() == 11, "a returned closure keeps its capture box");
    assert(closureKeptInArray() == 23, "a stored closure keeps its capture box");
    assert(closuresInALoop() == 72, "a loop's capture boxes do not disturb one another");
    assert(boundMethodKeepsItsObject() == 30, "a bound method does not take ownership of its object");
    assert(capturedObjectEscapes() == 42, "a captured object leaves the frame that made it");
    assert(constCaptureIsOwned() == 15, "a box owns what it captured by value");
    assert(capturedCellSharedByTwoClosures() == 10, "two closures share one captured variable");
    assert(frameOutlivesTheClosure() == 3, "a captured variable outlives the closures over it");
    assert(mutateThroughCapture() == 3, "the frame and the closure see one variable");
    assert(capturedParameterEscapes() == 50, "a cell owns the argument a captured parameter arrived with");
    assert(mutateCapturedParameter() == 3, "assigning through a captured parameter leaves the caller's value alone");
    assert(capturedParameterReassignedInFrame() == 15, "a captured parameter's cell takes what the frame stores in it");

    print("done.");
}
