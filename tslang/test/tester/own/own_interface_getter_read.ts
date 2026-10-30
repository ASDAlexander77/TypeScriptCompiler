// -mm=own: a read through an interface's field a class fills with a getter is the getter's result, a value
// nothing else holds; through a plain field the same read is a borrow (spec 28). It was never freed: reading
// `i.item` two million times kept two million Boxes (617 MB). Now a read whose value - and what is borrowed
// from it - stays in its block is released after its last use, on the getter's path only
// (OwnershipInferencePass::releaseCellReads, ReleaseOpLowering). A read that escapes - returned, stored, put
// in a `let` - is not, and still leaks on the getter's path. A getter returning a field is rejected under own,
// so only fresh values come through a getter here. Values are read after churn() has reused anything freed
// too early.
class Box {
    data: number[] = [1, 2, 3];
    constructor(public n: number) {}
}

interface I {
    item: Box;
}

class Fresh implements I {
    private k = 0;
    get item() { this.k = this.k + 1; return new Box(10 * this.k); }
    set item(x: Box) { this.k = x.n; }
}

class Plain implements I {
    item = new Box(2);
}

function churn() {
    const junk: Box[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Box(999));
    }
}

function asTemporary(i: I) {
    return i.item.n + i.item.data.length;
}

function folded(i: I) {
    const x = i.item;
    churn();
    return x.n + x.data.length;
}

function use(b: Box) {
    churn();
    return b.n;
}

function asArgument(i: I) {
    return use(i.item);
}

function both(a: Box, b: Box) {
    return a.n + b.n;
}

// the same field read twice, alike: CSE must not make them one read freed twice
function readTwice(i: I) {
    let total: number = 0;
    for (let k = 0; k < 1000; k++) {
        total = total + both(i.item, i.item);
    }

    return total;
}

function inLoop(i: I) {
    let total: number = 0;
    for (let k = 0; k < 100; k++) {
        total = total + i.item.data.length;
    }

    return total;
}

function main() {
    const f: I = new Fresh();
    const p: I = new Plain();

    assert(asTemporary(f) == 13, "fresh objects read as temporaries");
    assert(asTemporary(p) == 5, "a plain field read as a temporary");

    assert(folded(f) == 33, "a fresh object read into a const");
    assert(folded(p) == 5, "a plain field read into a const");

    assert(asArgument(f) == 40, "a fresh object passed on");
    assert(asArgument(p) == 2, "a plain field passed on");

    assert(inLoop(f) == 300, "fresh objects read in a loop");
    assert(inLoop(p) == 300, "a plain field read in a loop");

    assert(readTwice(p) == 4000, "a plain field read twice");
    churn();
    assert(p.item.n == 2, "the plain field is still there");

    // an assignment calls the setter only, not the getter: the next read is 10 * (7 + 1)
    f.item = new Box(7);
    assert(f.item.n == 80, "an assignment does not read the field");

    print("done.");
}
