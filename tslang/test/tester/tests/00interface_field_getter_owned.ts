// A read through an interface's field that a class fills with accessors calls the getter (#719), and
// under rc a getter returns its result retained, as every function does. The read was treated as a
// borrowed one, the plain field's way, so nothing gave that reference back: reading `i.item` two
// million times kept two million Boxes alive (274 MB, against 3.4 MB through a plain field). And an
// assignment `i.item = v` left behind an unused read of the place it wrote, which called the getter
// and leaked its result as well.
//
// Under rc such a read is now owned on both paths (InterfaceFieldAccess): the lowering retains what a
// plain slot holds, and a receiver takes the reference over or it is given back as a temporary. An
// unused read is dropped. Each such read carries a number of its own, or CSE (--opt) folds two reads
// of the same field into one, retained once and released twice (`readTwice`). interface-field-getter-owned.cmake checks the releases in the IR; here each
// value is read after allocations of the same size have reused anything freed too early.
class Box {
    data: number[] = [1, 2, 3];
    constructor(public n: number) {}
}

interface I {
    item: Box;
}

// a getter of a stored object
class Stored implements I {
    private b = new Box(1);
    get item() { return this.b; }
    set item(x: Box) { this.b = x; }
}

// a getter of a fresh object: the read holds its only reference
class Fresh implements I {
    private k = 0;
    get item() { this.k = this.k + 1; return new Box(10 * this.k); }
    set item(x: Box) { this.k = x.n; }
}

// a plain field: the slot path of the same read
class Plain implements I {
    item = new Box(2);
}

let gets = 0;

class Counted implements I {
    private b = new Box(3);
    get item() { gets = gets + 1; return this.b; }
    set item(x: Box) { this.b = x; }
}

function churn() {
    const junk: Box[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new Box(999));
    }
}

function intoLocal(i: I) {
    const x = i.item;
    churn();
    return x.n + x.data.length;
}

function asTemporary(i: I) {
    return i.item.n + i.item.data.length;
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

// the same field read twice, alike: CSE must not make them one read given back twice
function readTwice(i: I) {
    let total: number = 0;
    for (let k = 0; k < 1000; k++) {
        total = total + both(i.item, i.item);
    }

    return total;
}

function returned(i: I) {
    return i.item;
}

function inLoop(i: I) {
    let total: number = 0;
    for (let k = 0; k < 100; k++) {
        total = total + i.item.data.length;
    }

    return total;
}

function stored(i: I, n: number) {
    i.item = new Box(n);
}

function storedLocal(i: I) {
    const b = new Box(7);
    i.item = b;
    churn();
    return b.n + i.item.n;
}

function main() {
    const s: I = new Stored();
    const f: I = new Fresh();
    const p: I = new Plain();

    assert(intoLocal(s) == 4, "a stored object read into a local");
    assert(intoLocal(f) == 13, "a fresh object read into a local");
    assert(intoLocal(p) == 5, "a plain field read into a local");

    assert(asTemporary(s) == 4, "a stored object read as a temporary");
    assert(asTemporary(f) == 23, "a fresh object read as a temporary");
    assert(asTemporary(p) == 5, "a plain field read as a temporary");

    assert(asArgument(s) == 1, "a stored object passed on");
    assert(asArgument(f) == 40, "a fresh object passed on");
    assert(asArgument(p) == 2, "a plain field passed on");

    assert(readTwice(s) == 2000, "a stored object read twice");
    assert(readTwice(p) == 4000, "a plain field read twice");

    const r = returned(f);
    churn();
    assert(r.n == 50, "a fresh object returned");
    assert(returned(p).n == 2, "a plain field returned");

    assert(inLoop(s) == 300, "read in a loop");
    assert(inLoop(f) == 300, "a fresh object read in a loop");

    stored(s, 5);
    churn();
    assert(s.item.n == 5, "a store through the setter");
    stored(p, 6);
    churn();
    assert(p.item.n == 6, "a store into a plain field");

    assert(storedLocal(s) == 14, "a stored local is still the local's");
    assert(storedLocal(p) == 14, "a stored local is still the local's, through a plain field");

    // an assignment calls the setter only, not the getter
    const c: I = new Counted();
    c.item = new Box(4);
    c.item = new Box(5);
    assert(gets == 0, "an assignment does not read the field");
    assert(c.item.n == 5 && gets == 1, "a read calls the getter once");

    print("done.");
}
