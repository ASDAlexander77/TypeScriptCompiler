// regression test: a write through an interface to its function-typed field (`h.f = show`) stores
// to the class field that fills it. The slot points at a cell holding a thunk that calls the field
// (INTERFACE_METHOD_CELL_TAG, Defines.h), and the cell names the field's offset, so the store goes
// to the instance behind the interface: "saving to constant object" before.

class Item {
    constructor(public n: number) {}
}

interface Holder {
    f: (v: Item) => void;
}

class Walker {
    constructor(public f: (v: Item) => void) {}
}

let seen: number = 0;

function show(v: Item) {
    print(v.n);
    seen += v.n;
}

function twice(v: Item) {
    seen += v.n * 2;
}

function set(h: Holder) {
    h.f = show;
}

interface Counter {
    next: () => int;
}

class Steps {
    next: () => int = () => 0;
}

function count(c: Counter, by: int) {
    let k = 0;
    // a closure over a local, kept by the object after the call
    c.next = () => (k += by);
}

function main() {
    const w = new Walker(twice);
    set(w);
    // read back on the instance itself: the write landed in the object
    w.f(new Item(3));
    assert(seen == 3);

    // and through the interface
    const h: Holder = w;
    h.f(new Item(4));
    assert(seen == 7);

    h.f = twice;
    w.f(new Item(5));
    assert(seen == 17);

    // each instance keeps its own
    const other = new Walker(show);
    const ho: Holder = other;
    ho.f = twice;
    w.f = show;
    other.f(new Item(1));
    assert(seen == 19);

    const s = new Steps();
    count(s, 5);
    assert(s.next() == 5);
    const c: Counter = s;
    assert(c.next() == 10);

    print("done.");
}
