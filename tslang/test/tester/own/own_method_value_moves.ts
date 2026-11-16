// -mm=own: a method taken as a value (`v.get`) is `{fn, this, tag}`, released through its class's tag,
// so it is its object. Stored into a field or returned it takes the object (USER RULING 2026-11-16:
// the object moves in); kept in a `let` while the object is still used, it borrows it. Objects hold
// arrays made at run time, so a double release or a lost one shows.
class W {
    items: number[] = [];
    constructor(public x: number) {
        this.items.push(x);
    }
}

class V {
    constructor(public w: W) {}

    get() {
        return this.w.x + this.w.items.length;
    }
}

class H {
    f: () => number = () => 0;
}

function mkv(n: number) {
    return new V(new W(n));
}

function churn() {
    const a: V[] = [];
    for (let i = 0; i < 64; i++) a.push(mkv(1000 + i));
    return a.length;
}

// into a field: the holder owns the object now
function stored(h: H, n: number) {
    const v = mkv(n);
    h.f = v.get;
}

// returned: the caller owns it
function returned(n: number): () => number {
    const v = mkv(n);
    return v.get;
}

// kept in a `let` while the object is used: a borrow
function borrowed(n: number) {
    let v = mkv(n);
    let f = v.get;
    return f() + v.w.x;
}

// the `let` it was read from given a new object: the method value took the old one
function movedThenAssigned(n: number) {
    let v = mkv(n);
    let f = v.get;
    v = mkv(n + 1);
    churn();
    return f() * 100 + v.w.x;
}

// a method of a class with a subclass: reached through the vtable, the object moves in the same way
class Base {
    constructor(public x: number) {}

    get() {
        return this.x;
    }
}

class Twice extends Base {
    get() {
        return this.x * 2;
    }
}

function storedVirtual(h: H, n: number) {
    const b: Base = new Twice(n);
    h.f = b.get;
}

function main() {
    const h = new H();
    for (let i = 0; i < 200; i++) {
        storedVirtual(h, i);
        churn();
        assert(h.f() == 2 * i, "a virtual method");
    }

    for (let i = 0; i < 2000; i++) {
        const k = i % 10;
        stored(h, k);
        churn();
        assert(h.f() == k + 1, "stored");

        const g = returned(k);
        churn();
        assert(g() == k + 1, "returned");

        assert(borrowed(k) == 2 * k + 1, "borrowed");
        assert(movedThenAssigned(k) == (k + 1) * 100 + k + 1, "moved, then assigned");
    }

    print("done.");
}
