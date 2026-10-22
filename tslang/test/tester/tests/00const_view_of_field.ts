// A `const` interface or bound method made over a field read keeps the object alive when the field
// is given a new value, as a `let` does. Folded, the const was the view itself and held no
// reference of its own: under rc `h.v = mkv(..)` freed the object `m()` and `i.get()` then ran on.
// Each call follows allocations of the same size, so a freed object is reused.
interface I {
    get(): number;
}

class W {
    constructor(public x: number) {}
}

class V {
    items: number[] = [1, 2, 3];
    constructor(public w: W) {}

    get() {
        return this.w.x;
    }
}

class H {
    v: V = mkv(0);
}

function mkv(n: number) {
    return new V(new W(n));
}

function churn(): number {
    let s: number = 0;
    const a: V[] = [];
    for (let i = 0; i < 64; i++) a.push(mkv(1000 + i));
    for (const e of a) s += e.w.x;
    return s;
}

function boundMethod(n: number) {
    const h = new H();
    h.v = mkv(n);
    const m = h.v.get;
    h.v = mkv(n + 1);
    churn();
    return m() == n && h.v.w.x == n + 1;
}

function interfaceView(n: number) {
    const h = new H();
    h.v = mkv(n);
    const i: I = h.v;
    h.v = mkv(n + 1);
    churn();
    return i.get() == n && h.v.w.x == n + 1;
}

function main() {
    for (let n = 0; n < 20; n++) {
        assert(boundMethod(n), "a const bound method over a field");
        assert(interfaceView(n), "a const interface over a field");
    }

    print("done.");
}
