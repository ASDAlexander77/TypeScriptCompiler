// A method bound to an object and kept - in a `let`, a field, an element - keeps that object alive
// when the variable it was read from is given a new value. Under rc a bound method carried no type
// tag, so keeping it counted nothing, and `v = mkv(2)` freed the object `m()` then ran on. Each
// call follows allocations of the same size, so a freed object is reused.
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
    f: () => number = () => 0;
}

function mkv(n: number) {
    return new V(new W(n));
}

function churn(): number {
    let s = 0;
    const a: V[] = [];
    for (let i = 0; i < 64; i++) a.push(mkv(1000 + i));
    for (const e of a) s += e.w.x;
    return s;
}

function inLet(n: number) {
    let v = mkv(n);
    let m = v.get;
    v = mkv(n + 1);
    churn();
    return m() == n && v.w.x == n + 1;
}

function inField(n: number) {
    let v = mkv(n);
    const h = new H();
    h.f = v.get;
    v = mkv(n + 1);
    churn();
    return h.f() == n;
}

function inElement(n: number) {
    let v = mkv(n);
    const methods = [v.get];
    v = mkv(n + 1);
    churn();
    return methods[0]() == n;
}

function main() {
    for (let n = 0; n < 20; n++) {
        assert(inLet(n), "a bound method in a let");
        assert(inField(n), "a bound method in a field");
        assert(inElement(n), "a bound method in an element");
    }

    print("done.");
}
