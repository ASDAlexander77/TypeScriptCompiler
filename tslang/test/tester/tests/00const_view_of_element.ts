// A `const` interface or bound method made over an element read keeps the object alive when the
// element is given a new value, as a `let` does. Folded, the const was the view itself and held no
// reference of its own: under rc `arr[0] = mkv(..)` freed the object `m()` and `i.get()` then ran
// on. Each call follows allocations of the same size, so a freed object is reused.
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
    const arr: V[] = [mkv(n), mkv(0)];
    const m = arr[0].get;
    arr[0] = mkv(n + 1);
    churn();
    return m() == n && arr[0].w.x == n + 1;
}

function interfaceView(n: number) {
    const arr: V[] = [mkv(0), mkv(n)];
    const i: I = arr[1];
    arr[1] = mkv(n + 1);
    churn();
    return i.get() == n && arr[1].w.x == n + 1;
}

function main() {
    for (let n = 0; n < 20; n++) {
        assert(boundMethod(n), "a const bound method over an element");
        assert(interfaceView(n), "a const interface over an element");
    }

    print("done.");
}
