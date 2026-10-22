// A `const` interface made over a local (`const i: I = v`) keeps the object alive when the local
// is given a new value. Folded, the interface was a view of the read and held no reference of its
// own: under rc and own the assignment freed the object it still showed. Allocations of the same
// size reuse anything freed before each read.
class W {
    constructor(public x: number) {}
}

class V {
    items: number[] = [1, 2, 3];
    constructor(public w: W) {}
}

interface I {
    w: W;
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

function view(n: number) {
    let v = mkv(n);
    const i: I = v;
    v = mkv(n + 1);
    churn();
    return i.w.x == n && v.w.x == n + 1;
}

function main() {
    for (let n = 0; n < 20; n++) {
        assert(view(n), "an interface over a local outlives the local's assignment");
    }

    print("done.");
}
