// -mm=own: `m` is `v.get`, a method value, which takes the object `v` holds (USER RULING 2026-11-16);
// `v = mkv(2)` then gives `v` a new object and leaves `m`'s alone, and `m`'s release frees it.
class W {
    constructor(public x: number) {}
}

class V {
    constructor(public w: W) {}
    get() {
        return this.w.x;
    }
}

function mkv(n: number) {
    return new V(new W(n));
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) junk.push(mkv(999));
}

function main() {
    for (let i = 0; i < 500; i++) {
        let v = mkv(i);
        const m = v.get;
        v = mkv(i + 1);
        churn();
        assert(m() == i && v.w.x == i + 1, "the method value keeps the first object");
    }

    print("done.");
}
