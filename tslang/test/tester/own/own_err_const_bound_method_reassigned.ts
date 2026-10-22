// -mm=own rejects: `m` is `v.get` bound to the object `v` holds, and `v = mkv(2)` frees that
// object before `m()` runs. The method keeps no count of its own.
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

function main() {
    let v = mkv(1);
    const m = v.get;
    v = mkv(2);
    print(m() + v.w.x);
}
