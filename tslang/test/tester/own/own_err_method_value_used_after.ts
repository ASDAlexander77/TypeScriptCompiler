// -mm=own rejects: `v` moves into the method value `make` returns, and the array keeps it too.
class V {
    constructor(public x: number) {}

    get() {
        return this.x;
    }
}

function make(keep: V[], i: number) {
    const v = new V(i);
    const f = v.get;
    keep.push(v);
    return f;
}

function main() {
    const keep: V[] = [];
    print(make(keep, 3)());
}
