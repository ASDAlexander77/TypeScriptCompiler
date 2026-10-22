// -mm=own rejects: `add` only borrows its accumulator, but `pick`, passed to the same
// `___array_reduce<V, V>`, keeps it, so `add` is given it to keep too and releases it (spec 33).
// A direct call of `add` then moves `x` in, and `x` is read after it.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) {
        return new V(a.x + b.x);
    }
}

function add(acc: V, v: V) {
    return V.plus(acc, v);
}

function pick(acc: V, v: V) {
    return v.x > 1 ? acc : V.plus(acc, v);
}

function main() {
    const a = [new V(1), new V(2)];
    const r = a.reduce(add, new V(0));
    const q = a.reduce(pick, new V(0));
    const x = new V(5);
    const y = add(x, new V(1));
    print(r.x, q.x, y.x, x.x);
}
