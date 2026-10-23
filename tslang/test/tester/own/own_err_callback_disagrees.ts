// -mm=own rejects: both callbacks reach the one `___array_reduce<V, V>`, and its call through the
// callback gets what they agree on (spec 33). `pick` keeps its accumulator (it returns it on some
// paths only), `first` returns a borrow of it on every path: no caller can be told both, so both
// lose their facts, and their bodies fail, saying why.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) {
        return new V(a.x + b.x);
    }
}

function pick(acc: V, v: V) {
    return v.x > 1 ? acc : V.plus(acc, v);
}

function first(acc: V, v: V) {
    return acc;
}

function main() {
    const a = [new V(1), new V(2)];
    const q = a.reduce(pick, new V(0));
    const r = a.reduce(first, new V(4));
    print(q.x, r.x);
}
