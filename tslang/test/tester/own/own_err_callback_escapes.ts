// -mm=own rejects: `pick` is passed to `reduce`, whose calls through its callback get the facts
// of the functions its callers pass there (spec 33), but `pick` is also held in a `let` and called
// through it, where -mm=own cannot see which function runs: those callers cannot be told that it
// keeps `acc`, and `pick`'s body fails, saying why.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) {
        return new V(a.x + b.x);
    }
}

function pick(acc: V, v: V) {
    return v.x > 2 ? acc : V.plus(acc, v);
}

function main() {
    const a = [new V(1), new V(3)];
    const q = a.reduce(pick, new V(0));
    let g = pick;
    const r = g(new V(1), new V(5));
    print(q.x, r.x);
}
