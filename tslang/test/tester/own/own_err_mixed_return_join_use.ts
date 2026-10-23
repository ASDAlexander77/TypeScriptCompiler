// -mm=own rejects: `g` returns `n` on one path and a new object on the other, so it keeps `n` and
// releases it where the paths join on the path that did not return it (spec 33). `v`, read out of
// `n` before the join, is used after it: on that path its block was released.
class V {
    constructor(public x: number) {}
}

class N {
    constructor(public v: V) {}
}

function g(n: N, k: number) {
    const v = n.v;
    const r = k > 1 ? n : new N(new V(1));
    print(v.x);
    return r;
}

function main() {
    print(g(new N(new V(3)), 0).v.x);
}
