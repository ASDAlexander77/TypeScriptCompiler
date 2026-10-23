// -mm=own rejects: a conditional argument may give `n` a new value after the first argument read
// it, which frees what `g` is lent.
class V {
    constructor(public x: number) {}
}

function mk(n: number) {
    return new V(n);
}

function g(a: V, b: number) {
    return a.x * 10 + b;
}

function main() {
    let n = mk(1);
    const flag = n.x > 0;
    print(g(n, flag ? (n = mk(2), 0) : 1));
}
