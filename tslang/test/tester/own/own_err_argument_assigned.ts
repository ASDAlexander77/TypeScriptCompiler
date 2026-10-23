// -mm=own rejects: `g` borrows `n`'s object as `a`, but the second argument gives `n` a new value
// first, which frees that object before the call reads it.
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
    print(g(n, (n = mk(2), 0)));
}
