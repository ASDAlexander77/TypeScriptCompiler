// -mm=own rejects: the closure called for the second argument gives the captured `n` a new value
// after the first argument read it, which frees what `g` is lent.
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
    const set = () => {
        n = mk(2);
        return 0;
    };
    print(g(n, set()));
}
