// The field given to a call that keeps it, and the same field given again: the callee would own one
// argument and borrow the other, the same tree.
class T {
    l: T | null = null;
    constructor(public v: number) {}
}

function f(a: T | null, b: T | null): T { const n = new T(1); n.l = a; return n; }

function main() {
    const t = new T(0);
    t.l = new T(2);
    t.l = f(t.l, t.l);
    print(t.v);
}
