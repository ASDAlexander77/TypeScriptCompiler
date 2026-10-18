// The outer object goes on the stack; the inner one is stored in a field, so it stays on the heap.
class Inner {
    v: number;
    constructor(v: number) {
        this.v = v;
    }
}

class Outer {
    inner: Inner;
    constructor(inner: Inner) {
        this.inner = inner;
    }
}

function f(i: number) {
    const o = new Outer(new Inner(i));
    return o.inner.v;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        t += f(i);
    }
    assert(t == 499500);
    print("done.");
}
