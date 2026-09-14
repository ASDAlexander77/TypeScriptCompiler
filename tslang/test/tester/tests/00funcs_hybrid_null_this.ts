// A plain function cast to a function type is a hybrid function with a null `this`;
// calling it directly must not pass a `this` (and is lowered to a direct call).
function add1(x: number) {
    return x + 1;
}

class C {
    x: number = 10;
    m(): number {
        return this.x;
    }
}

function main() {
    const p: Opaque = <Opaque>add1;
    const r = (<(x: number) => number>p)(41);
    assert(r == 42, "direct call");

    let r2 = 0;
    try {
        r2 = (<(x: number) => number>p)(1);
    } finally {
        assert(r2 == 2, "direct call inside try");
    }

    // a real bound method in a function-typed variable still passes `this`
    const c = new C();
    const g: () => number = c.m;
    assert(g() == 10, "bound method");

    print("done.");
}
