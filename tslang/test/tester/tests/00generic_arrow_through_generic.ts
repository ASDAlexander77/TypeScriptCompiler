// TSLANG 183 (94): a generic arrow given to a parameter typed by a type parameter (`wrap(<T>(x: T) => x)`), and a
// function given where the type is `(...args: A) => T` with A inferred from it.
function wrap<X>(x: X): { x: X } {
    return { x };
}

function apply<F>(f: F) {
    return f;
}

function call<A extends unknown[], T>(x: { x: (...args: A) => T }, ...args: A): T {
    return x.x(...args);
}

function call2<A extends unknown[], T>(f: (...args: A) => T, ...args: A): T {
    return f(...args);
}

const kept = call(wrap(<T>(x: T) => x), 1);

function main() {
    const w = wrap(<T>(x: T) => x);
    assert(w.x(5) == 5, "wrap");

    const g = apply(<T>(x: T) => x);
    assert(g(5) == 5, "apply");

    assert(call({ x: (n: number) => n + 1 }, 4) == 5, "call");
    assert(call2((n: number) => n + 1, 4) == 5, "call2");
    assert(kept == 1, "kept");

    // parameters of different types: A is the tuple of them (TSLANG 184)
    assert(call2((a: number, b: string) => b + a, 1, "x") == "x1", "call2 tuple");
    assert(call2((a: string, b: number, c: boolean) => c ? a + b : a, "n", 7, true) == "n7", "call2 three");
    assert(call({ x: (a: number, b: string) => b + a }, 2, "y") == "y2", "call tuple");

    print("done.");
}
