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

    print("done.");
}
