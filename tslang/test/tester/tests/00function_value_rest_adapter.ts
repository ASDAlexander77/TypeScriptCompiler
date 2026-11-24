// TSLANG 188: a function read at run time - an object literal's field, a function's result - given where the type is
// `(...args: A) => T`, in every memory model: -mm=own too, where it is held by an adapter in a box (it was refused,
// "a function value read at run time is not given as another function type yet")
function wrap<X>(x: X): { x: X } {
    return { x };
}

function call<A extends unknown[], T>(x: { x: (...args: A) => T }, ...args: A): T {
    return x.x(...args);
}

function main() {
    assert(call({ x: (n: number) => n + 1 }, 4) == 5, "a field");
    assert(call(wrap(<T>(x: T) => x), 1) == 1, "a generic arrow through a generic");
    assert(call({ x: (a: number, b: string) => b + a }, 2, "y") == "y2", "a tuple");

    let t: number = 0;
    for (let i = 0; i < 20000; i++) {
        t += call({ x: (n: number) => n * 2 }, i % 5);
    }
    assert(t == 80000, "many times");

    print("done.");
}
