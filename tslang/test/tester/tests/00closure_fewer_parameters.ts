// An arrow that captures, taking fewer parameters than the function type it is given as, ignores the rest: it
// is bound over its captures to a function that takes them all and calls it with the first ones. It was an
// error, "it takes 2 argument(s), 2 expected" (the captures counted as a parameter).
class Box {
    constructor(public n: number) {}
}

function each(names: string[], visit: (name: string, index: number) => void) {
    for (let i = 0; i < names.length; i++) {
        visit(names[i], i);
    }
}

function apply(f: (a: number, b: number) => number) {
    return f(2, 3);
}

function eachOf<T>(xs: T[], f: (v: T, i: number) => void) {
    for (let i = 0; i < xs.length; i++) f(xs[i], i);
}

function mapOf<T, V>(xs: T[], f: (v: T, i: number) => V) {
    const result: V[] = [];
    for (let i = 0; i < xs.length; i++) result.push(f(xs[i], i));
    return result;
}

function churn() {
    const junk: Box[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Box(999));
}

function main() {
    const k: number = 10;
    assert(apply((a: number) => a + k) == 12);

    let count = 0;
    each(["ab", "c"], (name: string) => { count += name.length; });
    assert(count == 3);

    const b = new Box(4);
    const f = (a: number) => a * b.n;
    assert(apply(f) == 8);
    churn();
    assert(apply(f) == 8);
    assert(b.n == 4);

    let calls = 0;
    each(["x", "y", "z"], () => { calls++; });
    assert(calls == 3);

    // a generic callback: the arrow's captures are not a parameter `T` is inferred from (it was
    // `string | Reference<...>`)
    let joined = "";
    eachOf(["p", "q"], (v) => { joined += v; });
    assert(joined == "pq");

    // a result the thunk hands on owns its block (-mm=own took it for a second reference)
    assert(mapOf(["m", "n"], (v) => v + "!")[1] == "n!");

    print("done.");
}
