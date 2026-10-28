// a callback whose result is ignored may return one: given to a function type that returns
// nothing, an arrow (or function expression) without a return type of its own evaluates what it
// returns and drops it (docs/not_impl/not_impl27.ts)

class Box {
    v = 1;
}

let count = 0;

function bump(s: string) {
    count++;
    return s.length;
}

function each(a: string[], f: (s: string) => void) {
    for (const s of a) f(s);
}

function someGenerics2a<T extends string>(n: (x: T) => void) {
    n("a" as T);
}

function someGenerics2b<T extends string, U extends number>(n: (x: T, y: U) => void) {
    n("b" as T, 2 as U);
}

function someGenerics4<T, U extends number>(n: T, f: (x: U) => void) {
    f(1 as U);
}

// with every type argument given, an arrow takes its return type from the specialised parameter
function produce<T>(producer: () => T) {
    return producer();
}

function main() {
    someGenerics2a((n: string) => n);
    someGenerics2a<string>((n: string) => n);
    someGenerics2b((n: string, x: number) => n);
    someGenerics2b<string, number>((n: string, t: number) => n);
    someGenerics4(4, (x) => null);

    const cb: (x: string) => void = (n: string) => n;
    cb("x");

    each(["a", "bb"], s => bump(s));
    each(["a", "bb"], s => { return bump(s); });
    each(["a"], function (s) { return new Box(); });
    each(["a"], s => new Box());
    assert(count == 4, "count");

    assert(produce<any>(() => undefined) === undefined, "any");
    assert(produce<number | undefined>(() => undefined) === undefined, "number | undefined");
    assert(produce<number>(() => 2) == 2, "number");

    print("done.");
}
