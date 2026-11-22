// A generator whose next() takes a value (Generator<T, TReturn, TNext>: `next: (value?: TNext) =>`) given where an
// iterator whose next() takes none is wanted (`next: () =>`): iterated by a generator over its values, not read as
// it is - an Iterable's next() called a function of another signature (wasm's "function signature mismatch",
// natively a yield read a value never given).
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
}

interface Steps<T> {
    next: () => IterateResult<T>;
}

function* numbers(): Generator<number> {
    yield 1.5;
    yield 2;
}

function* running(): Generator<number, void, number> {
    let total: number = 0;
    for (let i = 0; i < 3; i++) {
        const v = yield total;
        total += 1;
    }
}

function sum(values: Steps<number>) {
    let s: number = 0;
    for (const v of values) {
        s += v;
    }

    return s;
}

// a Generator<T, TReturn, TNext> given to a Generator<T>: its next() takes a value of another type (a number, not
// an any) - adapted the same way
function drive(g: Generator<number>) {
    g.next();
    const a = g.next();
    const b = g.next(5);
    return a.value + b.value;
}

function main() {
    assert(drive(running()) == 3, "a Generator<T, TReturn, TNext> as a Generator<T>");
    assert(sum(numbers()) == 3.5, "a Generator<T>");
    assert(sum(running()) == 3, "a Generator<T, TReturn, TNext>");

    const r = running();
    r.next();
    assert(r.next(5).value == 1, "next(value) still");

    print("done.");
}
