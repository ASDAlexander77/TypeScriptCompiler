// TSLANG 188: -mm=own, a generator's `next()` and `throw()` called by hand through Generator<T> give a `{ value,
// done }` the caller owns: what a generator of strings yielded is given back where nothing keeps it (it was never
// freed - the heap grew with every call). Run many times, kept in a variable or not, the generator finished or not.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
    throw?: (e: any) => IterateResult<T>;
}

function* words(k: number): Generator<string> {
    try {
        yield "a" + k;
        yield "b" + k;
    } catch (e) {
    }
    yield "c" + k;
}

function main() {
    let n: number = 0;
    for (let i = 0; i < 20000; i++) {
        const g = words(i);
        const r = g.next();
        n += r.value.length;
        g.next();
        if (i % 2 == 0) {
            n += g.throw!("x" + i).value.length;
        }
    }

    assert(n > 0, "read");

    const g = words(7);
    assert(g.next().value == "a7" && g.next().value == "b7" && g.next().value == "c7" && g.next().done, "to the end");
    print("done.");
}
