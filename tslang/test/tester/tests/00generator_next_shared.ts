// next(value) of a Generator<T, TReturn, TNext> sending objects: under -mm=own the generator keeps what it is sent,
// so TNext is a Shared<T> there (one of another type is an error that says so:
// diagnostic-location/generator_next_object_own.ts); in the other models the same program runs as it is.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
}
class Box { constructor(public n: number) {} }

function* summer(): Generator<number, void, Shared<Box>> {
    let total = 0;
    while (true) {
        const b = yield total;
        if (b) total += b.value.n;
    }
}

function main() {
    const g = summer();
    g.next();
    g.next(new Shared<Box>(new Box(2)));
    const r = g.next(new Shared<Box>(new Box(3)));
    assert(r.value == 5);
    print("done.");
}
