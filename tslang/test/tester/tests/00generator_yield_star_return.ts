// return() of a generator paused in a `yield*` over a Generator: the delegate's return() runs first, with the same
// value - its finally blocks run, as in JavaScript - and then the generator finishes (its own finally blocks too).
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
}

let log = "";

function* inner(): Generator<number> {
    try {
        yield 1;
        yield 2;
    } finally {
        log += "inner;";
    }
}

function* outer(): Generator<number> {
    try {
        yield 0;
        yield* inner();
        yield 3;
    } finally {
        log += "outer;";
    }
}

function main() {
    const g = outer();
    assert(g.next().value == 0, "0");
    assert(g.next().value == 1, "1 from the delegate");
    const r = g.return(7);
    assert(r.done && r.value == 7, "finished with the value");
    assert(log == "inner;outer;", "both finally blocks, the delegate's first: " + log);
    assert(g.next().done, "finished after");

    // run to the end: the delegate's finally runs once, as it finishes
    log = "";
    let sum: number = 0;
    for (const v of outer()) sum += v;
    assert(sum == 6 && log == "inner;outer;", "to the end: " + log);

    print("done.");
}
