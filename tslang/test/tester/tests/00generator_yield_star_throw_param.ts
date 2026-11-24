// TSLANG 186: a yield* of a generator a parameter holds (`yield* it`): a throw() of the generator at that yield is
// handed to the delegate's throw() in every model, -mm=own too (for elements that own nothing)
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
    throw?: (e: any) => IterateResult<T>;
}

let log = "";

function* inner(k: number): Generator<number> {
    try {
        yield k;
        yield k + 1;
    } catch (e) {
        log += "caught " + e + ";";
    }
    yield k + 2;
}

function* outer(it: Generator<number>): Generator<number> {
    yield* it;
    yield 9;
}

function main() {
    const g = outer(inner(1));
    assert(g.next().value == 1, "1");
    assert(g.throw!("x").value == 3, "the delegate's next value");
    assert(log == "caught x;", "caught: " + log);
    assert(g.next().value == 9, "then the generator's own");
    assert(g.next().done, "finished");

    let sum: number = 0;
    for (const v of outer(inner(10))) sum += v;
    assert(sum == 42, "to the end");

    print("done.");
}
