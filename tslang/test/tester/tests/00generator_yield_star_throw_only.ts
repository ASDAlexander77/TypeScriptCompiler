// TSLANG 186: a yield* over a generator whose type has throw() but not return(): a throw() of the generator at that
// yield is handed to the delegate's throw() as with a Generator that has both (183/184), and nothing calls a return()
// the type does not have
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
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

function* outer(k: number): Generator<number> {
    yield* inner(k);
    yield 9;
}

function main() {
    const g = outer(1);
    assert(g.next().value == 1, "1");
    assert(g.throw!("x").value == 3, "the delegate's next value");
    assert(log == "caught x;", "caught: " + log);
    assert(g.next().value == 9, "then the generator's own");
    assert(g.next().done, "finished");

    let sum: number = 0;
    for (const v of outer(10)) sum += v;
    assert(sum == 10 + 11 + 12 + 9, "to the end");

    print("done.");
}
