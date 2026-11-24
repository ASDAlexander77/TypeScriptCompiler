// TSLANG 187: throw() of a generator paused in a `yield*` of strings is handed to the delegate's throw() under -mm=own
// too (183 for the other models, 184/186 for elements that own nothing): a delegate made there, and one a parameter
// holds
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
    throw?: (e: any) => IterateResult<T>;
}

let log = "";

function* inner(k: number): Generator<string> {
    try {
        yield "a" + k;
        yield "b" + k;
    } catch (e) {
        log += "caught " + e + ";";
    }
    yield "c" + k;
}

function* made(k: number): Generator<string> {
    yield* inner(k);
    yield "d" + k;
}

function* held(it: Generator<string>): Generator<string> {
    yield* it;
    yield "d";
}

function main() {
    const g = made(1);
    assert(g.next().value == "a1", "made a1");
    assert(g.throw!("x").value == "c1", "made: the delegate's next value");
    assert(g.next().value == "d1", "made d1");
    assert(log == "caught x;", "made: " + log);

    log = "";
    const h = held(inner(2));
    assert(h.next().value == "a2", "held a2");
    assert(h.throw!("y").value == "c2", "held: the delegate's next value");
    assert(h.next().value == "d", "held d");
    assert(log == "caught y;", "held: " + log);

    let all = "";
    for (const v of made(4)) all += v;
    assert(all == "a4b4c4d4", "to the end: " + all);

    print("done.");
}
