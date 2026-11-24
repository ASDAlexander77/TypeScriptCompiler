// TSLANG 183: throw() of a generator paused in a `yield*` over a Generator is handed to the delegate's throw(), as in
// JavaScript (156): the delegate may catch the exception and go on yielding, or finish, and the generator goes on
// with what it gives; one that does not catch it throws it on, through the generator's own try/catch.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
    throw?: (e: any) => IterateResult<T>;
}

let log = "";

// catches, and goes on yielding
function* catching(): Generator<number> {
    try {
        yield 1;
        yield 2;
    } catch (e) {
        log += "caught " + e + ";";
    }
    yield 99;
}

function* outer(): Generator<number> {
    yield* catching();
    yield 100;
}

// does not catch: its finally runs, and the generator's own catch gets it
function* plain(): Generator<number> {
    try {
        yield 1;
        yield 2;
    } finally {
        log += "plain finally;";
    }
}

function* guarded(): Generator<number> {
    try {
        yield* plain();
    } catch (e) {
        log += "guarded " + e + ";";
    }
    yield 50;
    yield 51;
}

// catches, and finishes: the yield* is done
function* finishing(): Generator<number> {
    try {
        yield 1;
    } catch (e) {
        log += "finishing;";
    }
}

function* afterFinish(): Generator<number> {
    yield* finishing();
    yield 7;
}

// through two yield*s
function* twice(): Generator<number> {
    yield* outer();
}

function main() {
    const g = outer();
    assert(g.next().value == 1, "1");
    const r = g.throw!("boom");
    assert(!r.done && r.value == 99, "the delegate's next value");
    assert(log == "caught boom;", "the delegate caught it: " + log);
    assert(g.next().value == 100, "then the generator's own");
    assert(g.next().done, "finished");

    log = "";
    const h = guarded();
    assert(h.next().value == 1, "h 1");
    const r2 = h.throw!("x");
    assert(!r2.done && r2.value == 50, "the generator's catch: " + log);
    assert(log == "plain finally;guarded x;", "the delegate's finally first: " + log);
    assert(h.next().value == 51, "h 51");
    assert(h.next().done, "h finished");

    log = "";
    const f = afterFinish();
    assert(f.next().value == 1, "f 1");
    const r3 = f.throw!("y");
    assert(!r3.done && r3.value == 7, "past the finished yield*");
    assert(log == "finishing;", "finishing: " + log);

    log = "";
    const t = twice();
    assert(t.next().value == 1, "t 1");
    assert(t.throw!("z").value == 99, "t through two");
    assert(log == "caught z;", "t: " + log);
    assert(t.next().value == 100, "t 100");

    // not thrown: as before
    let sum: number = 0;
    for (const v of outer()) sum += v;
    assert(sum == 202, "to the end");

    // thrown at a generator nothing catches in: thrown on
    let threw = false;
    const p = plain();
    p.next();
    try {
        p.throw!("w");
    } catch (e) {
        threw = true;
    }
    assert(threw, "thrown on");

    print("done.");
}
