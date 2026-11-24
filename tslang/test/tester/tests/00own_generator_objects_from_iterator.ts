// TSLANG 189: a generator that yields objects it got from another iterator - by `yield*`, by for-of, by `next()` by
// hand - in every memory model. Under -mm=own each was refused ("a value that borrows a field is used here after it
// may be released or overwritten"): the `{ value, done }` the iterator gave is kept in the generator's state, and
// the next `next()` releases it. The yield now moves the value out and leaves the field empty.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
    throw?: (e: any) => IterateResult<T>;
}

class Box {
    constructor(public s: string) {}
}

function* inner(k: number): Generator<Box> {
    yield new Box("a" + k);
    yield new Box("bb" + k);
    yield new Box("ccc" + k);
}

function* byHand(it: Generator<Box>): Generator<Box> {
    for (;;) {
        const r = it.next();
        if (r.done) break;
        yield r.value;
    }
}

// the value read before it is yielded, and one passed over
function* skipping(it: Generator<Box>): Generator<Box> {
    let r = it.next();
    while (!r.done) {
        if (r.value.s.length == 4) {
            r = it.next();
            continue;
        }

        yield r.value;
        r = it.next();
    }
}

function* byForOf(k: number): Generator<Box> {
    for (const b of inner(k)) yield b;
}

function* byStar(k: number): Generator<Box> {
    yield* inner(k);
    yield new Box("d" + k);
}

function* ofNumbers(k: number): Generator<number> {
    for (const n of [k, k + 1]) yield n;
}

function* byForOfNumbers(k: number): Generator<number> {
    for (const n of ofNumbers(k)) yield n;
}

function join(g: Generator<Box>): string {
    let s = "";
    for (const b of g) s += b.s + ",";
    return s;
}

function main() {
    assert(join(byHand(inner(1))) == "a1,bb1,ccc1,", "by hand");
    assert(join(skipping(inner(1))) == "a1,bb1,", "skipping");
    assert(join(byForOf(2)) == "a2,bb2,ccc2,", "for-of");
    assert(join(byStar(3)) == "a3,bb3,ccc3,d3,", "yield*");

    let sum: number = 0;
    for (const n of byForOfNumbers(4)) sum += n;
    assert(sum == 9, "numbers");

    // a value kept past the loop, and generators left before they finish
    const kept: Box[] = [];
    for (const b of byForOf(5)) kept.push(b);
    for (const b of byHand(inner(6))) kept.push(b);
    assert(kept.length == 6 && kept[2].s == "ccc5" && kept[3].s == "a6", "kept");

    let n: number = 0;
    for (let i = 0; i < 20000; i++) {
        for (const b of byHand(inner(i))) {
            n += b.s.length;
            break;
        }

        const g = byStar(i);
        n += g.next().value.s.length;
    }

    assert(n > 0, "many times");

    print("done.");
}
