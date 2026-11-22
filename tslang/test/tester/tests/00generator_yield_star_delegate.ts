// `yield*` of a generator a call makes (`yield* inner(3)`, a Generator<T>): the delegate is the outer generator's,
// stepped until it is done. Under -mm=own it was "a value that borrows a field is used here after it may be
// released or overwritten": next() was called on a delegate borrowed from a variable of the generator, and what
// next() runs may be, as far as its type says, this same generator function's next(), which writes that
// variable. The delegate is now exchanged into a local of next() for each call and back (TSLANG 159) - also when
// it throws, and it goes with the generator when the generator is given up early.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
}

function* inner(n: number): Generator<number> {
    for (let i = 0; i < n; i++) {
        yield i;
    }
}

function* outer(): Generator<number> {
    yield 100;
    yield* inner(3);
    yield 200;
}

// a generator delegating to itself, through a function that makes it
function deeper(d: number): Generator<number> {
    return deep(d);
}

function* deep(d: number): Generator<number> {
    yield d;
    if (d > 0) {
        yield* deeper(d - 1);
        yield* deeper(d - 1);
    }
}

// strings: each is the consumer's
function* words(n: number): Generator<string> {
    for (let i = 0; i < n; i++) {
        yield "w" + i;
    }
}

function* sentence(n: number): Generator<string> {
    yield "<" + n;
    yield* words(n);
    yield* words(1);
    yield ">" + n;
}

// two levels
function* shout(n: number): Generator<string> {
    yield* sentence(n);
    yield "!" + n;
}

// a delegate that throws: through the generator, and caught in it
function* boom(): Generator<number> {
    yield 1;
    throw "boom" + 1;
}

function* overBoom(): Generator<number> {
    yield 0;
    yield* boom();
    yield 9;
}

function* caught(): Generator<number> {
    let failed = false;
    try {
        yield* boom();
    } catch (e) {
        failed = true;
    }

    if (failed) {
        yield -1;
    }

    yield 7;
}

function sum(g: Generator<number>) {
    let s: number = 0;
    for (const v of g) {
        s += v;
    }

    return s;
}

function cat(g: Generator<string>) {
    let s = "";
    for (const v of g) {
        s += v;
    }

    return s;
}

// given up after two values: the delegate goes with the generator
function firstTwo(g: Generator<number>) {
    let s: number = 0;
    let n = 0;
    for (const v of g) {
        s += v;
        if (++n == 2) {
            break;
        }
    }

    return s;
}

function overBoomSum() {
    const g = overBoom();
    let s: number = 0;
    try {
        for (const v of g) {
            s += v;
        }
    } catch (e) {
        s += 1000;
    }

    return s;
}

function main() {
    for (let round = 0; round < 1000; round++) {
        assert(sum(outer()) == 303);
        assert(sum(deep(3)) == 11);
        assert(firstTwo(outer()) == 100 && firstTwo(deep(4)) == 7);
        assert(cat(sentence(2)) == "<2w0w1w0>2");
        assert(cat(shout(1)) == "<1w0w0>1!1");
        assert(overBoomSum() == 1001);
        assert(sum(caught()) == 7);
    }

    print("done.");
}
