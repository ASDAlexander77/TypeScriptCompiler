// A generator typed Generator<T> (or Iterator<T>, Iterable<T>, IterableIterator<T>) yields T: `yield 1` in a
// Generator<number> is a number. Its next() was typed by what it yields, the literal's s32, and the generator
// could not be a Generator<number>: "can't assign '() => {value:s32, done:boolean}' to '() => {value:number, ...}'".
// (-mm=own: a generator typed as an interface other than the library's is still refused.)
// as the default library declares them (the tests compile without it)
type IterateResult<T> = { value: T, done: boolean };
interface Iterable<T> {
    next: () => IterateResult<T>;
}
interface IterableIterator<T> {
    next: () => IterateResult<T>;
}
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: () => IterateResult<T>;
}

function* g(): Generator<number> {
    yield 1;
    yield 2.5;
}

function* k(): IterableIterator<number> {
    let i = 0;
    while (i < 3) {
        yield i++;
    }
}

function* names(): Iterable<string> {
    yield "a";
    yield "b";
}

function* stop(): Generator<number, void> {
    yield 1;
    return;
}

function main() {
    let s = 0.0;
    for (const x of g()) {
        s += x;
    }

    assert(s == 3.5);

    let t = 0;
    for (const x of k()) {
        t += x;
    }

    assert(t == 3);

    let n = "";
    for (const x of names()) {
        n += x;
    }

    assert(n == "ab");

    let c = 0;
    for (const x of stop()) {
        c += x;
    }

    assert(c == 1);
    print("done.");
}
