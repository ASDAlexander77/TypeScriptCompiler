// -mm=own: `yield*` of a delegate the generator is given - a parameter, a Generator<T> or an Iterable<T> - and a
// Generator<T> whose T is an object. `yield* it` was "a value that borrows a field is used here after it may be
// released or overwritten" (the loop called next() on a copy of the parameter kept in a variable of the
// generator: next() is called on the parameter itself now); a `Generator<P>` of objects did not compile at all (its
// return(value: P) kept the value in an `any` it could not be read back out of as its owner: such a generator has
// no return() and throw() under -mm=own). The other models run the same (TSLANG 161).
type IterateResult<T> = { value: T, done: boolean };
interface Iterable<T> {
    next: () => IterateResult<T>;
}

interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
}

let trail = "";

function* words(n: number): Generator<string> {
    for (let i = 0; i < n; i++) {
        yield "w" + i;
    }
}

function* numbers(n: number): Generator<number> {
    try {
        for (let i = 0; i < n; i++) {
            yield i;
        }
    } finally {
        trail += "f;";
    }
}

function* viaGenerator(a: Generator<string>): Generator<string> {
    yield "<";
    yield* a;
    yield ">";
}

function* viaIterable(a: Iterable<string>): Generator<string> {
    yield* a;
}

@keeps("a")
function* kept(a: Generator<number>): Generator<number> {
    yield -1;
    yield* a;
    yield -2;
}

class P {
    constructor(public n: string) {
    }
}

function* people(k: number): Generator<P> {
    for (let i = 0; i < k; i++) {
        yield new P("p" + i);
    }
}

function cat(g: Generator<string>) {
    let s = "";
    for (const v of g) {
        s += v;
    }

    return s;
}

function sum(g: Generator<number>) {
    let s: number = 0;
    for (const v of g) {
        s += v;
    }

    return s;
}

function names(g: Generator<P>) {
    let s = "";
    for (const p of g) {
        s += p.n;
    }

    return s;
}

function firstPerson(g: Generator<P>) {
    for (const p of g) {
        return p.n;
    }

    return "none" + 0;
}

function main() {
    for (let round = 0; round < 100; round++) {
        const w = words(3);
        assert(cat(viaGenerator(w)) == "<w0w1w2>");
        assert(cat(viaGenerator(words(2))) == "<w0w1>");
        assert(cat(viaIterable(words(2))) == "w0w1");
        assert(cat(viaIterable(["a" + round, "b"])) == "a" + round + "b");

        trail = "";
        assert(sum(kept(numbers(4))) == -1 + 0 + 1 + 2 + 3 - 2 && trail == "f;");

        // return() of the generator reaches the delegate it was given: its finally runs
        trail = "";
        const k = kept(numbers(5));
        k.next();
        k.next();
        const r = k.return(9);
        assert(r.done && r.value == 9 && trail == "f;");

        assert(names(people(3)) == "p0p1p2" && firstPerson(people(2)) == "p0");
    }

    print("done.");
}
