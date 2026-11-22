// A generator that names itself: `yield* deep(d - 1)` in `deep`, and two that name each other. It was "can't resolve
// name: deep" in every memory model: the generator's body is next() of the object its wrapper function returns,
// its variables are fields of that object and need their types before the wrapper has its own - and a function is
// known by its name only once its type is found. With a declared type the wrapper is known by its name while its
// body is discovered (TSLANG 160). (One with no declared type is still refused: the type of the call is what is
// being found.) The call in a for...of, and methods: 00generator_names_itself_methods.ts.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
}

function* deep(d: number): Generator<number> {
    yield d;
    if (d > 0) {
        yield* deep(d - 1);
        yield* deep(d - 1);
    }
}

// strings
function* nested(n: number): Generator<string> {
    yield "<" + n;
    if (n > 0) {
        yield* nested(n - 1);
    }

    yield n + ">";
}

// two generators naming each other
function* ping(n: number): Generator<number> {
    yield n;
    if (n > 0) {
        yield* pong(n - 1);
    }
}

function* pong(n: number): Generator<number> {
    yield -n;
    if (n > 0) {
        yield* ping(n - 1);
    }
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

function main() {
    for (let round = 0; round < 100; round++) {
        assert(sum(deep(3)) == 11);
        assert(cat(nested(2)) == "<2<1<00>1>2>");
        assert(sum(ping(3)) == 3 - 2 + 1 - 0);
    }

    print("done.");
}
