// A generator that names itself: `yield* deep(d - 1)` in `deep`, `yield* this.left.walk()` in `walk`, the call
// kept in a variable, a for...of over it. It was "can't resolve name: deep" ("Class member 'walk' can't be found")
// in every memory model: the generator's body is next() of the object its wrapper function returns, its
// variables are fields of that object and need their types before the wrapper has its own - and a function is
// known by its name only once its type is found. With a declared type the wrapper is known by its name while its
// body is discovered (TSLANG 160). (One with no declared type is still refused: the type of the call is what is
// being found.)
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

// the call in a for...of
function* countdown(n: number): Generator<number> {
    yield n;
    if (n > 0) {
        for (const v of countdown(n - 1)) {
            yield v * 10;
        }
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

// a method naming itself through another object of its class
class Tree {
    left: Tree | null = null;
    right: Tree | null = null;

    constructor(public v: number) {
    }

    *walk(): Generator<number> {
        if (this.left) {
            yield* this.left.walk();
        }

        yield this.v;
        if (this.right) {
            yield* this.right.walk();
        }
    }
}

// two methods naming each other
class Steps {
    *up(n: number): Generator<number> {
        yield n;
        if (n < 3) {
            yield* this.down(n + 1);
        }
    }

    *down(n: number): Generator<number> {
        yield -n;
        if (n < 3) {
            yield* this.up(n + 1);
        }
    }
}

function sum(g: Generator<number>) {
    let s: number = 0;
    for (const v of g) {
        s += v;
    }

    return s;
}

function digits(g: Generator<number>) {
    let s: number = 0;
    for (const v of g) {
        s = s * 10 + v;
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
        assert(sum(countdown(3)) == 3 + 20 + 100 + 0);
        assert(cat(nested(2)) == "<2<1<00>1>2>");
        assert(sum(ping(3)) == 3 - 2 + 1 - 0);

        const t = new Tree(2);
        t.left = new Tree(1);
        t.right = new Tree(3);
        t.right.right = new Tree(4);
        assert(digits(t.walk()) == 1234);
        assert(sum(new Steps().up(0)) == 0 - 1 + 2 - 3);
    }

    print("done.");
}
