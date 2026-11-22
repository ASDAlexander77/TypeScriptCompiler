// A generator that names itself (TSLANG 160), in the shapes -mm=own does not take yet (159): the call iterated by a
// for...of in the generator, and a class's generator method naming itself through another object of its class
// (`yield* this.left.walk()` was "Class member 'walk' can't be found"), or two methods naming each other.
type IterateResult<T> = { value: T, done: boolean };
interface Generator<T = unknown, TReturn = any, TNext = any> {
    next: (value?: TNext) => IterateResult<T>;
    return?: (value: T) => IterateResult<T>;
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

function main() {
    for (let round = 0; round < 100; round++) {
        assert(sum(countdown(3)) == 3 + 20 + 100 + 0);

        const t = new Tree(2);
        t.left = new Tree(1);
        t.right = new Tree(3);
        t.right.right = new Tree(4);
        assert(digits(t.walk()) == 1234);
        assert(sum(new Steps().up(0)) == 0 - 1 + 2 - 3);
    }

    print("done.");
}
