// A generator that says `@keeps`, given one that borrows its argument, owns that generator and is
// bounded by the same argument (spec 38): walked in place - a for...of, a call in the statement -
// while the argument lives, as `new Set(a).values().take(2)` is. Each case churns between steps, so
// a source freed early reads garbage.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

type It = { next(): { value: any, done: boolean } };

class Bag {
    items: C[] = [];
    constructor(n: number) {
        for (let i = 1; i <= n; i++) this.items.push(new C(i));
    }
}

// borrows its argument: its result is bounded by `b`. It reads through `b` at each step, as a
// Set's `values()` reads through the set.
function* each(b: Bag) {
    for (let i = 0; i < b.items.length; i++) yield b.items[i].x;
}

@keeps("iter")
function* take<TIter extends It>(iter: TIter, limit: number) {
    let count = 0;
    for (const v of iter) {
        if (count++ >= limit) break;
        yield v;
    }
}

@keeps("iter")
function* drop<TIter extends It>(iter: TIter, limit: number) {
    let count = 0;
    for (const v of iter) {
        if (count++ < limit) continue;
        yield v;
    }
}

function main() {
    let t = 0;
    for (let i = 0; i < 1000; i++) {
        // a temporary source, in a for...of header
        for (const v of take(each(new Bag(3)), 2)) {
            churn();
            t += v;
        }

        // two makers deep
        for (const v of take(drop(each(new Bag(4)), 1), 2)) {
            churn();
            t += v;
        }

        // one step, in the statement that made it
        t += drop(each(new Bag(2)), 1).next().value;

        // a named source, the chain held in a local
        const b = new Bag(3);
        const g = drop(take(each(b), 3), 2);
        churn();
        t += g.next().value;
    }

    assert(t == 1000 * (3 + 5 + 2 + 3), "t");
    print("done.");
}
