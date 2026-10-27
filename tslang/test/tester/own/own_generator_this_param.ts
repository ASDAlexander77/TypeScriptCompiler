// A generator whose first parameter is `this` (`function* keys<T>(this: T[])`, called `a.keys()`)
// borrows it as it does any other parameter (spec 19, 40): it is walked while the array lives, and a
// function that returns it says `@borrows("this")`. Each case churns between steps, so an array freed
// early, or a copy of it kept, reads garbage.
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

function* xs(this: C[]) {
    for (let i = 0; i < this.length; i++) yield this[i].x;
}

function* indices<T>(this: T[]) {
    for (let i = 0; i in this; i++) yield i;
}

@borrows("this")
function walk(this: C[]) {
    return this.xs();
}

function make(n: number) {
    const a: C[] = [];
    for (let i = 1; i <= n; i++) a.push(new C(i));
    return a;
}

function main() {
    let t = 0;
    for (let i = 0; i < 1000; i++) {
        const a = make(3);
        for (const v of a.xs()) {
            churn();
            t += v;
        }

        const g = a.indices();
        churn();
        for (const k of g) t += k;

        for (const v of a.walk()) t += v;

        // a temporary array, in the statement that walks it
        t += make(2).xs().next().value;
    }

    assert(t == 1000 * (6 + 3 + 6 + 1), "t");
    print("done.");
}
