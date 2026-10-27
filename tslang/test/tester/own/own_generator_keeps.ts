// A generator that says `@keeps` owns the arguments its state holds (spec 38): its callers move them
// in, the state's release frees them, and the generator is bounded by nothing - it may be stored,
// returned, or given to another generator that keeps it. What the default library's lazy iterator
// helpers (`map`, `filter`, `take`, `drop`) are. Each case churns between steps, so an argument freed
// early reads garbage.
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

// a class argument
@keeps("c")
function* twice(c: C) {
    yield c.x;
    churn();
    yield c.x * 2;
}

// an iterator argument
@keeps("iter")
function* plusOne<TIter extends It>(iter: TIter) {
    for (const v of iter) yield v + 1;
}

// an iterator and a callback
@keeps("iter", "f")
function* mapped<TIter extends It>(iter: TIter, f: (x: number) => number) {
    for (const v of iter) yield f(v);
}

function* count(n: number) {
    for (let i = 0; i < n; i++) yield i;
}

function double(x: number) {
    return x * 2;
}

// returned past the argument's expression
function make() {
    return twice(new C(5));
}

const top = twice(new C(3));

function main() {
    let t = 0;
    for (let i = 0; i < 1000; i++) {
        const g = make();
        churn();
        for (const v of g) t += v;

        const h = plusOne(plusOne(count(3)));
        churn();
        for (const v of h) t += v;

        for (const v of mapped(count(3), (x) => x * 10)) t += v;
        for (const v of mapped(plusOne(count(2)), double)) t += v;

        // given up after one step
        const k = twice(new C(1));
        t += k.next().value;
    }

    t += top.next().value;
    assert(t == 1000 * (15 + 9 + 30 + 6 + 1) + 3, "t");
    print("done.");
}
