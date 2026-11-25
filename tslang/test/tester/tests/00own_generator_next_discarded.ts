// TSLANG 191: a generator object's own `next()` called by hand (`const g = gen(k); g.next()`, typed by its shape,
// not Generator<T>), its `{ value, done }` dropped or kept in a local. Under -mm=own what it yielded was never
// freed: a generator of strings, stepped and then left, grew the heap with every call (TSLANG 188 did the call
// through Generator<T>).
class Box {
    constructor(public s: string) {}
}

function* strings(k: number) {
    yield "a" + k;
    yield "bb" + k;
    yield "ccc" + k;
}

function* boxes(k: number) {
    yield new Box("a" + k);
    yield new Box("bb" + k);
}

function* holding(k: number) {
    const s = "x" + k;
    const b = new Box("y" + k);
    yield s.length;
    yield b.s.length;
}

// one step, the result dropped
function dropped(k: number): number {
    const g = strings(k);
    g.next();
    return 1;
}

// every step, past the end
function drained(k: number): number {
    const g = strings(k);
    g.next();
    g.next();
    g.next();
    g.next();
    return 1;
}

// results kept in locals, one read
function kept(k: number): number {
    const g = strings(k);
    const a = g.next();
    const b = g.next();
    return (<string>b.value).length;
}

function keptBoxes(k: number): number {
    const g = boxes(k);
    const a = g.next();
    g.next();
    return (<Box>a.value).s.length;
}

// left with its state holding a string and an object
function leftHolding(k: number): number {
    const g = holding(k);
    g.next();
    const r = g.next();
    return <number>r.value;
}

function main() {
    assert(kept(1) == 3, "kept");
    assert(keptBoxes(1) == 2, "kept boxes");
    assert(leftHolding(1) == 2, "left holding");

    let n = 0;
    for (let i = 0; i < 100000; i++) {
        n += dropped(i) + drained(i) + kept(i) + keptBoxes(i) + leftHolding(i);
    }

    assert(n > 0, "many times");

    print("done.");
}
