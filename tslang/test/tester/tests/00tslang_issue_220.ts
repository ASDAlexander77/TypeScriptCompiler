// TSLANG 220: an array literal of closures given as `(() => number)[]` failed in every model ("invalid cast from
// (p0: Reference<{k:41}>) => s32[] to () => number[]"):
// - an element was the closure over its own capture box, so the array's type named the box (and two closures over
//   different boxes merged into the first one's type, or not at all: then the literal was a tuple);
// - an element returning an integer (`() => k + 1` of a literal `k`, `() => s.length`) could not be given as one
//   returning a number, as a function can be on its own: the array is copied, each element adapted.
// Under -mm=own the adapter's box holds an extra owner of the closure it calls.

function count(fs: (() => number)[]) {
    return fs.length;
}

function first(fs: (() => number)[]) {
    return fs[0]();
}

function sumAll(fs: (() => number)[]) {
    let t: number = 0;
    for (const f of fs) t += f();
    return t;
}

function keep(kept: (() => number)[][], fs: (() => number)[]) {
    kept.push(fs);
    while (kept.length > 2) kept.shift();
}

class B {
    v = 0;
    constructor(v: number) {
        this.v = v;
    }
}

const sink: B[] = [];
function churn(k: number) {
    for (let i = 0; i < 8; i++) sink.push(new B(-k - i));
    while (sink.length > 4) sink.pop();
}

// copied with adapters, the original emptied: the copy's closures live on
function step(kept: (() => number)[][], k: number) {
    const s = "x" + k;
    const fs = [() => s.length];
    keep(kept, fs);
    while (fs.length > 0) fs.pop();
    churn(k);
    const copy = kept[kept.length - 1];
    return copy.length == 1 && copy[0]() == s.length;
}

function assert(cond: boolean, msg: string) {
    if (!cond) throw msg;
}

function main() {
    const k = 41;
    const m = 100;
    const s = "abc";

    const fs = [() => k + 1];
    assert(count(fs) == 1 && fs[0]() == 42, "one closure");

    const gs = [() => 7];
    assert(first(gs) == 7, "no capture");

    const hs = [() => k + 1, () => k + 2];
    assert(first(hs) == 42 && sumAll(hs) == 85, "two closures of one box type");

    const two = [() => k, () => m];
    assert(sumAll(two) == 141 && two[0]() == 41 && two[1]() == 100, "two boxes");

    const mixed = [() => 7, () => m];
    assert(sumAll(mixed) == 107, "a function and a closure");

    const widened = [() => s.length, () => k + 0.5];
    assert(sumAll(widened) == 44.5, "an index and a number");

    const typed: (() => number)[] = [() => k, () => m];
    assert(sumAll(typed) == 141, "typed");

    let bad = 0;
    const kept: (() => number)[][] = [];
    for (let i = 0; i < 2000; i++) {
        if (!step(kept, i)) bad++;
    }

    assert(bad == 0, "a copied closure outlived");
    print("done.");
}
