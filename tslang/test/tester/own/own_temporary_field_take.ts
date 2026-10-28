// A field read out of a call's own result in one branch of a `??` or a `?:`, kept, and the result
// released right after (spec 63): rc retains it in the branch, before the release, and own takes it out
// of the field, so the result's release frees nothing of it. `initial ?? next(xs, 0).value` of
// DefaultLib's reduce, a string, and a class object given a `const` or a `let`. Each step churns, so a
// value freed with the temporary reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

class Res {
    constructor(public value: any, public done: boolean) {}
}

class Pair {
    constructor(public item: Item, public tag: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function next(xs: string[], i: number): Res {
    return new Res(xs[i], i >= xs.length);
}

function pair(n: number): Pair {
    return new Pair(new Item(n), n);
}

function red(xs: string[], f: (a: string, b: string) => string, initial?: string): string {
    let result = initial ?? next(xs, 0).value;
    churn();
    for (let i = initial ? 0 : 1; i < xs.length; i++) {
        result = f(result, xs[i]);
    }

    return result;
}

function pick(xs: string[], initial?: string): string {
    let result = initial ? initial : next(xs, 0).value;
    churn();
    return result;
}

function itemOf(n: number): number {
    const item = n > 0 ? pair(n).item : pair(-n).item;
    churn();
    let other = n > 0 ? pair(n + 1).item : pair(0).item;
    churn();
    return item.n * 10 + other.n;
}

function test() {
    let s: number = 0;
    if (red(["a", "b", "c"], (a, b) => a + b) == "abc") s++;
    if (red(["a", "b"], (a, b) => a + b, "z") == "zab") s++;
    if (pick(["q"]) == "q") s++;
    if (pick(["q"], "p") == "p") s++;
    s += itemOf(5);
    s += itemOf(-2);
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        if (test() == 4 + 56 + 20) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
