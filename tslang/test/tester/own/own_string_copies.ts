// A string stored by a statement that runs again - into each element in a loop, as `fill` does -
// is no one move of the parameter: each store takes a copy, and the parameter stays a borrow (spec
// 50). A string a generator's state holds, yielded on some paths only (`if (c) yield v`), goes into
// the record `next()` returns as a copy of its own. Each step churns, so a string freed too early
// reads garbage; every model runs it.
class C {
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) junk.push(new C(999));
}

function label(i: number) {
    return "s" + i;
}

function fillAll(a: string[], value: string) {
    for (let i = 0; i < a.length; i++) a[i] = value;
}

function* longer(a: string[], n: int) {
    for (const v of a) if (v.length > n) yield v;
}

class Names {
    constructor(private data: string[] = []) {}

    fill(value: string) {
        fillAll(this.data, value);
    }

    at(i: int) {
        return this.data[i];
    }
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = [label(1), label(2), label(3)];
        const v = label(i);
        fillAll(a, v);
        churn();
        if (a[0] == v && a[2] == v) ok++;

        const n = new Names([label(1), label(2)]);
        n.fill(label(7));
        churn();
        if (n.at(1) == "s7") ok++;

        const kept: string[] = [];
        for (const s of longer([label(1), label(22), label(333)], 2)) kept.push(s);
        churn();
        if (kept.length == 2 && kept[1] == "s333") ok++;
    }

    assert(ok == 300 * 3, "ok");
    print("done.");
}
