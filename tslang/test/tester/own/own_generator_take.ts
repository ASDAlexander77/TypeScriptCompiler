// What a generator yields is the caller's: a for...of or a spread that keeps it - pushes it into
// an array - moves it out of the record `next()` returns, and the record's release then frees
// nothing of it (spec 48). A value not kept stays in the record and goes with it. A local declared
// in a `try` (given its value by a store, where the cleanup can release it) owns what it is given
// as any local does. Each step churns, so a value freed too early reads garbage; every model runs it.
class C {
    constructor(public x: number) {}
}

function label(i: number) {
    return "s" + i;
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) junk.push(new C(999));
}

function* made(n: int) {
    for (let i = 0; i < n; i++) yield new C(i);
}

function* rows(n: int) {
    for (let i = 0; i < n; i++) yield [label(i), label(i + 1)];
}

function pushed(n: int) {
    const kept: C[] = [];
    for (const c of made(n)) kept.push(c);
    return kept;
}

function evens(n: int) {
    const kept: C[] = [];
    for (const c of made(n)) {
        if (c.x % 2 == 0) kept.push(c);
    }

    return kept;
}

function spread(n: int) {
    const all = [...made(n)];
    return all;
}

function inTry(n: int) {
    let count = 0;
    try {
        const all = [...rows(n)];
        churn();
        count = all.length + all[n - 1][1].length;
        const kept = pushed(n);
        count += kept.length;
    } catch {
        count = -1;
    }

    return count;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = pushed(4);
        churn();
        if (a.length == 4 && a[3].x == 3) ok++;
        const e = evens(5);
        churn();
        if (e.length == 3 && e[2].x == 4) ok++;
        const s = spread(3);
        churn();
        if (s.length == 3 && s[1].x == 1) ok++;
        const r = [...rows(2)];
        churn();
        if (r[1][0] == "s1" && r[1][1] == "s2") ok++;
        if (inTry(3) == 3 + 2 + 3) ok++;
    }

    assert(ok == 300 * 5, "ok");
    print("done.");
}
