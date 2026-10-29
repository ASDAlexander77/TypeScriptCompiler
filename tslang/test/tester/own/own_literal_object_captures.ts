// An object literal whose arrows capture an object (spec 67): when the literal stays in the function - its
// members called, the literal given to a parameter of an object type - its `.captured` box borrows the
// object, as a closure that does not escape does, and the object is used after. Each step churns, so
// anything freed too early reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function apply(o: { w: (x: number) => number }) {
    return o.w(3);
}

function viaParam(item: Item) {
    const q = { w: (x: number) => x + item.n };
    return q.w(1) + apply(q);
}

function test(i: number) {
    let s: number = 0;
    const b = new Item(i);
    const q = { w: (x: number) => x * b.n };
    s += q.w(1);
    churn();
    s += apply(q);
    s += b.n;

    const c = new Item(2);
    const r = { w: (x: number) => x + b.n + c.n, v: () => c.n };
    s += apply(r) + r.v();
    churn();
    s += c.n + b.n;

    s += viaParam(c);
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        // i + 3i + i, then (3 + i + 2) + 2, 2 + i, then 3 + 5
        if (test(i) == 5 * i + 7 + 2 + i + 8 + i) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
