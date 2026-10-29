// An object literal whose arrows capture a `let` or a parameter (spec 69): a literal that stays in the
// function borrows the `let`'s cell, as a closure that does not escape does - inline in a call, held in
// a `const`, given to an object-type or an interface parameter - and the variable is read and assigned
// after. A parameter captured by a literal given to a call that returns a number is a borrow too. Each
// step churns, so anything freed too early reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

interface Scaler {
    w(x: number): number;
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function apply(o: { w: (x: number) => number }) {
    return o.w(3);
}

function applyI(s: Scaler) {
    return s.w(3);
}

function viaParam(item: Item) {
    return apply({ w: (x: number) => x + item.n });
}

function viaParamI(item: Item) {
    const q = { w: (x: number) => x + item.n };
    return applyI(q) + q.w(1);
}

function test(i: number) {
    let s: number = 0;
    let d = new Item(i);
    s += apply({ w: (x: number) => x + d.n });
    churn();
    s += d.n;

    d = new Item(2);
    const q = { w: (x: number) => x * d.n };
    s += q.w(1) + apply(q);
    churn();
    s += d.n;

    let e = new Item(4);
    s += applyI({ w: (x: number) => x + d.n + e.n });
    churn();
    e = new Item(5);
    s += e.n;

    s += viaParam(d) + viaParamI(e);
    churn();
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        // (3 + i) + i, (2 + 6) + 2, (3 + 2 + 4) + 5, (3 + 2) + (3 + 5) + (1 + 5)
        if (test(i) == 3 + 2 * i + 10 + 14 + 19) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
