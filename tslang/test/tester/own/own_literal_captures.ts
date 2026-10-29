// An object literal a variable is declared with, with no type, whose arrow functions capture locals
// around it (spec 66): given later to a parameter of an object type (`apply(o)`), each member is read
// out as a method bound to the literal, which the call only borrows; the literal is used after, given
// again, and given to an interface. Each step churns, so anything freed too early reads garbage; every
// model runs it.
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

function applyBoth(o: { w: (x: number) => number; v: (x: number) => number }) {
    return o.w(1) + o.v(2);
}

function useScaler(s: Scaler) {
    return s.w(2);
}

function test(i: number) {
    let s: number = 0;
    const k: number = 2;
    const o = { w: (x: number) => x * k };
    s += apply(o);
    churn();
    s += o.w(1);
    s += apply(o);
    s += useScaler(o);

    const tag = "n" + i;
    const t = { w: (x: number) => x + tag.length, v: (x: number) => x * k + tag.length };
    s += applyBoth(t);
    churn();
    s += apply(t);
    s += tag.length;
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const len = ("n" + i).length;
        // 6 + 2 + 6 + 4, then (1 + len) + (4 + len), (3 + len), len
        if (test(i) == 18 + 8 + 4 * len) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
