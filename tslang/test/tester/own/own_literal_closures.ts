// An arrow function member of an object literal that never says `this` is a closure in a field, not a
// method bound to the literal (spec 60): read into a function-typed field of the literal's receiver, it
// hands out no literal object that nothing owns. Typed by a variable, by a parameter, by a generic call's
// parameter, capturing a local or not, several in one literal. Each step churns, so a closure or capture
// freed too early reads garbage; every model runs it.
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

function both(o: { w: (x: number) => number; r: () => number }) {
    const r = o.r();
    return o.w(r);
}

function f1<T, U>(a: { w: (x: T) => U; r: () => T }, b: T): U {
    return a.w(b);
}

function test(k: number) {
    let s: number = 0;
    const o: { w: (x: number) => number } = { w: x => x * 2 };
    churn();
    s += o.w(4);
    s += apply({ w: x => x + 1 });
    churn();
    s += apply({ w: x => x + k });
    s += both({ w: x => x * 10, r: () => 2 });
    churn();
    s += f1({ w: x => x + 1, r: () => 0 }, 5);
    const item = new Item(7);
    s += apply({ w: x => x + item.n });
    churn();
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        if (test(2) == 8 + 4 + 5 + 20 + 6 + 10) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
