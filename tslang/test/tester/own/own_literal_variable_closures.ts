// An arrow function member of an object literal a variable is declared with, with no type, is a closure
// in a field when it says neither `this` nor a local around it (spec 62): the literal given later to a
// function-typed field (`apply(o)`), and used after, hands out no literal object, and a cast to an
// interface copies the function as it would a method. Each step churns, so anything freed too early reads
// garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

interface Adder {
    add(x: number): number;
}

let seen = 0;

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function apply(o: { w: (x: number) => number }) {
    return o.w(3);
}

function useAdder(a: Adder) {
    return a.add(2);
}

function test() {
    let s: number = 0;
    const o = { w: (x: number) => x * 2 };
    s += apply(o);
    churn();
    s += o.w(1);
    s += apply(o);

    const a = { add: (x: number) => x + 10, n: 4 };
    s += useAdder(a);
    churn();
    s += a.n;
    const i: Adder = a;
    s += i.add(1);

    const h = { w: (x: number) => x + 1, item: new Item(5) };
    s += apply(h);
    churn();
    s += h.item.n;

    const g = { w: (x: number) => x + seen };
    seen = 1;
    s += apply(g);
    churn();
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        seen = 0;
        if (test() == 6 + 2 + 6 + 12 + 4 + 11 + 4 + 5 + 4) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
