// An object literal whose arrows capture an object, given to a parameter of an interface type (spec 68):
// the interface is a view of the literal's box, and when nothing keeps it the box borrows what it
// captures - the objects are used after. A literal that is kept or returned still takes its copies.
// Each step churns, so anything freed too early reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

interface Scaler {
    w(x: number): number;
}

interface Namer {
    name(): string;
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function apply(s: Scaler) {
    return s.w(3);
}

function nameOf(n: Namer) {
    return n.name();
}

const kept: Scaler[] = [];
function keep(s: Scaler) {
    const r = s.w(1);
    kept.push(s);
    return r;
}

function viaParam(item: Item) {
    return apply({ w: (x: number) => x + item.n });
}

function made(item: Item): Scaler {
    const n = item.n;
    return { w: (x: number) => x + n };
}

function test(i: number) {
    let s: number = 0;
    const b = new Item(i);
    s += apply({ w: (x: number) => x * b.n });
    churn();
    s += b.n;

    const c = new Item(2);
    s += apply({ w: (x: number) => x + b.n + c.n });
    churn();
    s += c.n + b.n;

    const label = "n" + i;
    if (nameOf({ name: () => label + c.n }) == "n" + i + "2") s += 1;
    churn();
    if (label == "n" + i) s += 1;

    s += viaParam(c);

    s += made(c).w(1);

    const e = new Item(5);
    s += keep({ w: (x: number) => x + e.n });
    churn();
    return s;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        // 3i + i, (3 + i + 2) + 2 + i, 1 + 1, 3 + 2, 1 + 2, 1 + 5
        if (test(i) == 4 * i + 7 + 2 * i + 2 + 5 + 3 + 6) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    assert(kept.length == 300, "kept");
    assert(kept[7].w(0) == 5, "kept value");
    print("done.");
}
