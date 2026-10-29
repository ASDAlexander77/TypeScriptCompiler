// Generic interface methods (erased `any` slots) under every model: an object given through `map<T>`
// comes back as the caller's, unboxed from the `any` the slot returns, and a string argument beside a
// generic one is borrowed. Each step churns, so anything freed too early reads garbage.
interface Mapper {
    map<T>(x: T): T;
    tag<T>(x: T, label: string): string;
}

class Box {
    constructor(public n: number) {}
}

class Id implements Mapper {
    map<T>(x: T): T {
        return x;
    }

    tag<T>(x: T, label: string): string {
        return "id:" + label;
    }
}

function churn() {
    const junk: Box[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Box(999));
}

function use(m: Mapper, i: number) {
    return m.map<number>(i) + 1;
}

function test(m: Mapper, i: number) {
    let s: number = 0;
    const b = m.map(new Box(i));
    churn();
    s += b.n;

    const label = "l" + i;
    if (m.tag(i, label) == "id:l" + i) s += 1;
    churn();
    if (label.length > 1) s += 1;

    const t = m.map<string>("t" + i);
    churn();
    s += t.length;

    s += use(m, i);
    return s;
}

function main() {
    const m: Mapper = new Id();
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        // i, 1, 1, ("t" + i).length, i + 1
        if (test(m, i) == 2 * i + 3 + ("t" + i).length) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
