// A callback parameter captured by a closure that only calls it (`a.forEach((v) => f(v))`), or
// given a second name (`const g = f`), is still known by the family of functions its callers pass
// (spec 54); and a callback that assigns a captured number (`t += v.n`) frees nothing (spec 52).
// A call through either releases nothing of the elements it is given, and the array a field holds
// is not dropped by it. Each step churns, so an element freed too early reads garbage; every model
// runs it.
class Item {
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

function each(a: Item[], f: (v: Item) => void) {
    a.forEach((v) => f(v));
}

// reads the element again after the call: the call must release nothing of it
function walk(a: Item[], g: (v: Item) => void) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const v = a[i];
        g(v);
        s += v.n;
    }

    return s;
}

function eachWalked(a: Item[], f: (v: Item) => void) {
    return walk(a, (v) => f(v));
}

function eachNamed(a: Item[], f: (v: Item) => void) {
    const g = f;
    for (const v of a) g(v);
}

class Bag {
    constructor(private data: Item[] = []) {}

    total() {
        let t = 0;
        each(this.data, (v) => {
            t += v.n;
        });
        return t;
    }

    largest() {
        let m = 0;
        eachNamed(this.data, (v) => {
            if (v.n > m) m = v.n;
        });
        return m;
    }

    first() {
        return this.data[0];
    }

    walked() {
        let seen = 0;
        const s = eachWalked(this.data, (v) => {
            seen++;
        });
        return s * 10 + seen;
    }
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const b = new Bag([new Item(1), new Item(5), new Item(2)]);
        if (b.total() == 8) ok++;
        churn();
        if (b.largest() == 5) ok++;
        churn();
        if (b.first().n == 1) ok++;
        if (b.walked() == 83) ok++;
        churn();
    }

    assert(ok == 300 * 4, "ok");
    print("done.");
}
