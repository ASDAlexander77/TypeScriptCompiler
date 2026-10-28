// A callback stored into a class's field keeps its family when the class's objects are seen through
// an interface (spec 58): the interface reaches the class's fields only through vtables built from
// the class, here. Each walk reads the element again after the call, so the call must release
// nothing of it. Each step churns, so an element freed too early reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

let seen = 0;

function once(v: Item) {
    seen++;
}

function twice(v: Item) {
    seen += 2;
}

interface Runner {
    walk(a: Item[]): number;
}

interface Named {
    name: string;
}

class Walker implements Runner, Named {
    name = "w";

    constructor(private f: (v: Item) => void) {}

    use(f: (v: Item) => void) {
        this.f = f;
    }

    walk(a: Item[]): number {
        let s: number = 0;
        for (let i = 0; i < a.length; i++) {
            const v = a[i];
            this.f(v);
            s += v.n;
        }

        return s;
    }
}

function run(r: Runner, a: Item[]) {
    return r.walk(a);
}

function nameOf(n: Named) {
    return n.name;
}

function test(a: Item[]) {
    seen = 0;
    const w = new Walker(once);
    let r = run(w, a);
    churn();
    w.use(twice);
    r += run(w, a) + w.walk(a);
    return nameOf(w) == "w" ? r * 100 + seen : -1;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = [new Item(1), new Item(5), new Item(2)];
        if (test(a) == 2415) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
