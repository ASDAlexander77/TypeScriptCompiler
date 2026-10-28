// A callback stored into a class's field keeps its family when the class's objects are cast to `any`
// or to a union (spec 59): each way back is a cast to the class, whose reads of the field are the
// family's again. Each walk reads the element again after the call, so the call must release nothing
// of it. Each step churns, so an element freed too early reads garbage; every model runs it.
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

class Walker {
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

class Other {
    walk(a: Item[]): number {
        return 0;
    }
}

function viaAny(x: any, a: Item[]) {
    const w = <Walker>x;
    return w.walk(a);
}

function viaUnion(u: Walker | Other, a: Item[]) {
    return u instanceof Walker ? u.walk(a) : -1;
}

function testAny(a: Item[]) {
    seen = 0;
    const w = new Walker(once);
    let r: number = w.walk(a);
    churn();
    w.use(twice);
    r += w.walk(a) + viaAny(w, a);
    return r * 100 + seen;
}

function testUnion(a: Item[]) {
    seen = 0;
    const w = new Walker(once);
    let r: number = w.walk(a);
    churn();
    w.use(twice);
    r += w.walk(a) + viaUnion(w, a);
    return r * 100 + seen;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = [new Item(1), new Item(5), new Item(2)];
        if (testAny(a) == 2415) ok++;
        churn();
        if (testUnion(a) == 2415) ok++;
        churn();
    }

    assert(ok == 600, "ok");
    print("done.");
}
