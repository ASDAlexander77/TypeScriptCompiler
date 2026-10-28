// A callback stored into a class's field keeps its family when other classes extend the class
// (spec 57): a derived class reaches the field in its base's storage, by the same key, both to call
// it and to store into it. Each walk reads the element again after the call, so the call must
// release nothing of it. Each step churns, so an element freed too early reads garbage; every model
// runs it.
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
    constructor(protected f: (v: Item) => void) {}

    walk(a: Item[]) {
        let s: number = 0;
        for (let i = 0; i < a.length; i++) {
            const v = a[i];
            this.f(v);
            s += v.n;
        }

        return s;
    }
}

// calls the field and stores into it from the derived class
class Sub extends Walker {
    constructor(f: (v: Item) => void) {
        super(f);
    }

    use(f: (v: Item) => void) {
        this.f = f;
    }

    each(a: Item[]) {
        let s: number = 0;
        for (const v of a) {
            this.f(v);
            s += v.n;
        }

        return s;
    }
}

// two levels down
class Leaf extends Sub {
    constructor() {
        super(twice);
    }

    reset() {
        this.f = once;
    }
}

function run(a: Item[]) {
    seen = 0;
    const s = new Sub(once);
    let r = s.walk(a) + s.each(a);
    s.use(twice);
    r += s.walk(a);
    churn();

    const l = new Leaf();
    r += l.each(a);
    l.reset();
    r += l.walk(a);

    const w = new Walker(twice);
    r += w.walk(a);
    return r * 100 + seen;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = [new Item(1), new Item(5), new Item(2)];
        if (run(a) == 4827) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
