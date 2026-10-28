// -mm=own rejects: a callback given through an interface's method is stored into the class's field
// (spec 58), here one that replaces the global array `walk` walks, releasing what it held, while
// `walk` reads the element again after the call.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

interface Runner {
    use(f: (v: Item) => void): void;
    walk(a: Item[]): number;
}

class Walker implements Runner {
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

function show(v: Item) {
    print(v.n);
}

function drain(v: Item) {
    pool = [];
}

function main() {
    pool = [new Item(7)];
    const r: Runner = new Walker(show);
    r.use(drain);
    print(r.walk(pool));
}
