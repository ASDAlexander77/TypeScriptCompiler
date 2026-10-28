// -mm=own rejects: a method called through an interface may be given any callback (spec 58), here
// one that replaces the global array the method walks, releasing what it held, while the method
// reads the element again after the call.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

interface Runner {
    each(a: Item[], f: (v: Item) => void): number;
}

class Walker implements Runner {
    each(a: Item[], f: (v: Item) => void): number {
        let s: number = 0;
        for (let i = 0; i < a.length; i++) {
            const v = a[i];
            f(v);
            s += v.n;
        }

        return s;
    }
}

function drain(v: Item) {
    pool = [];
}

function main() {
    pool = [new Item(7)];
    const r: Runner = new Walker();
    print(r.each(pool, drain));
}
