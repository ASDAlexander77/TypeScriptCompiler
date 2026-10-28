// -mm=own rejects: a callback given through an interface's slot (spec 59) is in the family of the
// method's parameter for its direct calls too. `drain` replaces the global array, releasing what it
// held, so the direct call that lends that array may have it released under it.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

interface Each {
    each(a: Item[], f: (v: Item) => void): number;
}

class Walker implements Each {
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

function once(v: Item) {}

function drain(v: Item) {
    pool = [];
}

function viaEach(e: Each) {
    return e.each([new Item(1)], drain);
}

function main() {
    pool = [new Item(7)];
    const w = new Walker();
    print(viaEach(w));
    print(w.each(pool, once));
}
