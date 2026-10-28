// -mm=own rejects: a callback parameter given on through an interface's slot reaches every class
// that fills the slot (spec 59). `drain`, given to `giveOn` for one class, is in the family of both
// classes' methods, and replaces the global array `giveOn` lends to them.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

interface Each {
    each(a: Item[], f: (v: Item) => void): number;
}

class Forward implements Each {
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

class Backward implements Each {
    each(a: Item[], f: (v: Item) => void): number {
        let s: number = 0;
        for (let i = a.length - 1; i >= 0; i--) {
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

function giveOn(e: Each, f: (v: Item) => void) {
    return e.each(pool, f);
}

function main() {
    pool = [new Item(7)];
    print(giveOn(new Forward(), once));
    print(giveOn(new Backward(), drain));
}
