// -mm=own rejects: `each` gives its callback to a closure that calls it (spec 54), so the function
// passed to `each` is the one the closure calls - and it replaces a global array, releasing what
// it held. `walk` calls the closure with an element of an array that may be that global, and reads
// the element again after the call.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

function walk(a: Item[], g: (v: Item) => void) {
    for (let i = 0; i < a.length; i++) {
        const v = a[i];
        g(v);
        print(v.n);
    }
}

function each(a: Item[], f: (v: Item) => void) {
    walk(a, (v) => f(v));
}

function main() {
    pool = [new Item(7), new Item(8)];
    const mine = [new Item(1)];
    each(mine, (v) => {
        pool = [];
    });
    print(pool.length);
}
