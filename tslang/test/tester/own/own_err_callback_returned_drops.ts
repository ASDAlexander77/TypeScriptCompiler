// -mm=own rejects: one of the functions `pick` returns (spec 56) replaces a global array, releasing
// what it held. `walk` calls what `pick` returned with an element of an array that may be that
// global, and reads the element again after the call.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

function pick(clear: boolean) {
    return clear
        ? (v: Item) => {
              pool = [];
          }
        : (v: Item) => {};
}

function walk(a: Item[], f: (v: Item) => void) {
    for (let i = 0; i < a.length; i++) {
        const v = a[i];
        f(v);
        print(v.n);
    }
}

function main() {
    pool = [new Item(7)];
    walk([new Item(1)], pick(true));
    print(pool.length);
}
