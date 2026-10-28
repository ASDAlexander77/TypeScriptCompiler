// -mm=own rejects: the function stored into `Walker`'s field (spec 55) replaces a global array,
// releasing what it held. `walk` calls it with an element of an array that may be that global, and
// reads the element again after the call.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

class Walker {
    constructor(private f: (v: Item) => void) {}

    walk(a: Item[]) {
        for (let i = 0; i < a.length; i++) {
            const v = a[i];
            this.f(v);
            print(v.n);
        }
    }
}

function main() {
    pool = [new Item(7)];
    const w = new Walker((v) => {
        pool = [];
    });
    w.walk([new Item(1)]);
    print(pool.length);
}
