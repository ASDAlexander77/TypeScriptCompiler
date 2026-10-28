// -mm=own rejects: a class extending `Walker` stores into its field (spec 57) a function that
// replaces a global array, releasing what it held. `walk` calls it with an element of an array that
// may be that global, and reads the element again after the call.
class Item {
    constructor(public n: number) {}
}

let pool: Item[] = [];

class Walker {
    constructor(protected f: (v: Item) => void) {}

    walk(a: Item[]) {
        for (let i = 0; i < a.length; i++) {
            const v = a[i];
            this.f(v);
            print(v.n);
        }
    }
}

function show(v: Item) {
    print(v.n);
}

function drain(v: Item) {
    pool = [];
}

class Sub extends Walker {
    constructor() {
        super(show);
    }

    swap() {
        this.f = drain;
    }
}

function main() {
    pool = [new Item(7)];
    const s = new Sub();
    s.swap();
    s.walk([new Item(1)]);
    print(pool.length);
}
