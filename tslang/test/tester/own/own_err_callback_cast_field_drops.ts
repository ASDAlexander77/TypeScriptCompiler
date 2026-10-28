// -mm=own rejects: the function stored into `Walker`'s field replaces a global array, releasing what
// it held. The class is cast to `any` and back before `walk` calls it (spec 59), with an element of an
// array that may be that global, and reads the element again after the call.
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

function viaAny(x: any) {
    (<Walker>x).walk([new Item(1)]);
}

function main() {
    pool = [new Item(7)];
    viaAny(new Walker((v) => {
        pool = [];
    }));
    print(pool.length);
}
