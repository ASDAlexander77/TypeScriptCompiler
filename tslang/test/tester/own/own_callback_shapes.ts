// Callbacks known by the functions that can reach them, in four shapes: a closure kept in a local
// and called (spec 54), a callback stored into a class's field - by the constructor or by a method
// - and called through it (spec 55), and one a function returns, called through a local or given on
// (spec 56). Each walk reads the element again after the call, so the call must release nothing of
// it. Each step churns, so an element freed too early reads garbage; every model runs it.
class Item {
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

// a closure in a local, over the callback parameter
function walkLocal(a: Item[], f: (v: Item) => void) {
    const call = (v: Item) => f(v);
    let s = 0;
    for (let i = 0; i < a.length; i++) {
        const v = a[i];
        call(v);
        s += v.n;
    }

    return s;
}

let seen = 0;

// a callback in a field
class Walker {
    constructor(private f: (v: Item) => void) {}

    use(f: (v: Item) => void) {
        this.f = f;
    }

    walk(a: Item[]) {
        let s = 0;
        for (let i = 0; i < a.length; i++) {
            const v = a[i];
            this.f(v);
            s += v.n;
        }

        return s;
    }
}

// a callback returned
function above(n: number) {
    return n > 1 ? (v: Item) => v.n > 1 : (v: Item) => v.n > 0;
}

function count(a: Item[], f: (v: Item) => boolean) {
    let c = 0;
    for (let i = 0; i < a.length; i++) {
        const v = a[i];
        if (f(v)) c++;
        c += v.n * 10;
    }

    return c;
}

class Bag {
    constructor(private data: Item[] = []) {}

    local() {
        let seen = 0;
        return walkLocal(this.data, (v) => {
            seen++;
        }) * 10 + seen;
    }

    // the callbacks count into a global: a closure that captures, stored into an object, escapes
    field() {
        seen = 0;
        const w = new Walker((v) => {
            seen++;
        });
        const first = w.walk(this.data);
        w.use((v) => {
            seen += 2;
        });
        return (first + w.walk(this.data)) * 100 + seen;
    }

    returned() {
        const big = above(2);
        let c = 0;
        for (const v of this.data) if (big(v)) c++;
        return count(this.data, above(0)) * 10 + c;
    }
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const b = new Bag([new Item(1), new Item(5), new Item(2)]);
        if (b.local() == 83) ok++;
        churn();
        if (b.field() == 1609) ok++;
        churn();
        if (b.returned() == 832) ok++;
        churn();
    }

    assert(ok == 300 * 3, "ok");
    print("done.");
}
