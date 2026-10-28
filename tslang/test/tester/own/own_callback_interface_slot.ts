// A method an interface's vtable holds takes its callback parameter's family from every call through
// the interface's slot as well as from its direct calls (spec 59): the interface is closed, so no
// other module calls through it. Two classes fill the slot; a caller's own callback parameter is given
// on through the interface to both. Each walk reads the element again after the call, so the call
// must release nothing of it. Each step churns, so an element freed too early reads garbage; every
// model runs it.
class Item {
    constructor(public n: number) {}
}

function churn() {
    const junk: Item[] = [];
    for (let i = 0; i < 64; i++) junk.push(new Item(999));
}

let seen = 0;

function once(v: Item) {
    seen++;
}

function twice(v: Item) {
    seen += 2;
}

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
            s += v.n * 10;
        }

        return s;
    }
}

function viaEach(e: Each, a: Item[]) {
    return e.each(a, once);
}

function giveOn(e: Each, a: Item[], f: (v: Item) => void) {
    return e.each(a, f);
}

function test(a: Item[]) {
    seen = 0;
    const fw = new Forward();
    const bw = new Backward();
    let r: number = fw.each(a, twice) + bw.each(a, once);
    churn();
    r += viaEach(fw, a) + viaEach(bw, a);
    churn();
    r += giveOn(fw, a, twice) + giveOn(bw, a, once);
    return r * 100 + seen;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = [new Item(1), new Item(5), new Item(2)];
        if (test(a) == 26424) ok++;
        churn();
    }

    assert(ok == 300, "ok");
    print("done.");
}
