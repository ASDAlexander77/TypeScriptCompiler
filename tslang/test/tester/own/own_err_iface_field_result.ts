// -mm=own rejects: `dirOf` returns a borrow of its parameter's field, read through an interface;
// the result is read after that field is overwritten (spec 28).
class V {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new V(999);
        c.v.push(99);
        junk.push(c);
    }
}

interface R { dir: V; }

function dirOf(r: R) {
    return r.dir;
}

function main() {
    const r: R = { dir: new V(1) };
    const d = dirOf(r);
    r.dir = new V(2);
    churn();
    print(d.x);
}
