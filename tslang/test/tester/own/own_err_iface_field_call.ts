// -mm=own rejects: `d` borrows `r.dir`, read through an interface, and a call given `r` overwrites
// that field through the interface (spec 28).
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

function setDir(r: R) {
    r.dir = new V(5);
}

function f(r: R) {
    let d = r.dir;
    setDir(r);
    churn();
    print(d.x);
}

function main() {
    const r: R = { dir: new V(1) };
    f(r);
}
