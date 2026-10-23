// -mm=own rejects: the literal borrows the value `d` holds (spec 32), and `d` is given a new value
// - its old one freed - before the literal is read.
class V {
    constructor(public x: number) {}
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 200; i++) {
        keep.push(new V(77777));
    }
    return keep.length;
}

interface Ray { start: V; dir: V; }

function len(r: Ray) { return r.start.x + r.dir.x; }

function main() {
    let d = new V(1);
    const r: Ray = { start: d, dir: new V(2) };
    d = new V(3);
    churn();
    print(len(r));
}
