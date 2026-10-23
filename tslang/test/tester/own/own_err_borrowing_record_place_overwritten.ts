// -mm=own rejects: the literal borrows the field `c.pos` (spec 32), which is given a new value - its
// old one freed - before the literal is read.
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

class Cam {
    pos: V = new V(3);
}

function main() {
    const c = new Cam();
    const r: Ray = { start: c.pos, dir: new V(1) };
    c.pos = new V(9);
    churn();
    print(len(r));
}
