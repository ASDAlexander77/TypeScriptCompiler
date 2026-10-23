// -mm=own rejects: a field written through a borrowing record (spec 32) would release the value the
// record borrows, and the record's release would leave the one written.
class V {
    constructor(public x: number) {}
}

interface Ray { start: V; dir: V; }

function len(r: Ray) { return r.start.x + r.dir.x; }

class Cam {
    pos: V = new V(3);
    go() {
        const r: Ray = { start: this.pos, dir: new V(1) };
        r.start = new V(7);
        return len(r);
    }
}

function main() {
    const c = new Cam();
    print(c.go(), c.pos.x);
}
