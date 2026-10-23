// -mm=own rejects: the literal borrows the parameter `p` (spec 32), which this function keeps and
// moves into the sink before the literal is read; the sink may free it.
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

class Sink {
    vs: V[] = [];
    take(v: V) { this.vs.push(v); }
    clear() { this.vs.pop(); }
}

function give(s: Sink, p: V) {
    const r: Ray = { start: p, dir: new V(2) };
    s.take(p);
    s.clear();
    churn();
    return len(r);
}

function main() {
    const s = new Sink();
    print(give(s, new V(1)));
}
