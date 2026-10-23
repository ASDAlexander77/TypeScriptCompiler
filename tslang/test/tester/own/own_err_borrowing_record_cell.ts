// -mm=own rejects: the literal borrows the value of a captured variable (spec 32), and the closure
// gives the variable a new value - its old one freed - before the literal is read.
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

function run() {
    let v = new V(1);
    const g = () => {
        const r: Ray = { start: v, dir: new V(2) };
        v = new V(5);
        churn();
        return len(r);
    };
    return g();
}

function main() {
    print(run());
}
