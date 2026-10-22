// -mm=own rejects: the literal borrows `this.pos` (spec 32), and `take` keeps its argument, where
// it would outlive the field it borrows.
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

class Sink {
    rays: Ray[] = [];
    take(r: Ray) { this.rays.push(r); }
}

class Cam {
    pos: V = new V(3);
    send(s: Sink) { s.take({ start: this.pos, dir: new V(1) }); }
}

function main() {
    const s = new Sink();
    let c = new Cam();
    c.send(s);
    c = new Cam();
    churn();
    print(s.rays[0].start.x);
}
