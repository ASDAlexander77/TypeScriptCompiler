// A conditional between two object literals, one of which borrows a field (`{ start: this.pos, ... }`),
// each holding an object the branch makes (`new V(i)`). Under rc the branch gave its new object back
// before the receiver after the conditional retained the literal's fields, so the field was freed under
// it (heap corruption). The branch now retains the literal it yields. Each value is read after churn()
// has reused anything freed too early.
class V {
    constructor(public x: number) {}
}

interface Ray {
    start: V;
    dir: V;
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 200; i++) keep.push(new V(77777));
    return keep.length;
}

class Cam {
    pos: V = new V(3);

    pick(i: number) {
        const t: Ray = i == 1 ? { start: this.pos, dir: new V(i) } : { start: new V(i), dir: new V(i + 10) };
        churn();
        return t.start.x + t.dir.x;
    }
}

function main() {
    const c = new Cam();
    let s = 0;
    for (let i = 0; i < 4; i++) {
        s += c.pick(i);
        churn();
        assert(c.pos.x == 3, "pos");
    }

    assert(s == 0 + 10 + (3 + 1) + 2 + 12 + 3 + 13, "sum");
    print("done.");
}
