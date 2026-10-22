// @strict-null false
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
    pick(c: boolean) {
        let r: Ray = { start: new V(10), dir: new V(20) };
        churn();
        if (c) { r = { start: this.pos, dir: new V(1) }; }
        churn();
        const n = len(r);
        let best: Ray = null;
        for (let i = 0; i < 3; i++) {
            const t: Ray = i == 1 ? { start: this.pos, dir: new V(i) } : { start: new V(i), dir: new V(i) };
            churn();
            if (len(t) > 3 && c) { best = t; }
        }
        churn();
        return n + (best == null ? 0 : len(best));
    }
}
function main() {
    const c = new Cam();
    let s = 0;
    for (let i = 0; i < 500; i++) {
        s += c.pick(i % 2 == 0);
        churn();
        assert(c.pos.x == 3);
    }
    print(s, c.pos.x);
}
