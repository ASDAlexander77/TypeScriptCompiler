// -mm=own: a `let` given to an object literal that a call only reads as an interface is borrowed
// by it, not moved in (spec 32): the `let` is used after the call and freed once, by its own release.
class V {
    constructor(public x: number) {}
}

interface Ray {
    start: V;
    dir: V;
}

function show(r: Ray) {
    return r.start.x + r.dir.x;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 2000; i++) {
        let p = new V(i);
        t += show({ start: p, dir: new V(2) });
        t += p.x;
    }

    assert(t == 4002000, "t");
    print("done.");
}
