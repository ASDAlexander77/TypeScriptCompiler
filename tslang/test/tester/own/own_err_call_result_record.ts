// A borrowing record (spec 32) returned by a call points into the arguments it was made from
// (spec 34): unlike the result of another call of a function this module defines, a use of it is a
// use of the borrow `s.cam`, and `replace` may free that block first.
class V {
    constructor(public x: number) {}
}

class Cam {
    constructor(public f: V) {}
}

class Scene {
    constructor(public cam: Cam) {}
}

interface Hit {
    v: V;
    n: number;
}

@noinline
function makeScene(n: number) {
    return new Scene(new Cam(new V(n)));
}

function replace(s: Scene) {
    s.cam = new Cam(new V(2));
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 50; i++) {
        junk.push(new V(999));
    }

    return junk.length;
}

function hit(c: Cam): Hit {
    return { v: c.f, n: 1 };
}

function total(s: Scene) {
    const h = hit(s.cam);
    replace(s);
    churn();
    return h.v.x + h.n;
}

function main() {
    const s = makeScene(5);
    const t = total(s);
    print(t);
    assert(t == 6);
    print("done.");
}
