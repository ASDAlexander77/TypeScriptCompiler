// A generator's state borrows what it was made from (spec 34): unlike the result of another call of
// a function this module defines, it points into its argument, so iterating it is a use of the
// borrow `s.cam`, and `replace` may free that block first.
class V {
    constructor(public x: number) {}
}

class Cam {
    constructor(public f: V) {}
}

class Scene {
    constructor(public cam: Cam) {}
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

function* xs(c: Cam) {
    yield c.f.x;
}

function total(s: Scene) {
    const it = xs(s.cam);
    replace(s);
    churn();
    let t: number = 0;
    for (const v of it) {
        t += v;
    }

    return t;
}

function main() {
    const s = makeScene(5);
    const t = total(s);
    print(t);
    assert(t == 5);
    print("done.");
}
