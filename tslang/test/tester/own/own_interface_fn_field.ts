// -mm=own: a function-typed field of an interface closed in its module (`f: (x: number) => number`),
// called through the interface: every function ever stored into the fields its implementing object
// literals give it is known (spec 31), so the call gets their facts - here, that it destroys
// nothing - and the borrow of the object it is called on lives across it. Through a global
// `Shared<S>` of a literal (the ray tracer's surfaces), a local one, and a parameter with no handle,
// each after churn() reuses freed memory: a literal freed while it is borrowed reads reused memory.
interface S { f: (x: number) => number; roughness: number; }
let g = new Shared<S>({ f: function (x: number) { return x * 2; }, roughness: 2 });

class Junk {
    a = -1;
    b = -2;
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) {
        const j = new Junk();
        j.a = -i;
        keep.push(j);
    }

    return keep.length;
}

class P {
    s: S = { f: function (x: number) { return x + this.roughness; }, roughness: 3 };
}

function viaParam(p: P) {
    return p.s.f(3);
}

function viaHandle(h: Shared<S>) {
    return h.value.f(4) + h.value.roughness;
}

function main() {
    let l = new Shared<S>({ f: function (x: number) { return x + 1; }, roughness: 3 });
    const p = new P();

    assert(churn() == 1000);
    assert(g.value.f(3) == 6, "through the global's handle");
    assert(churn() == 1000);
    assert(l.value.f(3) == 4, "through the local's handle");
    assert(churn() == 1000);
    assert(viaParam(p) == 6, "through a parameter");
    assert(churn() == 1000);
    assert(viaHandle(g) == 10 && viaHandle(l) == 8, "through a handle given as a parameter");
    assert(churn() == 1000);
    assert(g.value.roughness == 2 && l.value.roughness == 3 && p.s.roughness == 3, "the literals are alive");

    print(g.value.f(3), l.value.f(3), viaParam(p), viaHandle(g), viaHandle(l));   // 6 4 6 10 8
    print("done.");
}
