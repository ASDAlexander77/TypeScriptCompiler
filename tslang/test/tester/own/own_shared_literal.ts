// Shared<I> of an object literal (spec 26): the literal, a boxed object behind an interface with
// function fields (the ray tracer's surfaces), moves into the handle the way `new C()` does - in a
// module-level global's initializer and in a function. The global's handle and the local's two
// handles are read after churn() reuses freed memory: a literal freed while its handle points at it
// reads reused memory. The function fields are not called through the handles here: a call through
// an interface slot has no facts yet (G9 of the ray tracer plan), and a handle's borrow ends at such
// a call (spec 23.3).
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

function main() {
    let l = new Shared<S>({ f: function (x: number) { return x + 1; }, roughness: 3 });
    const m = l;
    assert(churn() == 1000);
    assert(g.value.roughness == 2, "the global's literal is alive");
    assert(l.value.roughness == 3 && m.value.roughness == 3, "the local's literal is alive");
    print(g.value.roughness, l.value.roughness, m.value.roughness);   // 2 3 3
    print("done.");
}
