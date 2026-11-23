// TSLANG_ISSUES 165: `<Shape>a` of an `any` holding a member of a union of shapes is that member (it was "can't be
// stored"), and `<object>` of an object literal is a block of its own (it was the address of a stack slot: returned,
// or kept by -mm=rc, it crashed). Not under -mm=own: the box is read once per shape checked, which own counts as a
// second reference.
type Shape = { kind: "sq"; size: number } | { kind: "ci"; r: number };

function area(a: any) {
    const s = <Shape>a;
    if (s.kind === "ci") return s.r * s.r;
    return s.size * s.size;
}

function id(x: object | string) {
    return x;
}

function make(i: number): object | string {
    if (i > 0) return <object>{ n: i, s: "x" + i };
    return "s";
}

function main() {
    const c: Shape = { kind: "ci", r: 3 };
    const q: Shape = { kind: "sq", size: 2 };
    assert(area(c) == 9, "ci");
    assert(area(q) == 4, "sq");

    let thrown = false;
    try {
        area(5);
    } catch {
        thrown = true;
    }
    assert(thrown, "a number is no shape");

    const o = <object>{};
    assert(id(o) === o, "the same object");

    let objects = 0;
    for (let i = 0; i < 10000; i++) {
        const m = make(i % 2);
        if (typeof m !== "string" && id(m) === m) objects++;
    }
    assert(objects == 5000, "objects made in a loop");
    print("done.");
}
