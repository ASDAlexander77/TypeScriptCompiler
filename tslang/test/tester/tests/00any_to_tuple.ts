// `any` to an object type (`<{ n: number }>x`) reads a tuple out of the box, and throws for anything
// else. A class object in the box was read as the tuple's layout: its vtable pointer as the first
// field, and a write through the result (`p.f = g`) overwrote the vtable.
class W {
    constructor(public n: number) {}
}

function asTuple(x: any) {
    return <{ n: number }>x;
}

function main() {
    const n: number = 5;
    const t: any = { n: n };
    assert(asTuple(t).n == 5);

    let thrown = false;
    try {
        asTuple(new W(7));
    } catch (e) {
        thrown = true;
    }

    assert(thrown);

    thrown = false;
    try {
        const k: number = 3;
        asTuple(k);
    } catch (e) {
        thrown = true;
    }

    assert(thrown);

    // a tuple of other fields throws too: `{ n: 5 }` holds an s32, which was read as a number, and
    // `{ m: 1.5 }` was read as if its field were `n`
    thrown = false;
    try {
        const s: any = { n: 5 };
        asTuple(s);
    } catch (e) {
        thrown = true;
    }

    assert(thrown);

    thrown = false;
    try {
        const m: any = { m: 1.5 };
        asTuple(m);
    } catch (e) {
        thrown = true;
    }

    assert(thrown);

    // the same fields, from a literal of numbers and nested in another object
    const f: any = { n: 2.5 };
    assert(asTuple(f).n == 2.5);

    const o: any = { p: { n: n }, s: "x" };
    const q = <{ p: { n: number }; s: string }>o;
    assert(q.p.n == 5 && q.s == "x");

    print("done.");
}
