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

    print("done.");
}
