// A closure over a `const` read out of a field, an element or a global sees
// the value the const was given, not what the place holds by the time the closure is made. Folded
// into the read, the const had no storage of its own, and the capture was taken through the
// place it was read from: `const v = n.v; n.v = d; const c = () => v.x;` gave `d.x` under gc and
// none (rc gives such a const storage, and was right).
class V {
    constructor(public x: number) {}
}

class N {
    v = new V(3);
    items: V[] = [new V(4)];
}

let G = new V(5);

function field() {
    const n = new N();
    const v = n.v;
    n.v = new V(2);
    const c = (): number => v.x;
    return c();
}

function element() {
    const n = new N();
    const v = n.items[0];
    n.items[0] = new V(2);
    const c = (): number => v.x;
    return c();
}

function global() {
    const v = G;
    G = new V(2);
    const c = (): number => v.x;
    return c();
}

function main() {
    assert(field() == 3, "a closure over a const read out of a field");
    assert(element() == 4, "a closure over a const read out of an element");
    assert(global() == 5, "a closure over a const read out of a global");
    print("done.");
}
