// ...and with the generic declared first, before the @value types (on the first pass K is not even
// declared yet when L's field is typed)
class Box<T> {
    constructor(public l: T | null) {}
}

@value
interface L {
    readonly v: number;
    readonly box: Box<K>;
}

@value
interface K {
    readonly v: number;
}

@value
interface M {
    readonly k: Box<K>;
}

function main() {
    const a: L = { v: 1, box: new Box<K>({ v: 2 }) };
    assert(a.v == 1);
    const k = a.box.l;
    assert(k !== null && k.v == 2);
    const m: M = { k: new Box<K>({ v: 3 }) };
    const mk = m.k.l;
    assert(mk !== null && mk.v == 3);
    print("done.");
}
