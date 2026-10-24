// A field of a @value type whose generic's type argument is another @value type declared after it
// (and before): typed once that type is laid out, never as the unspecialized generic.
@value
interface L {
    readonly v: number;
    readonly box: Box<K>;
}

@value
interface K {
    readonly v: number;
}

class Box<T> {
    constructor(public l: T | null) {}
}

@value
interface M {
    readonly k: Box<K>;
}

function main() {
    const a: L = { v: 1, box: new Box<K>({ v: 2 }) };
    const m: M = { k: a.box };
    assert(a.v == 1);
    const k = m.k.l;
    assert(k !== null && k.v == 2);
    print("done.");
}
