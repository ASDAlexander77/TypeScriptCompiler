// A @value type named in a type argument of its own field's generic (`box: Box<L>`): the specialized
// class's type is made from L's, so L cannot hold it. The unspecialized `Box` was taken instead, and
// the compiler crashed at the literal.
class Box<T> {
    constructor(public l: T | null) {}
}

@value
interface L {
    readonly v: number;
    readonly box: Box<L>;
}

function main() {
    const bx = new Box<L>(null);
    const a: L = { v: 1, box: bx };
    print(a.v);
}
