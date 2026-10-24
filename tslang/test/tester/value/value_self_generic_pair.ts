// ...in any of the generic's type arguments
class Pair<X, Y> {
    constructor(public a: X | null, public b: Y | null) {}
}

@value
interface A {
    readonly v: number;
    readonly p: Pair<B, number>;
}

@value
interface B {
    readonly w: number;
    readonly a: A;
}

function main() {
    const a: A = { v: 1, p: new Pair<B, number>(null, 3) };
    print(a.v);
}
