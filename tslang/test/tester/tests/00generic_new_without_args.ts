// `new C()` of a generic class with no constructor and no base class, given no arguments: nothing gives its type
// arguments, and a type parameter with no default is `any`, as TypeScript's `unknown` - "Generic class 'C' is missing
// type arguments". A class whose constructor arguments give them is as before (`new Pair(5)`).
class SS<T> {
    v: T;
    n = 2;
}

class Pair<A, B = string> {
    constructor(public a: A) {}
}

function main() {
    const s = new SS();
    assert(s.n == 2);
    const s2 = new SS<number>();
    assert(s2.n == 2);
    const p = new Pair(5);
    assert(p.a == 5);
    print("done.");
}
