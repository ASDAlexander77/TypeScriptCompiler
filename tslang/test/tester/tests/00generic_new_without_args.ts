// `new C()` of a generic class whose type arguments nothing gives (no constructor arguments reach them, no
// receiver types the value): a type parameter with no default is `any`, as TypeScript's `unknown` - "Generic class
// 'C' is missing type arguments". A receiver still gives its own (`const s: Set<number> = new Set()`).
class SS<T> {
    v: T;
}

class Box<T> {
    constructor(public x?: T) {}
}

class Pair<A, B = string> {
    constructor(public a: A) {}
}

function main() {
    const s = new SS();
    const b = new Box();
    assert(b.x === undefined);
    const p = new Pair(5);
    assert(p.a == 5);
    const typed: Set<number> = new Set();
    typed.add(1);
    assert(typed.size == 1);
    print("done.");
}
