// -mm=own rejects: the signature says the constructor keeps `a`, but its body keeps `b`. A caller
// in another module trusts the signature, so it would move `a` in and free `b` while it is kept.
class C {
    constructor(public n: string) {}
}

export class Pair {
    b: C;

    @keeps("a")
    constructor(a: C, b: C) {
        this.b = b;
        print(a.n);
    }
}

function main() {
    const p = new Pair(new C("a"), new C("b"));
    print(p.b.n);
}
