// -mm=own rejects: an override says other than the method it overrides. A call through the base
// reaches either, and its caller can follow only one signature.
class C {
    constructor(public n: string) {}
}

class Base {
    c: C | null = null;

    @keeps("c")
    set(c: C) {
        this.c = c;
    }
}

class Derived extends Base {
    set(c: C) {
        print(c.n);
    }
}

function main() {
    const b: Base = new Derived();
    b.set(new C("a"));
}
