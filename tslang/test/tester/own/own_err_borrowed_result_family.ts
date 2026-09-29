// -mm=own, phase 4 rejects: `B.get` returns a field, but its override `D.get` returns a new
// object, so a call through `B` cannot know whether it owns the result.
class C {
    constructor(public x: number) {}
}

class B {
    c: C = new C(1);
    get() {
        return this.c;
    }
}

class D extends B {
    get() {
        return new C(2);
    }
}

function main() {
    const b: B = new D();
    print(b.get().x);
}
