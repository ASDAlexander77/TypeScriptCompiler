// -mm=own, phase 7 rejects: the generator's state object takes `c`'s cell with it, but `c` is the
// caller's, so the state object cannot own it.
class C {
    constructor(public x: number) {}
}

function* g(c: C) {
    yield c.x;
}

function main() {
    const c = new C(1);
    for (const v of g(c)) {
        print(v);
    }
}
