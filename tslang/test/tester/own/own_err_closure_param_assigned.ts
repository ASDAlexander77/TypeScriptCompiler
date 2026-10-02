// -mm=own, phase 5 rejects: `c` is captured, so its cell borrows the caller's value, and assigning
// it would free that value.
class C {
    constructor(public x: number) {}
}

function read(c: C) {
    const g = () => c.x;
    c = new C(2);
    return g();
}

function main() {
    const c = new C(1);
    print(read(c), c.x);
}
