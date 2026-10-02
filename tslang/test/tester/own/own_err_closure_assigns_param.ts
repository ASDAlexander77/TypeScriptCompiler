// -mm=own, phase 5 rejects: the closure assigns the captured parameter `c`, whose value is the
// caller's.
class C {
    constructor(public x: number) {}
}

function reset(c: C) {
    const g = () => {
        c = new C(2);
    };
    g();
    return c.x;
}

function main() {
    const c = new C(1);
    print(reset(c), c.x);
}
