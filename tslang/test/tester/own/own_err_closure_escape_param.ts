// -mm=own, phase 5b rejects: `c` is the caller's, so a closure that escapes cannot own it.
class C {
    constructor(public x: number) {}
}

function make(c: C) {
    return () => c.x;
}

function main() {
    const c = new C(1);
    print(make(c)());
}
