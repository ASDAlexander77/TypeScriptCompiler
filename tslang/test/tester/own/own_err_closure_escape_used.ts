// -mm=own, phase 5b rejects: the closure takes `a` with it, and the frame reads `a` afterwards.
class C {
    constructor(public x: number) {}
}

function make() {
    let a = new C(1);
    const f = () => a.x;
    print(a.x);
    return f;
}

function main() {
    print(make()());
}
