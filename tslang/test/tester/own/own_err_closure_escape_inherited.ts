// -mm=own, phase 5b rejects: the inner closure escapes from `g` with a cell that `g`'s own box
// holds for `make`.
class C {
    constructor(public x: number) {}
}

function make() {
    let a = new C(1);
    const g = () => () => a.x;
    const h = g();
    return h();
}

function main() {
    print(make());
}
