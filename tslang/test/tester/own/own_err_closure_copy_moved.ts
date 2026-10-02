// -mm=own, phase 5 rejects: the closure borrows its copy of `h`, and is called after `h` moved
// into the array.
class C {
    constructor(public x: number) {}
}

function main() {
    const keep: C[] = [];
    const h = new C(1);
    let k = 0;
    const f = () => h.x + k;
    keep.push(h);
    print(f(), keep.length);
}
