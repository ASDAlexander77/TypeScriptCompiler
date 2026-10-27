// -mm=own rejects: an element read before a call that swaps its array's elements, used after the
// call. The callee frees nothing, but it moves what was read to another place (spec 42).
class C {
    constructor(public x: number) {}
}

function swapFirst(a: C[]) {
    [a[0], a[1]] = [a[1], a[0]];
}

function main() {
    const a = [new C(1), new C(2)];
    const first = a[0];
    swapFirst(a);
    print(first.x);
}
