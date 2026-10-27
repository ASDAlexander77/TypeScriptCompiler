// -mm=own rejects: an element read before a swap of its array's elements, used after it. The swap
// frees nothing, but what was read is held by another place now, which may end first (spec 42).
class C {
    constructor(public x: number) {}
}

function main() {
    const a = [new C(1), new C(2)];
    const first = a[0];
    [a[0], a[1]] = [a[1], a[0]];
    print(first.x);
}
