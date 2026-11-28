// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
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
