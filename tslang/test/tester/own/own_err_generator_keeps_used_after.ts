// -mm=own rejects: an argument a `@keeps` generator's state owns moved into it, and is read after.
class C {
    constructor(public x: number) {}
}

@keeps("c")
function* twice(c: C) {
    yield c.x;
    yield c.x * 2;
}

function main() {
    const c = new C(5);
    const g = twice(c);
    print(c.x);
    for (const v of g) print(v);
}
