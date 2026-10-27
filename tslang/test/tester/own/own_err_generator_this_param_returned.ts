// -mm=own rejects: a generator that borrows its `this` parameter returned by a function that does not
// say `@borrows("this")`.
class C {
    constructor(public x: number) {}
}

function* xs(this: C[]) {
    for (let i = 0; i < this.length; i++) yield this[i].x;
}

function walk(this: C[]) {
    return this.xs();
}

function main() {
    const a = [new C(1)];
    for (const v of a.walk()) print(v);
}
