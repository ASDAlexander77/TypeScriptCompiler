// A field read out of the record a borrowing `T | undefined` result holds borrows what the record
// borrows: pushing a class-typed field into another array is still a second reference.
class Tag { constructor(public v: string) {} }
type Fruit = { name: string, tag: Tag };

function findIn(this: Fruit[], test: (f: Fruit) => boolean): Fruit | undefined {
    for (const v of this) if (test(v)) return v;
    return undefined;
}

function main() {
    const inventory: Fruit[] = [];
    const kept: Tag[] = [];
    const r = inventory.findIn((f) => f.name == "f1");
    if (r) kept.push(r.tag);
    print(kept.length);
}
