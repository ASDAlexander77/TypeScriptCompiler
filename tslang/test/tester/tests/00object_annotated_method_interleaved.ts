// Extends 00object_annotated_method_params.ts's coverage: that test's
// type-literal annotations always declare all fields before all methods
// (total; add(); addTwice()). Here fields and methods INTERLEAVE in
// declaration order (base; addBase(); total; add(); ...) - this used to
// silently corrupt data: the object literal's own storage laid out as
// "fields, then methods" regardless of the annotation's order, so a method
// compiled against that layout read/wrote the wrong byte offset once
// invoked against the annotation-ordered variable, up to overwriting a
// sibling method's function pointer with a field's value. See
// docs/object-literal-boxing-design.md-adjacent history; root-caused and
// fixed via a field/value reorder pass in mlirGen(ObjectLiteralExpression).

function main() {
    let acc: { base: number; addBase(n: number): void; total: number; add(n: number): void; addTwice(n: number): void; scaled(factor: number): number } = {
        base: 100,
        addBase(n: number) { this.base = this.base + n; },
        total: 0,
        add(n: number) { this.total = this.total + n; },
        addTwice(n: number) { this.add(n); this.add(n); },
        scaled(factor: number) { return this.total * factor; },
    };

    acc.addBase(5);
    assert(acc.base == 105);
    print(acc.base);

    acc.add(3);
    assert(acc.total == 3);
    print(acc.total);

    acc.addTwice(2);
    assert(acc.total == 7);
    print(acc.total);

    const result = acc.scaled(2);
    assert(result == 14);
    print(result);

    print("done.");
}
