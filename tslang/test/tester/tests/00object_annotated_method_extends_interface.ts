// Casting a structurally-typed object literal to an interface that EXTENDS
// another interface used to fail outright: interface B { base; addBase() }
// interface Accumulator extends B { total; add(); addTwice(); scaled() }.
//
// Two separate bugs, both in the interface vtable machinery:
// 1. InterfaceInfo::getVirtualTable's recursion into `extends` forgot to
//    propagate the `methodsAsFields` flag to the recursive call, so an
//    inherited (non-conditional) method always hit a stub resolver that
//    unconditionally returns failure - the CAST STATEMENT ITSELF failed to
//    compile ("error: failed statement", no specific diagnostic).
// 2. Once (1) was fixed, the interface cast's runtime vtable-patch loop
//    (mlirGenCreateInterfaceVTableForObject) only walked the interface's
//    OWN methods, never inherited ones - an inherited method's vtable slot
//    was left holding its unpatched offset-placeholder value forever,
//    crashing with an access violation on the first call through it.

function main() {
    interface Base {
        base: number;
        addBase(n: number): void;
    }

    interface Accumulator extends Base {
        total: number;
        add(n: number): void;
        addTwice(n: number): void;
        scaled(factor: number): number;
    }

    let raw = {
        base: 100.0,
        addBase(n: number) { this.base = this.base + n; },
        total: 0.0,
        add(n: number) { this.total = this.total + n; },
        addTwice(n: number) { this.add(n); this.add(n); },
        scaled(factor: number) { return this.total * factor; },
    };

    let acc: Accumulator = <Accumulator>raw;

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
