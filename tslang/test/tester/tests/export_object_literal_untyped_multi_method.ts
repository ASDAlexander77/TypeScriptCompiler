namespace A {

    export interface Accumulator {
        total: number;
        add(n: number): void;
        addTwice(n: number): void;
        scaled(factor: number): number;
    }

    // untyped export (infers to a boxed ObjectType, like
    // export_object_literal_untyped.ts) but extends that test's coverage to
    // MULTIPLE methods - export_object_literal_untyped.ts only ever exercised
    // a single method (inc()). Regression coverage for the combination of the
    // @boxed dynamic-import mechanism (PR #262) and the decl-text
    // object-vs-tuple printer fix (PR #263, which is what actually made
    // multi-method fields round-trip correctly at all) applied together.
    export var acc = {
        total: 0.0,
        add(n: number) { this.total = this.total + n; },
        addTwice(n: number) { this.add(n); this.add(n); },
        scaled(factor: number) { return this.total * factor; },
    };
}
