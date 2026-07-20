namespace A {

    export interface Accumulator {
        base: number;
        addBase(n: number): void;
        total: number;
        add(n: number): void;
        addTwice(n: number): void;
        scaled(factor: number): number;
    }

    // extends export_object_literal_structural_typed_multi_method.ts's
    // coverage: that test's methods are all grouped after every field
    // (total; add(); addTwice(); scaled()). Here fields and methods
    // INTERLEAVE in declaration order (base; addBase(); total; add(); ...) -
    // see 00object_annotated_method_interleaved.ts for the same-module
    // version of this coverage and the mechanism this used to break.
    export var acc: {
        base: number;
        addBase(n: number): void;
        total: number;
        add(n: number): void;
        addTwice(n: number): void;
        scaled(factor: number): number;
    } = {
        base: 100.0,
        addBase(n: number) { this.base = this.base + n; },
        total: 0.0,
        add(n: number) { this.total = this.total + n; },
        addTwice(n: number) { this.add(n); this.add(n); },
        scaled(factor: number) { return this.total * factor; },
    };
}
