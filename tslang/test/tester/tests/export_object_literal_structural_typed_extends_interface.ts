namespace A {

    export interface Base {
        base: number;
        addBase(n: number): void;
    }

    // extends export_object_literal_structural_typed_interleaved.ts's
    // coverage: that test's interface is flat (no `extends`). Here
    // Accumulator inherits base/addBase from Base - see
    // 00object_annotated_method_extends_interface.ts for the same-module
    // version of this coverage and the two bugs this used to hit (interface
    // vtable construction and patching both need to walk inherited members,
    // not just an interface's own directly-declared ones).
    export interface Accumulator extends Base {
        total: number;
        add(n: number): void;
        addTwice(n: number): void;
        scaled(factor: number): number;
    }

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
