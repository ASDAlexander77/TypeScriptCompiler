namespace A {

    export interface Accumulator {
        total: number;
        add(n: number): void;
    }

    // structurally-typed (not interface-typed) export, like
    // export_object_literal_structural_typed.ts, but extends that test's
    // coverage: the method takes a PARAMETER (that test's inc() took none).
    //
    // NOTE: deliberately kept to ONE method. A multi-method version of this
    // (total; add(n); addTwice(n); scaled(factor): number) was tried and
    // found broken cross-module for any method beyond vtable slot 0 - see
    // docs/interface-vtable-simplification-design.md's "multi-method
    // cross-module vtable slot bug" section. That's a distinct, deeper,
    // not-yet-fixed bug; this test intentionally stays within the
    // currently-working single-method shape.
    export var acc: { total: number; add(n: number): void } = {
        total: 0.0,
        add(n: number) { this.total = this.total + n; },
    };
}
