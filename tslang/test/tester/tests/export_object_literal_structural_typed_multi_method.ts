namespace A {

    export interface Accumulator {
        total: number;
        add(n: number): void;
        addTwice(n: number): void;
        scaled(factor: number): number;
    }

    // extends export_object_literal_structural_typed_params.ts's coverage: that
    // test deliberately stayed within a SINGLE-method shape because a genuinely
    // multi-method structurally-typed export used to corrupt every method field
    // after the first when reconstructed cross-module - see
    // docs/interface-vtable-simplification-design.md's "multi-method
    // cross-module vtable slot bug" section. Root cause turned out to be in the
    // decl-text printer (MLIRPrinter.h's printType), not the vtable-patch code:
    // a named-field tuple type was printed with positional tuple syntax
    // ("[name: (args) => result]") instead of object syntax
    // ("{name(args): result}"), so method fields re-imported as the wider
    // HybridFunctionType (16-byte {data,func} pair) instead of the exporter's
    // actual plain FunctionType (8-byte raw pointer) storage - misaligning
    // every field after the first.
    export var acc: { total: number; add(n: number): void; addTwice(n: number): void; scaled(factor: number): number } = {
        total: 0.0,
        add(n: number) { this.total = this.total + n; },
        addTwice(n: number) { this.add(n); this.add(n); },
        scaled(factor: number) { return this.total * factor; },
    };
}
