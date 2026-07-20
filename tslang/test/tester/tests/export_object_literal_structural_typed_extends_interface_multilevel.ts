namespace M {

    // Extends export_object_literal_structural_typed_extends_interface.ts's
    // coverage (single-level: Accumulator extends Base) to a cross-module
    // 3-level chain (C extends B extends A) and an interface with TWO
    // extends targets (Combined extends Left, Right) - the same coverage
    // 00object_annotated_method_extends_interface_multilevel.ts locked in
    // same-module, verified here to also survive the module boundary.

    export interface A {
        a: number;
        addA(n: number): void;
    }
    export interface B extends A {
        b: number;
        addB(n: number): void;
    }
    export interface C extends B {
        c: number;
        addC(n: number): void;
    }

    export var raw: {
        a: number;
        addA(n: number): void;
        b: number;
        addB(n: number): void;
        c: number;
        addC(n: number): void;
    } = {
        a: 1.0,
        addA(n: number) { this.a = this.a + n; },
        b: 2.0,
        addB(n: number) { this.b = this.b + n; },
        c: 3.0,
        addC(n: number) { this.c = this.c + n; },
    };

    export interface Left {
        left: number;
        addLeft(n: number): void;
    }
    export interface Right {
        right: number;
        addRight(n: number): void;
    }
    export interface Combined extends Left, Right {
        combined: number;
        addCombined(n: number): void;
    }

    export var rawCombined: {
        left: number;
        addLeft(n: number): void;
        right: number;
        addRight(n: number): void;
        combined: number;
        addCombined(n: number): void;
    } = {
        left: 1.0,
        addLeft(n: number) { this.left = this.left + n; },
        right: 2.0,
        addRight(n: number) { this.right = this.right + n; },
        combined: 3.0,
        addCombined(n: number) { this.combined = this.combined + n; },
    };
}
