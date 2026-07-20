// Extends 00object_annotated_method_extends_interface.ts's coverage (single
// level: Accumulator extends Base) to a 3-level chain (C extends B extends
// A) and an interface with TWO extends targets (Combined extends Left,
// Right) - the "known limitation, not verified" gap flagged when that
// single-level fix landed (PR #266). Both work correctly with the same
// fix: InterfaceInfo::getVirtualTable's extends recursion and
// getAllMethods() are themselves recursive/multi-target already, not
// hardcoded to one level - this test just locks that in.

function main() {
    interface A {
        a: number;
        addA(n: number): void;
    }
    interface B extends A {
        b: number;
        addB(n: number): void;
    }
    interface C extends B {
        c: number;
        addC(n: number): void;
    }

    let raw = {
        a: 1.0,
        addA(n: number) { this.a = this.a + n; },
        b: 2.0,
        addB(n: number) { this.b = this.b + n; },
        c: 3.0,
        addC(n: number) { this.c = this.c + n; },
    };

    let obj: C = <C>raw;

    obj.addA(10);
    assert(obj.a == 11);
    print(obj.a);

    obj.addB(20);
    assert(obj.b == 22);
    print(obj.b);

    obj.addC(30);
    assert(obj.c == 33);
    print(obj.c);

    interface Left {
        left: number;
        addLeft(n: number): void;
    }
    interface Right {
        right: number;
        addRight(n: number): void;
    }
    interface Combined extends Left, Right {
        combined: number;
        addCombined(n: number): void;
    }

    let rawCombined = {
        left: 1.0,
        addLeft(n: number) { this.left = this.left + n; },
        right: 2.0,
        addRight(n: number) { this.right = this.right + n; },
        combined: 3.0,
        addCombined(n: number) { this.combined = this.combined + n; },
    };

    let combined: Combined = <Combined>rawCombined;

    combined.addLeft(10);
    assert(combined.left == 11);
    print(combined.left);

    combined.addRight(20);
    assert(combined.right == 22);
    print(combined.right);

    combined.addCombined(30);
    assert(combined.combined == 33);
    print(combined.combined);

    print("done.");
}
