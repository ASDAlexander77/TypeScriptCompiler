// Combines two previously-separate axes of interface coverage that had never
// been tested together: an optional member (00interface_optional_cast_order.ts
// - optional member's virtualIndex can get clobbered to -1 on the SHARED
// InterfaceInfo by whichever cast ran most recently) and `extends`
// (00object_annotated_method_extends_interface*.ts - inherited members need
// vtableOffset added to their own-interface-relative virtualIndex). The
// optional member here is declared on the BASE interface, so resolving it
// through the DERIVED interface's combined vtable exercises both the
// extends-offset math and the optional-member vtable-patch path at once.

function main() {
    interface Base {
        base: number;
        opt?: number;
    }

    interface Derived extends Base {
        derived: number;
    }

    let present: Derived = <Derived>{ base: 1, opt: 5, derived: 10 };
    let missing: Derived = <Derived>{ base: 2, derived: 20 };

    assert(present.base == 1);
    assert(present.opt == 5);
    assert(present.derived == 10);

    assert(missing.base == 2);
    assert(missing.opt == undefined);
    assert(missing.derived == 20);

    // re-read the providing object after the non-providing one was cast, to
    // catch the shared-virtualIndex-clobber bug class if it resurfaces here
    print(present.opt);
    assert(present.opt == 5);

    print("done.");
}
