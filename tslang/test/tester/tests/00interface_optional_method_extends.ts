// Sibling of 00interface_optional_extends.ts (optional FIELD inherited via
// extends), but for an optional METHOD instead. Methods go through a
// different vtable-patch path than data fields
// (mlirGenCreateInterfaceVTableForObject clones/patches function pointers,
// not byte offsets), so a fix proven for optional fields isn't automatically
// proven for optional methods.
//
// Found and fixed: casting an object literal that OMITS an optional method
// inherited via extends used to crash the compiler outright
// (llvm_unreachable("not implemented yet") in
// mlirGenObjectVirtualTableDefinitionForInterface - building the placeholder
// vtable for a missing METHOD had never been implemented at all, only for a
// missing FIELD). Fixed by mirroring the missing-field's -1-sentinel
// placeholder pattern for a missing method's vtable slot.
//
// KNOWN LIMITATION (not fixed here, deferred): comparing an optional
// interface METHOD against `undefined` (`someObj.optMethod == undefined`)
// does not work correctly for ANY interface, extends or not - it's hardcoded
// to a compile-time-constant result in UndefLogicHelper.h's
// processUndefVale, which only special-cases InterfaceType/ClassType against
// undefined, not BoundFunctionType. A fix was attempted (zeroing the
// bound_func's `this` pointer as a missing-sentinel, checking it in the
// undefined-comparison lowering) but caused a control-flow crash deep in
// LLVM dialect conversion (execution ran off the end of a JIT-compiled
// block) that wasn't resolved before this test landed. Do not use
// `== undefined` / `!= undefined` on an optional interface method in a test
// until that's fixed - only test presence via calling it or (like here)
// simply never calling an omitted one.

function main() {
    interface Base {
        base: number;
        opt?(n: number): number;
    }

    interface Derived extends Base {
        derived: number;
    }

    let present: Derived = <Derived>{
        base: 1.0,
        opt(n: number) { return this.base + n; },
        derived: 10.0,
    };
    let missing: Derived = <Derived>{ base: 2.0, derived: 20.0 };

    assert(present.base == 1);
    assert(present.opt(4) == 5);
    assert(present.derived == 10);

    // does not call missing.opt (per the known limitation above, there is no
    // reliable way yet to check it's absent before calling it) - just
    // exercises that constructing/casting this object at all doesn't crash
    // building its vtable, which is the bug that was actually fixed here.
    assert(missing.base == 2);
    assert(missing.derived == 20);

    // re-read the providing object's method after the non-providing one was
    // cast, to catch the shared-virtualIndex-clobber bug class if it
    // resurfaces here
    print(present.opt(9));
    assert(present.opt(9) == 10);

    print("done.");
}
