// Sibling of 00interface_optional_extends.ts (optional FIELD inherited via
// extends), but for an optional METHOD instead. Methods go through a
// different vtable-patch path than data fields
// (mlirGenCreateInterfaceVTableForObject clones/patches function pointers,
// not byte offsets), so a fix proven for optional fields isn't automatically
// proven for optional methods.
//
// Two bugs found and fixed here:
//
// 1. Casting an object literal that OMITS an optional method inherited via
// extends used to crash the compiler outright
// (llvm_unreachable("not implemented yet") in
// mlirGenObjectVirtualTableDefinitionForInterface - building the placeholder
// vtable for a missing METHOD had never been implemented at all, only for a
// missing FIELD). Fixed by mirroring the missing-field's -1-sentinel
// placeholder pattern for a missing method's vtable slot.
//
// 2. Comparing an optional interface METHOD against `undefined`
// (`someObj.optMethod == undefined`) never worked for ANY interface, extends
// or not - hardcoded to a compile-time-constant result in
// UndefLogicHelper.h's processUndefVale, which only special-cased
// InterfaceType/ClassType against undefined, not BoundFunctionType. Fixed by:
// (a) InterfaceSymbolRefOpLowering now selects a null `this` pointer
// (branchless LLVM::SelectOp) when an optional method's vtable slot holds the
// "missing" -1 sentinel - a real bound method's `this` is never null; (b)
// UndefLogicHelper.h's new BoundFunctionType branch checks that null-or-not
// via a directly-emitted LLVM::ICmpOp - NOT the shared LogicOp<...> helper,
// whose comparison predicate is a template parameter baked in from the
// OUTER comparison operator that triggered the whole call (passing a
// different SyntaxKind at the call site is silently ignored and inverts the
// result).

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
    assert(present.opt != undefined);
    assert(present.opt(4) == 5);
    assert(present.derived == 10);

    assert(missing.base == 2);
    assert(missing.opt == undefined);
    assert(missing.derived == 20);

    // re-read the providing object's method after the non-providing one was
    // cast, to catch the shared-virtualIndex-clobber bug class if it
    // resurfaces here
    print(present.opt(9));
    assert(present.opt(9) == 10);
    assert(present.opt != undefined);

    print("done.");
}
