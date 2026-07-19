// regression test: InterfaceInfo::getVirtualTable() (MLIRGenStore.h) marks an
// optional interface member's virtualIndex as -1 on the SHARED InterfaceInfo
// whenever it builds a vtable for an object that doesn't provide that member
// -- a side effect of resolving that ONE specific cast target, not a property
// of the interface itself. InterfaceFieldAccess (MLIRGenImpl.h) then reads
// that shared, mutable virtualIndex at COMPILE TIME: if it is -1 it bakes in
// OptionalUndefOp directly, bypassing the runtime InterfaceSymbolRefOp read
// entirely (InterfaceSymbolRefOpLowering already has a correct per-object
// runtime slot==-1 check -- LowerToLLVM.cpp -- but this compile-time shortcut
// never reaches it).
//
// So: cast an object that DOES provide the optional member, then cast a
// DIFFERENT object of the same interface that does NOT provide it, then
// access the member on the FIRST (providing) object -- the access is
// compiled after the second cast clobbered the shared virtualIndex to -1,
// so it wrongly resolves to "undefined" instead of reading the real value.
// See docs/interface-vtable-simplification-design.md §5.

interface Box {
    a: number;
    m?: number;
}

const present: Box = { a: 2, m: 5 };
const missing: Box = { a: 1 };

function main() {
    assert(missing.m == undefined);
    print(present.m);
    assert(present.m == 5);

    print("done.");
}
