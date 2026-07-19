// regression test: an interface member declared as a PropertySignature with a
// function type (`toString: () => string`) is categorized as a FIELD by the
// interface itself (InterfaceInfo::fields, not ::methods), even though the
// object literal implementing it stores it as a func-typed field internally
// (methodsAsFields, getInterfaceVirtualTableForObject). The access site
// (InterfaceFieldAccess) computes thisVal + slotValue expecting an OFFSET,
// unlike a real MethodSignature (`inc(): void`, InterfaceMethodAccess, raw
// function-pointer slot semantics). mlirGenObjectVirtualTableDefinitionForInterface
// must gate its constant-symbol vtable optimization (see
// docs/interface-vtable-simplification-design.md §3) on the interface's own
// method/field categorization, not merely "is this slot func-typed" - doing
// otherwise crashed the JIT with 0xC0000005 (call through a garbage address
// computed as thisVal + a raw function pointer misread as an offset).

interface Formattable {
    prefix: string;
    toString: () => string;
}

function main() {
    const a = {
        prefix: "A: ",
        toString() {
            return this.prefix + "hello";
        },
    };

    const iface: Formattable = a;
    print(iface.toString());
    assert(iface.toString() == "A: hello");

    print("done.");
}
