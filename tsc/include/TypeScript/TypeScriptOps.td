#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def TypeScript_IdentifierReference : TypeScript_Op<"identifier_reference", [NoSideEffect, Symbol]> {
    let summary = "Internal op.";
    let description = [{
        The `typescript.identifier_reference` operation is internal
    }];

    let arguments = (ins FlatSymbolRefAttr:$identifier);
    let results = (outs NoneType);

    let assemblyFormat = [{
        $identifier attr-dict
    }];

    let extraClassDeclaration = [{
        static IdentifierReference create(Location location, StringRef name);
    }];
}

#endif // TYPESCRIPT_OPS
