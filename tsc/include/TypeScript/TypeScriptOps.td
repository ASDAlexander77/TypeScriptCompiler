#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def TypeScript_IdentifierReference : TypeScript_Op<"identifier_reference", [NoSideEffect]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `typescript.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        typescript.identifier_reference @my_identifier_name
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$identifier);
    let results = (outs NoneType);

    let assemblyFormat = [{
        $identifier attr-dict
    }];

    let builders = [
        OpBuilderDAG<(ins "StringRef":$identifier)>
    ];
}

#endif // TYPESCRIPT_OPS
