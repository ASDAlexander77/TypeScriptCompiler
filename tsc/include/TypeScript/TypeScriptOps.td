#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// TypeScript op definitions
//===----------------------------------------------------------------------===//

def TypeScript_IdentifierReference : TypeScript_Op<"identifier_reference", [NoSideEffect, Symbol]> {
    let arguments = (ins FlatSymbolRefAttr:$identifier);
    let results = (outs NoneType);

    let extraClassDeclaration = [{
        static IdentifierReference create(Location location, StringRef name);
    }];
}

def TypeScript_PrintOp : TypeScript_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input string, and produces no results.
  }];

  let arguments = (ins AnyType:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def TypeScript_AssertOp : TypeScript_Op<"assert"> {
  let summary = "Assert operation";
  let description = [{
    Assert operation

    Example:

    ```mlir
    assert %b, "Expected ... to be true"
    ```
  }];

  let arguments = (ins I1:$arg, StrAttr:$msg);

  let assemblyFormat = "$arg `,` $msg attr-dict";

  // AssertOp is fully verified by its traits.
  let verifier = ?;

  let hasCanonicalizer = 1;
}

#endif // TYPESCRIPT_OPS
