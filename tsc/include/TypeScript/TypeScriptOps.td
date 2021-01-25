#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TypeScript_FooOp : TypeScript_Op<"foo", [NoSideEffect,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `typescript.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = typescript.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def IdentifierReferenceOp : TypeScript_Op<"identifier_reference", [NoSideEffect]> {
  let summary = "identifier reference operation";
  let description = [{
    Identifier reference
  }];

  // The identifier operation takes a symbol reference attribute as the
  // identifier
  let arguments = (ins FlatSymbolRefAttr:$identifier);

  let results = (outs TypeScript_IdentifierReferenceType);

  // Add custom build methods for the identifier reference
  let builders = [
    OpBuilderDAG<(ins "StringRef":$identifier)>
  ];
  let extraClassDeclaration = [{
    static IdentifierReferenceOp create(Location location, StringRef name);
  }];
}


#endif // TYPESCRIPT_OPS
