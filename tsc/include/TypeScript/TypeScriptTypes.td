#ifndef TYPESCRIPT_TYPEDEFS
#define TYPESCRIPT_TYPEDEFS

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TypeScript types definition.
//===----------------------------------------------------------------------===//

// To get the typescript dialect def.
include "TypeScriptDialect.td"

// All of the types will extend this class.
class TypeScript_Type<string name> : TypeDef<TypeScript_Dialect, name> { }

def Optional : TypeScript_Type<"Optional"> {
  let mnemonic = "optional";

  // List of type parameters.
  let parameters = (
    ins
    "bool":$hasValue,
    "::mlir::Type":$value
  );
}

#endif // TYPESCRIPT_TYPEDEFS
