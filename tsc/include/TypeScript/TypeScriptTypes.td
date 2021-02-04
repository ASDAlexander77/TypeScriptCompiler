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

def SimpleTypeA : TypeScript_Type<"SimpleA"> {
  let mnemonic = "smpla";
}

#endif // TYPESCRIPT_TYPEDEFS
