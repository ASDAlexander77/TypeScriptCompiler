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

  let summary = "Complex number with a parameterized element type";
  let description = [{
    Syntax:

    ```
    optional-type ::= `optional` `<` type `>`
    ```

    Examples:

    ```mlir
    optional<f32>
    optional<i32>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  // We define the printer inline.
  let printer = [{ $_printer << "optional<" << getImpl()->elementType << ">"; }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }], [{
      return Base::getChecked($_loc, elementType);
    }]>
  ];

  // Parses the above format
  let parser = [{
    Type elementType;
    if ($_parser.parseLess()) return Type();
    if ($_parser.parseLBrace()) return Type();
    if ($_parser.parseType(elementType)) return Type();
    if ($_parser.parseRBrace()) return Type();
    if ($_parser.parseGreater()) return Type();
    return get(elementType);
  }];

  let skipDefaultBuilders = 1;
  let genVerifyInvariantsDecl = 1;

}

#endif // TYPESCRIPT_TYPEDEFS
