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

def TypeScript_Void : TypeScript_Type<"Void"> {
    let mnemonic = "void";
}

def TypeScript_Any : TypeScript_Type<"Any"> {
    let mnemonic = "any";
}

def TypeScript_Boolean : TypeScript_Type<"Boolean"> {
    let mnemonic = "boolean";
}

def TypeScript_Byte : TypeScript_Type<"Byte"> {
    let mnemonic = "byte";
}

def TypeScript_Number : TypeScript_Type<"Number"> {
    let mnemonic = "number";
}

def TypeScript_BigInt : TypeScript_Type<"BigInt"> {
    let mnemonic = "bigint";
}

def TypeScript_String : TypeScript_Type<"String"> {
    let mnemonic = "string";
}

def TypeScript_Char : TypeScript_Type<"Char"> {
    let mnemonic = "char";
}

def TypeScript_Object : TypeScript_Type<"Object"> {
    let mnemonic = "object";
}

def TypeScript_Ref : TypeScript_Type<"Ref"> {
  let mnemonic = "ref";

  let description = [{
    Syntax:

    ```
    ref-type ::= `ref` `<` type `>`
    ```

    Examples:

    ```mlir
    ref<f32>
    ref<i32>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  // We define the printer inline.
  let printer = [{ $_printer << "ref<" << getImpl()->elementType << ">"; }];

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

}

def TypeScript_IsRefType : CPred<"$_self.isa<::mlir::typescript::RefType>()">;

def TypeScript_AnyRef : DialectType<TypeScript_Dialect, TypeScript_IsRefType, "any reference type">;

def TypeScript_Optional : TypeScript_Type<"Optional"> {
  let mnemonic = "optional";

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

def TypeScript_Array : TypeScript_Type<"Array"> {
  let mnemonic = "array";

  let description = [{
    Syntax:

    ```
    array-type ::= `array<` type `>`
    ```

    Examples:

    ```mlir
    array<f32>
    array<i32>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  // We define the printer inline.
  let printer = [{ $_printer << "array<" << getImpl()->elementType << ">"; }];

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
    if ($_parser.parseType(elementType)) return Type();
    if ($_parser.parseGreater()) return Type();
    return get(elementType);
  }];

  let skipDefaultBuilders = 1;
  let genVerifyInvariantsDecl = 1;

}

#endif // TYPESCRIPT_TYPEDEFS
