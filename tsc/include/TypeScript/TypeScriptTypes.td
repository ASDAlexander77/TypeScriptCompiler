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
    if ($_parser.parseType(elementType)) return Type();
    if ($_parser.parseGreater()) return Type();
    return get(elementType);
  }];

  let skipDefaultBuilders = 1;

}

def TypeScript_IsRefType : CPred<"$_self.isa<::mlir::typescript::RefType>()">;

def TypeScript_AnyRef : DialectType<TypeScript_Dialect, TypeScript_IsRefType, "any reference type">;

def TypeScript_IsTupleType : CPred<"$_self.isa<::mlir::typescript::TupleType>()">;

def TypeScript_AnyTuple : DialectType<TypeScript_Dialect, TypeScript_IsTupleType, "any tuple type">;

def TypeScript_IsTupleRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::TupleType>()">;

def TypeScript_AnyTupleRef : DialectType<TypeScript_Dialect, TypeScript_IsTupleRefType, "any tuple reference type">;

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
  let genVerifyInvariantsDecl = 1;

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
}

def TypeScript_Tuple : TypeScript_Type<"Tuple"> {
  let mnemonic = "tuple";

  let summary = "Fixed-sized collection of other types";
  let description = [{
    Syntax:

    ```
    tuple-type ::= `tuple` `<` (type ( `,` type)*)? `>`
    ```

    The value of `tuple` type represents a fixed-size collection of elements,
    where each element may be of a different type.

    Examples:

    ```mlir
    // Empty tuple.
    tuple<>

    // Single element
    tuple<f32>

    // Many elements.
    tuple<i32, f32, tensor<i1>, i5>
    ```
  }];
  let parameters = (ins "SmallVector<Type>":$types);

  let printer = [{
    $_printer << "tuple" << "<";
    size_t i = 0;
    for (auto typeItem : getImpl()->types) 
    {
      $_printer << typeItem;
      if (i++ < getImpl()->types.size() - 1)
          $_printer << ",";    
    }
    $_printer << ">";
  }];

  // Parses the above format
  let parser = [{
    Type elementType;
    SmallVector<Type> types;
    if ($_parser.parseLess()) return Type();
    while (!mlir::succeeded($_parser.parseOptionalGreater())) 
    {
        if ($_parser.parseType(elementType)) return Type();
        if ($_parser.parseOptionalComma()) break;
    }
    if ($_parser.parseGreater()) return Type();
    return get(context, types);
  }];  

  let extraClassDeclaration = [{
    /// Accumulate the types contained in this tuple and tuples nested within
    /// it. Note that this only flattens nested tuples, not any other container
    /// type, e.g. a tuple<i32, tensor<i32>, tuple<f32, tuple<i64>>> is
    /// flattened to (i32, tensor<i32>, f32, i64)
    void getFlattenedTypes(SmallVector<Type> &types);

    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<Type>::iterator;
    iterator begin() const { return getTypes().begin(); }
    iterator end() const { return getTypes().end(); }

    /// Return the element type at index 'index'.
    Type getType(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getTypes()[index];
    }
  }];
}

def TypeScript_ArrayLike : AnyTypeOf<[
    TypeScript_Array, TypeScript_String
  ]>;

def TypeScript_AnyStructLike : AnyTypeOf<[
    TypeScript_AnyTuple
  ]>;

#endif // TYPESCRIPT_TYPEDEFS
