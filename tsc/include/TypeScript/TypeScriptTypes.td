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

def TypeScript_UndefPlaceHolder : TypeScript_Type<"UndefPlaceHolder"> {
    let mnemonic = "undef_ph";
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

def TypeScript_ValueRef : TypeScript_Type<"ValueRef"> {
  let mnemonic = "value_ref";

  let description = [{
    Syntax:

    ```
    value-ref-type ::= `value_ref` `<` type `>`
    ```

    Examples:

    ```mlir
    value_ref<f32>
    value_ref<i32>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  // We define the printer inline.
  let printer = [{ $_printer << "value_ref<" << getImpl()->elementType << ">"; }];

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
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

def TypeScript_Enum : TypeScript_Type<"Enum"> {
  let mnemonic = "enum";

  let description = [{
    Syntax:

    ```
    enum-type ::= `enum<` type `>`
    ```

    Examples:

    ```mlir
    enum<f32>
    enum<i32>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  // We define the printer inline.
  let printer = [{ $_printer << "enum<" << getImpl()->elementType << ">"; }];  

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
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

def TypeScript_ConstArray : TypeScript_Type<"ConstArray"> {
  let mnemonic = "const_array";

  let description = [{
    Syntax:

    ```
    array-type ::= `const_array<` type `,` size `>`
    ```

    Examples:

    ```mlir
    const_array<f32, 1>
    const_array<i32, 10>
    ```
  }];
  let parameters = (ins "Type":$elementType, "unsigned":$size);

  // We define the printer inline.
  let printer = [{ $_printer << "const_array<" << getImpl()->elementType << "," << getImpl()->size << ">"; }];  

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "unsigned":$size), [{
      return Base::get(elementType.getContext(), elementType, size);
    }]>
  ];

  // Parses the above format
  let parser = [{
    Type elementType;
    if ($_parser.parseLess()) return Type();
    if ($_parser.parseType(elementType)) return Type();
    if ($_parser.parseComma()) return Type();
    int size;
    if ($_parser.parseInteger(size)) return Type();
    if ($_parser.parseGreater()) return Type();
    return get(elementType, size);
  }];  

  let skipDefaultBuilders = 1;  
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

def TypeScript_ConstTuple : TypeScript_Type<"ConstTuple"> {
  let mnemonic = "const_tuple";

  let summary = "Fixed-sized collection of other types";
  let description = [{
    Syntax:

    ```
    const-tuple-type ::= `const_tuple` `<` (type ( `,` type)*)? `>`
    ```

    The value of `tuple` type represents a fixed-size collection of elements,
    where each element may be of a different type.

    Examples:

    ```mlir
    // Empty tuple.
    const_tuple<>

    // Single element
    const_tuple<f32>

    // Many elements.
    const_tuple<i32, f32, tensor<i1>, i5>
    ```
  }];
  let parameters = (ins ArrayRefOfSelfAllocationParameter<"::mlir::typescript::FieldInfo", "fields">:$fields);

  let printer = [{
    $_printer << "const_tuple" << "<";
    for (size_t i=0, e = getImpl()->fields.size(); i < e; i++) {
      const auto& field = getImpl()->fields[i];
      if (field.id) 
      {
          $_printer << "{" << field.id << "," << field.type << "}";
      }
      else
      {
          $_printer << field.type;
      }
      
      if (i < getImpl()->fields.size() - 1)
          $_printer << ",";
    }
    $_printer << ">";
  }];

  // Parses the above format
  let parser = [{
    SmallVector<FieldInfo, 4> parameters;
    if ($_parser.parseLess()) return Type();
    while (mlir::succeeded($_parser.parseOptionalLBrace())) {
      Attribute id;
      if ($_parser.parseAttribute(id)) return Type();
      if ($_parser.parseComma()) return Type();
      Type type;
      if ($_parser.parseType(type)) return Type();
      if ($_parser.parseRBrace()) return Type();
      parameters.push_back(FieldInfo {id, type});
      if ($_parser.parseOptionalComma()) break;
    }
    if ($_parser.parseGreater()) return Type();
    return get($_ctxt, parameters);
  }];  

  let extraClassDeclaration = [{
    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const { return getFields().begin(); }
    iterator end() const { return getFields().end(); }

    int getIndex(Attribute id)
    {
        return std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf){ return id == fldInf.id; }));
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index];
    }

    Attribute getId(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index].id;
    }

    Type getType(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index].type;
    }
  }];
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
    tuple<{f32}>

    // Many elements.
    tuple<{i32}, {f32}, {tensor<i1>}, {i5}>
    ```
  }];
  let parameters = (ins ArrayRefOfSelfAllocationParameter<"::mlir::typescript::FieldInfo", "fields">:$fields);

  let printer = [{
    $_printer << "tuple" << "<";
    for (size_t i=0, e = getImpl()->fields.size(); i < e; i++) {
      const auto& field = getImpl()->fields[i];
      if (field.id) 
      {
          $_printer << "{" << field.id << "," << field.type << "}";
      }
      else
      {
          $_printer << field.type;
      }

      if (i < getImpl()->fields.size() - 1)
          $_printer << ",";
    }
    $_printer << ">";
  }];

  // Parses the above format
  let parser = [{
    SmallVector<FieldInfo, 4> parameters;
    if ($_parser.parseLess()) return Type();
    while (mlir::succeeded($_parser.parseOptionalLBrace())) {
      Attribute id;
      if ($_parser.parseAttribute(id)) return Type();
      if ($_parser.parseComma()) return Type();
      Type type;
      if ($_parser.parseType(type)) return Type();
      if ($_parser.parseRBrace()) return Type();
      parameters.push_back(FieldInfo {id, type});
      if ($_parser.parseOptionalComma()) break;
    }
    if ($_parser.parseGreater()) return Type();
    return get($_ctxt, parameters);
  }];  

  let extraClassDeclaration = [{
    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const { return getFields().begin(); }
    iterator end() const { return getFields().end(); }

    int getIndex(Attribute id)
    {
        return std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf){ return id == fldInf.id; }));
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index];
    }

    Attribute getId(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index].id;
    }

    Type getType(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index].type;
    }
  }];
}

def TypeScript_ClassStorage : TypeScript_Type<"ClassStorage"> {
  let mnemonic = "class_storage";

  let summary = "Fixed-sized collection of other types for class";
  let description = [{
    Syntax:

    ```
    class-storage-type ::= `class_storage` `<` (type ( `,` type)*)? `>`
    ```

    The value of `class_storage` type represents a fixed-size collection of elements,
    where each element may be of a different type.

    Examples:

    ```mlir
    // Empty tuple.
    class_storage<>

    // Single element
    class_storage<{f32}>

    // Many elements.
    class_storage<{i32}, {f32}, {tensor<i1>}, {i5}>
    ```
  }];
  let parameters = (ins "FlatSymbolRefAttr":$name, ArrayRefOfSelfAllocationParameter<"::mlir::typescript::FieldInfo", "fields">:$fields);

  let printer = [{
    $_printer << "class_storage<" <<  getImpl()->name << ",";
    for (size_t i=0, e = getImpl()->fields.size(); i < e; i++) {
      const auto& field = getImpl()->fields[i];
      if (field.id) 
      {
          $_printer << "{" << field.id << "," << field.type << "}";
      }
      else
      {
          $_printer << field.type;
      }

      if (i < getImpl()->fields.size() - 1)
          $_printer << ",";
    }
    $_printer << ">";
  }];

  // Parses the above format
  let parser = [{
    Attribute name;
    SmallVector<FieldInfo, 4> parameters;
    if ($_parser.parseLess()) return Type();
    if ($_parser.parseAttribute(name)) return Type();    
    if ($_parser.parseComma()) return Type();
    while (mlir::succeeded($_parser.parseOptionalLBrace())) {
      Attribute id;
      if ($_parser.parseAttribute(id)) return Type();
      if ($_parser.parseComma()) return Type();
      Type type;
      if ($_parser.parseType(type)) return Type();
      if ($_parser.parseRBrace()) return Type();
      parameters.push_back(FieldInfo {id, type});
      if ($_parser.parseOptionalComma()) break;
    }
    if ($_parser.parseGreater()) return Type();
    return get($_ctxt, name.dyn_cast_or_null<FlatSymbolRefAttr>(), parameters);
  }];  

  let extraClassDeclaration = [{
    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const { return getFields().begin(); }
    iterator end() const { return getFields().end(); }

    int getIndex(Attribute id)
    {
        return std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf){ return id == fldInf.id; }));
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index];
    }

    Attribute getId(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index].id;
    }

    Type getType(size_t index) const {
      assert(index < size() && "invalid index for tuple type");
      return getFields()[index].type;
    }
  }];
}

def TypeScript_Class : TypeScript_Type<"Class"> {
  let mnemonic = "class";

  let description = [{
    Syntax:

    ```
    class-type ::= `class<` type `>`
    ```

    Examples:

    ```mlir
    class<f32>
    class<i32>
    ```
  }];
  let parameters = (ins "FlatSymbolRefAttr":$name, "Type":$storageType);

  // We define the printer inline.
  let printer = [{ $_printer << "class<" << getImpl()->name << "," << getImpl()->storageType << ">"; }];  

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name, "Type":$storageType), [{
      return Base::get(storageType.getContext(), name, storageType);
    }]>
  ];

  // Parses the above format
  let parser = [{
    Attribute name;
    Type storageType;
    if ($_parser.parseLess()) return Type();
    if ($_parser.parseAttribute(name)) return Type();
    if ($_parser.parseComma()) return Type();
    if ($_parser.parseType(storageType)) return Type();
    if ($_parser.parseGreater()) return Type();
    return get(name.dyn_cast_or_null<FlatSymbolRefAttr>(), storageType);
  }];  

  let skipDefaultBuilders = 1;  
}

def TypeScript_Namespace : TypeScript_Type<"Namespace"> {
  let mnemonic = "namespace";

  let description = [{
    Syntax:

    ```
    namespace-type ::= `namespace`
    ```

    Examples:

    ```mlir
    namespace<@n1>
    ```
  }];
  let parameters = (ins "FlatSymbolRefAttr":$name);

  // We define the printer inline.
  let printer = [{ $_printer << "namespace<" << getImpl()->name << ">"; }];  

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name), [{
      return Base::get(name.getContext(), name);
    }]>
  ];

  // Parses the above format
  let parser = [{
    Attribute name;
    if ($_parser.parseLess()) return Type();
    if ($_parser.parseAttribute(name)) return Type();
    return get(name.dyn_cast_or_null<FlatSymbolRefAttr>());
  }];  

  let skipDefaultBuilders = 1;  
}

def TypeScript_Interface : TypeScript_Type<"Interface"> {
  let mnemonic = "iface";

  let description = [{
    Syntax:

    ```
    iface-type ::= `iface`
    ```

    Examples:

    ```mlir
    iface<@f1>
    ```
  }];
  let parameters = (ins "FlatSymbolRefAttr":$name);

  // We define the printer inline.
  let printer = [{ $_printer << "iface<" << getImpl()->name << ">"; }];  

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name), [{
      return Base::get(name.getContext(), name);
    }]>
  ];

  // Parses the above format
  let parser = [{
    Attribute name;
    if ($_parser.parseLess()) return Type();
    if ($_parser.parseAttribute(name)) return Type();
    return get(name.dyn_cast_or_null<FlatSymbolRefAttr>());
  }];  

  let skipDefaultBuilders = 1;  
}

def TypeScript_Union : TypeScript_Type<"Union"> {
  let mnemonic = "union";

  let summary = "Fixed-sized union of other types";
  let description = [{
    Syntax:

    ```
    union-type ::= `union` `<` (type ( `,` type)*)? `>`
    ```

    The value of `union` type represents a fixed-size union of elements,
    where each element may be of a different type.

    Examples:

    ```mlir
    // Empty union.
    union<>

    // Single union
    union<f32>

    // Many elements.
    union<i32, f32, tensor<i1>, i5>
    ```
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$types);

  let printer = [{
    $_printer << "union" << "<";
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

def TypeScript_Intersection : TypeScript_Type<"Intersection"> {
  let mnemonic = "intersection";

  let summary = "Fixed-sized intersection of other types";
  let description = [{
    Syntax:

    ```
    union-type ::= `intersection` `<` (type ( `,` type)*)? `>`
    ```

    The value of `intersection` type represents a fixed-size intersection of elements,
    where each element may be of a different type.

    Examples:

    ```mlir
    // Empty union.
    intersection<>

    // Single union
    intersection<f32>

    // Many elements.
    intersection<i32, f32, tensor<i1>, i5>
    ```
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$types);

  let printer = [{
    $_printer << "union" << "<";
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

def TypeScript_IsRefType : CPred<"$_self.isa<::mlir::typescript::RefType>()">;

def TypeScript_AnyRef : DialectType<TypeScript_Dialect, TypeScript_IsRefType, "any reference type">;

def TypeScript_IsTupleType : CPred<"$_self.isa<::mlir::typescript::TupleType>()">;

def TypeScript_AnyTuple : DialectType<TypeScript_Dialect, TypeScript_IsTupleType, "any tuple type">;

def TypeScript_IsConstTupleRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::ConstTupleType>()">;

def TypeScript_AnyConstTupleRef : DialectType<TypeScript_Dialect, TypeScript_IsConstTupleRefType, "any const tuple reference type">;

def TypeScript_IsTupleRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::TupleType>()">;

def TypeScript_AnyTupleRef : DialectType<TypeScript_Dialect, TypeScript_IsTupleRefType, "any tuple reference type">;

def TypeScript_IsClassStorageRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::ClassStorageType>()">;

def TypeScript_AnyClassStorageRef : DialectType<TypeScript_Dialect, TypeScript_IsClassStorageRefType, "any tuple reference type">;

def TypeScript_IsOptionalType : CPred<"$_self.isa<::mlir::typescript::OptionalType>()">;

def TypeScript_AnyOptional : DialectType<TypeScript_Dialect, TypeScript_IsOptionalType, "any optional type">;

def TypeScript_IsArrayRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::ArrayType>()">;

def TypeScript_AnyArrayRef : DialectType<TypeScript_Dialect, TypeScript_IsArrayRefType, "any array reference type">;

def TypeScript_ArrayLike : AnyTypeOf<[
    TypeScript_ConstArray, TypeScript_Array, TypeScript_String
  ]>;

def TypeScript_AnyStructRefLike : AnyTypeOf<[
    TypeScript_AnyConstTupleRef, TypeScript_AnyTupleRef, TypeScript_Class, TypeScript_Object, TypeScript_AnyClassStorageRef
  ]>;

def TypeScript_AnyRefLike : AnyTypeOf<[
    TypeScript_Ref, TypeScript_ValueRef, TypeScript_AnyStructRefLike, TypeScript_Array, TypeScript_String, TypeScript_Any
  ]>;

def TypeScript_AnyStructLike : AnyTypeOf<[
    TypeScript_ConstTuple, TypeScript_Tuple, TypeScript_Class, TypeScript_ClassStorage
  ]>;

#endif // TYPESCRIPT_TYPEDEFS
