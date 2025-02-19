#ifndef TYPESCRIPT_TYPEDEFS
#define TYPESCRIPT_TYPEDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TypeScript types definition.
//===----------------------------------------------------------------------===//

// To get the typescript dialect def.
include "TypeScriptDialect.td"

// All of the types will extend this class.
class TypeScript_Type<string name, list<Trait> traits = []> : TypeDef<TypeScript_Dialect, name, traits> { }

def IsMutable : NativeTypeTrait<"IsMutable">;

def TypeScript_Void : TypeScript_Type<"Void"> {
    let mnemonic = "void";
}

def TypeScript_Any : TypeScript_Type<"Any"> {
    let mnemonic = "any";
}

def TypeScript_Unknown : TypeScript_Type<"Unknown"> {
    let mnemonic = "unknown";
}

def TypeScript_Never : TypeScript_Type<"Never"> {
    let mnemonic = "never";
}

def TypeScript_Const : TypeScript_Type<"Const"> {
    let mnemonic = "const";
}

def TypeScript_Symbol : TypeScript_Type<"Symbol"> {
    let mnemonic = "symbol";
}

def TypeScript_Undefined : TypeScript_Type<"Undefined"> {
    let mnemonic = "undefined";
}

def TypeScript_Null : TypeScript_Type<"Null"> {
    let mnemonic = "null";
}

def TypeScript_Opaque : TypeScript_Type<"Opaque"> {
    let mnemonic = "opaque";
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

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
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

  let assemblyFormat = "`<` params `>`";
  
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
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

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
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
  let parameters = (ins "Type":$elementType, "DictionaryAttr":$values);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "DictionaryAttr":$values), [{
      return Base::get(elementType.getContext(), elementType, values);
    }]>
  ];
}

def TypeScript_LiteralType : TypeScript_Type<"Literal"> {
  let mnemonic = "literal";

  let description = [{
    Syntax:

    ```
    enum-type ::= `literal<` value `,` type `>`
    ```

    Examples:

    ```mlir
    literal<1.0, f32>
    literal<1.0, i32>
    ```
  }];
  let parameters = (ins "Attribute":$value, "Type":$elementType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Attribute":$value, "Type":$elementType), [{
      return Base::get(elementType.getContext(), value, elementType);
    }]>
  ];
}

def TypeScript_ConstArrayValue : TypeScript_Type<"ConstArrayValue"> {
  let mnemonic = "const_array_value";

  let description = [{
    Syntax:

    ```
    array-type ::= `const_array_value<` type `,` size `>`
    ```

    Examples:

    ```mlir
    const_array_value<f32, 1>
    const_array_value<i32, 10>
    ```
  }];
  let parameters = (ins "Type":$elementType, "unsigned":$size);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "unsigned":$size), [{
      return Base::get(elementType.getContext(), elementType, size);
    }]>
  ];
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

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "unsigned":$size), [{
      return Base::get(elementType.getContext(), elementType, size);
    }]>
  ];
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

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
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

  let hasCustomAssemblyFormat = 1;  

  let extraClassDeclaration = [{
    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const { return getFields().begin(); }
    iterator end() const { return getFields().end(); }

    int getIndex(Attribute id)
    {
        auto dist = std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf){ return id == fldInf.id; }));
        return (signed)dist >= (signed)size() ? -1 : dist;
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index];
    }

    Attribute getId(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index].id;
    }

    Type getType(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
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

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const { return getFields().begin(); }
    iterator end() const { return getFields().end(); }

    int getIndex(Attribute id)
    {
        auto dist = std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf){ return id == fldInf.id; }));
        return (signed)dist >= (signed)size() ? -1 : dist;
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index];
    }

    Attribute getId(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index].id;
    }

    Type getType(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index].type;
    }
  }];
}

def TypeScript_Object : TypeScript_Type<"Object"> {
  let mnemonic = "object";

  let description = [{
    Syntax:

    ```
    object-type ::= `object<` type `>`
    ```

    Examples:

    ```mlir
    object<tuple<i32>>
    object<any>
    ```
  }];
  let parameters = (ins "Type":$storageType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$storageType), [{
      return Base::get(storageType.getContext(), storageType);
    }]>
  ];
}

def TypeScript_ObjectStorage : TypeScript_Type<"ObjectStorage", [IsMutable]> {
  let mnemonic = "object_storage";

  let summary = "Fixed-sized collection of other types for class";
  let description = [{
    Syntax:

    ```
    object-storage-type ::= `object_storage` `<` (type ( `,` type)*)? `>`
    ```

    The value of `object_storage` type represents a fixed-size collection of elements,
    where each element may be of a different type.

    Examples:

    ```mlir
    // Empty tuple.
    object_storage<>

    // Single element
    object_storage<{f32}>

    // Many elements.
    object_storage<{i32}, {f32}, {tensor<i1>}, {i5}>
    ```
  }];
  let parameters = (ins "FlatSymbolRefAttr":$name);

  let assemblyFormat = "`<` params `>`";

  let extraClassDeclaration = [{
    /// Return the number of held types.
    size_t size() const {
        return std::distance(begin(), end());
    }

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const { return getFields().begin(); }
    iterator end() const { return getFields().end(); }

    int getIndex(Attribute id)
    {
        auto dist = std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf){ return id == fldInf.id; }));
        return (signed)dist >= (signed)size() ? -1 : dist;
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index];
    }

    Attribute getId(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index].id;
    }

    Type getType(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index].type;
    }

    ::llvm::ArrayRef<::mlir::typescript::FieldInfo> getFields() const;
    // Mutator
    LogicalResult setFields(::llvm::ArrayRef<::mlir::typescript::FieldInfo> newFields);
  }];

  let genStorageClass = 0;
}

def TypeScript_ClassStorage : TypeScript_Type<"ClassStorage", [IsMutable]> {
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
  let parameters = (ins "FlatSymbolRefAttr":$name);

  let assemblyFormat = "`<` params `>`";

  let extraClassDeclaration = [{
    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const { return getFields().begin(); }
    iterator end() const { return getFields().end(); }

    int getIndex(Attribute id)
    {
        auto dist = std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf){ return id == fldInf.id; }));
        return (signed)dist >= (signed)size() ? -1 : dist;
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index];
    }

    Attribute getId(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index].id;
    }

    Type getType(size_t index) const {
      assert(index >= 0 && index < size() && "invalid index for tuple type");
      return getFields()[index].type;
    }

    ::llvm::ArrayRef<::mlir::typescript::FieldInfo> getFields() const;
    // Mutator
    LogicalResult setFields(::llvm::ArrayRef<::mlir::typescript::FieldInfo> newFields);
  }];

  let storageClass = "ClassStorageTypeStorage";
  let storageNamespace = "mlir::typescript::detail";
  let genStorageClass = 0;
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

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name, "Type":$storageType), [{
      return Base::get(storageType.getContext(), name, storageType);
    }]>
  ];
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

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name), [{
      return Base::get(name.getContext(), name);
    }]>
  ];
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

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$name), [{
      return Base::get(name.getContext(), name);
    }]>
  ];
}

def TypeScript_BoundRef : TypeScript_Type<"BoundRef"> {
  let mnemonic = "bound_ref";

  let description = [{
    Syntax:

    ```
    bound-ref-type ::= `bound_ref` `<` type `>`
    ```

    Examples:

    ```mlir
    bound-ref<f32>
    bound-ref<i32>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
}

def TypeScript_Function : TypeScript_Type<"Function"> {
  let mnemonic = "func";

  let summary = "Map from a list of inputs to a list of results";
  let description = [{
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$inputs, ArrayRefParameter<"Type", "type">:$results, "bool":$isVarArg);

  let assemblyFormat = "`<` params `>`";

  let builders = [
     TypeBuilder<(ins CArg<"ArrayRef<Type>", "{}">:$inputs, CArg<"ArrayRef<Type>", "{}">:$results, CArg<"bool", "false">:$isVarArg), [{
       return Base::get($_ctxt, inputs, results, isVarArg);
     }]>,
     TypeBuilder<(ins "mlir::FunctionType":$funcType, CArg<"bool", "false">:$isVarArg), [{
       return Base::get(funcType.getContext(), funcType.getInputs(), funcType.getResults(), isVarArg);
     }]>
  ];
  
  let skipDefaultBuilders = 1;  
  //let genStorageClass = 0;
  let extraClassDeclaration = [{
    bool isVarArg() const;

    /// Returns the result type of the function.
    Type getReturnType();

    /// Returns the number of arguments to the function.
    unsigned getNumParams();

    /// Returns `i`-th argument of the function. Asserts on out-of-bounds.
    Type getParamType(unsigned i);

    /// Returns a list of argument types of the function.
    ArrayRef<Type> getParams();
    ArrayRef<Type> params() { return getParams(); }

    /// Input types.
    unsigned getNumInputs() const;
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Result types.
    unsigned getNumResults() const;
    Type getResult(unsigned i) const { return getResults()[i]; }

    /// Returns a clone of this function type with the given argument
    /// and result types.
    FunctionType clone(TypeRange inputs, TypeRange results) const;

    /// Returns a new function type with the specified arguments and results
    /// inserted.
    FunctionType getWithArgsAndResults(ArrayRef<unsigned> argIndices,
                                       TypeRange argTypes,
                                       ArrayRef<unsigned> resultIndices,
                                       TypeRange resultTypes);

    /// Returns a new function type without the specified arguments and results.
    FunctionType getWithoutArgsAndResults(const BitVector &argIndices,
                                          const BitVector &resultIndices);
  }];
}

def TypeScript_HybridFunction : TypeScript_Type<"HybridFunction"> {
  let mnemonic = "hybrid_func";

  let summary = "Map from a list of inputs to a list of results with/without 'this' reference";
  let description = [{
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$inputs, ArrayRefParameter<"Type", "type">:$results, "bool":$isVarArg);

  let assemblyFormat = "`<` params `>`";

  let builders = [
     TypeBuilder<(ins "ArrayRef<Type>":$inputs, "ArrayRef<Type>":$results, "bool":$isVarArg), [{
       return Base::get($_ctxt, inputs, results, isVarArg);
     }]>,
     TypeBuilder<(ins "FunctionType":$funcType), [{
       return Base::get(funcType.getContext(), funcType.getInputs(), funcType.getResults(), funcType.isVarArg());
     }]>
  ]; 

  let skipDefaultBuilders = 1;  
  //let genStorageClass = 0;
  let extraClassDeclaration = [{
    bool isVarArg() const;

    /// Input types.
    unsigned getNumInputs() const;
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Result types.
    unsigned getNumResults() const;
    Type getResult(unsigned i) const { return getResults()[i]; }
  }];
}

def TypeScript_BoundFunction : TypeScript_Type<"BoundFunction"> {
  let mnemonic = "bound_func";

  let summary = "Map from a list of inputs to a list of results with 'this' reference";
  let description = [{
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$inputs, ArrayRefParameter<"Type", "type">:$results, "bool":$isVarArg);

  let assemblyFormat = "`<` params `>`";

  let builders = [
     TypeBuilder<(ins "ArrayRef<Type>":$inputs, "ArrayRef<Type>":$results, "bool":$isVarArg), [{
       return Base::get($_ctxt, inputs, results, isVarArg);
     }]>,
     TypeBuilder<(ins "FunctionType":$funcType), [{
       return Base::get(funcType.getContext(), funcType.getInputs(), funcType.getResults(), funcType.isVarArg());
     }]>
  ]; 

  let skipDefaultBuilders = 1;  
  //let genStorageClass = 0;
  let extraClassDeclaration = [{
    bool isVarArg() const;

    /// Input types.
    unsigned getNumInputs() const;
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Result types.
    unsigned getNumResults() const;
    Type getResult(unsigned i) const { return getResults()[i]; }
  }];
}

def TypeScript_ExtensionFunction : TypeScript_Type<"ExtensionFunction"> {
  let mnemonic = "extent_func";

  let summary = "Map from a list of inputs to a list of results with 'this' reference";
  let description = [{
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$inputs, ArrayRefParameter<"Type", "type">:$results, "bool":$isVarArg);

  let assemblyFormat = "`<` params `>`";

  let builders = [
     TypeBuilder<(ins "ArrayRef<Type>":$inputs, "ArrayRef<Type>":$results, "bool":$isVarArg), [{
       return Base::get($_ctxt, inputs, results, isVarArg);
     }]>,
     TypeBuilder<(ins "FunctionType":$funcType), [{
       return Base::get(funcType.getContext(), funcType.getInputs(), funcType.getResults(), funcType.isVarArg());
     }]>
  ]; 

  let skipDefaultBuilders = 1;  
  //let genStorageClass = 0;
  let extraClassDeclaration = [{
    bool isVarArg() const;

    /// Input types.
    unsigned getNumInputs() const;
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Result types.
    unsigned getNumResults() const;
    Type getResult(unsigned i) const { return getResults()[i]; }
  }];
}

def TypeScript_ConstructFunction : TypeScript_Type<"ConstructFunction"> {
  let mnemonic = "construct_func";

  let summary = "Map from a list of inputs to a list of results with 'this' reference";
  let description = [{
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$inputs, ArrayRefParameter<"Type", "type">:$results, "bool":$isVarArg, "bool":$isAbstract);

  let assemblyFormat = "`<` params `>`";

  let builders = [
     TypeBuilder<(ins CArg<"ArrayRef<Type>", "{}">:$inputs, CArg<"ArrayRef<Type>", "{}">:$results, CArg<"bool", "false">:$isVarArg, CArg<"bool", "false">:$isAbstract), [{
       return Base::get($_ctxt, inputs, results, isVarArg, isAbstract);
     }]>,
     TypeBuilder<(ins "FunctionType":$funcType, CArg<"bool", "false">:$isAbstract), [{
       return Base::get(funcType.getContext(), funcType.getInputs(), funcType.getResults(), funcType.isVarArg(), isAbstract);
     }]>
  ]; 

  let skipDefaultBuilders = 1;  
  //let genStorageClass = 0;
  let extraClassDeclaration = [{
    bool isVarArg() const;

    bool isAbstract() const;

    /// Input types.
    unsigned getNumInputs() const;
    Type getInput(unsigned i) const { return getInputs()[i]; }

    /// Result types.
    unsigned getNumResults() const;
    Type getResult(unsigned i) const { return getResults()[i]; }
  }];
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

  let assemblyFormat = "`<` params `>`";

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

  let summary = "intersection of other types";
  let description = [{
    Syntax:

    ```
    union-type ::= `intersection` `<` (type ( `,` type)*)? `>`
    ```

    The value of `intersection` type represents an intersection of elements,
    where each element may be of a different type.

    Examples:

    ```mlir
    // Empty intersection.
    intersection<>

    // Single intersection
    intersection<f32>

    // Many elements.
    intersection<i32, f32, tensor<i1>, i5>
    ```
  }];
  let parameters = (ins ArrayRefParameter<"Type", "type">:$types);

  let assemblyFormat = "`<` params `>`";

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

def TypeScript_NamedGeneric : TypeScript_Type<"NamedGeneric"> {
    let mnemonic = "named_generic";

    let parameters = (ins "FlatSymbolRefAttr":$name);

    let assemblyFormat = "`<` params `>`";
}

def TypeScript_Infer : TypeScript_Type<"Infer"> {
  let mnemonic = "infer";

  let description = [{
    Syntax:

    ```
    infer-type ::= `Infer<` type `>`
    ```

    Examples:

    ```mlir
    infer<!Name>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
}

def TypeScript_Conditional : TypeScript_Type<"Conditional"> {
  let mnemonic = "conditional";

  let description = [{
    Syntax:

    ```
    conditional-type ::= `Conditional<` checkType `,` extendsType `,` trueType `,` falseType  `>`
    ```

    Examples:

    ```mlir
    conditional<!Type, !Type, !Type, !Type>
    ```
  }];
  let parameters = (ins "Type":$checkType, "Type":$extendsType, "Type":$trueType, "Type":$falseType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$checkType, "Type":$extendsType, "Type":$trueType, "Type":$falseType), [{
      return Base::get(checkType.getContext(), checkType, extendsType, trueType, falseType);
    }]>
  ];
}

def TypeScript_IndexAccess : TypeScript_Type<"IndexAccess"> {
  let mnemonic = "indexAccess";

  let description = [{
    Syntax:

    ```
    index-access-type ::= `IndexAccess<` type `,` indexType `>`
    ```

    Examples:

    ```mlir
    index-access<!Name, !Name>
    ```
  }];
  let parameters = (ins "Type":$type, "Type":$indexType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$type, "Type":$indexType), [{
      return Base::get(indexType.getContext(), type, indexType);
    }]>
  ];
}

def TypeScript_KeyOf : TypeScript_Type<"KeyOf"> {
  let mnemonic = "keyOf";

  let description = [{
    Syntax:

    ```
    key-of-type ::= `KeyOf<` type `>`
    ```

    Examples:

    ```mlir
    key-of-type<!Name>
    ```
  }];
  let parameters = (ins "Type":$elementType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType), [{
      return Base::get(elementType.getContext(), elementType);
    }]>
  ];
}

def TypeScript_Mapped : TypeScript_Type<"Mapped"> {
  let mnemonic = "mapped";

  let description = [{
    Syntax:

    ```
    mapped-type ::= `Mapped<` type `,` nameType `,` constrainType  `>`
    ```

    Examples:

    ```mlir
    mapped<!Name, !Name, !Name>
    ```
  }];
  let parameters = (ins "Type":$elementType, "Type":$nameType, "Type":$constrainType);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType, "Type":$nameType, "Type":$constrainType), [{
      return Base::get(elementType.getContext(), elementType, nameType, constrainType);
    }]>
  ];
}

def TypeScript_TypeReference : TypeScript_Type<"TypeReference"> {
  let mnemonic = "typeref";

  let description = [{
    Syntax:

    ```
    typeref-type ::= `typeref` `<` name `,` (type ( `,` type)*)? `>`
    ```

    Examples:

    ```mlir
    typeref<'name', @f1>
    ```
  }];
  let parameters = (ins "FlatSymbolRefAttr":$name, ArrayRefParameter<"Type", "type">:$types);

  let assemblyFormat = "`<` params `>`";
}

def TypeScript_TypePredicate : TypeScript_Type<"TypePredicate"> {
  let mnemonic = "Is";

  let description = [{
    Syntax:

    ```
    type-predicate ::= `Is<` name `,` type `>`
    ```

    Examples:

    ```mlir
    is<!Name, !Type>
    ```
  }];
  let parameters = (ins "FlatSymbolRefAttr":$parameterName, "Type":$elementType, "bool":$asserts, "int":$parameterIndex);

  let assemblyFormat = "`<` params `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "FlatSymbolRefAttr":$parameterName, "Type":$elementType, CArg<"bool", "false">:$asserts, CArg<"int", "-1">:$parameterIndex), [{
      return Base::get(elementType.getContext(), parameterName, elementType, asserts, parameterIndex);
    }]>
  ];
}

def TypeScript_IsConstTupleRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::ConstTupleType>()">;

def TypeScript_AnyConstTupleRef : DialectType<TypeScript_Dialect, TypeScript_IsConstTupleRefType, "any const tuple reference type">;

def TypeScript_IsTupleRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::TupleType>()">;

def TypeScript_AnyTupleRef : DialectType<TypeScript_Dialect, TypeScript_IsTupleRefType, "any tuple reference type">;

def TypeScript_IsOptionalRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::OptionalType>()">;

def TypeScript_AnyOptionalRef : DialectType<TypeScript_Dialect, TypeScript_IsOptionalRefType, "any optional reference type">;

def TypeScript_IsStringRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::StringType>()">;

def TypeScript_StringRef : DialectType<TypeScript_Dialect, TypeScript_IsStringRefType, "string reference type">;

def TypeScript_IsClassStorageType : CPred<"$_self.isa<::mlir::typescript::ClassStorageType>()">;

def TypeScript_AnyClassStorageType : DialectType<TypeScript_Dialect, TypeScript_IsClassStorageType, "any class storage type">;

def TypeScript_IsClassStorageRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::ClassStorageType>()">;

def TypeScript_AnyClassStorageRef : DialectType<TypeScript_Dialect, TypeScript_IsClassStorageRefType, "any tuple reference type">;

def TypeScript_IsOptionalType : CPred<"$_self.isa<::mlir::typescript::OptionalType>()">;

def TypeScript_AnyOptional : DialectType<TypeScript_Dialect, TypeScript_IsOptionalType, "any optional type">;

def TypeScript_IsArrayRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::ArrayType>()">;

def TypeScript_AnyArrayRef : DialectType<TypeScript_Dialect, TypeScript_IsArrayRefType, "any array reference type">;

def TypeScript_IsConstArrayValueRefType : CPred<"$_self.isa<::mlir::typescript::RefType>() && $_self.cast<::mlir::typescript::RefType>().getElementType().isa<::mlir::typescript::ConstArrayValueType>()">;

def TypeScript_AnyConstArrayValueRef : DialectType<TypeScript_Dialect, TypeScript_IsConstArrayValueRefType, "any const array value reference type">;

def TypeScript_IsNotUnionType : CPred<"!$_self.isa<::mlir::typescript::UnionType>()">;

def TypeScript_NonUnion : DialectType<TypeScript_Dialect, TypeScript_IsNotUnionType, "non-union type">;

def TypeScript_ArrayLike : AnyTypeOf<[
    TypeScript_ConstArray, TypeScript_Array, TypeScript_String, TypeScript_AnyConstArrayValueRef, TypeScript_Ref
  ]>;

def TypeScript_TupleLike : AnyTypeOf<[
    TypeScript_ConstTuple, TypeScript_Tuple
  ]>;

def TypeScript_AnyStructRefLike : AnyTypeOf<[
    TypeScript_AnyConstTupleRef, TypeScript_AnyTupleRef, TypeScript_Class, TypeScript_Object, TypeScript_AnyClassStorageRef, TypeScript_AnyOptionalRef
  ]>;

def TypeScript_AnyRefLike : AnyTypeOf<[
    TypeScript_Ref, TypeScript_ValueRef, TypeScript_AnyStructRefLike, TypeScript_Array, TypeScript_String, TypeScript_Any, TypeScript_Null, TypeScript_Opaque
  ]>;

def TypeScript_AnyRefLikeOrInterface : AnyTypeOf<[
    TypeScript_AnyRefLike, TypeScript_Interface
  ]>;

def TypeScript_AnyStructLike : AnyTypeOf<[
    TypeScript_ConstTuple, TypeScript_Tuple, TypeScript_ConstArrayValue, TypeScript_Class, TypeScript_AnyClassStorageType
  ]>;

def TypeScript_RefOrBoundRef : AnyTypeOf<[
    TypeScript_Ref, TypeScript_BoundRef
  ]>;

def TypeScript_RefOrOpaque : AnyTypeOf<[
    TypeScript_Ref, TypeScript_Opaque
  ]>;

def TypeScript_RefOrValueRef : AnyTypeOf<[
    TypeScript_Ref, TypeScript_ValueRef
  ]>;

def TypeScript_RefOrBoundRefOrValueRef : AnyTypeOf<[
    TypeScript_Ref, TypeScript_BoundRef, TypeScript_ValueRef
  ]>;

def TypeScript_Callable : AnyTypeOf<[
    TypeScript_Function, TypeScript_BoundFunction, TypeScript_HybridFunction
  ]>;

def TypeScript_AnyRefOrCallable : AnyTypeOf<[
    TypeScript_RefOrValueRef, TypeScript_Callable
  ]>;

def TypeScript_AnyRefOrCallableOrNone : AnyTypeOf<[
    TypeScript_RefOrValueRef, TypeScript_Callable, NoneType
  ]>;  

def TypeScript_BoundFunctionLike : AnyTypeOf<[
    TypeScript_BoundFunction, TypeScript_HybridFunction
  ]>;

def TypeScript_AnyClassOrValueRef : AnyTypeOf<[
    TypeScript_Class, TypeScript_ValueRef
  ]>;


#endif // TYPESCRIPT_TYPEDEFS
