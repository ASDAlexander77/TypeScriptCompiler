#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "TypeScriptTypes.td"
include "TypeScriptInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

//===----------------------------------------------------------------------===//
// TypeScript op definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : TypeScript_Op<"Constant",
    [ConstantLike, Pure]> {
  let summary = "constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$constValue);

  let builders = [
    OpBuilder<(ins "TypedAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];

  let hasFolder = 1;

  
}

def SymbolRefOp : TypeScript_Op<"SymbolRef", 
    [Pure]> {
  let summary = "symbol reference as constant";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType:$value);

  let hasCanonicalizer = 1;

  
}

def ThisSymbolRefOp : TypeScript_Op<"ThisSymbolRef", 
    [Pure]> {
  let summary = "symbol reference as to this member";

  let arguments = (ins TypeScript_AnyRefLike:$thisVal, FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_AnyRefOrCallableOrNone:$value);

  let hasCanonicalizer = 1;

  
}

def VTableOffsetRefOp : TypeScript_Op<"VTableOffsetRef", 
    [Pure]> {
  let summary = "gets reference in vtable";

  let arguments = (ins TypeScript_AnyRefLike:$vtable, I32Attr:$index);
  let results = (outs TypeScript_AnyRefLike:$value);

  
}

def VirtualSymbolRefOp : TypeScript_Op<"VirtualSymbolRef", 
    [Pure]> {
  let summary = "symbol reference as to virtual static member";

  let arguments = (ins TypeScript_AnyRefLike:$vtable, I32Attr:$index, FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_AnyRefOrCallable:$value);

  let hasCanonicalizer = 1;

  
}

def ThisVirtualSymbolRefOp : TypeScript_Op<"ThisVirtualSymbolRef", 
    [Pure]> {
  let summary = "symbol reference as to virtual this member";

  let arguments = (ins TypeScript_AnyRefLike:$thisVal, TypeScript_AnyRefLike:$vtable, I32Attr:$index, FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_AnyRefOrCallable:$value);

  let hasCanonicalizer = 1;

  
}

def AccessorOp : TypeScript_Op<"Accessor", 
    [Pure]> {
  let summary = "symbol reference as to static accessor";

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$getAccessor, OptionalAttr<FlatSymbolRefAttr>:$setAccessor);
  let results = (outs AnyType:$value);

  let hasCanonicalizer = 1;

  
}

def ThisAccessorOp : TypeScript_Op<"ThisAccessor", 
    [Pure]> {
  let summary = "accessor of class";

  let arguments = (ins AnyType:$thisVal, OptionalAttr<FlatSymbolRefAttr>:$getAccessor, OptionalAttr<FlatSymbolRefAttr>:$setAccessor);
  let results = (outs AnyType:$value);

  let hasCanonicalizer = 1;
}

def InterfaceSymbolRefOp : TypeScript_Op<"InterfaceSymbolRef", 
    [Pure]> {
  let summary = "symbol reference as to interface member";

  let arguments = (ins TypeScript_Interface:$interfaceVal, I32Attr:$index, StrAttr:$identifier, OptionalAttr<BoolAttr>:$optional);
  let results = (outs TypeScript_AnyRefOrCallable:$funcOrFieldReference);

  let hasCanonicalizer = 1;

  
}

def TypeRefOp : TypeScript_Op<"TypeRef", 
    [Pure]> {
  let summary = "symbol reference to type";

  let results = (outs AnyType:$value);

  let hasCanonicalizer = 1;
}

def NamespaceRefOp : TypeScript_Op<"NamespaceRef", 
    [Pure]> {
  let summary = "symbol reference to namespace";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType:$value);

  let hasCanonicalizer = 1;
}

def ClassRefOp : TypeScript_Op<"ClassRef", 
    [Pure]> {
  let summary = "symbol reference to class";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_Class:$value);

  let hasCanonicalizer = 1;
}

def InterfaceRefOp : TypeScript_Op<"InterfaceRef", 
    [Pure]> {
  let summary = "symbol reference to interface";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_Interface:$value);

  let hasCanonicalizer = 1;
}

def TypeScript_FuncOp : TypeScript_Op<"Func", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove, Symbol, RegionKindInterface
]> {
  let description = [{
    Example:

    ```mlir
    // External function definitions.
    ts.func @abort()
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<TypeScript_Function>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<BoolAttr>:$personality,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Block *addEntryBlock();

    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    bool isVarArg() {
        return getFunctionType().isVarArg();
    }

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncOp clone(IRMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ArrayAttr getCallableArgAttrs() {
        return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ArrayAttr getCallableResAttrs() {
        return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // RegionKindInterface Methods
    //===------------------------------------------------------------------===//
    static ::mlir::RegionKind getRegionKind(unsigned index) {
      return ::mlir::RegionKind::SSACFG;
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let assemblyFormat =
      [{ $sym_name $function_type $body attr-dict }];
}

def TypeScript_NullOp : TypeScript_Op<"Null", [Pure]> {
  let summary = "Make an intermediate object whose value is null.";

  let description = [{
    ```
    null-op ::= `ts.null` `:` any-type
    ```

    #### Example:

    ```mlir
    %0 = ts.null : ts.any
    ```
  }];

  let arguments = (ins);

  let results = (outs
    TypeScript_Null:$result
  );

  let hasCanonicalizer = 1;

  
}

def TypeScript_UndefOp : TypeScript_Op<"Undef", [Pure]> {
  let summary = "Make an intermediate object whose value is undefined.";

  let description = [{
    Result Type is the type of object to make.

    ```
    undef-op ::= `ts.undef` `:` any-type
    ```

    #### Example:

    ```mlir
    %0 = ts.undef : f32
    ```
  }];

  let arguments = (ins);

  let results = (outs
    AnyType:$result
  );

  let hasCanonicalizer = 1;

  
}

def TypeScript_PrintOp : TypeScript_Op<"Print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input string, and produces no results.
  }];

  let arguments = (ins Variadic<TypeScript_String>:$inputs);
}

def TypeScript_ConvertFOp : TypeScript_Op<"ConvertF"> {
  let summary = "format convert operation";
  let description = [{
    The "convertf" builtin operation to convert values into string with format.
  }];

  let arguments = (ins Index:$bufferSize, TypeScript_String:$format, Variadic<AnyType>:$inputs);
  let results = (outs TypeScript_String:$result);
}

def TypeScript_AssertOp : TypeScript_Op<"Assert"> {
  let summary = "Assert operation";
  let description = [{
    Assert operation

    Example:

    ```mlir
    assert %b, "Expected ... to be true"
    ```
  }];

  let arguments = (ins TypeScript_Boolean:$arg, StrAttr:$msg);

  

  let hasCanonicalizer = 1;
}

def TypeScript_ParseIntOp : TypeScript_Op<"parseInt"> {
  let summary = "Parse int operation";
  let description = [{
    Parse int operation

    Example:

    ```mlir
    %0 = parseInt %b, %base
    ```
  }];

  let arguments = (ins AnyType:$arg, Optional<I32>:$base);
  let results = (outs AnyInteger:$res);
}

def TypeScript_ParseFloatOp : TypeScript_Op<"parseFloat"> {
  let summary = "Parse float operation";
  let description = [{
    Parse float operation

    Example:

    ```mlir
    %0 = parseFloat %b
    ```
  }];

  let arguments = (ins AnyType:$arg);
  let results = (outs TypeScript_Number:$res);
}

def TypeScript_IsNaNOp : TypeScript_Op<"isNaN"> {
  let summary = "isNaN float operation";
  let description = [{
    isNaN float operation

    Example:

    ```mlir
    %0 = isNaN %b
    ```
  }];

  let arguments = (ins AnyType:$arg);
  let results = (outs TypeScript_Boolean:$res);
}

def TypeScript_TypeOfOp : TypeScript_Op<"TypeOf"> {
  let summary = "name of type";
  let description = [{
    name of type
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs TypeScript_String:$typeOf);
}

def TypeScript_TypeOfAnyOp : TypeScript_Op<"TypeOfAny"> {
  let summary = "name of type from any";
  let description = [{
    name of type from any
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs TypeScript_String:$typeOf);
}

def TypeScript_SizeOfOp : TypeScript_Op<"SizeOf"> {
  let summary = "size of type";
  let description = [{
    size of type
  }];

  let arguments = (ins TypeAttr:$type);
  let results = (outs Index:$size);
}

def TypeScript_NewOp : TypeScript_Op<"New"> {
  let summary = "allocate in memory";
  let description = [{
    allocate in memory
  }];

  let arguments = (ins OptionalAttr<BoolAttr>:$stackAlloc);
  let results = (outs Res<TypeScript_AnyClassOrValueRef, "", [MemAlloc]>:$instance);

  
}

def TypeScript_NewInterfaceOp : TypeScript_Op<"NewInterface"> {
  let summary = "create new interface";
  let description = [{
    create new interface
  }];

  let arguments = (ins TypeScript_AnyRefLike:$thisVal, TypeScript_AnyRefLike:$interfaceVTable);
  let results = (outs TypeScript_Interface:$instance);

  
}

def ExtractInterfaceThisOp : TypeScript_Op<"ExtractInterfaceThis", 
    [Pure]> {
  let summary = "extract interface this  ref.";

  let arguments = (ins TypeScript_Interface:$interfaceVal);
  let results = (outs TypeScript_Opaque:$thisVal);

  
}

def ExtractInterfaceVTableOp : TypeScript_Op<"ExtractInterfaceVTable", 
    [Pure]> {
  let summary = "extract interface vtable";

  let arguments = (ins TypeScript_Interface:$interfaceVal);
  let results = (outs TypeScript_Opaque:$vtableVal);

  
}

def TypeScript_CreateTupleOp : TypeScript_Op<"CreateTuple"> {
  let summary = "create tuple";
  let description = [{
    create tuple
  }];

  let arguments = (ins Variadic<AnyType>:$items);
  let results = (outs TypeScript_Tuple:$instance);

  
}

def TypeScript_DeconstructTupleOp : TypeScript_Op<"DeconstructTuple"> {
  let summary = "deconstruct tuple";
  let description = [{
    deconstruct tuple
  }];

  let arguments = (ins TypeScript_TupleLike:$instance);
  let results = (outs Variadic<AnyType>:$results);

  
}

def TypeScript_CreateArrayOp : TypeScript_Op<"CreateArray", [
    AllTypesMatch<["items"]>    
]> {
  let summary = "create array";
  let description = [{
    create array
  }];

  let arguments = (ins Variadic<AnyType>:$items);
  let results = (outs Res<TypeScript_Array, "", [MemAlloc]>:$instance);

  
}

def TypeScript_NewEmptyArrayOp : TypeScript_Op<"NewEmptyArray"> {
  let summary = "create empty array";
  let description = [{
    create empty array
  }];

  let arguments = (ins );
  let results = (outs Res<TypeScript_Array, "", [MemAlloc]>:$instance);

  
}

def TypeScript_NewArrayOp : TypeScript_Op<"NewArray"> {
  let summary = "allocate in memory few instances";
  let description = [{
    allocate in memory few instances
  }];

  let arguments = (ins I32:$count);
  let results = (outs Res<TypeScript_Array, "", [MemAlloc]>:$instance);

  
}

def TypeScript_DeleteOp : TypeScript_Op<"Delete"> {
  let summary = "delete allocated in memory";
  let description = [{
    delete allocated in memory
  }];

  let arguments = (ins Arg<TypeScript_AnyRefLike, "", [MemFree]>:$reference);
  let results = (outs );

  
}

def TypeScript_CallOp : TypeScript_Op<"Call", [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the call.

    Example:

    ```mlir
    %2 = call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$callOperands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$callOperands), [{
      $_state.addOperands(callOperands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$callOperands), [{
      $_state.addOperands(callOperands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$callOperands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, callOperands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$callOperands), [{
      build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), results,
            callOperands);
    }]>];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() 
    {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return (*this)->getAttrOfType<SymbolRefAttr>("callee"); }

    void setCalleeFromCallable(CallInterfaceCallable callee) { (*this)->setAttr("callee", callee.get<SymbolRefAttr>()); }
  }];
}

def TypeScript_CallIndirectOp : TypeScript_Op<"CallIndirect", [
      CallOpInterface,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>,
      TypesMatchWith<"callee result types match result types",
                     "callee", "results",
                     "($_self.isa<HybridFunctionType>() ? $_self.cast<HybridFunctionType>().getResults() : $_self.cast<FunctionType>().getResults())">
    ]> {
  let summary = "indirect call operation";
  let description = [{
    The `call_indirect` operation represents an indirect call to a value of
    function type. Functions are first class types in MLIR, and may be passed as
    arguments and merged together with block arguments. The operands and result
    types of the call must match the specified function type.
  }];

  let arguments = (ins TypeScript_Callable:$callee, Variadic<AnyType>:$callOperands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$callOperands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(callOperands);
      if (auto funcType = callee.getType().dyn_cast<FunctionType>())
      {
        $_state.addTypes(funcType.getResults());
      }
      else if (auto hybridFuncType = callee.getType().dyn_cast<mlir::typescript::HybridFunctionType>())
      {
        $_state.addTypes(hybridFuncType.getResults());
      }
      else
      {
          llvm_unreachable("not implemented");
      }
    }]>];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, callee.get<Value>());
    }
  }];

  let hasCanonicalizer = 1;

  
}

def TypeScript_InvokeOp : TypeScript_Op<"Invoke", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      Terminator
                      ]> {
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<AnyType>:$callOperands,
                   Variadic<AnyType>:$normalDestOperands,
                   Variadic<AnyType>:$unwindDestOperands);
  let results = (outs Variadic<AnyType>:$results);
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);

  let builders = [
    OpBuilder<(ins "TypeRange":$tys, "FlatSymbolRefAttr":$callee,
      "ValueRange":$ops, "Block*":$normal, "ValueRange":$normalOps,
      "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      $_state.addAttribute("callee", callee);
      build($_builder, $_state, tys, ops, normal, normalOps, unwind, unwindOps);
    }]>,
    OpBuilder<(ins "TypeRange":$tys, "ValueRange":$ops, "Block*":$normal,
      "ValueRange":$normalOps, "Block*":$unwind, "ValueRange":$unwindOps),
    [{
      build($_builder, $_state, tys, /*callee=*/FlatSymbolRefAttr(), ops, normalOps,
            unwindOps, normal, unwind);
    }]>];
  let hasVerifier = 1;
}

def TypeScript_InvokeHybridOp : TypeScript_Op<"InvokeHybrid", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      Terminator]> {
  let arguments = (ins TypeScript_HybridFunction:$callee,
                   Variadic<AnyType>:$callOperands,
                   Variadic<AnyType>:$normalDestOperands,
                   Variadic<AnyType>:$unwindDestOperands);
  let results = (outs Variadic<AnyType>:$results);
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);
}

def TypeScript_SymbolCallInternalOp : TypeScript_Op<"SymbolCallInternal",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "symbol call internal operation";
  let description = [{
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$callOperands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$callOperands), [{
      $_state.addOperands(callOperands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$callOperands), [{
      $_state.addOperands(callOperands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$callOperands), [{
      build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), results,
            callOperands);
    }]>];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() 
    {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return (*this)->getAttrOfType<SymbolRefAttr>("callee"); }

    void setCalleeFromCallable(CallInterfaceCallable callee) { (*this)->setAttr("callee", callee.get<SymbolRefAttr>()); }
  }];
}

def TypeScript_CallInternalOp : TypeScript_Op<"CallInternal",
    [CallOpInterface]> {
  let summary = "call internal operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$callOperands);
  let results = (outs Variadic<AnyType>:$results);
  
  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$callOperands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(callOperands);
      $_state.addTypes(callee.getType().cast<FunctionType>().getResults());
    }]>];

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() 
    {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getOperand(0); }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, callee.get<Value>());
    }    
  }];
}

def TypeScript_CallHybridInternalOp : TypeScript_Op<"CallHybridInternal",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>,
     TypesMatchWith<"callee input types match argument types",
                    "callee", "callOperands",
                    "$_self.cast<mlir::typescript::HybridFunctionType>().getInputs()">,
     TypesMatchWith<"callee result types match result types",
                    "callee", "results",
                    "$_self.cast<mlir::typescript::HybridFunctionType>().getResults()">
    ]> {
  let summary = "call (hybrid) internal operation";
  let description = [{
  }];

  let arguments = (ins TypeScript_HybridFunction:$callee, Variadic<AnyType>:$callOperands);
  let results = (outs Variadic<AnyType>:$results);

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() 
    {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    void setCalleeFromCallable(CallInterfaceCallable callee) {
      setOperand(0, callee.get<Value>());
    }    
  }];
}

def TypeScript_CaptureOp : TypeScript_Op<"Capture", []> {
  let summary = "capture variables";

  let arguments = (ins Arg<Variadic<AnyType>, "", [MemRead]>:$captured);
  let results = (outs Res<AnyType, "", [MemAlloc, MemWrite]>);
}

def TypeScript_ParamOp : TypeScript_Op<"Param", []> {
  let summary = [{
    Allocate an parameter object in memory.
  }];

  let description = [{

    ```mlir
    %0 = ts.param : f32
    ```
  }];

  let arguments = (ins
    AnyType:$argValue, OptionalAttr<BoolAttr>:$captured, OptionalAttr<IndexAttr>:$diArgNumber
  );

  let results = (outs
    Res<TypeScript_AnyRefLike, "", [MemAlloc]>:$reference
  );
}

def TypeScript_ParamOptionalOp : TypeScript_Op<"ParamOpt", [SingleBlockImplicitTerminator<"typescript::ParamDefaultValueOp">]> {
  let summary = [{
    Allocate an parameter object in memory.
  }];

  let description = [{

    ```mlir
    %0 = ts.param_opt : !ts.optional<f32>
    ```
  }];

  let arguments = (ins
    TypeScript_AnyOptional:$argValue, OptionalAttr<BoolAttr>:$captured, OptionalAttr<IndexAttr>:$diArgNumber
  );

  let results = (outs
    Res<TypeScript_AnyRefLike, "", [MemAlloc]>:$reference
  );

  let regions = (region AnyRegion:$defaultValueRegion);

  let extraClassDeclaration = [{
    OpBuilder getDefaultValueBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return OpBuilder::atBlockEnd(body, listener);
    }
  }];

}

def TypeScript_ParamDefaultValueOp : TypeScript_Op<"DefaultValue", [Pure, ReturnLike, Terminator,
                               ParentOneOf<["ParamOptionalOp"]>]> {
  let arguments = (ins Variadic<AnyType>:$results);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>]; 
}

def TypeScript_OptionalOp : TypeScript_Op<"Optional", 
    [TypesMatchWith<"type of 'value' matches element type of 'optional'",
                     "res", "in",
                     "$_self.cast<OptionalType>().getElementType()">]> {
  let description = [{
    Example:
      ts.optional %v, %f : i32, bool to ts.optional<i32>
  }];
  let arguments = (ins AnyType:$in, TypeScript_Boolean:$flag);
  let results = (outs TypeScript_AnyOptional:$res);
}

def TypeScript_VariableOp : TypeScript_Op<"Variable", []> {
  let summary = [{
    Allocate an object in stack or heap memory.
  }];

  let description = [{

    ```mlir
    %0 = ts.constant ...

    %1 = ts.variable : f32
    %2 = ts.variable init(%0): f32
    ```
  }];

  let arguments = (ins
    Optional<AnyType>:$initializer, OptionalAttr<BoolAttr>:$captured, OptionalAttr<IndexAttr>:$diArgNumber
  );

  let results = (outs
    Res<TypeScript_Ref, "", [MemAlloc]>:$reference
  );

  
}

def TypeScript_DebugVariableOp : TypeScript_Op<"DebugVariable", []> {
  let summary = [{
    Placeholder to Generate DebugInfo for constant values.
  }];

  let description = [{

    ```mlir
    %0 = ts.constant ...

    %1 = ts.debug_variable : f32
    ```
  }];

  let arguments = (ins
    AnyType:$initializer
  );

  let results = (outs
  ); 
}

def TypeScript_OptionalValueOp : TypeScript_Op<"OptionalValue", 
    [TypesMatchWith<"type of 'value' matches element type of 'optional'",
                     "res", "in",
                     "$_self.cast<OptionalType>().getElementType()">]> {
  let description = [{
    Example:
      ts.optional_value %v : i32 to ts.optional<i32>
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs TypeScript_AnyOptional:$res);
}

def TypeScript_OptionalUndefOp : TypeScript_Op<"OptionalUndef"> {
  let description = [{
    Example:
      ts.optional_undef : ts.optional<i32>
  }];
  let arguments = (ins );
  let results = (outs TypeScript_AnyOptional:$res);
}

def TypeScript_HasValueOp : TypeScript_Op<"HasValue"> {
  let description = [{
    Example:
      ts.has_value %v : ts.optional<i32> to bool
  }];
  let arguments = (ins TypeScript_AnyOptional:$in);
  let results = (outs TypeScript_Boolean:$res);
}

def TypeScript_ValueOp : TypeScript_Op<"Value", 
    [TypesMatchWith<"type of 'value' matches element type of 'optional'",
                     "in", "res",
                     "$_self.cast<OptionalType>().getElementType()">]> {
  let description = [{
    Example:
      ts.value %v : ts.optional<i32> to i32
  }];
  let arguments = (ins TypeScript_AnyOptional:$in);
  let results = (outs AnyType:$res);

  let hasCanonicalizer = 1;  
}

def TypeScript_ValueOrDefaultOp : TypeScript_Op<"ValueOrDefault", 
    [TypesMatchWith<"type of 'value' matches element type of 'optional'",
                     "in", "res",
                     "$_self.cast<OptionalType>().getElementType()">]> {
  let description = [{
    Example:
      ts.value_of_default %v : ts.optional<i32> to i32
  }];
  let arguments = (ins TypeScript_AnyOptional:$in);
  let results = (outs AnyType:$res);
}

def TypeScript_OptionalValueOrDefaultOp : TypeScript_Op<"OptValueOrDefault", [SingleBlockImplicitTerminator<"typescript::ResultOp">]> {
  let summary = [{
  }];

  let description = [{

    ```mlir
    %0 = ts.opt_value_or_default : !ts.optional<f32>
    ```
  }];

  let arguments = (ins
    TypeScript_AnyOptional:$argValue
  );

  let results = (outs
    AnyType:$value
  );

  let regions = (region AnyRegion:$defaultValueRegion);

  let extraClassDeclaration = [{
    OpBuilder getDefaultValueBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return OpBuilder::atBlockEnd(body, listener);
    }
  }];

}

def TypeScript_AllocaOp : TypeScript_Op<"Alloca", []> {
  let summary = [{
    Allocate an object in stack memory.
  }];

  let arguments = (ins
    Index:$count
  );

  let results = (outs
    Res<TypeScript_Ref, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$reference
  );

  
}

// TODO: add verify logic to match types for BoundRef and BoundFunction
def TypeScript_StoreOp : TypeScript_Op<"Store", []> {
  let summary = "stores the value into the reference";
  let description = [{
    The `store` operation stores the value into the reference storage.
  }];

  let arguments = (ins AnyType:$value, Arg<TypeScript_RefOrBoundRefOrValueRef, "the reference to store to", [MemWrite]>:$reference);
  
}

def TypeScript_LoadOp : TypeScript_Op<"Load", []> {
  let summary = "loads the value from the reference type";
  let description = [{
    The `load` operation loads the value from the reference value.
  }];

  let arguments = (ins Arg<TypeScript_RefOrBoundRefOrValueRef, "the reference to load from", [MemRead]>:$reference);
  let results = (outs AnyType:$result);
  

  let hasCanonicalizer = 1;
}

def TypeScript_ExtractPropertyOp : TypeScript_Op<"ExtractProperty", [Pure]> {
  let description = [{
    ```mlir
    %2 = ts.extract_property %1, %2 : f32
    ```
  }];

  let arguments = (ins
    TypeScript_AnyStructLike:$object,
    DenseI64ArrayAttr:$position
  );

  let results = (outs
    AnyType:$result
  );

  
}

def TypeScript_InsertPropertyOp : TypeScript_Op<"InsertProperty"> {
  let description = [{
    ```mlir
    ts.insert_property %1, %2, %3
    ```
  }];

  let arguments = (ins
    AnyType:$value,
    TypeScript_AnyStructLike:$object,
    DenseI64ArrayAttr:$position
  );

  let results = (outs 
    TypeScript_AnyStructLike:$res
  );
}

def TypeScript_ElementRefOp : TypeScript_Op<"ElementRef"> {
  let description = [{
    ```mlir
    %3 = ts.element_ref %1, %2 : !ts.ref<f32>
    ```
  }];

  let arguments = (ins
    TypeScript_ArrayLike:$array,
    I32:$index
  );

  let results = (outs
    TypeScript_Ref:$result
  );

  
}

def TypeScript_PropertyRefOp : TypeScript_Op<"PropertyRef"> {
  let description = [{
    ```mlir
    %2 = ts.property_ref %1, %2 : !ts.ref<f32>
    ```
  }];

  let arguments = (ins
    TypeScript_AnyStructRefLike:$objectRef,
    I32Attr:$position
  );

  let results = (outs
    TypeScript_RefOrBoundRef:$result
  );

  
}

def TypeScript_PointerOffsetRefOp : TypeScript_Op<"PointerOffsetRef"> {
  let description = [{
    ```mlir
    %3 = ts.pointer_offset_ref %1, %2 : !ts.ref<f32>
    ```
  }];

  let arguments = (ins
    TypeScript_Ref:$ref,
    I32:$index
  );

  let results = (outs
    TypeScript_Ref:$result
  );

  
}

def TypeScript_StoreBoundRefOp : TypeScript_Op<"StoreBoundRef", []> {
  let summary = "stores the value into the bound reference";
  let description = [{
    The `store` operation stores the value into the bound reference storage.
  }];

  let arguments = (ins AnyType:$value, Arg<TypeScript_BoundRef, "the bound reference to store to", [MemWrite]>:$reference);
  
}

def TypeScript_LoadBoundRefOp : TypeScript_Op<"LoadBoundRef", []> {
  let summary = "loads the value from the bound reference type";
  let description = [{
    The `load` operation loads the value from the bound reference value.
  }];

  let arguments = (ins Arg<TypeScript_BoundRef, "the bound reference to load from", [MemRead]>:$reference);
  let results = (outs AnyType:$result);
  
}

def TypeScript_CreateBoundRefOp : TypeScript_Op<"CreateBoundRef"> {
  let description = [{
    ```mlir
    %2 = ts.create_bound_ref %1, %2 : !ts.bound_ref<i32>
    ```
  }];

  let arguments = (ins
    TypeScript_AnyRefLike:$thisVal, TypeScript_Ref:$valueRef
  );

  let results = (outs
    TypeScript_BoundRef:$result
  );

  
}

def TypeScript_CreateExtensionFunctionOp : TypeScript_Op<"CreateExtensionFunction"> {
  let description = [{
    ```mlir
    %2 = ts.extention_function %1, %2 : !ts.this_func<()->void>
    ```
  }];

  let arguments = (ins
    AnyType:$thisVal, TypeScript_Function:$func
  );

  let results = (outs
    TypeScript_ExtensionFunction:$result
  );

  

  let hasCanonicalizer = 1;
}

def TypeScript_CreateBoundFunctionOp : TypeScript_Op<"CreateBoundFunction"> {
  let description = [{
    ```mlir
    %2 = ts.create_bound_function %1, %2 : !ts.this_func<()->void>
    ```
  }];

  let arguments = (ins
    TypeScript_AnyRefLike:$thisVal, TypeScript_Callable:$func
  );

  let results = (outs
    TypeScript_BoundFunctionLike:$result
  );

  
}

def TypeScript_GetThisOp : TypeScript_Op<"GetThis"> {
  let description = [{
    ```mlir
    %2 = ts.get_this %1 : !ts.ref<any>
    ```
  }];

  let arguments = (ins
    TypeScript_BoundFunctionLike:$boundFunc
  );

  let results = (outs
    TypeScript_AnyRefLike:$result
  );

  
}

def TypeScript_GetMethodOp : TypeScript_Op<"GetMethod"> {
  let description = [{
    ```mlir
    %2 = ts.get_method %1 : !ts.ref<()->void>
    ```
  }];

  let arguments = (ins
    TypeScript_BoundFunctionLike:$boundFunc
  );

  let results = (outs
    TypeScript_Function:$result
  );

  
}

def TypeScript_ArithmeticUnaryOp : TypeScript_Op<"ArithmeticUnary"> {
  let description = [{
    ```mlir
    %2 = ts.arithmetic_unary 1, %1 : f32
    ```
  }];

  let arguments = (ins
    I32Attr:$opCode,
    AnyType:$operand1
  );

  let results = (outs
    AnyType:$result
  );

  
}

def TypeScript_PrefixUnaryOp : TypeScript_Op<"PrefixUnary"> {
  let description = [{
    ```mlir
    %2 = ts.prefix_unary 1, %1 : f32
    ```
  }];

  let arguments = (ins
    I32Attr:$opCode,
    AnyType:$operand1
  );

  let results = (outs
    AnyType:$result
  );

  
}

def TypeScript_PostfixUnaryOp : TypeScript_Op<"PostfixUnary"> {
  let description = [{
    ```mlir
    %2 = ts.postfix_unary 1, %1 : f32
    ```
  }];

  let arguments = (ins
    I32Attr:$opCode,
    AnyType:$operand1
  );

  let results = (outs
    AnyType:$result
  );

  
}

def TypeScript_ArithmeticBinaryOp : TypeScript_Op<"ArithmeticBinary"> {
  let description = [{
    ```mlir
    %2 = ts.arithmetic_binary 1, %0, %1 : f32
    ```
  }];

  let arguments = (ins
    I32Attr:$opCode,
    AnyType:$operand1, 
    AnyType:$operand2
  );

  let results = (outs
    AnyType:$result
  );

  
}

def TypeScript_LogicalBinaryOp : TypeScript_Op<"LogicalBinary"> {
  let description = [{
    ```mlir
    %2 = ts.logical_binary 1, %0, %1 : ts.boolean
    ```
  }];

  let arguments = (ins
    I32Attr:$opCode,
    AnyType:$operand1, 
    AnyType:$operand2
  );

  let results = (outs
    TypeScript_Boolean:$result
  );

  
  let hasCanonicalizer = 1;
}

def TypeScript_CastOp : TypeScript_Op<"Cast", [DeclareOpInterfaceMethods<CastOpInterface>, Pure]> {
  let description = [{
    Example:
      ts.cast %v : i32 to f16
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$res);
  

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def TypeScript_BoxOp : TypeScript_Op<"Box", [Pure]> {
  let description = [{
    Example:
      %ti = ts.typeof %1
      ts.box %v, %ti : i32 to ts.any
  }];
  let arguments = (ins AnyType:$in, TypeScript_String:$typeInfo);
  let results = (outs TypeScript_Any:$res);
  
}

def TypeScript_UnboxOp : TypeScript_Op<"Unbox", [Pure]> {
  let description = [{
    Example:
      ts.unbox %v : ts.any to i32
  }];
  let arguments = (ins TypeScript_Any:$in);
  let results = (outs AnyType:$res);
  
}

def TypeScript_CreateUnionInstanceOp : TypeScript_Op<"CreateUnionInstance", [Pure]> {
  let description = [{
    Example:
      %ti = ts.typeof %1
      ts.create_union %v, %ti : i32 to ts.union
  }];
  let arguments = (ins AnyType:$in, TypeScript_String:$typeInfo);
  let results = (outs TypeScript_Union:$res);
  
}

def TypeScript_GetValueFromUnionOp : TypeScript_Op<"GetValueFromUnionOp", [Pure]> {
  let description = [{
    Example:
      %1 = ts.getvalue_from_union %v : ts.union to i32
  }];
  let arguments = (ins TypeScript_Union:$in);
  let results = (outs TypeScript_NonUnion:$res);
  
}

def TypeScript_GetTypeInfoFromUnionOp : TypeScript_Op<"GetTypeInfoFromUnionOp", [Pure]> {
  let description = [{
    Example:
      %1 = ts.gettype_from_union %v : ts.union to i32
  }];
  let arguments = (ins TypeScript_Union:$in);
  let results = (outs TypeScript_String:$res);
  
}

def TypeScript_DialectCastOp : TypeScript_Op<"DialectCast", [DeclareOpInterfaceMethods<CastOpInterface>, Pure]> {
  let description = [{
    Example:
      ts.dialect_cast %v : ts.number to f64
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$res);
  
}

def TypeScript_EntryOp : TypeScript_Op<"Entry"> {
  let description = [{
    To prepare data for return value

    #### Example:

    ```mlir
    ts.entry
    ```
  }];

  let arguments = (ins
  );  

  let results = (outs
    Optional<TypeScript_Ref>:$reference
  );  
}

def TypeScript_ReturnOp : TypeScript_Op<"Return"> {
  let description = [{
    ```
    return-value-op ::= `ts.return` ssa-use `:` typescript-type
    ```

    #### Example:

    ```mlir
    ts.return
    ```
  }];

  let arguments = (ins
  );
}

def TypeScript_ReturnValOp : TypeScript_Op<"ReturnVal"> {
  let description = [{
    ```
    return-value-op ::= `ts.returnVal` ssa-use `:` typescript-type
    ```

    #### Example:

    ```mlir
    ts.return %0 : f32
    ```
  }];

  let arguments = (ins
    AnyType:$operand,
    Res<TypeScript_Ref, "", [MemWrite]>:$reference
  );
}

def TypeScript_YieldReturnValOp : TypeScript_Op<"YieldReturnVal"> {
  let description = [{
    ```
    return-value-op ::= `ts.yieldReturnVal` ssa-use `:` typescript-type
    ```

    #### Example:

    ```mlir
    ts.yieldReturn %0 : f32
    ```
  }];

  let arguments = (ins
    AnyType:$operand,
    Res<TypeScript_Ref, "", [MemWrite]>:$reference
  );
}

def TypeScript_ExitOp : TypeScript_Op<"Exit", [Pure, ReturnLike, Terminator]> {
  let description = [{
    To prepare exit data for return

    #### Example:

    ```mlir
    ts.exit
    ```
  }];

  let arguments = (ins
    Optional<TypeScript_Ref>:$reference
  );
}

def TypeScript_ReturnInternalOp : TypeScript_Op<"ReturnInternal", [Pure, HasParent<"FuncOp">,
                                ReturnLike, Terminator]> {
  let summary = "return 'internal' operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$retOperands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, std::nullopt); }]>];

  
}

def TypeScript_IfOp : TypeScript_Op<"If",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>, SingleBlockImplicitTerminator<"typescript::ResultOp">, 
      NoRegionArguments]> {
  let summary = "if-then-else operation";

  let arguments = (ins TypeScript_Boolean:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$withElseRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
    OpBuilder<(ins "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
  }];

  let hasCanonicalizer = 1;
}

def TypeScript_ResultOp : TypeScript_Op<"Result", [Pure, ReturnLike, Terminator,
                               ParentOneOf<["IfOp",  "DoWhileOp", "WhileOp", "ForOp", "TryOp", "OptionalValueOrDefaultOp"]>]> {
  let summary = "loop yield and termination operation";

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def TypeScript_WhileOp : TypeScript_Op<"While",
    [
        LoopLikeOpInterface,
        ConditionallySpeculatable,
        DeclareOpInterfaceMethods<RegionBranchOpInterface>
    ]> {
  let summary = "a generic 'while' loop";
  let description = [{
    This operation represents a generic "while" loop that keeps
    iterating as long as a condition is satisfied. There is no restriction on
    the complexity of the condition. It consists of two regions (with single
    block each): "before" region and "body" region. The names of regions
    indicates whether they execute before or after the condition check.
    Therefore,

    The "cond" region terminates with a special operation, `ts.condition`,
    that accepts as its first operand an `i1` value indicating whether to
    proceed to the "body" region (value is `true`) or not. The two regions
    communicate by means of region arguments. Initially, the "cond" region
    accepts as arguments the operands of the `ts.while` operation and uses them
    to evaluate the condition. It forwards the trailing, non-condition operands
    of the `ts.condition` terminator either to the "body" region if the
    control flow is transferred there or to results of the `ts.while` operation
    otherwise. The "body" region takes as arguments the values produced by the
    "before" region and uses `ts.yield` to supply new arguments for the "body"
    region, into which it transfers the control flow unconditionally.
  }];

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$cond, AnyRegion:$body);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(std::optional<unsigned int> index);
    mlir::Region &getLoopBody() { return getBody(); }    
    Speculation::Speculatability getSpeculatability() {  
        return isTrue(getCond()) 
            ? Speculation::Speculatable
            : Speculation::NotSpeculatable;
    };
  }];

  let assemblyFormat =
      [{ ($inits^)? attr-dict `:` type($inits) `->` type($results) `(` $cond `)` $body }];
}

def TypeScript_DoWhileOp : TypeScript_Op<"DoWhile",
    [
        LoopLikeOpInterface,
        ConditionallySpeculatable,
        DeclareOpInterfaceMethods<RegionBranchOpInterface>
    ]> {
  let summary = "a generic 'do-while' loop";
  let description = [{
    This operation represents a generic "do-while" loop.
  }];

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body, SizedRegion<1>:$cond);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(std::optional<unsigned int> index);
    mlir::Region &getLoopBody() { return getBody(); }
    Speculation::Speculatability getSpeculatability() {  
        return isTrue(getCond()) 
            ? Speculation::Speculatable
            : Speculation::NotSpeculatable;
    };    
  }];

  let assemblyFormat =
      [{ ($inits^)? attr-dict `:` type($inits) `->` type($results) $body `(` $cond `)` }];  
}

def TypeScript_ForOp : TypeScript_Op<"For",
    [
        LoopLikeOpInterface,
        ConditionallySpeculatable,
        DeclareOpInterfaceMethods<RegionBranchOpInterface>
    ]> {
  let summary = "a generic 'for' loop";

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$cond, AnyRegion:$body, SizedRegion<1>:$incr);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(std::optional<unsigned int> index);
    mlir::Region &getLoopBody() { return getBody(); }
    Speculation::Speculatability getSpeculatability() {  
        return isTrue(getCond()) || isEmpty(getCond()) 
            ? Speculation::Speculatable
            : Speculation::NotSpeculatable;
    };    
  }];

  let assemblyFormat =
      [{ attr-dict ($inits^)? `:` type($inits) `->` type($results) `(` $cond `,` $incr `)` $body }];   
}

def TypeScript_ConditionOp : TypeScript_Op<"Condition",
                         [HasParent<"DoWhileOp, WhileOp, ForOp">, Pure, Terminator]> {
  let summary = "loop continuation condition";
  let description = [{
    This operation accepts the continuation (i.e., inverse of exit) condition
    of the `ts.while` construct. If its first argument is true, the "after"
    region of `ts.while` is executed, with the remaining arguments forwarded
    to the entry block of the region. Otherwise, the loop terminates.
  }];

  let arguments = (ins TypeScript_Boolean:$condition, Variadic<AnyType>:$args);

  let assemblyFormat =
      [{ `(` $condition `)` ($args^)? attr-dict `:` type($args) }];
}

def TypeScript_NoConditionOp : TypeScript_Op<"NoCondition",
                         [HasParent<"ForOp">, Terminator]> {
  let summary = "loop continuation condition";
  let description = [{
    This operation accepts the continuation (i.e., inverse of exit) condition
    of the `ts.while` construct. If its first argument is true, the "after"
    region of `ts.while` is executed, with the remaining arguments forwarded
    to the entry block of the region. Otherwise, the loop terminates.
  }];

  let arguments = (ins Variadic<AnyType>:$args);

  let assemblyFormat =
      [{ attr-dict ($args^ `:` type($args))? }];
}

def TypeScript_ContinueOp : TypeScript_Op<"Continue"> {
  let summary = "loop continuation";

  let arguments = (ins OptionalAttr<StrAttr>:$label);
}

def TypeScript_BreakOp : TypeScript_Op<"Break"> {
  let summary = "loop termination";

  let arguments = (ins OptionalAttr<StrAttr>:$label);
}

def TypeScript_LabelOp : TypeScript_Op<"Label",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       NoRegionArguments]> {
  let summary = "label operation";

  let arguments = (ins  StrAttr:$label);
  let results = (outs );
  let regions = (region AnyRegion:$labelRegion);

  let extraClassDeclaration = [{
    /// Returns the selection merge block.
    Block *getMergeBlock();

    /// Adds a selection merge block containing one ts.merge op.
    void addMergeBlock();
  }];  
}

def TypeScript_SwitchStateOp : TypeScript_Op<"SwitchState",
      [Terminator]> {
  let summary = "switch state operation";

  let arguments = (ins  I32:$state);
  let results = (outs );
  let successors = (successor AnySuccessor:$defaultDest,
                              VariadicSuccessor<AnySuccessor>:$cases);
}

def TypeScript_SwitchStateInternalOp : TypeScript_Op<"SwitchStateInternal",
      [Terminator]> {
  let summary = "switch state (internal) operation";

  let arguments = (ins  I32:$state);
  let results = (outs );
  let successors = (successor AnySuccessor:$defaultDest,
                              VariadicSuccessor<AnySuccessor>:$cases);
}

def TypeScript_StateLabelOp : TypeScript_Op<"StateLabel",
      []> {
  let summary = "state label operation";

  let arguments = (ins  StrAttr:$label);
  let results = (outs );
}

def TypeScript_SwitchOp : TypeScript_Op<"Switch",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>, NoRegionArguments]> {
  let summary = "switch operation";

  let arguments = (ins AnyType:$value);
  let results = (outs );
  let regions = (region AnyRegion:$casesRegion);
  
  let extraClassDeclaration = [{
    /// Returns the selection header block.
    Block *getHeaderBlock();

    /// Returns the selection merge block.
    Block *getMergeBlock();

    /// Adds a selection merge block containing one ts.merge op.
    void addMergeBlock();
  }];
}

def TypeScript_MergeOp : TypeScript_Op<"Merge", [Pure, HasParent<"SwitchOp, LabelOp">, Terminator]> {
  let summary = "A special terminator for merging a switch or label.";

  let arguments = (ins);

  let results = (outs);
}

def TypeScript_TryOp : TypeScript_Op<"Try",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "a generic 'try' clause";
  let description = [{
    This operation represents a generic "try" clause.
  }];

  let arguments = (ins );
  let results = (outs );
  let regions = (region AnyRegion:$body, AnyRegion:$cleanup, AnyRegion:$catches, AnyRegion:$finally);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(std::optional<unsigned int> index);
  }];

  
}

def TypeScript_CatchOp : TypeScript_Op<"Catch",
                         [HasParent<"TryOp">]> {
  let summary = "try catch";
  let description = [{
  }];

  let arguments = (ins TypeScript_Ref:$catchArg);
}

def TypeScript_ThrowOp : TypeScript_Op<"Throw"> {
  let arguments = (ins AnyType:$exception);
  let results = (outs );

}

def TypeScript_LandingPadOp : TypeScript_Op<"LandingPad"> {
  let summary = "Landing Pad - for affine layer";
  let description = [{
  }];

  let arguments = (ins BoolAttr:$cleanup, Variadic<AnyType>:$catches);
  let results = (outs TypeScript_Tuple:$result);
}

def TypeScript_CompareCatchTypeOp : TypeScript_Op<"CompareCatchType"> {
  let summary = "CompareCatchType - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_Tuple:$landingPad, TypeScript_Ref:$throwTypeInfo);
  let results = (outs TypeScript_Boolean:$result);
}

def TypeScript_BeginCatchOp : TypeScript_Op<"BeginCatch"> {
  let summary = "BeginCatch - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_Tuple:$landingPad);
  let results = (outs TypeScript_Opaque:$exceptionInfo);
}

def TypeScript_SaveCatchVarOp : TypeScript_Op<"SaveCatchVar"> {
  let summary = "SaveCatchVar - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_Opaque:$exceptionInfo, TypeScript_Ref:$varStore);
  let results = (outs );
}

def TypeScript_EndCatchOp : TypeScript_Op<"EndCatch"> {
  let summary = "EndCatch - for affine layer";
  let description = [{
  }];

  let arguments = (ins );
  let results = (outs );
}

def TypeScript_BeginCleanupOp : TypeScript_Op<"BeginCleanup"> {
  let summary = "BeginCleanup - for affine layer";
  let description = [{
  }];

  let arguments = (ins );
  let results = (outs );
}

def TypeScript_EndCleanupOp : TypeScript_Op<"EndCleanup", [Terminator]> {
  let summary = "EndCleanup - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_Tuple:$landingPad);
  let results = (outs );
  let successors = (successor VariadicSuccessor<AnySuccessor>:$unwindDest);
}

def TypeScript_ThrowUnwindOp : TypeScript_Op<"ThrowUnwind", [Terminator]> {
  let summary = "Throw (unwind) - for affine layer";
  let arguments = (ins AnyType:$exception);
  let results = (outs );
  let successors = (successor AnySuccessor:$unwindDest);
  
}

def TypeScript_ThrowCallOp : TypeScript_Op<"ThrowCall", [Terminator]> {
  let summary = "Throw (just call) - for affine layer";
  let arguments = (ins AnyType:$exception);
  let results = (outs );
  
}

def TypeScript_AddressOfOp : TypeScript_Op<"AddressOf"> {
  let arguments = (ins FlatSymbolRefAttr:$global_name, OptionalAttr<I32Attr>:$offset);
  let results = (outs TypeScript_RefOrOpaque:$reference);

  let builders = [
    OpBuilder<(ins "GlobalOp":$global,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, RefType::get(global.getType()), global.getSymName(), mlir::IntegerAttr());
      $_state.addAttributes(attrs);
    }]>,
    OpBuilder<(ins "FuncOp":$func, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, RefType::get(func.getFunctionType()), func.getName(), mlir::IntegerAttr());
      $_state.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the ts.global operation that defined the value referenced here.
    GlobalOp getGlobal();

    /// Return the ts.func operation that is referenced here.
    FuncOp getFunction();
  }];
}

def TypeScript_AddressOfConstStringOp : TypeScript_Op<"AddressOfConstString"> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs TypeScript_String:$instance);
}

def TypeScript_AddressOfElementOp : TypeScript_Op<"AddressOfElement"> {
  let arguments = (ins 
        TypeScript_Ref:$reference,
        Index:$elementIndex
  );
  let results = (outs TypeScript_Ref:$reference_of_element);
}

def TypeScript_GlobalOp : TypeScript_Op<"Global",
    [IsolatedFromAbove, SingleBlockImplicitTerminator<"GlobalResultOp">, Symbol]> {
  let arguments = (ins
    TypeAttr:$type,
    UnitAttr:$constant,
    StrAttr:$sym_name,
    OptionalAttr<AnyAttr>:$value
  );

  let regions = (region AnyRegion:$initializer);

  let builders = [
    OpBuilder<(ins "Type":$type, "bool":$isConstant, "StringRef":$name, 
        "Attribute":$value, "ArrayRef<NamedAttribute>":$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Return the initializer region. This may be empty, but if it is not it
    /// terminates in an `llvm.return` op with the initializer value.
    Region &getInitializerRegion() {
      return getOperation()->getRegion(0);
    }
    /// Return the initializer block. If the initializer region is empty this
    /// is nullptr. If it is not nullptr, it terminates with an `llvm.return`
    /// op with the initializer value.
    Block *getInitializerBlock() {
      return getInitializerRegion().empty() ?
        nullptr : &getInitializerRegion().front();
    }
  }];
}

def TypeScript_GlobalResultOp : TypeScript_Op<"GlobalResult", [Pure, ReturnLike, Terminator,
                               ParentOneOf<["GlobalOp"]>]> {
  let summary = "global init yield and termination operation";

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def TypeScript_GlobalConstructorOp : TypeScript_Op<"GlobalConstructor"> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs );

  let builders = [
    OpBuilder<(ins "FuncOp":$func, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, RefType::get(func.getFunctionType()), func.getName());
      $_state.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the ts.func operation that is referenced here.
    FuncOp getFunction();
  }];
}

def TypeScript_BodyInternalOp : TypeScript_Op<"BodyInternal",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>, SingleBlockImplicitTerminator<"typescript::BodyResultInternalOp">]> {
  let summary = "a generic 'body'";

  let arguments = (ins );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body);

  let assemblyFormat =
      [{ attr-dict `->` type($results) `:` $body }];

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(std::optional<unsigned int> index);

    Region &getBodyRegion() {
      return getOperation()->getRegion(0);
    }

    Block *getBodyBlock() {
      return getBodyRegion().empty() ?
        nullptr : &getBodyRegion().front();
    }
  }];      
}

def TypeScript_BodyResultInternalOp : TypeScript_Op<"BodyResultInternal", [Pure, ReturnLike, Terminator,
                               ParentOneOf<["typescript::BodyInternalOp"]>]> {
  let summary = "termination operation";

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def TypeScript_ArrayPushOp : TypeScript_Op<"ArrayPush"> {
  let arguments = (ins Arg<TypeScript_AnyArrayRef, "", [MemAlloc, MemWrite]>:$op, Variadic<AnyType>:$items);
  let results = (outs I32:$new_size);
}

def TypeScript_ArrayPopOp : TypeScript_Op<"ArrayPop"> {
  let arguments = (ins Arg<TypeScript_AnyArrayRef, "", [MemFree, MemRead]>:$op);
  let results = (outs AnyType:$item);
}

def TypeScript_ArrayUnshiftOp : TypeScript_Op<"ArrayUnshift"> {
  let arguments = (ins Arg<TypeScript_AnyArrayRef, "", [MemAlloc, MemWrite]>:$op, Variadic<AnyType>:$items);
  let results = (outs I32:$new_size);
}

def TypeScript_ArrayShiftOp : TypeScript_Op<"ArrayShift"> {
  let arguments = (ins Arg<TypeScript_AnyArrayRef, "", [MemFree, MemRead]>:$op);
  let results = (outs AnyType:$item);
}

def TypeScript_ArraySpliceOp : TypeScript_Op<"ArraySplice"> {
  let arguments = (ins Arg<TypeScript_AnyArrayRef, "", [MemAlloc, MemWrite]>:$op, Index:$start, Index:$deleteCount, Variadic<AnyType>:$items);
  let results = (outs I32:$new_size);
}

def TypeScript_ArrayViewOp : TypeScript_Op<"ArrayView"> {
  let arguments = (ins TypeScript_Array:$op, I32:$offset, I32:$count);
  let results = (outs TypeScript_Array:$result);
}

def TypeScript_LengthOfOp : TypeScript_Op<"LengthOf"> {
  let arguments = (ins TypeScript_ArrayLike:$op);
  let results = (outs I32:$result);
}

def TypeScript_SetLengthOfOp : TypeScript_Op<"SetLengthOf"> {
  let arguments = (ins Arg<TypeScript_AnyArrayRef, "", [MemAlloc, MemWrite]>:$op, I32:$newLength);
  let results = (outs );
}

def TypeScript_SetStringLengthOp : TypeScript_Op<"SetStringLength"> {
  let arguments = (ins Arg<TypeScript_StringRef, "", [MemAlloc, MemWrite]>:$op,  I32:$size);
  let results = (outs );
}

def TypeScript_StringLengthOp : TypeScript_Op<"StringLength"> {
  let arguments = (ins TypeScript_String:$op);
  let results = (outs I32:$result);
}

def TypeScript_StringConcatOp : TypeScript_Op<"StringConcat"> {
  let arguments = (ins Variadic<TypeScript_String>:$ops, OptionalAttr<BoolAttr>:$allocInStack);
  let results = (outs Res<TypeScript_String, "", [MemAlloc]>:$result);
}

def TypeScript_StringCompareOp : TypeScript_Op<"StringCompare"> {
  let arguments = (ins TypeScript_String:$op1, TypeScript_String:$op2, I32Attr:$code);
  let results = (outs TypeScript_Boolean:$result);
}

def TypeScript_CharToStringOp : TypeScript_Op<"CharToString"> {
  let arguments = (ins TypeScript_Char:$op);
  let results = (outs TypeScript_String:$result);
}

def TypeScript_LoadSaveOp : TypeScript_Op<"LoadSave", [CopyOpInterface, AllTypesMatch<["dst", "src"]>]> {
  let arguments = (ins Arg<AnyType, "copy to", [MemWrite]>:$dst, Arg<AnyType, "copy from", [MemRead]>:$src);
  let results = (outs );

  let extraClassDeclaration = [{
    Value getSource() { return getSrc(); }
    Value getTarget() { return getDst(); }
  }];
}

def TypeScript_CopyStructOp : TypeScript_Op<"CopyStruct", [CopyOpInterface, AllTypesMatch<["dst", "src"]>]> {
  let arguments = (ins Arg<AnyType, "copy to", [MemWrite]>:$dst, Arg<AnyType, "copy from", [MemRead]>:$src);
  let results = (outs );

  let extraClassDeclaration = [{
    Value getSource() { return getSrc(); }
    Value getTarget() { return getDst(); }
  }];
}

def TypeScript_MemoryCopyOp : TypeScript_Op<"MemoryCopy", [CopyOpInterface, AllTypesMatch<["dst", "src"]>]> {
  let arguments = (ins Arg<AnyType, "copy to", [MemWrite]>:$dst, Arg<AnyType, "copy from", [MemRead]>:$src, I32:$count);
  let results = (outs );

  let extraClassDeclaration = [{
    Value getSource() { return getSrc(); }
    Value getTarget() { return getDst(); }
  }];
}

def TypeScript_MemoryMoveOp : TypeScript_Op<"MemoryMove", [CopyOpInterface, AllTypesMatch<["dst", "src"]>]> {
  let arguments = (ins Arg<AnyType, "move to", [MemWrite]>:$dst, Arg<AnyType, "copy from", [MemRead]>:$src, I32:$count);
  let results = (outs );

  let extraClassDeclaration = [{
    Value getSource() { return getSrc(); }
    Value getTarget() { return getDst(); }
  }];
}

def TypeScript_DebuggerOp : TypeScript_Op<"Debugger"> {
  let summary = "debugger";
}

def TypeScript_UnreachableOp : TypeScript_Op<"unreachable", [Terminator]> {
  let summary = "unreachable";
}

def TypeScript_NoOp : TypeScript_Op<"noop", [Pure]> {
  let summary = "no operation";
}

// GC support
def TypeScript_GCMakeDescriptorOp : TypeScript_Op<"GCMakeDescriptor"> {
  let summary = "GC make descriptor operation";

  let arguments = (ins TypeScript_Ref:$typeBitmap, Index:$sizeOfBitmapInElements);
  let results = (outs I64:$descr);
}

def TypeScript_GCNewExplicitlyTypedOp : TypeScript_Op<"GCNewExplicitlyTyped"> {
  let summary = "GC new exmplicitly typed operation";

  let arguments = (ins I64:$typeDescr);
  let results = (outs Res<TypeScript_AnyClassOrValueRef, "", [MemAlloc]>:$instance);
}

// Shared libraries support
def TypeScript_LoadLibraryPermanentlyOp : TypeScript_Op<"LoadLibraryPermanentlyOp"> {
  let description = [{
    Example:
      %1 = ts.load_library_permanently %file_name : i32
  }];
  let arguments = (ins TypeScript_String:$filename);
  let results = (outs I32:$result);
}

def TypeScript_SearchForAddressOfSymbolOp : TypeScript_Op<"SearchForAddressOfSymbolOp"> {
  let description = [{
    Example:
      %1 = ts.search_for_address_of_symbol %symbol_name : ptr
  }];
  let arguments = (ins TypeScript_String:$symbolName);
  let results = (outs TypeScript_Opaque:$addr);
}

#endif // TYPESCRIPT_OPS
