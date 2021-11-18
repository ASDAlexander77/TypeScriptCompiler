#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "TypeScriptTypes.td"
include "TypeScriptInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"


//===----------------------------------------------------------------------===//
// TypeScript op definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : TypeScript_Op<"Constant",
    [ConstantLike, NoSideEffect]> {
  let summary = "constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$constValue);

  let builders = [
    OpBuilder<(ins "Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];

  let hasFolder = 1;

  let assemblyFormat = "attr-dict `:` type($constValue)";
}

def UnresolvedSymbolRefOp : TypeScript_Op<"UnresolvedSymbolRef", 
    [NoSideEffect]> {
  let summary = "unresolved symbol reference as constant";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType:$value);

  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$identifier),
    [{ build($_builder, $_state, identifier.getType(), identifier); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict `:` type($value)";
}

def SymbolRefOp : TypeScript_Op<"SymbolRef", 
    [NoSideEffect]> {
  let summary = "symbol reference as constant";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType:$value);

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict `:` type($value)";
}

def ThisSymbolRefOp : TypeScript_Op<"ThisSymbolRef", 
    [NoSideEffect]> {
  let summary = "symbol reference as to this member";

  let arguments = (ins TypeScript_AnyRefLike:$thisVal, FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType:$value);

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "`(` $thisVal `)` attr-dict `:` type($thisVal) `->` type($value)";
}

def VTableOffsetRefOp : TypeScript_Op<"VTableOffsetRef", 
    [NoSideEffect]> {
  let summary = "gets reference in vtable";

  let arguments = (ins TypeScript_AnyRefLike:$vtable, I32Attr:$index);
  let results = (outs TypeScript_AnyRefLike:$value);

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let assemblyFormat = "`(` $vtable `[` $index `]` `)` attr-dict `:` type($vtable) `->` type($value)";
}


def ThisVirtualSymbolRefOp : TypeScript_Op<"ThisVirtualSymbolRef", 
    [NoSideEffect]> {
  let summary = "symbol reference as to virtual this member";

  let arguments = (ins TypeScript_AnyRefLike:$thisVal, TypeScript_AnyRefLike:$vtable, I32Attr:$index, FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_AnyRefOrCallable:$value);

  let builders = [
    OpBuilder<(ins "Value":$thisVal, "Value":$vtable, "IntegerAttr":$index, "FlatSymbolRefAttr":$identifier),
    [{ build($_builder, $_state, identifier.getType(), thisVal, vtable, index, identifier); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "`(` $thisVal `,` $vtable `[` $index `]` `)` attr-dict `:`  `:` type($thisVal) `,` type($vtable) `->` type($value)";
}

def AccessorRefOp : TypeScript_Op<"AccessorRef", 
    [NoSideEffect]> {
  let summary = "symbol reference as to static accessor";

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$getAccessor, OptionalAttr<FlatSymbolRefAttr>:$setAccessor);
  let results = (outs AnyType:$value);

  let builders = [
    OpBuilder<(ins "FlatSymbolRefAttr":$getAccessor, "FlatSymbolRefAttr":$setAccessor),
    [{ build($_builder, $_state, getAccessor.getType(), getAccessor, setAccessor); }]>];

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict `:` type($value)";
}

def ThisAccessorRefOp : TypeScript_Op<"ThisAccessorRef", 
    [NoSideEffect]> {
  let summary = "symbol reference as to accessor";

  let arguments = (ins AnyType:$thisVal, OptionalAttr<FlatSymbolRefAttr>:$getAccessor, OptionalAttr<FlatSymbolRefAttr>:$setAccessor);
  let results = (outs AnyType:$value);

  let builders = [
    OpBuilder<(ins "Value":$thisVal, "FlatSymbolRefAttr":$getAccessor, "FlatSymbolRefAttr":$setAccessor),
    [{ build($_builder, $_state, getAccessor.getType(), thisVal, getAccessor, setAccessor); }]>];

  let hasCanonicalizer = 1;

  let assemblyFormat = "`(` $thisVal `)` $getAccessor $setAccessor attr-dict `:` type($thisVal) `->` type($value)";  
}

def InterfaceSymbolRefOp : TypeScript_Op<"InterfaceSymbolRef", 
    [NoSideEffect,DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
  let summary = "symbol reference as to interface member";

  let arguments = (ins TypeScript_Interface:$interfaceVal, I32Attr:$index, StrAttr:$identifier);
  let results = (outs TypeScript_AnyRefOrCallable:$funcOrFieldReference, TypeScript_AnyRefLike:$thisRef);

  let builders = [
    OpBuilder<(ins "Value":$interfaceVal, "IntegerAttr":$index, "StringAttr":$identifier),
    [{ build($_builder, $_state, identifier.getType(), interfaceVal, index, identifier); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "`(` $interfaceVal `)` `[` $index `]` attr-dict `:` type($interfaceVal) `->` type($funcOrFieldReference) `,` type($thisRef)";
}

def TypeRefOp : TypeScript_Op<"TypeRef", 
    [NoSideEffect]> {
  let summary = "symbol reference to type";

  let results = (outs AnyType:$value);

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict `:` type($value)";   
}

def NamespaceRefOp : TypeScript_Op<"NamespaceRef", 
    [NoSideEffect]> {
  let summary = "symbol reference to namespace";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType:$value);

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict `:` type($value)";     
}

def ClassRefOp : TypeScript_Op<"ClassRef", 
    [NoSideEffect]> {
  let summary = "symbol reference to class";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_Class:$value);

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict `:` type($value)";    
}

def InterfaceRefOp : TypeScript_Op<"InterfaceRef", 
    [NoSideEffect]> {
  let summary = "symbol reference to interface";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs TypeScript_Interface:$value);

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];

  let hasCanonicalizer = 1;

  let assemblyFormat = "attr-dict `:` type($value)";    
}

def TypeScript_FuncOp : TypeScript_Op<"Func", [
  AffineScope, AutomaticAllocationScope, CallableOpInterface, FunctionLike,
  IsolatedFromAbove, Symbol
]> {
  let description = [{
    Example:

    ```mlir
    // External function definitions.
    ts.func @abort()
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttr:$type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<BoolAttr>:$personality);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncOp clone(BlockAndValueMapping &mapper);
    FuncOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncOp dest, BlockAndValueMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

  private:
    // This trait needs access to the hooks defined below.
    friend class OpTrait::FunctionLike<FuncOp>;

    /// Returns the number of arguments. This is a hook for
    /// OpTrait::FunctionLike.
    unsigned getNumFuncArguments() { return getType().getInputs().size(); }

    /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
    unsigned getNumFuncResults() { return getType().getResults().size(); }

    /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
    /// attribute is present and checks if it holds a function type. Ensures
    /// getType, getNumFuncArguments, and getNumFuncResults can be called
    /// safely.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }
  }];

  let verifier = [{ return ::verify(*this); }];

  let assemblyFormat =
      [{ $sym_name $type attr-dict $body }];
}

def TypeScript_NullOp : TypeScript_Op<"Null", []> {
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

  let verifier = [{ return success(); }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TypeScript_UndefOp : TypeScript_Op<"Undef", []> {
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
  let verifier = [{ return success(); }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TypeScript_PrintOp : TypeScript_Op<"Print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input string, and produces no results.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);

  let assemblyFormat = "`(` $inputs `)` attr-dict `:` type($inputs) ";  
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

  let assemblyFormat = "$arg `,` $msg attr-dict";

  // AssertOp is fully verified by its traits.
  let verifier = ?;

  let hasCanonicalizer = 1;
}

def TypeScript_ParseIntOp : TypeScript_Op<"parseInt"> {
  let summary = "Parse int operation";
  let description = [{
    Parse int operation

    Example:

    ```mlir
    %0 = parseInt %b
    ```
  }];

  let arguments = (ins AnyType:$arg);
  let results = (outs I32:$res);
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
  let results = (outs I64:$size);
}

def TypeScript_NewOp : TypeScript_Op<"New"> {
  let summary = "allocate in memory";
  let description = [{
    allocate in memory
  }];

  let arguments = (ins OptionalAttr<BoolAttr>:$stackAlloc);
  let results = (outs AnyType:$instance);

  let assemblyFormat = "`{` $stackAlloc `}` attr-dict `:` type($instance)";
}

def TypeScript_NewInterfaceOp : TypeScript_Op<"NewInterface"> {
  let summary = "create new interface";
  let description = [{
    create new interface
  }];

  let arguments = (ins TypeScript_AnyRefLike:$thisVal, TypeScript_AnyRefLike:$interfaceVTable);
  let results = (outs TypeScript_Interface:$instance);

  let assemblyFormat = "`(` $thisVal `,` $interfaceVTable `)` attr-dict `:` type($thisVal) `,` type($interfaceVTable) `->` type($instance)";
}

def ExtractInterfaceVTableOp : TypeScript_Op<"ExtractInterfaceVTable", 
    [NoSideEffect]> {
  let summary = "extract interface vtable";

  let arguments = (ins TypeScript_Interface:$interfaceVal);
  let results = (outs TypeScript_Opaque:$vtableVal);

  let assemblyFormat = "`(` $interfaceVal `)` attr-dict `:` type($interfaceVal) `->` type($vtableVal)";
}

def TypeScript_CreateTupleOp : TypeScript_Op<"CreateTuple"> {
  let summary = "create tuple";
  let description = [{
    create tuple
  }];

  let arguments = (ins Variadic<AnyType>:$items);
  let results = (outs TypeScript_Tuple:$instance);

  let assemblyFormat = "`[` $items `]` attr-dict `:` type($items) `->` type($instance)";
}

def TypeScript_DeconstructTupleOp : TypeScript_Op<"DeconstructTuple"> {
  let summary = "deconstruct tuple";
  let description = [{
    deconstruct tuple
  }];

  let arguments = (ins TypeScript_Tuple:$instance);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = "$instance attr-dict `:` type($instance) `->` `{` type($results) `}`";
}

def TypeScript_CreateArrayOp : TypeScript_Op<"CreateArray"> {
  let summary = "create array";
  let description = [{
    create array
  }];

  let arguments = (ins Variadic<AnyType>:$items);
  let results = (outs TypeScript_Array:$instance);

  let assemblyFormat = "`[` $items `]` attr-dict `:` type($items) `->` type($instance)";
}

def TypeScript_NewEmptyArrayOp : TypeScript_Op<"NewEmptyArray"> {
  let summary = "create empty array";
  let description = [{
    create empty array
  }];

  let arguments = (ins );
  let results = (outs TypeScript_Array:$instance);

  let assemblyFormat = "attr-dict `:` type($instance)";
}

def TypeScript_NewArrayOp : TypeScript_Op<"NewArray"> {
  let summary = "allocate in memory few instances";
  let description = [{
    allocate in memory few instances
  }];

  let arguments = (ins I32:$count);
  let results = (outs TypeScript_Array:$instance);

  let assemblyFormat = "`[` $count `]` attr-dict `:` type($instance)";
}

def TypeScript_DeleteOp : TypeScript_Op<"Delete"> {
  let summary = "delete allocated in memory";
  let description = [{
    delete allocated in memory
  }];

  let arguments = (ins TypeScript_AnyRefLike:$reference);
  let results = (outs );

  let assemblyFormat = "`(` $reference `)` attr-dict `:` type($reference)";
}

def TypeScript_CallOp : TypeScript_Op<"Call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the call.

    Example:

    ```mlir
    %2 = call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee",$_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(callee), results,
            operands);
    }]>];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() 
    {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() 
    {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let verifier = ?;
}

def TypeScript_CallIndirectOp : TypeScript_Op<"CallIndirect", [
      CallOpInterface,
      DeclareOpInterfaceMethods<SymbolUserOpInterface>,
      TypesMatchWith<"callee input types match argument types",
                     "callee", "operands",
                     "($_self.isa<HybridFunctionType>() ? $_self.cast<HybridFunctionType>().getInputs() : $_self.cast<FunctionType>().getInputs())">,
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

  let arguments = (ins TypeScript_Callable:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins "Value":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(operands);
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
    Value getCallee() { return getOperand(0); }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return ++operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() { return getCallee(); }
  }];

  let verifier = ?;
  let hasCanonicalizer = 1;

  let assemblyFormat = "$callee `(` $operands `)` attr-dict `:` type($callee)";
}

def TypeScript_InvokeOp : TypeScript_Op<"Invoke", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      Terminator]> {
  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee,
                   Variadic<AnyType>:$operands,
                   Variadic<AnyType>:$normalDestOperands,
                   Variadic<AnyType>:$unwindDestOperands);
  let results = (outs Variadic<AnyType>);
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
  let verifier = [{ return ::verify(*this);  }];
}

def TypeScript_InvokeHybridOp : TypeScript_Op<"InvokeHybrid", [
                      AttrSizedOperandSegments,
                      DeclareOpInterfaceMethods<BranchOpInterface>,
                      Terminator]> {
  let arguments = (ins TypeScript_HybridFunction:$callee,
                   Variadic<AnyType>:$operands,
                   Variadic<AnyType>:$normalDestOperands,
                   Variadic<AnyType>:$unwindDestOperands);
  let results = (outs Variadic<AnyType>);
  let successors = (successor AnySuccessor:$normalDest,
                              AnySuccessor:$unwindDest);

  let verifier = ?;
}

def TypeScript_CallInternalOp : TypeScript_Op<"CallInternal",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call internal operation";
  let description = [{
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let builders = [
    OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee",$_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, $_builder.getSymbolRefAttr(callee), results,
            operands);
    }]>];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee().hasValue() ? callee().getValue() : ""; }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() 
    {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() 
    {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let verifier = ?;
}

def TypeScript_CallHybridInternalOp : TypeScript_Op<"CallHybridInternal",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call (hybrid) internal operation";
  let description = [{
  }];

  let arguments = (ins TypeScript_HybridFunction:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    StringRef getCallee() { return ""; }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() 
    {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() 
    {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` type($callee) `+` functional-type($operands, results)
  }];
  let verifier = ?;
}

def TypeScript_CaptureOp : TypeScript_Op<"Capture", []> {
  let summary = "capture variables";

  let arguments = (ins Variadic<AnyType>:$captured);
  let results = (outs AnyType);
}

def TypeScript_TrampolineOp : TypeScript_Op<"Trampoline", []> {
  let summary = "trampoline logic";

  let arguments = (ins FunctionType:$callee, AnyType:$data_reference, OptionalAttr<BoolAttr>:$allocInHeap);
  let results = (outs FunctionType:$newCallee);
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
    AnyType:$argValue
  );

  let results = (outs
    TypeScript_AnyRefLike:$reference
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
    TypeScript_AnyOptional:$argValue
  );

  let results = (outs
    TypeScript_AnyRefLike:$reference
  );

  let regions = (region AnyRegion:$defaultValueRegion);

  let extraClassDeclaration = [{
    OpBuilder getDefaultValueBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return OpBuilder::atBlockEnd(body, listener);
    }
  }];

}

def TypeScript_ParamDefaultValueOp : TypeScript_Op<"DefaultValue", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["ParamOptionalOp"]>]> {
  let arguments = (ins Variadic<AnyType>:$results);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let verifier = ?;

  let assemblyFormat = "$results attr-dict `:` type($results)";
}

def TypeScript_CreateOptionalOp : TypeScript_Op<"CreateOptional", 
    [TypesMatchWith<"type of 'value' matches element type of 'optional'",
                     "res", "in",
                     "$_self.cast<OptionalType>().getElementType()">]> {
  let description = [{
    Example:
      ts.create_optional %v : i32 to ts.optional<i32>
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs TypeScript_AnyOptional:$res);
}

def TypeScript_UndefOptionalOp : TypeScript_Op<"UndefOptional"> {
  let description = [{
    Example:
      ts.undef_optional : ts.optional<i32>
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
}

def TypeScript_VariableOp : TypeScript_Op<"Variable", []> {
  let summary = [{
    Allocate an object in stack memory.
  }];

  let description = [{

    ```mlir
    %0 = ts.constant ...

    %1 = ts.variable : f32
    %2 = ts.variable init(%0): f32
    ```
  }];

  let arguments = (ins
    Optional<AnyType>:$initializer, OptionalAttr<BoolAttr>:$captured
  );

  let results = (outs
    TypeScript_Ref:$reference
  );

  let assemblyFormat = "`(` $initializer `)` `{` $captured `}` attr-dict `:` type($initializer) `->` type($reference)";
}

// TODO: add verify logic to match types for BoundRef and BoundFunction
def TypeScript_StoreOp : TypeScript_Op<"Store", []> {
  let summary = "stores the value into the reference";
  let description = [{
    The `store` operation stores the value into the reference storage.
  }];

  let arguments = (ins AnyType:$value, Arg<TypeScript_RefOrBoundRefOrValueRef, "the reference to store to", [MemWrite]>:$reference);
  let assemblyFormat = "$value `,` $reference attr-dict `:` type($value) `->` type($reference)";
}

def TypeScript_LoadOp : TypeScript_Op<"Load", []> {
  let summary = "loads the value from the reference type";
  let description = [{
    The `load` operation loads the value from the reference value.
  }];

  let arguments = (ins Arg<TypeScript_RefOrBoundRefOrValueRef, "the reference to load from", [MemRead]>:$reference);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $reference `)` attr-dict `:` type($reference) `->` type($result)";

  let hasCanonicalizer = 1;
}

def TypeScript_ExtractPropertyOp : TypeScript_Op<"ExtractProperty"> {
  let description = [{
    ```mlir
    %2 = ts.extract_property %1, %2 : f32
    ```
  }];

  let arguments = (ins
    TypeScript_AnyStructLike:$object,
    ArrayAttr:$position
  );

  let results = (outs
    AnyType:$result
  );

  let assemblyFormat = "`(` $object `,` $position `)` attr-dict `:` type($object) `->` type($result)";
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
    ArrayAttr:$position
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

  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array) `[` type($index) `]` `->` type($result)";
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

  let assemblyFormat = "$objectRef `<` $position `>` attr-dict `:` type($objectRef) `->` type($result)";
}

def TypeScript_ThisPropertyRefOp : TypeScript_Op<"ThisPropertyRef"> {
  let description = [{
    ```mlir
    %2 = ts.this_property_ref %1, %2 : !ts.ref<f32>
    ```
  }];

  let arguments = (ins
    TypeScript_AnyRefLike:$objectRef,
    TypeScript_Ref:$offset
  );

  let results = (outs
    TypeScript_Ref:$result
  );

  let assemblyFormat = "$objectRef `+` $offset attr-dict `:` type($objectRef) `+` type($offset) `->` type($result)";
}

def TypeScript_StoreBoundRefOp : TypeScript_Op<"StoreBoundRef", []> {
  let summary = "stores the value into the bound reference";
  let description = [{
    The `store` operation stores the value into the bound reference storage.
  }];

  let arguments = (ins AnyType:$value, Arg<TypeScript_BoundRef, "the bound reference to store to", [MemWrite]>:$reference);
  let assemblyFormat = "$value `,` $reference attr-dict `:` type($value) `->` type($reference)";
}

def TypeScript_LoadBoundRefOp : TypeScript_Op<"LoadBoundRef", []> {
  let summary = "loads the value from the bound reference type";
  let description = [{
    The `load` operation loads the value from the bound reference value.
  }];

  let arguments = (ins Arg<TypeScript_BoundRef, "the bound reference to load from", [MemRead]>:$reference);
  let results = (outs AnyType:$result);
  let assemblyFormat = "`(` $reference `)` attr-dict `:` type($reference) `->` type($result)";
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

  let assemblyFormat = "$thisVal `:``:` $valueRef attr-dict `:` type($thisVal) `,` type($valueRef) `->` type($result)";
}

def TypeScript_CreateBoundFunctionOp : TypeScript_Op<"CreateBoundFunction"> {
  let description = [{
    ```mlir
    %2 = ts.create_bound_function %1, %2 : !ts.this_func<()->void>
    ```
  }];

  let arguments = (ins
    TypeScript_AnyRefLike:$thisVal, FunctionType:$func
  );

  let results = (outs
    TypeScript_BoundFunctionLike:$result
  );

  let assemblyFormat = "$thisVal `:``:` $func `:` type($thisVal) `,` type($func) attr-dict `->` type($result)";
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

  let assemblyFormat = "$boundFunc attr-dict `:` type($boundFunc) `->` type($result)";
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
    FunctionType:$result
  );

  let assemblyFormat = "$boundFunc attr-dict `:` type($boundFunc) `->` type($result)";
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

  let assemblyFormat = "$operand1 `(` $opCode `)` attr-dict `:` type($operand1) `->` type($result)";
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

  let assemblyFormat = "$operand1 `(` $opCode `)` attr-dict `:` type($operand1) `->` type($result)";
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

  let assemblyFormat = "$operand1 `(` $opCode `)` attr-dict `:` type($operand1) `->` type($result)";
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

  let assemblyFormat = "$operand1 `(` $opCode `)` $operand2 attr-dict `:` type($operand1) `,` type($operand2) `->` type($result)";
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

  let assemblyFormat = "$operand1 `(` $opCode `)` $operand2 attr-dict `:` type($operand1) `,` type($operand2) `->` type($result)";
}

def TypeScript_CastOp : TypeScript_Op<"Cast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let description = [{
    Example:
      ts.cast %v : i32 to f16
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($res)";

  let hasCanonicalizer = 1;
  let verifier = [{ return ::verify(*this); }];
}

def TypeScript_BoxOp : TypeScript_Op<"Box", [NoSideEffect]> {
  let description = [{
    Example:
      %ti = ts.typeof %1
      ts.box %v, %ti : i32 to ts.any
  }];
  let arguments = (ins AnyType:$in, TypeScript_String:$typeInfo);
  let results = (outs TypeScript_Any:$res);
  let assemblyFormat = "$in `,` $typeInfo attr-dict `:` type($in) `,` type($typeInfo) `to` type($res)";
}

def TypeScript_UnboxOp : TypeScript_Op<"Unbox", [NoSideEffect]> {
  let description = [{
    Example:
      ts.unbox %v : ts.any to i32
  }];
  let arguments = (ins TypeScript_Any:$in);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($res)";
}

def TypeScript_DialectCastOp : TypeScript_Op<"DialectCast", [DeclareOpInterfaceMethods<CastOpInterface>, NoSideEffect]> {
  let description = [{
    Example:
      ts.dialect_cast %v : ts.number to f64
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($res)";
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
    TypeScript_Ref:$reference
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
    TypeScript_Ref:$reference
  );
}

def TypeScript_ExitOp : TypeScript_Op<"Exit", [NoSideEffect, ReturnLike, Terminator]> {
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

def TypeScript_ReturnInternalOp : TypeScript_Op<"ReturnInternal", [NoSideEffect, HasParent<"FuncOp">,
                                ReturnLike, Terminator]> {
  let summary = "return 'internal' operation";
  let description = [{
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def TypeScript_IfOp : TypeScript_Op<"If",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"typescript::ResultOp">, RecursiveSideEffects,
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
      return results().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return results().empty() ? OpBuilder::atBlockTerminator(body, listener)
                               : OpBuilder::atBlockEnd(body, listener);
    }
  }];

  let hasCanonicalizer = 1;
}

def TypeScript_ResultOp : TypeScript_Op<"Result", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["IfOp",  "DoWhileOp", "WhileOp", "ForOp", "TryOp"]>]> {
  let summary = "loop yield and termination operation";

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  // Override default verifier, no custom verification
  // needed.
  let verifier = ?;
}

def TypeScript_WhileOp : TypeScript_Op<"While",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveSideEffects]> {
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
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];

  let assemblyFormat =
      [{ ($inits^)? attr-dict `:` type($inits) `->` type($results) `(` $cond `)` $body }];
}

def TypeScript_DoWhileOp : TypeScript_Op<"DoWhile",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveSideEffects]> {
  let summary = "a generic 'do-while' loop";
  let description = [{
    This operation represents a generic "do-while" loop.
  }];

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region AnyRegion:$body, SizedRegion<1>:$cond);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];

  let assemblyFormat =
      [{ ($inits^)? attr-dict `:` type($inits) `->` type($results) $body `(` $cond `)` }];  
}

def TypeScript_ForOp : TypeScript_Op<"For",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveSideEffects]> {
  let summary = "a generic 'for' loop";

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$cond, AnyRegion:$body, SizedRegion<1>:$incr);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];

  let assemblyFormat =
      [{ attr-dict ($inits^)? `:` type($inits) `->` type($results) `(` $cond `,` $incr `)` $body }];   
}

def TypeScript_ConditionOp : TypeScript_Op<"Condition",
                         [HasParent<"DoWhileOp, WhileOp, ForOp">, NoSideEffect, Terminator]> {
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

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
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

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
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
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>, RecursiveSideEffects,
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
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       RecursiveSideEffects, NoRegionArguments]> {
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

def TypeScript_MergeOp : TypeScript_Op<"Merge", [NoSideEffect, HasParent<"SwitchOp, LabelOp">, Terminator]> {
  let summary = "A special terminator for merging a switch or label.";

  let arguments = (ins);

  let results = (outs);
}

def TypeScript_TryOp : TypeScript_Op<"Try",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveSideEffects]> {
  let summary = "a generic 'try' clause";
  let description = [{
    This operation represents a generic "try" clause.
  }];

  let arguments = (ins );
  let results = (outs );
  let regions = (region AnyRegion:$body, AnyRegion:$catches, AnyRegion:$finallyBlock);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];

  let assemblyFormat = "attr-dict $body $catches $finallyBlock";
}

def TypeScript_CatchOp : TypeScript_Op<"Catch",
                         [HasParent<"TryOp">]> {
  let summary = "try catch";
  let description = [{
  }];

  let arguments = (ins TypeScript_Ref:$catchArg);

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
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
  let results = (outs TypeScript_AnyTuple:$result);

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
}

def TypeScript_CompareCatchTypeOp : TypeScript_Op<"CompareCatchType"> {
  let summary = "CompareCatchType - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_AnyTuple:$landingPad, TypeScript_Ref:$throwTypeInfo);
  let results = (outs TypeScript_Boolean:$result);

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
}

def TypeScript_BeginCatchOp : TypeScript_Op<"BeginCatch"> {
  let summary = "BeginCatch - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_AnyTuple:$landingPad);
  let results = (outs TypeScript_Opaque:$exceptionInfo);

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
}

def TypeScript_SaveCatchVarOp : TypeScript_Op<"SaveCatchVar"> {
  let summary = "SaveCatchVar - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_Opaque:$exceptionInfo, TypeScript_Ref:$varStore);
  let results = (outs );

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
}

def TypeScript_EndCatchOp : TypeScript_Op<"EndCatch"> {
  let summary = "EndCatch - for affine layer";
  let description = [{
  }];

  let arguments = (ins );
  let results = (outs );

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
}

def TypeScript_BeginCleanupOp : TypeScript_Op<"BeginCleanup"> {
  let summary = "BeginCleanup - for affine layer";
  let description = [{
  }];

  let arguments = (ins );
  let results = (outs );

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
}

def TypeScript_EndCleanupOp : TypeScript_Op<"EndCleanup", [Terminator]> {
  let summary = "EndCleanup - for affine layer";
  let description = [{
  }];

  let arguments = (ins TypeScript_AnyTuple:$landingPad);
  let results = (outs );
  let successors = (successor VariadicSuccessor<AnySuccessor>:$unwindDest);

  // Override the default verifier, everything is checked by traits.
  let verifier = ?;
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
      build($_builder, $_state, RefType::get(global.getType()), global.sym_name(), mlir::IntegerAttr());
      $_state.addAttributes(attrs);
    }]>,
    OpBuilder<(ins "FuncOp":$func, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, RefType::get(func.getType()), func.getName(), mlir::IntegerAttr());
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
    /// Return the LLVM type of the global.
    Type getType() {
      return type();
    }
    /// Return the initializer attribute if it exists, or a null attribute.
    Attribute getValueOrNull() {
      return value().getValueOr(Attribute());
    }
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

def TypeScript_GlobalResultOp : TypeScript_Op<"GlobalResult", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["GlobalOp"]>]> {
  let summary = "global init yield and termination operation";

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  // Override default verifier, no custom verification
  // needed.
  let verifier = ?;
}

def TypeScript_GlobalConstructorOp : TypeScript_Op<"GlobalConstructor"> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs );

  let builders = [
    OpBuilder<(ins "FuncOp":$func, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, RefType::get(func.getType()), func.getName());
      $_state.addAttributes(attrs);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Return the ts.func operation that is referenced here.
    FuncOp getFunction();
  }];
}

def TypeScript_PushOp : TypeScript_Op<"Push"> {
  let arguments = (ins TypeScript_AnyArrayRef:$op, Variadic<AnyType>:$items);
  let results = (outs I64:$new_size);
}

def TypeScript_PopOp : TypeScript_Op<"Pop"> {
  let arguments = (ins TypeScript_AnyArrayRef:$op);
  let results = (outs AnyType:$item);
}

def TypeScript_LengthOfOp : TypeScript_Op<"LengthOf"> {
  let arguments = (ins TypeScript_ArrayLike:$op);
  let results = (outs I32:$result);
}

def TypeScript_StringLengthOp : TypeScript_Op<"StringLength"> {
  let arguments = (ins TypeScript_String:$op);
  let results = (outs I32:$result);
}

def TypeScript_StringConcatOp : TypeScript_Op<"StringConcat"> {
  let arguments = (ins Variadic<TypeScript_String>:$ops, OptionalAttr<BoolAttr>:$allocInStack);
  let results = (outs TypeScript_String:$result);
}

def TypeScript_StringCompareOp : TypeScript_Op<"StringCompare"> {
  let arguments = (ins TypeScript_String:$op1, TypeScript_String:$op2, I32Attr:$code);
  let results = (outs TypeScript_Boolean:$result);
}

def TypeScript_CharToStringOp : TypeScript_Op<"CharToString"> {
  let arguments = (ins TypeScript_Char:$op);
  let results = (outs TypeScript_String:$result);
}

def TypeScript_MemoryCopyOp : TypeScript_Op<"MemoryCopy"> {
  let arguments = (ins AnyType:$dst, AnyType:$src);
  let results = (outs );
}

def TypeScript_LoadSaveOp : TypeScript_Op<"LoadSave", [AllTypesMatch<["dst", "src"]>]> {
  let arguments = (ins AnyType:$dst, AnyType:$src);
  let results = (outs );
}

def TypeScript_DebuggerOp : TypeScript_Op<"Debugger"> {
  let summary = "debugger";
}

def TypeScript_UnreachableOp : TypeScript_Op<"unreachable", [Terminator]> {
  let summary = "unreachable";
}

def TypeScript_NoOp : TypeScript_Op<"noop"> {
  let summary = "no operation";
}

#endif // TYPESCRIPT_OPS
