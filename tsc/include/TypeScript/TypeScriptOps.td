#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "TypeScriptTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    Syntax:

    ```
    operation ::= ssa-id `=` `ts.constant` attribute-value `:` type
    ```

    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute.

    Example:

    ```mlir
    // Integer constant
    %1 = constant 42 : i32

    // Reference to function @myfn.
    %3 = constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

    // Equivalent generic forms
    %1 = "ts.constant"() {value = 42 : i32} : () -> i32
    %3 = "ts.constant"() {value = @myfn}
       : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$constValue);

  let builders = [
    OpBuilderDAG<(ins "Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("value"); }
  }];

  let hasFolder = 1;

  let assemblyFormat = "attr-dict `:` type($constValue)";
}

def SymbolRefOp : TypeScript_Op<"SymbolRef"> {
  let summary = "sumbol reference as constant";

  let arguments = (ins FlatSymbolRefAttr:$identifier);
  let results = (outs AnyType);

  let builders = [
    OpBuilderDAG<(ins "FlatSymbolRefAttr":$identifier),
    [{ build($_builder, $_state, identifier.getType(), identifier); }]>];

  let extraClassDeclaration = [{
    Attribute getValue() { return (*this)->getAttr("identifier"); }
  }];
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
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilderDAG<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];
  let extraClassDeclaration = [{
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static FuncOp create(Location location, StringRef name, FunctionType type,
                         iterator_range<dialect_attr_iterator> attrs);
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
  let parser = [{ return ::parseFuncOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  let verifier = [{ return ::verify(*this); }];
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
    TypeScript_Any:$result
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

  let verifier = [{ return success(); }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TypeScript_PrintOp : TypeScript_Op<"Print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input string, and produces no results.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
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
  let results = (outs F32:$res);
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
    OpBuilderDAG<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee",$_builder.getSymbolRefAttr(callee));
      $_state.addTypes(callee.getType().getResults());
    }]>,
    OpBuilderDAG<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilderDAG<(ins "StringRef":$callee, "TypeRange":$results,
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
      TypesMatchWith<"callee input types match argument types",
                     "callee", "operands",
                     "$_self.cast<FunctionType>().getInputs()">,
      TypesMatchWith<"callee result types match result types",
                     "callee", "results",
                     "$_self.cast<FunctionType>().getResults()">
    ]> {
  let summary = "indirect call operation";
  let description = [{
    The `call_indirect` operation represents an indirect call to a value of
    function type. Functions are first class types in MLIR, and may be passed as
    arguments and merged together with block arguments. The operands and result
    types of the call must match the specified function type.

    Function values can be created with the
    [`constant` operation](#stdconstant-constantop).

    Example:

    ```mlir
    %31 = call_indirect %15(%0, %1)
            : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
    ```
  }];

  let arguments = (ins FunctionType:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);

  let builders = [
    OpBuilderDAG<(ins "Value":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.operands.push_back(callee);
      $_state.addOperands(operands);
      $_state.addTypes(callee.getType().cast<FunctionType>().getResults());
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
    AnyType:$reference
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
    AnyType:$reference
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

  let builders = [OpBuilderDAG<(ins), [{ /* nothing to do */ }]>];
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
    Optional<AnyType>:$initializer, OptionalAttr<BoolAttr>:$copy
  );

  let results = (outs
    TypeScript_AnyRef:$reference
  );
}

def TypeScript_StoreOp : TypeScript_Op<"Store",
      [TypesMatchWith<"type of 'value' matches element type of 'reference'",
                     "reference", "value",
                     "$_self.cast<RefType>().getElementType()">]> {
  let summary = "stores the value into the reference";
  let description = [{
    The `store` operation stores the value into the reference storage.
  }];

  let arguments = (ins AnyType:$value, Arg<TypeScript_AnyRef, "the reference to store to", [MemWrite]>:$reference);
  let assemblyFormat = "$value `,` $reference attr-dict `:` type($reference)";
}

def TypeScript_LoadOp : TypeScript_Op<"Load",
      [TypesMatchWith<"type of 'value' matches element type of 'reference'",
                     "reference", "result",
                     "$_self.cast<RefType>().getElementType()">]> {
  let summary = "loads the value from the reference type";
  let description = [{
    The `load` operation loads the value from the reference value.
  }];

  let arguments = (ins Arg<TypeScript_AnyRef, "the reference to load from", [MemRead]>:$reference);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$reference attr-dict `:` type($reference) `->` type($result)";
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

  //let assemblyFormat = "$object_reference `,` $propertyIndex attr-dict `:` type($result)";
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
    TypeScript_AnyRef:$result
  );

  //let assemblyFormat = "$array_reference `[` $index `]` attr-dict `:` type($result)";
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
    TypeScript_AnyRef:$result
  );

  //let assemblyFormat = "$object_reference `,` $propertyIndex attr-dict `:` type($result)";
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

def TypeScript_CastOp : TypeScript_Op<"Cast"> {
  let description = [{
    Example:
      ts.cast %v : i32 to f16
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
    Optional<TypeScript_AnyRef>:$reference
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
    TypeScript_AnyRef:$reference
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
  );
}

def TypeScript_IfOp : TypeScript_Op<"If",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"typescript::YieldOp">, RecursiveSideEffects,
       NoRegionArguments]> {
  let summary = "if-then-else operation";

  let arguments = (ins TypeScript_Boolean:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilderDAG<(ins "Value":$cond, "bool":$withElseRegion)>,
    OpBuilderDAG<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$withElseRegion)>,
    OpBuilderDAG<(ins "TypeRange":$resultTypes, "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
    OpBuilderDAG<(ins "Value":$cond,
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

def TypeScript_YieldOp : TypeScript_Op<"Yield", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["IfOp",  "DoWhileOp", "WhileOp", "ForOp"]>]> {
  let summary = "loop yield and termination operation";

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilderDAG<(ins), [{ /* nothing to do */ }]>];
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
  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];
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
  let regions = (region SizedRegion<1>:$body, SizedRegion<1>:$cond);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];
}

def TypeScript_ForOp : TypeScript_Op<"For",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveSideEffects]> {
  let summary = "a generic 'for' loop";

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$cond, SizedRegion<1>:$body, SizedRegion<1>:$incr);

  let extraClassDeclaration = [{
    OperandRange getSuccessorEntryOperands(unsigned index);
  }];
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
      [{ `(` $condition `)` attr-dict ($args^ `:` type($args))? }];

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
}

def TypeScript_BreakOp : TypeScript_Op<"Break"> {
  let summary = "loop termination";
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

def TypeScript_MergeOp : TypeScript_Op<"Merge", [NoSideEffect, HasParent<"SwitchOp">, Terminator]> {
  let summary = "A special terminator for merging a switch.";

  let arguments = (ins);

  let results = (outs);
}

def TypeScript_AddressOfOp : TypeScript_Op<"Addressof"> {
  let arguments = (ins FlatSymbolRefAttr:$global_name);
  let results = (outs TypeScript_AnyRef:$reference);

  let builders = [
    OpBuilderDAG<(ins "GlobalOp":$global,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, RefType::get(global.getType()), global.sym_name());
      $_state.addAttributes(attrs);
    }]>,
    OpBuilderDAG<(ins "FuncOp":$func, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs),
    [{
      build($_builder, $_state, RefType::get(func.getType()), func.getName());
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
        TypeScript_AnyRef:$reference,
        Index:$elementIndex
  );
  let results = (outs TypeScript_AnyRef:$reference_of_element);
}

def TypeScript_GlobalOp : TypeScript_Op<"Global",
    [IsolatedFromAbove, SingleBlockImplicitTerminator<"ReturnOp">, Symbol]> {
  let arguments = (ins
    TypeAttr:$type,
    UnitAttr:$constant,
    StrAttr:$sym_name,
    OptionalAttr<AnyAttr>:$value
  );

  let regions = (region AnyRegion:$initializer);

  let builders = [
    OpBuilderDAG<(ins "Type":$type, "bool":$isConstant, "StringRef":$name, 
        "Attribute":$value, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
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

def TypeScript_StringLengthOp : TypeScript_Op<"StringLength"> {
  let arguments = (ins TypeScript_String:$op);
  let results = (outs I32:$result);
}

def TypeScript_StringConcatOp : TypeScript_Op<"StringConcat"> {
  let arguments = (ins Variadic<TypeScript_String>:$ops);
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

#endif // TYPESCRIPT_OPS
