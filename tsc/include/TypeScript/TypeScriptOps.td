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

def TypeScript_FuncOp : TypeScript_Op<"func", [
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

def TypeScript_IdentifierReference : TypeScript_Op<"identifier_reference", [NoSideEffect, Symbol]> {
    let arguments = (ins FlatSymbolRefAttr:$identifier);
    let results = (outs NoneType);

    let extraClassDeclaration = [{
        static IdentifierReference create(Location location, StringRef name);
    }];
}

def TypeScript_NullOp : TypeScript_Op<"null", []> {
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
}

def TypeScript_UndefOp : TypeScript_Op<"undef", []> {
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
}

def TypeScript_PrintOp : TypeScript_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input string, and produces no results.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
}

def TypeScript_AssertOp : TypeScript_Op<"assert"> {
  let summary = "Assert operation";
  let description = [{
    Assert operation

    Example:

    ```mlir
    assert %b, "Expected ... to be true"
    ```
  }];

  let arguments = (ins I1:$arg, StrAttr:$msg);

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
    parseInt %b
    ```
  }];

  let arguments = (ins AnyType:$arg);
}

def TypeScript_ParseFloatOp : TypeScript_Op<"parseFloat"> {
  let summary = "Parse float operation";
  let description = [{
    Parse float operation

    Example:

    ```mlir
    parseFloat %b
    ```
  }];

  let arguments = (ins AnyType:$arg);
}


def TypeScript_CallOp : TypeScript_Op<"call",
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

  let verifier = ?;
}

def TypeScript_ParamOp : TypeScript_Op<"param", []> {
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

def TypeScript_ParamOptionalOp : TypeScript_Op<"param_opt", [SingleBlockImplicitTerminator<"typescript::ParamDefaultValueOp">]> {
  let summary = [{
    Allocate an parameter object in memory.
  }];

  let description = [{

    ```mlir
    %0 = ts.param_opt : f32
    ```
  }];

  let arguments = (ins
    AnyType:$argValue,
    I32:$params_count,
    OptionalAttr<I32Attr>:$paramIndex
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

def TypeScript_ParamDefaultValueOp : TypeScript_Op<"defaultValue", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["ParamOptionalOp"]>]> {
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilderDAG<(ins), [{ /* nothing to do */ }]>];
  let verifier = ?;
}

def TypeScript_VariableOp : TypeScript_Op<"variable", []> {
  let summary = [{
    Allocate an object in memory.
  }];

  let description = [{

    ```mlir
    %0 = ts.constant ...

    %1 = ts.variable : f32
    %2 = ts.variable init(%0): f32
    ```
  }];

  let arguments = (ins
    Optional<AnyType>:$initializer
  );

  let results = (outs
    TypeScript_AnyRef:$reference
  );
}

def TypeScript_StoreOp : TypeScript_Op<"store",
      [TypesMatchWith<"type of 'value' matches element type of 'reference'",
                     "reference", "value",
                     "$_self.cast<RefType>().getElementType()">]> {
  let summary = "stores the value into the reference";
  let description = [{
    The `store` operation stores the value into the reference storage.
  }];

  let arguments = (ins AnyType:$value, TypeScript_AnyRef:$reference);
  let assemblyFormat = "$value `,` $reference attr-dict `:` type($reference)";
}

def TypeScript_LoadOp : TypeScript_Op<"load",
      [TypesMatchWith<"type of 'value' matches element type of 'reference'",
                     "reference", "result",
                     "$_self.cast<RefType>().getElementType()">]> {
  let summary = "loads the value from the reference type";
  let description = [{
    The `load` operation loads the value from the reference value.
  }];

  let arguments = (ins TypeScript_AnyRef:$reference);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$reference attr-dict `:` type($reference)";
}

def TypeScript_ArithmeticBinaryOp : TypeScript_Op<"arithmetic_binary"> {
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

def TypeScript_LogicalBinaryOp : TypeScript_Op<"logical_binary"> {
  let description = [{
    ```mlir
    %2 = ts.logical_binary 1, %0, %1 : i1
    ```
  }];

  let arguments = (ins
    I32Attr:$opCode,
    AnyType:$operand1, 
    AnyType:$operand2
  );

  let results = (outs
    I1:$result
  );
}

def TypeScript_CastOp : TypeScript_Op<"cast"> {
  let description = [{
    Example:
      ts.cast %v : i32 to f16
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($res)";
}

def TypeScript_EntryOp : TypeScript_Op<"entry"> {
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

def TypeScript_ReturnOp : TypeScript_Op<"return"> {
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

def TypeScript_ReturnValOp : TypeScript_Op<"returnVal"> {
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

def TypeScript_ExitOp : TypeScript_Op<"exit", [NoSideEffect, ReturnLike, Terminator]> {
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

def TypeScript_TerminatorOp : TypeScript_Op<"terminator", [Terminator]> {
  let summary = "Generic terminator for entry op.";

  let description = [{
  }];

  let verifier = ?;

  let assemblyFormat = "attr-dict";
}


#endif // TYPESCRIPT_OPS
