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
    null-op ::= `typescript.null` `:` any-type
    ```

    #### Example:

    ```mlir
    %0 = typescript.null : memref<*xi1>
    ```
  }];

  let arguments = (ins);

  let results = (outs
    AnyMemRef:$result
  );

  let verifier = [{ return success(); }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TypeScript_UndefOp : TypeScript_Op<"undef", []> {
  let summary = "Make an intermediate object whose value is undefined.";

  let description = [{
    Result Type is the type of object to make.

    ```
    undef-op ::= `typescript.undef` `:` any-type
    ```

    #### Example:

    ```mlir
    %0 = typescript.undef : f32
    ```
  }];

  let arguments = (ins);

  let results = (outs
    AnyType:$result
  );

  let verifier = [{ return success(); }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def TypeScript_PrintOp : TypeScript_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input string, and produces no results.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);

  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
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

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let verifier = ?;
}

def TypeScript_ParamOp : TypeScript_Op<"param", []> {
  let summary = [{
    Allocate an parameter object in memory.
  }];

  let description = [{

    ```mlir
    %0 = typescript.param : f32
    ```
  }];

  let arguments = (ins
    AnyType:$argValue
  );

  let results = (outs
    AnyType:$pointer
  );
}

def TypeScript_ParamOptionalOp : TypeScript_Op<"param_opt", [SingleBlockImplicitTerminator<"typescript::ParamDefaultValueOp">]> {
  let summary = [{
    Allocate an parameter object in memory.
  }];

  let description = [{

    ```mlir
    %0 = typescript.param_opt : f32
    ```
  }];

  let arguments = (ins
    AnyType:$argValue,
    I32:$params_count,
    OptionalAttr<I32Attr>:$paramIndex
  );

  let results = (outs
    AnyType:$pointer
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
    %0 = typescript.constant ...

    %1 = typescript.variable : f32
    %2 = typescript.variable init(%0): f32
    ```
  }];

  let arguments = (ins
    Optional<AnyType>:$initializer
  );

  let results = (outs
    AnyType:$pointer
  );
}

def TypeScript_ArithmeticBinaryOp : TypeScript_Op<"arithmetic_binary"> {
  let summary = [{
  }];

  let description = [{
    ```mlir
    %2 = typescript.arithmetic_binary 1, %0, %1 : f32
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
  let summary = [{
  }];

  let description = [{
    ```mlir
    %2 = typescript.logical_binary 1, %0, %1 : i1
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
      typescript.cast %v : i32 to f16
  }];
  let arguments = (ins AnyType:$in);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($res)";
}

def TypeScript_ReturnOp : TypeScript_Op<"return"> {
  let description = [{
    ```
    return-value-op ::= `typescript.return` ssa-use `:` typescript-type
    ```

    #### Example:

    ```mlir
    typescript.return
    typescript.return %0 : f32
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$operands
  );

  let builders = [
    OpBuilderDAG<(ins),
    [{ build($_builder, $_state, llvm::None); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def TypeScript_EntryOp : TypeScript_Op<"entry"> {
  let description = [{
    To prepare data for return value

    #### Example:

    ```mlir
    typescript.entry
    ```
  }];

  let arguments = (ins
    OptionalAttr<TypeAttr>:$returnType
  );  

  let assemblyFormat = "$returnType attr-dict";
}

def TypeScript_ExitOp : TypeScript_Op<"exit", [NoSideEffect, ReturnLike, Terminator]> {
  let description = [{
    To prepare exit data for return

    #### Example:

    ```mlir
    typescript.exit
    ```
  }];

  let arguments = (ins
    OptionalAttr<TypeAttr>:$returnType
  );

  let assemblyFormat = "$returnType attr-dict";
}

#endif // TYPESCRIPT_OPS
