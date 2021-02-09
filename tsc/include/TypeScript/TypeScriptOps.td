#ifndef TYPESCRIPT_OPS
#define TYPESCRIPT_OPS

include "TypeScriptDialect.td"
include "TypeScriptTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

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

def TypeScript_PrintOp : TypeScript_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input string, and produces no results.
  }];

  let arguments = (ins AnyType:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
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

def OptionalHasValueOp : TypeScript_Op<"optional_hasvalue",
    [NoSideEffect]> {
  let summary = "returns true if it has value";
  let description = [{
    Example:

    ```mlir
    %a = typescript.optional_hasvalue %b : (optional<any>) -> any
    ```
  }];

  let arguments = (ins Optional<AnyType>:$optional);
  let results = (outs I8:$hasValue);

  let assemblyFormat = "$optional attr-dict `:` type($optional) `->` type($hasValue)";
}

def OptionalCreateOp : TypeScript_Op<"optional_create",
    [NoSideEffect,
     TypesMatchWith<"optional element type matches value type",
                    "value", "optional",
                    "$_self.cast<OptionalType>().getElementType()">]> {
  let summary = "create new instance of the optional type";
  let description = [{
    Example:

    ```mlir
    %a = typescript.optional_create %b : (any) -> optional<any>
    ```
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs Optional<AnyType>:$optional);

  let assemblyFormat = "$value attr-dict `:` type($value) `to` type($optional)";
}

def OptionalCreateEmptyOp : TypeScript_Op<"optional_undef",
    [NoSideEffect,
     TypesMatchWith<"optional element type matches value type",
                    "type", "optional",
                    "$_self.cast<OptionalType>().getElementType()">]> {
  let summary = "create empty instance of the optional type";
  let description = [{
    Example:

    ```mlir
    %a = typescript.optional_undef : () -> optional<any>
    ```
  }];

  let arguments = (ins TypeAttr:$type);
  let results = (outs Optional<AnyType>:$optional);

  let assemblyFormat = "attr-dict `:` type($optional)";
}

def OptionalValueOp : TypeScript_Op<"optional_value",
    [NoSideEffect,
     TypesMatchWith<"optional element type matches result type",
                    "optional", "value",
                    "$_self.cast<OptionalType>().getElementType()">]> {
  let summary = "extracts the value of an optional type";
  let description = [{
    Example:

    ```mlir
    %a = typescript.optional_value %b : (optional<any>) -> i8
    ```
  }];

  let arguments = (ins Optional<AnyType>:$optional);
  let results = (outs AnyType:$value);

  let assemblyFormat = "$optional attr-dict `:` type($optional) `->` type($value)";
}

#endif // TYPESCRIPT_OPS
