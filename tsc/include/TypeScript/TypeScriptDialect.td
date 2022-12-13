#ifndef TYPESCRIPT_DIALECT
#define TYPESCRIPT_DIALECT

include "mlir/IR/OpBase.td"

//include "mlir/Dialect/Arithmetic/IR/ArithmeticBase.td"
//include "mlir/Dialect/Math/IR/MathBase.td"
//include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.td"
//include "mlir/Dialect/Func/IR/FuncOps.td"
//include "mlir/Dialect/Async/IR/AsyncDialect.td"

//===----------------------------------------------------------------------===//
// TypeScript dialect definition.
//===----------------------------------------------------------------------===//

def TypeScript_Dialect : Dialect {
    let name = "ts";    
    let summary = "A typescript out-of-tree MLIR dialect.";    
    let description = [{
        This dialect is a typescript out-of-tree MLIR dialect.
    }];
    let cppNamespace = "::mlir::typescript";
    //let dependentDialects = ["::mlir::arith::ArithmeticDialect", "::mlir::math::MathDialect", "::mlir::cf::ControlFlowDialect", "::mlir::func::FuncDialect", "::mlir::async::AsyncDialect"];
}

//===----------------------------------------------------------------------===//
// Base typescript operation definition.
//===----------------------------------------------------------------------===//

class TypeScript_Op<string mnemonic, list<Trait> traits = []> :
        Op<TypeScript_Dialect, mnemonic, traits> {
          //string llvmBuilder = "";
        }

#endif // TYPESCRIPT_DIALECT
