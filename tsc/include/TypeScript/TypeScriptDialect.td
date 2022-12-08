#ifndef TYPESCRIPT_DIALECT
#define TYPESCRIPT_DIALECT

include "mlir/IR/OpBase.td"

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
    let useDefaultTypePrinterParser = 0;
}

//===----------------------------------------------------------------------===//
// Base typescript operation definition.
//===----------------------------------------------------------------------===//

class TypeScript_Op<string mnemonic, list<Trait> traits = []> :
        Op<TypeScript_Dialect, mnemonic, traits> {
          //string llvmBuilder = "";
        }

#endif // TYPESCRIPT_DIALECT
