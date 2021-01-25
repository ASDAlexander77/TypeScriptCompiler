#ifndef TYPESCRIPT_DIALECT
#define TYPESCRIPT_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TypeScript dialect definition.
//===----------------------------------------------------------------------===//

def TypeScript_Dialect : Dialect {
    let name = "typescript";
    let summary = "A typescript out-of-tree MLIR dialect.";
    let description = [{
        This dialect is a typescript out-of-tree MLIR dialect.
    }];
    let cppNamespace = "::mlir::typescript";
}

//===----------------------------------------------------------------------===//
// Base typescript operation definition.
//===----------------------------------------------------------------------===//

class TypeScript_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<TypeScript_Dialect, mnemonic, traits>;

def TypeScript_IdentifierReferenceType :
    Type<CPred<"$_self.isa<IdentifierReferenceType>()">, "TypeScript identifier reference type">;

// Provide a definition of the types that are used within the TypeScript dialect.
def TypeScript_Type : AnyTypeOf<[TypeScript_IdentifierReferenceType]>;

#endif // TYPESCRIPT_DIALECT
