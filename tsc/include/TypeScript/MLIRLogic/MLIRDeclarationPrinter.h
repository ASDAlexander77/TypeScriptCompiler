#ifndef TYPESCRIPT_MLIRGENLOGIC_DECLARATIONPRINTER_H
#define TYPESCRIPT_MLIRGENLOGIC_DECLARATIONPRINTER_H

#include "TypeScript/MLIRLogic/MLIRGenStore.h"

#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream;

namespace typescript
{
    class MLIRDeclarationPrinter
    {
        raw_ostream &os;

    public:
        MLIRDeclarationPrinter(raw_ostream &os) : os(os) {};

        void newline();
        void printBeforeDeclaration();
        void printAsFieldName(mlir::Attribute);
        bool filterName(StringRef);
        bool filterField(mlir::Attribute);
        void printParams(ArrayRef<mlir::Type>);
        void printFunction(StringRef, ArrayRef<mlir::Type>, mlir::Type);
        void printMethod(bool, StringRef, ArrayRef<mlir::Type>, mlir::Type);
        void print(mlir::Type);
        void print(FunctionPrototypeDOM::TypePtr funcProto);
        void print(ClassInfo::TypePtr);
    };

} // namespace typescript 

#undef DEBUG_TYPE

#endif // TYPESCRIPT_MLIRGENLOGIC_DECLARATIONPRINTER_H
