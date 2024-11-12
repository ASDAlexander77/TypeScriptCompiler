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

        void printTypeDeclaration(StringRef, mlir::Type);
        void printEnum(StringRef, ArrayRef<mlir::NamedAttribute>);
        void printVariableDeclaration(StringRef, mlir::Type, bool);
        void print(FunctionPrototypeDOM::TypePtr);
        void print(ClassInfo::TypePtr);
        void print(InterfaceInfo::TypePtr);

    protected:
        void newline();
        void printBeforeDeclaration();
        void printAsFieldName(mlir::Attribute);
        void printAsValue(mlir::Attribute);
        void printFloatValue(const APFloat &, raw_ostream &, bool * = nullptr);
        bool filterName(StringRef);
        bool filterField(mlir::Attribute);
        void printParams(ArrayRef<mlir::Type>, mlir::Type);
        void printFunction(StringRef, ArrayRef<mlir::Type>, mlir::Type);
        void printMethod(bool, StringRef, ArrayRef<mlir::Type>, mlir::Type, mlir::Type);
        void print(mlir::Type);
    };

} // namespace typescript 

#undef DEBUG_TYPE

#endif // TYPESCRIPT_MLIRGENLOGIC_DECLARATIONPRINTER_H
