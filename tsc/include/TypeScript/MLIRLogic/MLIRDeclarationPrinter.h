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

        void printTypeDeclaration(StringRef, NamespaceInfo::TypePtr, mlir::Type);
        void printEnum(StringRef, NamespaceInfo::TypePtr, mlir::DictionaryAttr);
        void printVariableDeclaration(StringRef, NamespaceInfo::TypePtr, mlir::Type, bool);
        void print(StringRef, NamespaceInfo::TypePtr, mlir_ts::FunctionType);
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
        void printNamespaceBegin(NamespaceInfo::TypePtr);
        void printNamespaceEnd(NamespaceInfo::TypePtr);
        void print(mlir::Type);
    };

} // namespace typescript 

#undef DEBUG_TYPE

#endif // TYPESCRIPT_MLIRGENLOGIC_DECLARATIONPRINTER_H
