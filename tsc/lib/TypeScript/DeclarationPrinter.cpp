#include "TypeScript/MLIRLogic/MLIRDeclarationPrinter.h"
#include "TypeScript/MLIRLogic/MLIRPrinter.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mlir"

namespace typescript
{

void MLIRDeclarationPrinter::newline()
{
    os << "\n";
}

void MLIRDeclarationPrinter::printBeforeDeclaration() 
{
    os << "@dllimport";
    newline();
}

void MLIRDeclarationPrinter::print(mlir::Type type)
{
    MLIRPrinter mp{};
    mp.printType<raw_ostream>(os, type);
}

void MLIRDeclarationPrinter::printAsFieldName(mlir::Attribute attr)
{
    mlir::TypeSwitch<mlir::Attribute>(attr)
        .Case<mlir::FlatSymbolRefAttr>([&](auto strAttr) {
            os << strAttr.getValue();
        })
        .Case<mlir::StringAttr>([&](auto strAttr) {
            os << strAttr.getValue();
        })
        .Default([&](auto attr) {
            os << attr;
        });   
}

bool MLIRDeclarationPrinter::filterField(mlir::Attribute attr)
{
    return mlir::TypeSwitch<mlir::Attribute, bool>(attr)
        .Case<mlir::FlatSymbolRefAttr>([&](auto strAttr) {
            return strAttr.getValue().starts_with(".");
        })
        .Case<mlir::StringAttr>([&](auto strAttr) {
            return strAttr.getValue().starts_with(".");
        })
        .Default([&](auto attr) {
            return false;
        });   
}

void MLIRDeclarationPrinter::print(ClassInfo::TypePtr classType)
{
    // TODO:
    printBeforeDeclaration();
    os << "class " << classType->name;
    newline();
    os << "{";
    newline();

    auto storageType = cast<mlir_ts::ClassStorageType>(classType->classType.getStorageType());
    for (auto [index, field] : enumerate(storageType.getFields()))
    {
        if (filterField(field.id)) continue;

        os.indent(4);
        printAsFieldName(field.id);
        if (field.isConditional) os << "?";
        os << ": ";
        print(field.type);
        os << ";";
        newline();
    }

    os << "}";
    newline();
}

} // namespace typescript
