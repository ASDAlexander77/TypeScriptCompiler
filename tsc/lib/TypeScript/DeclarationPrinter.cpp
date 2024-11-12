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

bool MLIRDeclarationPrinter::filterName(StringRef name)
{
    return name.starts_with(".");
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

void MLIRDeclarationPrinter::printParams(ArrayRef<mlir::Type> params)
{
    os << "(";
    auto separator = false;
    for (auto [index, paramType] : enumerate(params))
    {
        if (separator) os << ", ";
        separator = true;

        auto actualType = paramType;
        auto isConditional = false;
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(paramType))
        {
            isConditional = true;
            actualType = optType.getElementType();
        }

        os << "p" << index << (isConditional ? "?" : "") << " : ";
        print(actualType);
    }
    
    os << ")";
}

void MLIRDeclarationPrinter::printFunction(StringRef name, ArrayRef<mlir::Type> params, mlir::Type returnType)
{
    os << "function " << name;

    printParams(params);

    if (returnType)
    {
        os << " : ";
        print(returnType);            
    }
}

void MLIRDeclarationPrinter::printMethod(bool isStatic, StringRef name, ArrayRef<mlir::Type> params, mlir::Type returnType)
{
    if (isStatic)
    {
        os << "static ";
    }

    os << name;

    printParams(params);

    if (returnType)
    {
        os << " : ";
        print(returnType);            
    }
}

void MLIRDeclarationPrinter::print(FunctionPrototypeDOM::TypePtr funcProto)
{
    auto funcType = funcProto->getFuncType();
    printFunction(
        funcProto->getName(), 
        funcType.getParams(), 
        funcType.getNumResults() > 0 ? funcType.getResult(0) : mlir::Type());    
}

void MLIRDeclarationPrinter::print(ClassInfo::TypePtr classType)
{
    // TODO:
    printBeforeDeclaration();
    os << "class " << classType->name;
    newline();
    os << "{";
    newline();

    // Static fields
    for (auto staticField : classType->staticFields)
    {
        if (filterField(staticField.id)) continue;

        os.indent(4);
        os << "static ";
        printAsFieldName(staticField.id);
        os << ": ";
        print(staticField.type);
        os << ";";
        newline();        
    }

    // Fields
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

    // methods (including static)
    for (auto method : classType->methods)
    {
        if (filterName(method.name)) continue;

        os.indent(4);

        printMethod(
            method.isStatic, 
            method.name, 
            method.funcType.getParams(), 
            method.funcType.getNumResults() > 0 ? method.funcType.getResult(0) : mlir::Type());

        newline();
    }

    // TODO: we can't declare generic methods, otherwise we need to declare body of methods
    // generic methods
    // for (auto method : classType->staticGenericMethods)
    // {
    //     if (filterName(method.name)) continue;

    //     os.indent(4);

    //     printMethod(
    //         method.isStatic, 
    //         method.name, 
    //         method.funcType.getParams(), 
    //         method.funcType.getNumResults() > 0 ? method.funcType.getResult(0) : mlir::Type());

    //     newline();
    // }

    os << "}";
    newline();
}

} // namespace typescript
