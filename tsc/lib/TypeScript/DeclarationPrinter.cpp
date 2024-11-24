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
    
    void MLIRDeclarationPrinter::printNamespaceBegin(NamespaceInfo::TypePtr elementNamespace) {
        if (elementNamespace && elementNamespace->name.size() > 0)
        {
            os << "namespace " << elementNamespace->fullName << " {";
            newline();
        }
    }

    void MLIRDeclarationPrinter::printNamespaceEnd(NamespaceInfo::TypePtr elementNamespace) {
        if (elementNamespace && elementNamespace->name.size() > 0)
        {
            newline();
            os << "}";
            newline();
        }
    }

    void MLIRDeclarationPrinter::printFloatValue(const APFloat &apValue, raw_ostream &os, bool *printedHex)
    {
        // We would like to output the FP constant value in exponential notation,
        // but we cannot do this if doing so will lose precision.  Check here to
        // make sure that we only output it in exponential format if we can parse
        // the value back and get the same value.
        bool isInf = apValue.isInfinity();
        bool isNaN = apValue.isNaN();
        if (!isInf && !isNaN)
        {
            SmallString<128> strValue;
            apValue.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0, /*TruncateZero=*/false);

            // Check to make sure that the stringized number is not some string like
            // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
            // that the string matches the "[-+]?[0-9]" regex.
            assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
                    ((strValue[0] == '-' || strValue[0] == '+') &&
                     (strValue[1] >= '0' && strValue[1] <= '9'))) &&
                   "[-+]?[0-9] regex does not match!");

            // Parse back the stringized version and check that the value is equal
            // (i.e., there is no precision loss).
            if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue))
            {
                os << strValue;
                return;
            }

            // If it is not, use the default format of APFloat instead of the
            // exponential notation.
            strValue.clear();
            apValue.toString(strValue);

            // Make sure that we can parse the default form as a float.
            if (strValue.str().contains('.'))
            {
                os << strValue;
                return;
            }
        }

        // Print special values in hexadecimal format. The sign bit should be included
        // in the literal.
        if (printedHex)
            *printedHex = true;
        SmallVector<char, 16> str;
        APInt apInt = apValue.bitcastToAPInt();
        apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                       /*formatAsCLiteral=*/true);
        os << str;
    }

    void MLIRDeclarationPrinter::printAsFieldName(mlir::Attribute attr)
    {
        mlir::TypeSwitch<mlir::Attribute>(attr)
            .Case<mlir::FlatSymbolRefAttr>([&](auto strAttr)
                                           { os << strAttr.getValue(); })
            .Case<mlir::StringAttr>([&](auto strAttr)
                                    { os << strAttr.getValue(); })
            .Case<mlir::IntegerAttr>([&](auto intAttr) {

                os << "[";

                auto intType = intAttr.getType();
                if (intType.isSignlessInteger(1)) {
                    os << (intAttr.getValue().getBoolValue() ? "true" : "false");
                    // Boolean integer attributes always elides the type.
                    return;
                }

                // Only print attributes as unsigned if they are explicitly unsigned or are
                // signless 1-bit values.  Indexes, signed values, and multi-bit signless
                // values print as signed.
                bool isUnsigned =
                    intType.isUnsignedInteger() || intType.isSignlessInteger(1);
                intAttr.getValue().print(os, !isUnsigned); 

                os << "]";
            })
            .Case<mlir::FloatAttr>([&](auto floatAttr) { 
                os << "[";

                bool printedHex = false;
                printFloatValue(floatAttr.getValue(), os, &printedHex); 

                os << "]";
            })
            .Default([&](auto attr) { os << attr; });
    }

    void MLIRDeclarationPrinter::printAsValue(mlir::Attribute attr)
    {
        mlir::TypeSwitch<mlir::Attribute>(attr)
            .Case<mlir::FlatSymbolRefAttr>([&](auto strAttr)
                                           { os << strAttr.getValue(); })
            .Case<mlir::StringAttr>([&](auto strAttr) {
                os << "\"";
                os.write_escaped(strAttr.getValue());
                os << "\""; 
            })
            .Case<mlir::IntegerAttr>([&](auto intAttr) {
                auto intType = intAttr.getType();
                if (intType.isSignlessInteger(1)) {
                    os << (intAttr.getValue().getBoolValue() ? "true" : "false");
                    // Boolean integer attributes always elides the type.
                    return;
                }

                // Only print attributes as unsigned if they are explicitly unsigned or are
                // signless 1-bit values.  Indexes, signed values, and multi-bit signless
                // values print as signed.
                bool isUnsigned =
                    intType.isUnsignedInteger() || intType.isSignlessInteger(1);
                intAttr.getValue().print(os, !isUnsigned); 
            })
            .Case<mlir::FloatAttr>([&](auto floatAttr) { 
                bool printedHex = false;
                printFloatValue(floatAttr.getValue(), os, &printedHex); 
            })
            .Default([&](auto attr) { os << attr; });
    }

    bool MLIRDeclarationPrinter::filterName(StringRef name)
    {
        return name.starts_with(".");
    }

    bool MLIRDeclarationPrinter::filterField(mlir::Attribute attr)
    {
        return mlir::TypeSwitch<mlir::Attribute, bool>(attr)
            .Case<mlir::FlatSymbolRefAttr>([&](auto strAttr)
                                           { return strAttr.getValue().starts_with("."); })
            .Case<mlir::StringAttr>([&](auto strAttr)
                                    { return strAttr.getValue().starts_with("."); })
            .Default([&](auto attr)
                     { return false; });
    }

    void MLIRDeclarationPrinter::printParams(ArrayRef<mlir::Type> params, mlir::Type thisType)
    {
        os << "(";
        auto separator = false;
        for (auto [index, paramType] : enumerate(params))
        {
            if (index == 0 && paramType == thisType)
                continue;

            if (separator)
                os << ", ";
            separator = true;

            auto actualType = paramType;
            auto isThisType = false;
            auto isConditional = false;
            if (auto optType = dyn_cast<mlir_ts::OptionalType>(paramType))
            {
                isConditional = true;
                actualType = optType.getElementType();
            }
            else if (index == 0 && paramType == thisType)
            {
                isThisType = true;
            }

            os << "p" << index << (isConditional ? "?" : "") << " : ";

            if (isThisType)
            {
                os << "this";
            }
            else
            {
                print(actualType);
            }
        }

        os << ")";
    }

    void MLIRDeclarationPrinter::printFunction(StringRef name, ArrayRef<mlir::Type> params, mlir::Type returnType)
    {
        os << "function " << name;

        printParams(params, mlir::Type());

        if (returnType)
        {
            os << " : ";
            print(returnType);
        }
    }

    void MLIRDeclarationPrinter::printMethod(bool isStatic, StringRef name, ArrayRef<mlir::Type> params, mlir::Type returnType, mlir::Type thisType)
    {
        if (isStatic)
        {
            os << "static ";
        }

        os << name;

        printParams(params, thisType);

        if (returnType)
        {
            os << " : ";
            print(returnType);
        }
    }

    void MLIRDeclarationPrinter::printTypeDeclaration(StringRef name, NamespaceInfo::TypePtr elementNamespace, mlir::Type type) 
    {
        printNamespaceBegin(elementNamespace);

        printBeforeDeclaration();

        os << "type " << name << " = ";
        print(type);
        os << ";";
        newline();

        printNamespaceEnd(elementNamespace);
    }

    void MLIRDeclarationPrinter::printEnum(StringRef name, NamespaceInfo::TypePtr elementNamespace, mlir::DictionaryAttr enumValues)
    {
        printNamespaceBegin(elementNamespace);

        printBeforeDeclaration();

        os << "enum " << name;
        newline();
        os << "{";
        newline();

        for (auto enumValue : enumValues)
        {
            os.indent(4);
            printAsFieldName(enumValue.getName());
            if (enumValue.getValue())
            {
                os << " = ";
                printAsValue(enumValue.getValue());
            }

            os << ",";
            newline();
        }

        os << "}";
        newline();

        printNamespaceEnd(elementNamespace);
    }

    void MLIRDeclarationPrinter::printVariableDeclaration(StringRef name, NamespaceInfo::TypePtr elementNamespace, mlir::Type type, bool isConst)
    {
        printNamespaceBegin(elementNamespace);

        printBeforeDeclaration();
        os << (isConst ? "const" : "let") << " " << name << " : ";
        print(type);
        os << ";";
        newline();

        printNamespaceEnd(elementNamespace);
    }

    void MLIRDeclarationPrinter::print(StringRef name, NamespaceInfo::TypePtr elementNamespace, mlir_ts::FunctionType funcType)
    {
        printNamespaceBegin(elementNamespace);

        printBeforeDeclaration();

        printFunction(
            name,
            funcType.getParams(),
            funcType.getNumResults() > 0 ? funcType.getResult(0) : mlir::Type());
        os << ";";
        newline();

        printNamespaceEnd(elementNamespace);
    }

    void MLIRDeclarationPrinter::print(ClassInfo::TypePtr classType)
    {
        printNamespaceBegin(classType->elementNamespace);

        printBeforeDeclaration();
        os << "class " << classType->name;

        if (classType->baseClasses.size() > 0)
        {
            os << " extends ";
            auto any = false;
            for (auto baseClass : classType->baseClasses)
            {
                if (any) 
                {
                    os << ", ";
                }

                os << classType->fullName;
                any = true;
            }
        }

        if (classType->implements.size() > 0)
        {
            os << " implements ";
            auto any = false;
            for (auto implement : classType->implements)
            {
                if (any) 
                {
                    os << ", ";
                }

                os << implement.interface->fullName;
                any = true;
            }
        }

        newline();
        os << "{";
        newline();

        // Static fields
        for (auto staticField : classType->staticFields)
        {
            if (filterField(staticField.id))
                continue;

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
        for (auto field : storageType.getFields())
        {
            if (filterField(field.id))
                continue;

            os.indent(4);
            printAsFieldName(field.id);
            if (field.isConditional)
                os << "?";
            os << ": ";
            print(field.type);
            os << ";";
            newline();
        }

        // methods (including static)
        for (auto method : classType->methods)
        {
            if (filterName(method.name))
                continue;

            os.indent(4);

            printMethod(
                method.isStatic,
                method.name,
                method.funcType.getParams(),
                method.funcType.getNumResults() > 0 ? method.funcType.getResult(0) : mlir::Type(),
                classType->classType);

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

        printNamespaceEnd(classType->elementNamespace);
    }

    void MLIRDeclarationPrinter::print(InterfaceInfo::TypePtr interfaceType)
    {
        printNamespaceBegin(interfaceType->elementNamespace);

        printBeforeDeclaration();
        os << "interface " << interfaceType->name;

        if (interfaceType->extends.size() > 0)
        {
            os << " extends ";
            auto any = false;
            for (auto interfaceInfo : interfaceType->extends)
            {
                if (any) 
                {
                    os << ", ";
                }

                os << interfaceInfo.second->fullName;
                any = true;
            }
        }

        newline();
        os << "{";
        newline();

        // Fields
        for (auto field : interfaceType->fields)
        {
            if (filterField(field.id))
                continue;

            os.indent(4);
            printAsFieldName(field.id);
            if (field.isConditional)
                os << "?";
            os << ": ";
            print(field.type);
            os << ";";
            newline();
        }

        // methods (including static)
        auto opaqueType = mlir_ts::OpaqueType::get(interfaceType->interfaceType.getContext());
        for (auto method : interfaceType->methods)
        {
            if (filterName(method.name))
                continue;

            os.indent(4);

            printMethod(
                false,
                method.name,
                method.funcType.getParams(),
                method.funcType.getNumResults() > 0 ? method.funcType.getResult(0) : mlir::Type(),
                /*interfaceType->interfaceType*/opaqueType);

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

        printNamespaceEnd(interfaceType->elementNamespace);
    }    

} // namespace typescript
