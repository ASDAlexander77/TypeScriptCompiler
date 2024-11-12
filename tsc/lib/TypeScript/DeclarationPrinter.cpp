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

    void MLIRDeclarationPrinter::printParams(ArrayRef<mlir::Type> params)
    {
        os << "(";
        auto separator = false;
        for (auto [index, paramType] : enumerate(params))
        {
            if (separator)
                os << ", ";
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

    void MLIRDeclarationPrinter::printTypeDeclaration(StringRef name, mlir::Type type) 
    {
        printBeforeDeclaration();

        os << "type " << name << " = ";
        print(type);
        os << ";";
        newline();
    }

    void MLIRDeclarationPrinter::printEnum(StringRef name, ArrayRef<mlir::NamedAttribute> enumValues)
    {
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
    }

    void MLIRDeclarationPrinter::printVariableDeclaration(StringRef name, mlir::Type type, bool isConst)
    {
        printBeforeDeclaration();
        os << (isConst ? "const" : "let") << " " << name << " : ";
        print(type);
        os << ";";
        newline();
    }

    void MLIRDeclarationPrinter::print(FunctionPrototypeDOM::TypePtr funcProto)
    {
        printBeforeDeclaration();

        auto funcType = funcProto->getFuncType();
        printFunction(
            funcProto->getName(),
            funcType.getParams(),
            funcType.getNumResults() > 0 ? funcType.getResult(0) : mlir::Type());
        os << ";";
        newline();
    }

    void MLIRDeclarationPrinter::print(ClassInfo::TypePtr classType)
    {
        printBeforeDeclaration();
        os << "class " << classType->name;
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
        for (auto [index, field] : enumerate(storageType.getFields()))
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
