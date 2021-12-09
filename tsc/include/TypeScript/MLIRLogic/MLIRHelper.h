#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_

#include "TypeScript/TypeScriptOps.h"
#include "llvm/Support/ConvertUTF.h"

#include "parser.h"

namespace mlir_ts = mlir::typescript;

namespace typescript
{

static std::string convertWideToUTF8(const std::wstring &ws)
{
    std::string s;
    llvm::convertWideToUTF8(ws, s);
    return s;
}

static std::wstring ConvertUTF8toWide(const std::string &s)
{
    std::wstring ws;
    llvm::ConvertUTF8toWide(s, ws);
    return ws;
}

class MLIRHelper
{
  public:
    static std::string getName(ts::Identifier identifier)
    {
        std::string nameValue;
        if (identifier)
        {
            nameValue = convertWideToUTF8(identifier->escapedText);
            assert(nameValue.size() > 0);
        }

        return nameValue;
    }

    static std::string getName(ts::PrivateIdentifier identifier)
    {
        std::string nameValue;
        if (identifier)
        {
            nameValue = convertWideToUTF8(identifier->escapedText);
            assert(nameValue.size() > 0);
        }

        return nameValue;
    }

    static std::string getName(ts::Node name)
    {
        std::string nameValue;
        if (name == SyntaxKind::Identifier)
        {
            return getName(name.as<ts::Identifier>());
        }

        if (name == SyntaxKind::PrivateIdentifier)
        {
            return getName(name.as<ts::PrivateIdentifier>());
        }

        return nameValue;
    }

    static mlir::StringRef getName(ts::Node name, llvm::BumpPtrAllocator &stringAllocator)
    {
        auto nameValue = getName(name);
        return mlir::StringRef(nameValue).copy(stringAllocator);
    }

    static std::string getAnonymousName(mlir::Location loc)
    {
        return getAnonymousName(loc, "__uf");
    }

    static std::string getAnonymousName(mlir::Location loc, const char *prefix)
    {
        // auto calculate name
        std::stringstream ssName;
        ssName << prefix << hash_value(loc);
        return ssName.str();
    }

    static bool matchLabelOrNotSet(mlir::StringAttr loopLabel, mlir::StringAttr opLabel)
    {
        auto loopHasValue = loopLabel && loopLabel.getValue().size() > 0;
        auto opLabelHasValue = opLabel && opLabel.getValue().size() > 0;

        if (!opLabelHasValue)
        {
            return true;
        }

        if (loopHasValue && opLabelHasValue)
        {
            auto eq = loopLabel.getValue() == opLabel.getValue();
            return eq;
        }

        return false;
    }

    static bool matchSimilarTypes(mlir::Type ty1, mlir::Type ty2)
    {
        if (ty1 == ty2)
        {
            return true;
        }

        if (auto constArray1 = ty1.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            if (auto constArray2 = ty2.dyn_cast_or_null<mlir_ts::ConstArrayType>())
            {
                return matchSimilarTypes(constArray1.getElementType(), constArray2.getElementType());
            }
        }

        return false;
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_
