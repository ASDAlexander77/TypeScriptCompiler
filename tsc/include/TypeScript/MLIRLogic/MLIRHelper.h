#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_

#include "TypeScript/TypeScriptOps.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/SmallPtrSet.h"

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

    static std::string getName(ts::StringLiteral stringLiteral)
    {
        std::string nameValue;
        if (stringLiteral)
        {
            // it can be empty
            nameValue = convertWideToUTF8(stringLiteral->text);            
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

        if (name == SyntaxKind::StringLiteral)
        {
            return getName(name.as<ts::StringLiteral>());
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
        return getAnonymousName(loc, ".unk");
    }

    static void getAnonymousNameStep(std::stringstream &ssName, mlir::Location loc)
    {
        mlir::TypeSwitch<mlir::LocationAttr>(loc)
        .Case<mlir::FileLineColLoc>([&](auto loc) {
            // auto fileName = loc.getFilename();
            auto line = loc.getLine();
            auto column = loc.getColumn();
            ssName << 'L' << line << 'C' << column;
        })
        .Case<mlir::FusedLoc>([&](auto loc) {
            for (auto subLoc : loc.getLocations())
            {
                getAnonymousNameStep(ssName, subLoc);
            }
        });        
    }

    static std::string getAnonymousName(mlir::Location loc, const char *prefix)
    {
        // auto calculate name
        std::stringstream ssName;
        ssName << prefix;
        getAnonymousNameStep(ssName, loc);
        ssName << 'H' << hash_value(loc);
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

    static void loadTypes(mlir::SmallVector<mlir::Type> &types, mlir::Type type)
    {
        if (auto sourceUnionType = type.dyn_cast<mlir_ts::UnionType>())
        {
            for (auto item : sourceUnionType.getTypes())
            {
                types.push_back(item);
            }
        }
        else
        {
            types.push_back(type);
        }
    }

    static void loadTypes(mlir::SmallPtrSet<mlir::Type, 2> &types, mlir::Type type)
    {
        if (auto sourceUnionType = type.dyn_cast<mlir_ts::UnionType>())
        {
            for (auto item : sourceUnionType.getTypes())
            {
                types.insert(item);
            }
        }
        else
        {
            types.insert(type);
        }
    }

    static mlir::Type stripLiteralType(mlir::Type type)
    {
        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            return literalType.getElementType();
        }

        return type;
    }    

    static mlir::Attribute TupleFieldName(mlir::StringRef name, mlir::MLIRContext *context)
    {
        assert(!name.empty());
        return mlir::StringAttr::get(context, name);
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_
