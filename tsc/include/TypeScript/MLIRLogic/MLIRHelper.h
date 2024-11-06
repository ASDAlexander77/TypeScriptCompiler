#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_

#include "TypeScript/TypeScriptOps.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "parser.h"
#include "node_factory.h"

#include <functional>
#include <string>

using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

static std::string convertWideToUTF8(const std::wstring &ws)
{
    std::string s;
    llvm::convertWideToUTF8(ws, s);
    return s;
}

static std::wstring convertUTF8toWide(const std::string &s)
{
    std::wstring ws;
    llvm::ConvertUTF8toWide(s, ws);
    return ws;
}

enum class MatchResultType
{
    Match,
    NotMatchArgCount,
    NotMatchArg,
    NotMatchResultCount,
    NotMatchResult
};

struct MatchResult
{
    MatchResultType result;
    unsigned index;
};

enum class ExtendsResult {
    False,
    True,
    Never,
    Any
};

inline bool isTrue(ExtendsResult val)
{
    return val == ExtendsResult::True || val == ExtendsResult::Any;
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
        SyntaxKind kind = name;
        if (kind == SyntaxKind::Identifier)
        {
            return getName(name.as<ts::Identifier>());
        }

        if (kind == SyntaxKind::PrivateIdentifier)
        {
            return getName(name.as<ts::PrivateIdentifier>());
        }

        if (kind == SyntaxKind::StringLiteral)
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

    static mlir::Location getCallSiteLocation(mlir::Location callee, mlir::Location caller, bool enable = true)
    {
        if (enable)
            return mlir::CallSiteLoc::get(callee, caller);
        return caller;
    }    

    static mlir::Location getCallSiteLocation(mlir::Value callee, mlir::Location caller, bool enable = true)
    {
        return getCallSiteLocation(callee.getDefiningOp()->getLoc(), caller, enable);
    }

    static void getAnonymousNameStep(std::stringstream &ssName, mlir::Location loc)
    {
        mlir::TypeSwitch<mlir::LocationAttr>(loc)
            .Case<mlir::FileLineColLoc>([&](auto loc) {
                auto fileName = loc.getFilename();
                auto line = loc.getLine();
                auto column = loc.getColumn();

                assert(line != 0 || column != 0);

                auto hashCode = hash_value(fileName);
                ssName << 'L' << line << 'C' << column << "FH" << hashCode;
            })
            .Case<mlir::NameLoc>([&](auto loc) {
                getAnonymousNameStep(ssName, loc.getChildLoc());
            })
            .Case<mlir::OpaqueLoc>([&](auto loc) {
                getAnonymousNameStep(ssName, loc.getFallbackLocation());
            })
            .Case<mlir::CallSiteLoc>([&](auto loc) {
                getAnonymousNameStep(ssName, loc.getCaller());
            })        
            .Case<mlir::FusedLoc>([&](mlir::FusedLoc loc) {
                for (auto subLoc : loc.getLocations())
                {
                    getAnonymousNameStep(ssName, subLoc);
                }
            });        
    }

    static std::string getAnonymousName(mlir::Location loc, const char *prefix, StringRef fullNamesapceName)
    {
        // auto calculate name
        std::stringstream ssName;
        if (!fullNamesapceName.empty())
            ssName << fullNamesapceName.str() << ".";
        ssName << prefix;
        getAnonymousNameStep(ssName, loc);
        return ssName.str();
    }

    static std::string getAnonymousName(mlir::Type type, const char *prefix)
    {
        std::string ssName;
        llvm::raw_string_ostream s(ssName);
        s << prefix;
        s << '_';
        s << type;
        return ssName;
    }

    static mlir::ArrayRef<int64_t> getStructIndex(mlir::OpBuilder &builder, int64_t index)
    {
        return builder.getDenseI64ArrayAttr(index);
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
        if (!type)
        {
            return;
        }

        if (auto sourceUnionType = dyn_cast<mlir_ts::UnionType>(type))
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

    static void flatUnionTypes(mlir::SmallPtrSet<mlir::Type, 2> &types, mlir::Type type)
    {
        if (!type)
        {
            return;
        }

        if (auto sourceUnionType = dyn_cast<mlir_ts::UnionType>(type))
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
        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
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

    static bool hasDecorator(Node node, const char* decoratorStr)
    {
        for (auto decorator : node->modifiers)
        {
            if (decorator != SyntaxKind::Decorator)
            {
                continue;
            }

            SmallVector<std::string> args;
            auto expr = decorator.as<Decorator>()->expression;
            if (expr == SyntaxKind::CallExpression)
            {
                auto callExpression = expr.as<CallExpression>();
                expr = callExpression->expression;
            }            

            if (expr == SyntaxKind::Identifier)
            {
                auto name = MLIRHelper::getName(expr.as<Node>());
                if (name == decoratorStr)
                {
                    return true;
                }
            }
        }

        return false;
    }

    static void iterateDecorators(Node node, std::function<void(std::string, SmallVector<std::string>)> functor)
    {
        for (auto decorator : node->modifiers)
        {
            if (decorator != SyntaxKind::Decorator)
            {
                continue;
            }

            SmallVector<std::string> args;
            auto expr = decorator.as<Decorator>()->expression;
            if (expr == SyntaxKind::CallExpression)
            {
                auto callExpression = expr.as<CallExpression>();
                expr = callExpression->expression;
                for (auto argExpr : callExpression->arguments)
                {
                    args.push_back(MLIRHelper::getName(argExpr.as<Node>()));
                }
            }            

            if (expr == SyntaxKind::Identifier)
            {
                auto name = MLIRHelper::getName(expr.as<Node>());
                functor(name, args);
            }
        }
    }

    static void addDecoratorIfNotPresent(Node node, StringRef decoratorName)
    {
        NodeFactory nf(NodeFactoryFlags::None);
        for (auto decorator : node->modifiers)
        {
            if (decorator != SyntaxKind::Decorator)
            {
                continue;
            }

            auto expr = decorator.as<Decorator>()->expression;
            if (expr == SyntaxKind::Identifier)
            {
                auto name = getName(expr.as<Node>());
                if (name == decoratorName)
                {
                    return;
                }
            }
        }            

        node->modifiers.push_back(nf.createDecorator(nf.createIdentifier(convertUTF8toWide(decoratorName.str()))));
    }

    static std::string replaceAll(const char* source, const char* oldStr, const char* newStr)
    {
        std::string result;
        result.append(source);

        // cycle
        size_t pos = 0;
        size_t posPrev = 0;
        std::string token;
        size_t oldLen = std::string(oldStr).length();
        size_t newLen = std::string(newStr).length();

        while ((pos = result.find(oldStr, posPrev)) != std::string::npos)
        {
            result.replace(pos, oldLen, newStr);
            posPrev = pos + newLen;
        }

        return result;
    }

    // TODO: review usage of it in SizeOf, in ArrayPush, etc to return correct sizes
    static mlir::Type getElementTypeOrSelf(mlir::Type type)
    {
        if (type)
        {
            if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
            {
                return arrayType.getElementType();
            }
            else if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(type))
            {
                return constArrayType.getElementType();
            }
            else if (isa<mlir_ts::StringType>(type))
            {
                return mlir_ts::CharType::get(type.getContext());
            }
            else if (auto classType = dyn_cast<mlir_ts::ClassType>(type))
            {
                return classType.getStorageType();
            }
            else if (auto objType = dyn_cast<mlir_ts::ObjectType>(type))
            {
                return objType.getStorageType();
            }
            else if (auto refType = dyn_cast<mlir_ts::RefType>(type))
            {
                return refType.getElementType();
            }
            else if (auto boundRefType = dyn_cast<mlir_ts::BoundRefType>(type))
            {
                return boundRefType.getElementType();
            }
            else if (auto valueRefType = dyn_cast<mlir_ts::ValueRefType>(type))
            {
                return valueRefType.getElementType();
            }
        }

        return type;
    }    
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRHELPER_H_
