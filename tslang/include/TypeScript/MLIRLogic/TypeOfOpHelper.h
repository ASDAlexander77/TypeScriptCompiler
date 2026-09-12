#ifndef MLIR_TYPESCRIPT_TYPEOFHELPER_H_
#define MLIR_TYPESCRIPT_TYPEOFHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "llvm/ADT/StringSwitch.h"

#define DEBUG_TYPE "mlir"

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class TypeOfOpHelper
{
    mlir::OpBuilder &rewriter;

  public:
    TypeOfOpHelper(mlir::OpBuilder &rewriter) : rewriter(rewriter)
    {
    }

    // The runtime type tag: the same string `typeOfAsString` reports, but pointing into the
    // static descriptor for `type` rather than at a bare literal, so the descriptor is
    // recoverable from any tag. Every producer of an "any" box tag or a union tag goes
    // through here - see TypeScript_TypeDescriptorOp and TYPE_DESCR_* in Defines.h.
    mlir::Value typeDescriptorValue(mlir::Location loc, mlir::Type type)
    {
        if (typeOfAsString(type).empty()) return mlir::Value();

        auto strType = mlir_ts::StringType::get(rewriter.getContext());
        return rewriter.create<mlir_ts::TypeDescriptorOp>(loc, strType, mlir::TypeAttr::get(typeOfBaseType(type)));
    }

    // The type a descriptor is actually keyed by: the wrappers `typeOfAsString` sees through
    // are stripped here too, so every string literal type shares one "string" descriptor
    // instead of minting its own. Each distinct class or object still gets its own, which is
    // the distinction the descriptor exists to preserve.
    static mlir::Type typeOfBaseType(mlir::Type type)
    {
        if (auto subType = dyn_cast<mlir_ts::RefType>(type))
        {
            return typeOfBaseType(subType.getElementType());
        }

        if (auto subType = dyn_cast<mlir_ts::ValueRefType>(type))
        {
            return typeOfBaseType(subType.getElementType());
        }

        if (auto subType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            return typeOfBaseType(subType.getElementType());
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return typeOfBaseType(literalType.getElementType());
        }

        return type;
    }

    // Coarse category for a name produced by `typeOfAsString`. Deriving the kind from the
    // name (rather than re-switching over the type) is what keeps the two from drifting
    // apart as `typeOfAsString` grows cases.
    static int typeKindFromName(llvm::StringRef name)
    {
        // "s32", "u64", "f64", "i1", ... - a numeric-width tag, as opposed to a name that
        // merely starts with one of those letters ("interface", "symbol", "function").
        if (name.size() > 1 && (name[0] == 'i' || name[0] == 's' || name[0] == 'u' || name[0] == 'f') &&
            llvm::all_of(name.drop_front(), [](char c) { return c >= '0' && c <= '9'; }))
        {
            return TYPE_KIND_NUMBER;
        }

        return llvm::StringSwitch<int>(name)
            .Case("number", TYPE_KIND_NUMBER)
            .Case("index", TYPE_KIND_NUMBER)
            .Case("string", TYPE_KIND_STRING)
            .Case("boolean", TYPE_KIND_BOOLEAN)
            .Case("char", TYPE_KIND_CHAR)
            .Case("array", TYPE_KIND_ARRAY)
            .Case("tuple", TYPE_KIND_TUPLE)
            .Case("object", TYPE_KIND_OBJECT)
            .Case("class", TYPE_KIND_CLASS)
            .Case("interface", TYPE_KIND_INTERFACE)
            .Case("function", TYPE_KIND_FUNCTION)
            .Case("symbol", TYPE_KIND_SYMBOL)
            .Case(UNDEFINED_NAME, TYPE_KIND_UNDEFINED)
            .Case("null", TYPE_KIND_NULL)
            .Default(TYPE_KIND_UNKNOWN);
    }

    std::string typeOfAsString(mlir::Type type)
    {
        if (type.isIntOrIndex() && !type.isIndex())
        {
            std::stringstream val;
            val << (type.isSignlessInteger() ? "i" : type.isSignedInteger() ? "s" : "u") << type.getIntOrFloatBitWidth();
            return val.str();
        }

        if (type.isIntOrFloat() && !type.isIntOrIndex())
        {
            std::stringstream val;
            val << "f" << type.getIntOrFloatBitWidth();
            return val.str();
        }

        if (type.isIndex())
        {
            return "index";
        }

        if (isa<mlir_ts::BooleanType>(type))
        {
            return "boolean";
        }

        // special case
        if (isa<mlir_ts::TypePredicateType>(type))
        {
            return "boolean";
        }        

        if (isa<mlir_ts::NumberType>(type))
        {
            return "number";
        }

        if (isa<mlir_ts::StringType>(type))
        {
            return "string";
        }

        if (isa<mlir_ts::ArrayType>(type))
        {
            return "array";
        }

        if (isa<mlir_ts::FunctionType>(type))
        {
            return "function";
        }

        if (isa<mlir_ts::HybridFunctionType>(type))
        {
            return "function";
        }

        if (isa<mlir_ts::BoundFunctionType>(type))
        {
            return "function";
        }

        if (isa<mlir_ts::ClassType>(type))
        {
            return "class";
        }

        if (isa<mlir_ts::ClassStorageType>(type))
        {
            return "class";
        }

        if (isa<mlir_ts::ObjectType>(type))
        {
            return "object";
        }

        if (isa<mlir_ts::InterfaceType>(type))
        {
            return "interface";
        }

        if (isa<mlir_ts::OpaqueType>(type))
        {
            return "object";
        }

        if (isa<mlir_ts::SymbolType>(type))
        {
            return "symbol";
        }

        if (isa<mlir_ts::UndefinedType>(type))
        {
            return UNDEFINED_NAME;
        }

        if (isa<mlir_ts::UnknownType>(type))
        {
            return "unknown";
        }

        if (isa<mlir_ts::ConstTupleType>(type))
        {
            return "tuple";
        }

        if (isa<mlir_ts::TupleType>(type))
        {
            return "tuple";
        }

        if (isa<mlir_ts::ArrayType>(type))
        {
            return "array";
        }

        if (isa<mlir_ts::ConstArrayType>(type))
        {
            return "array";
        }

        if (auto subType = dyn_cast<mlir_ts::RefType>(type))
        {
            return typeOfAsString(subType.getElementType());
        }

        if (auto subType = dyn_cast<mlir_ts::ValueRefType>(type))
        {
            return typeOfAsString(subType.getElementType());
        }

        if (auto subType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            return typeOfAsString(subType.getElementType());
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return typeOfAsString(literalType.getElementType());
        }

        if (isa<mlir_ts::NullType>(type))
        {
            return "null";
        }        

        if (isa<mlir_ts::CharType>(type))
        {
            return "char";
        }

        LLVM_DEBUG(llvm::dbgs() << "TypeOf: " << type << "\n");

        return "";
    }    

    mlir::Value typeOfLogic(mlir::Location loc, mlir::Type type)
    {
        return typeDescriptorValue(loc, type);
    }

    mlir::Value typeOfLogic(mlir::Location loc, mlir::Value value, mlir::Type origType, CompileOptions& compileOptions)
    {
        if (isa<mlir_ts::AnyType>(origType))
        {
            // AnyLogic al(op, rewriter, tch, loc);
            // return al.getTypeOfAny(value);
            return rewriter.create<mlir_ts::TypeOfAnyOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), value);
        }

        if (auto unionType = dyn_cast<mlir_ts::UnionType>(origType))
        {
            MLIRTypeHelper mth(rewriter.getContext(), compileOptions);

            mlir::Type baseType;
            bool needTag = mth.isUnionTypeNeedsTag(loc, unionType, baseType);
            if (needTag)
            {
                return rewriter.create<mlir_ts::GetTypeInfoFromUnionOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), value);
            }

            origType = baseType;
        }

        if (auto subType = dyn_cast<mlir_ts::OptionalType>(origType))
        {
            auto dataTypeIn = subType.getElementType();
            auto resultType = mlir_ts::StringType::get(value.getContext());

            // ts.if
            auto hasValue = rewriter.create<mlir_ts::HasValueOp>(loc, mlir_ts::BooleanType::get(value.getContext()), value);
            auto ifOp = rewriter.create<mlir_ts::IfOp>(loc, resultType, hasValue, true);

            // then block
            auto &thenRegion = ifOp.getThenRegion();

            rewriter.setInsertionPointToStart(&thenRegion.back());

            mlir::Value valueOfOpt = rewriter.create<mlir_ts::ValueOp>(loc, subType.getElementType(), value);
            auto typeOfValue = typeOfLogic(loc, valueOfOpt, valueOfOpt.getType(), compileOptions);
            rewriter.create<mlir_ts::ResultOp>(loc, typeOfValue);

            // else block
            auto &elseRegion = ifOp.getElseRegion();

            rewriter.setInsertionPointToStart(&elseRegion.back());

            // goes through a descriptor like every other tag, so an "any" holding an empty
            // optional carries a recoverable descriptor rather than a bare literal
            auto undefStrValue = typeDescriptorValue(loc, mlir_ts::UndefinedType::get(rewriter.getContext()));
            rewriter.create<mlir_ts::ResultOp>(loc, undefStrValue);

            rewriter.setInsertionPointAfter(ifOp);

            return ifOp.getResult(0);
        }

        return typeOfLogic(loc, origType);
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_TYPEOFHELPER_H_
