#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEOFHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEOFHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class TypeOfOpHelper
{
    PatternRewriter &rewriter;

  public:
    TypeOfOpHelper(PatternRewriter &rewriter) : rewriter(rewriter)
    {
    }

    mlir::Value strValue(mlir::Location loc, std::string value)
    {
        auto strType = mlir_ts::StringType::get(rewriter.getContext());
        auto typeOfValue = rewriter.create<mlir_ts::ConstantOp>(loc, strType, rewriter.getStringAttr(value));
        return typeOfValue;
    }

    mlir::Value typeOfLogic(mlir::Location loc, mlir::Type type)
    {
        if (type.isIntOrIndex() && !type.isIndex())
        {
            std::stringstream val;
            val << "i" << type.getIntOrFloatBitWidth();
            auto typeOfValue = strValue(loc, val.str());
            return typeOfValue;
        }

        if (type.isIntOrFloat() && !type.isIntOrIndex())
        {
            std::stringstream val;
            val << "f" << type.getIntOrFloatBitWidth();
            auto typeOfValue = strValue(loc, val.str());
            return typeOfValue;
        }

        if (type.isIndex())
        {
            auto typeOfValue = strValue(loc, "ptrint");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::BooleanType>())
        {
            auto typeOfValue = strValue(loc, "boolean");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::NumberType>())
        {
            auto typeOfValue = strValue(loc, "number");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::StringType>())
        {
            auto typeOfValue = strValue(loc, "string");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::ArrayType>())
        {
            auto typeOfValue = strValue(loc, "array");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::FunctionType>())
        {
            auto typeOfValue = strValue(loc, "function");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::BoundFunctionType>())
        {
            auto typeOfValue = strValue(loc, "function");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::ClassType>())
        {
            auto typeOfValue = strValue(loc, "class");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::ClassStorageType>())
        {
            auto typeOfValue = strValue(loc, "class");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::ObjectType>())
        {
            auto typeOfValue = strValue(loc, "object");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::InterfaceType>())
        {
            auto typeOfValue = strValue(loc, "interface");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::OpaqueType>())
        {
            auto typeOfValue = strValue(loc, "object");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::SymbolType>())
        {
            auto typeOfValue = strValue(loc, "symbol");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::UndefinedType>())
        {
            auto typeOfValue = strValue(loc, "undefined");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::UnknownType>())
        {
            auto typeOfValue = strValue(loc, "unknown");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::ConstTupleType>())
        {
            auto typeOfValue = strValue(loc, "tuple");
            return typeOfValue;
        }

        if (type.isa<mlir_ts::TupleType>())
        {
            auto typeOfValue = strValue(loc, "tuple");
            return typeOfValue;
        }

        if (auto subType = type.dyn_cast_or_null<mlir_ts::RefType>())
        {
            return typeOfLogic(loc, subType.getElementType());
        }

        if (auto subType = type.dyn_cast_or_null<mlir_ts::ValueRefType>())
        {
            return typeOfLogic(loc, subType.getElementType());
        }

        if (auto subType = type.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            return typeOfLogic(loc, subType.getElementType());
        }

        if (auto literalType = type.dyn_cast_or_null<mlir_ts::LiteralType>())
        {
            return typeOfLogic(loc, literalType.getElementType());
        }

        LLVM_DEBUG(llvm::dbgs() << "TypeOf: " << type << "\n");

        llvm_unreachable("not implemented");
    }

    mlir::Value typeOfLogic(mlir::Location loc, mlir::Value value, mlir::Type origType)
    {
        if (origType.isa<mlir_ts::AnyType>())
        {
            // AnyLogic al(op, rewriter, tch, loc);
            // return al.typeOfFromAny(value);
            return rewriter.create<mlir_ts::TypeOfAnyOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), value);
        }

        if (origType.isa<mlir_ts::UnionType>())
        {
            return rewriter.create<mlir_ts::GetTypeInfoFromUnionOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), value);
        }

        if (auto subType = origType.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            auto valueOfOpt = rewriter.create<mlir_ts::ValueOp>(loc, subType.getElementType(), value);
            return typeOfLogic(loc, valueOfOpt, subType.getElementType());
        }

        return typeOfLogic(loc, origType);
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEOFHELPER_H_
