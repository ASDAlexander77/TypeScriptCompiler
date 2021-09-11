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
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;

  public:
    TypeOfOpHelper(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch) : op(op), rewriter(rewriter), tch(tch)
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

        if (type.isa<mlir::FunctionType>())
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

        LLVM_DEBUG(llvm::dbgs() << "TypeOf: " << type << "\n");

        llvm_unreachable("not implemented");
    }

    mlir::Value typeOfLogic(mlir::Location loc, mlir::Value value)
    {
        if (value.getType().isa<mlir_ts::AnyType>())
        {
            AnyLogic al(op, rewriter, tch, loc);
            return al.typeOfFromAny(value);
        }

        return typeOfLogic(loc, value.getType());
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEOFHELPER_H_
