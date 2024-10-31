#ifndef MLIR_TYPESCRIPT_TYPEOFHELPER_H_
#define MLIR_TYPESCRIPT_TYPEOFHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

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
            auto typeOfValue = strValue(loc, "index");
            return typeOfValue;
        }

        if (isa<mlir_ts::BooleanType>(type))
        {
            auto typeOfValue = strValue(loc, "boolean");
            return typeOfValue;
        }

        // special case
        if (isa<mlir_ts::TypePredicateType>(type))
        {
            auto typeOfValue = strValue(loc, "boolean");
            return typeOfValue;
        }        

        if (isa<mlir_ts::NumberType>(type))
        {
            auto typeOfValue = strValue(loc, "number");
            return typeOfValue;
        }

        if (isa<mlir_ts::StringType>(type))
        {
            auto typeOfValue = strValue(loc, "string");
            return typeOfValue;
        }

        if (isa<mlir_ts::ArrayType>(type))
        {
            auto typeOfValue = strValue(loc, "array");
            return typeOfValue;
        }

        if (isa<mlir_ts::FunctionType>(type))
        {
            auto typeOfValue = strValue(loc, "function");
            return typeOfValue;
        }

        if (isa<mlir_ts::HybridFunctionType>(type))
        {
            auto typeOfValue = strValue(loc, "function");
            return typeOfValue;
        }

        if (isa<mlir_ts::BoundFunctionType>(type))
        {
            auto typeOfValue = strValue(loc, "function");
            return typeOfValue;
        }

        if (isa<mlir_ts::ClassType>(type))
        {
            auto typeOfValue = strValue(loc, "class");
            return typeOfValue;
        }

        if (isa<mlir_ts::ClassStorageType>(type))
        {
            auto typeOfValue = strValue(loc, "class");
            return typeOfValue;
        }

        if (isa<mlir_ts::ObjectType>(type))
        {
            auto typeOfValue = strValue(loc, "object");
            return typeOfValue;
        }

        if (isa<mlir_ts::InterfaceType>(type))
        {
            auto typeOfValue = strValue(loc, "interface");
            return typeOfValue;
        }

        if (isa<mlir_ts::OpaqueType>(type))
        {
            auto typeOfValue = strValue(loc, "object");
            return typeOfValue;
        }

        if (isa<mlir_ts::SymbolType>(type))
        {
            auto typeOfValue = strValue(loc, "symbol");
            return typeOfValue;
        }

        if (isa<mlir_ts::UndefinedType>(type))
        {
            auto typeOfValue = strValue(loc, UNDEFINED_NAME);
            return typeOfValue;
        }

        if (isa<mlir_ts::UnknownType>(type))
        {
            auto typeOfValue = strValue(loc, "unknown");
            return typeOfValue;
        }

        if (isa<mlir_ts::ConstTupleType>(type))
        {
            auto typeOfValue = strValue(loc, "tuple");
            return typeOfValue;
        }

        if (isa<mlir_ts::TupleType>(type))
        {
            auto typeOfValue = strValue(loc, "tuple");
            return typeOfValue;
        }

        if (isa<mlir_ts::ArrayType>(type))
        {
            auto typeOfValue = strValue(loc, "array");
            return typeOfValue;
        }

        if (isa<mlir_ts::ConstArrayType>(type))
        {
            auto typeOfValue = strValue(loc, "array");
            return typeOfValue;
        }

        if (auto subType = dyn_cast<mlir_ts::RefType>(type))
        {
            return typeOfLogic(loc, subType.getElementType());
        }

        if (auto subType = dyn_cast<mlir_ts::ValueRefType>(type))
        {
            return typeOfLogic(loc, subType.getElementType());
        }

        if (auto subType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            return typeOfLogic(loc, subType.getElementType());
        }

        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            return typeOfLogic(loc, literalType.getElementType());
        }

        if (isa<mlir_ts::NullType>(type))
        {
            auto typeOfValue = strValue(loc, "null");
            return typeOfValue;
        }        

        LLVM_DEBUG(llvm::dbgs() << "TypeOf: " << type << "\n");

        llvm_unreachable("not implemented");
    }

    mlir::Value typeOfLogic(mlir::Location loc, mlir::Value value, mlir::Type origType)
    {
        if (isa<mlir_ts::AnyType>(origType))
        {
            // AnyLogic al(op, rewriter, tch, loc);
            // return al.getTypeOfAny(value);
            return rewriter.create<mlir_ts::TypeOfAnyOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), value);
        }

        if (isa<mlir_ts::UnionType>(origType))
        {
            return rewriter.create<mlir_ts::GetTypeInfoFromUnionOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), value);
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
            auto typeOfValue = typeOfLogic(loc, valueOfOpt, valueOfOpt.getType());
            rewriter.create<mlir_ts::ResultOp>(loc, typeOfValue);

            // else block
            auto &elseRegion = ifOp.getElseRegion();

            rewriter.setInsertionPointToStart(&elseRegion.back());

            auto undefStrValue = strValue(loc, UNDEFINED_NAME);
            rewriter.create<mlir_ts::ResultOp>(loc, undefStrValue);

            rewriter.setInsertionPointAfter(ifOp);

            return ifOp.getResult(0);
        }

        return typeOfLogic(loc, origType);
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_TYPEOFHELPER_H_
