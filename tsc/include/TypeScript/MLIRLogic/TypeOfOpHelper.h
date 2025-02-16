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

        llvm_unreachable("not implemented");
    }    

    mlir::Value typeOfLogic(mlir::Location loc, mlir::Type type)
    {
        return strValue(loc, typeOfAsString(type));
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
