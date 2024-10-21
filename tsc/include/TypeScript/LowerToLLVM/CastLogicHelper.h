#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CASTLOGICHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CASTLOGICHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"
#include "TypeScript/LowerToLLVM/ConvertLogic.h"
#include "TypeScript/LowerToLLVM/AnyLogic.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#define DEBUG_TYPE "llvm"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class CastLogicHelper
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;
    TypeHelper th;
    LLVMCodeHelperBase ch;
    CodeLogicHelper clh;
    Location loc;
    CompileOptions &compileOptions;
    bool external;

  public:
    CastLogicHelper(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, &tch.typeConverter, compileOptions), clh(op, rewriter), loc(op->getLoc()),
          compileOptions(compileOptions), external(false)
    {
    }

    void setExternal()
    {
        external = true;
    }

    mlir::Value cast(mlir::Value in, mlir::Type inType, mlir::Type resType)
    {
        auto inLLVMType = tch.convertType(inType);
        auto resLLVMType = tch.convertType(resType);
        return cast(in, inType, inLLVMType, resType, resLLVMType);
    }

    mlir::Value cast(mlir::Value in, mlir::Type inType, mlir::Type inLLVMType, mlir::Type resType, mlir::Type resLLVMType)
    {
        auto val = castTypeScriptTypes(in, inType, inLLVMType, resType, resLLVMType);
        if (val)
        {
            return val;
        }

        return castLLVMTypes(in, inLLVMType, resType, resLLVMType);
    }

    mlir::Value castTypeScriptTypes(mlir::Value in, mlir::Type inType, mlir::Type inLLVMType, mlir::Type resType, mlir::Type resLLVMType)
    {
        if (inType == resType)
        {
            return in;
        }

        if (auto literalType = inType.dyn_cast<mlir_ts::LiteralType>())
        {
            return cast(in, literalType.getElementType(), tch.convertType(literalType.getElementType()), resType, resLLVMType);
        }

        if (inType.isa<mlir_ts::CharType>() && resType.isa<mlir_ts::StringType>())
        {
            // types are equals
            return rewriter.create<mlir_ts::CharToStringOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
        }

        auto isResString = resType.isa<mlir_ts::StringType>();
        if (inLLVMType.isInteger(1) && isResString)
        {
            return castBoolToString(in);
        }

        if (inType.isIndex() && isResString)
        {
            return castIntToString(in, tch.getIndexTypeBitwidth(), false);
        }

        if (inType.isa<mlir::IntegerType>() && isResString)
        {
            return castIntToString(in, inLLVMType.getIntOrFloatBitWidth(), inType.isSignedInteger());
        }

        if ((inLLVMType.isF16() || inLLVMType.isF32() || inLLVMType.isF64() || inLLVMType.isF128()) && isResString)
        {
            if (inLLVMType.isF16())
            {
                in = cast(in, inType, rewriter.getF64Type());
            }
            else if (inLLVMType.isF32())
            {
                in = cast(in, inType, rewriter.getF64Type());
            }
            else if (inLLVMType.isF128())
            {
                in = cast(in, inType, rewriter.getF64Type());
            }

            return castF64ToString(in);
        }

        if (inType.isIndex())
        {
            if (resType.isSignedInteger())
            {
                return rewriter.create<mlir::index::CastSOp>(loc, resLLVMType, in);
            }
            else
            {
                return rewriter.create<mlir::index::CastUOp>(loc, resLLVMType, in);
            }
        }

        if (inType.isSignedInteger() && resType.isSignedInteger() && resType.getIntOrFloatBitWidth() > inType.getIntOrFloatBitWidth())
        {
            return rewriter.create<LLVM::SExtOp>(loc, resLLVMType, in);
        }        

        auto isResAny = resType.isa<mlir_ts::AnyType>();
        if (isResAny)
        {
            return castToAny(in, inType, inLLVMType);
        }

        auto isInAny = inType.isa<mlir_ts::AnyType>();
        if (isInAny)
        {
            return castFromAny(in, resType);
        }

        if (auto numberType = resType.dyn_cast<mlir_ts::NumberType>())
        {
            if (auto boolType = inType.dyn_cast<mlir_ts::BooleanType>())
            {
                return castBoolToNumber(in);
            }
        }

        if (auto obj = resType.dyn_cast<mlir_ts::ObjectType>())
        {
            if (obj.getStorageType().isa<mlir_ts::AnyType>())
            {
                return castToOpaqueType(in, inLLVMType);
            }
        }

        if (auto obj = resType.dyn_cast<mlir_ts::UnknownType>())
        {
            return castToOpaqueType(in, inLLVMType);
        }

        auto isInString = inType.dyn_cast<mlir_ts::StringType>();
        if (isInString && (resLLVMType.isInteger(32) || resLLVMType.isInteger(64)))
        {
            auto castIntOp = rewriter.create<mlir_ts::ParseIntOp>(loc, resType, in);
            return rewriter.create<mlir_ts::DialectCastOp>(loc, resLLVMType, castIntOp);
        }

        if (isInString && (resLLVMType.isF32() || resLLVMType.isF64()))
        {
            auto castNumberOp = rewriter.create<mlir_ts::ParseFloatOp>(loc, resType, in);
            return rewriter.create<mlir_ts::DialectCastOp>(loc, resLLVMType, castNumberOp);
        }

        // array to ref of element
        if (auto arrayType = inType.dyn_cast<mlir_ts::ArrayType>())
        {
            if (auto refType = resType.dyn_cast<mlir_ts::RefType>())
            {
                if (arrayType.getElementType() == refType.getElementType())
                {
                    return rewriter.create<LLVM::ExtractValueOp>(loc, resLLVMType, in,
                        MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
                }
            }

            if (auto opaqueType = resType.dyn_cast<mlir_ts::OpaqueType>())
            {
                auto ptrOfElementValue = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPointerType(tch.convertType(arrayType.getElementType())), in,
                    MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));                

                return rewriter.create<LLVM::BitcastOp>(loc, th.getI8PtrType(), ptrOfElementValue);
            }            
        }

        if (auto resFuncType = resType.dyn_cast<mlir_ts::FunctionType>())
        {
            if (auto inBoundFunc = inType.dyn_cast<mlir_ts::BoundFunctionType>())
            {
                // somehow llvm.trampoline accepts only direct method symbol
                /*
                auto thisVal = rewriter.create<mlir_ts::GetThisOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()), in);
                auto methodVal = rewriter.create<mlir_ts::GetMethodOp>(loc, resFuncType, in);
                return rewriter.create<mlir_ts::TrampolineOp>(loc, resFuncType, methodVal, thisVal);
                */
                op->emitWarning("losing this reference");
                /*
                // you can wrap into () => {} lambda call to capture vars
                const user = {
                    firstName: "World",
                    sayHi() {
                        print(`Hello ${this.firstName}`);
                    },
                };

                let hi2 = user.sayHi;
                call_func_1(() => {
                     hi2();
                });
                */
                return rewriter.create<mlir_ts::GetMethodOp>(loc, resFuncType, in);
            }
        }

        if (auto resHybridFunc = resType.dyn_cast<mlir_ts::HybridFunctionType>())
        {
            if (auto inFuncType = inType.dyn_cast<mlir_ts::FunctionType>())
            {
                // BoundFunction is the same as HybridFunction
                // null this
                auto thisNullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(rewriter.getContext()));
                auto boundFuncVal = rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, resHybridFunc, thisNullVal, in);
                return boundFuncVal;
            }
        }

        if (auto resRefType = resType.dyn_cast<mlir_ts::RefType>())
        {
            if (auto inBoundRef = inType.dyn_cast<mlir_ts::BoundRefType>())
            {
                return castBoundRefToRef(in, inBoundRef, resRefType);
            }
        }

        if (auto tupleTypeRes = resType.dyn_cast<mlir_ts::TupleType>())
        {
            if (auto tupleTypeIn = inType.dyn_cast<mlir_ts::ConstTupleType>())
            {
                return castTupleToTuple(in, tupleTypeIn.getFields(), tupleTypeRes);
            }
            if (auto tupleTypeIn = inType.dyn_cast<mlir_ts::TupleType>())
            {
                return castTupleToTuple(in, tupleTypeIn.getFields(), tupleTypeRes);
            }
        }

        if (auto nullType = inType.dyn_cast<mlir_ts::NullType>())
        {
            if (auto ifaceType = resType.dyn_cast<mlir_ts::InterfaceType>())
            {
                // create null interface
                return rewriter.create<mlir_ts::NewInterfaceOp>(loc, ifaceType, in, in);
            }

            if (auto resHybridFunc = resType.dyn_cast<mlir_ts::HybridFunctionType>())
            {
                // null this
                auto thisNullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(rewriter.getContext()));
                auto funcType = mlir_ts::FunctionType::get(rewriter.getContext(), resHybridFunc.getInputs(), resHybridFunc.getResults(), resHybridFunc.isVarArg());
                auto castFuncNullVal = cast(in, inType, funcType);
                auto boundFuncVal = rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, resHybridFunc, thisNullVal, castFuncNullVal);
                return boundFuncVal;
            }
        }

        if (auto stringTypeRes = resType.dyn_cast<mlir_ts::StringType>())
        {
            if (auto tupleTypeIn = inType.dyn_cast<mlir_ts::ConstTupleType>())
            {
                return castTupleToString<mlir_ts::ConstTupleType>(in, inType, tupleTypeIn);
            }

            if (auto tupleTypeIn = inType.dyn_cast<mlir_ts::TupleType>())
            {
                return castTupleToString<mlir_ts::TupleType>(in, inType, tupleTypeIn);
            }
        }

        if (auto interfaceTypeRes = resType.dyn_cast<mlir_ts::InterfaceType>())
        {
            if (auto tupleTypeIn = inType.dyn_cast<mlir_ts::ConstTupleType>())
            {
                llvm_unreachable("not implemented, must be processed at MLIR pass");
            }

            if (auto tupleTypeIn = inType.dyn_cast<mlir_ts::TupleType>())
            {
                llvm_unreachable("not implemented, must be processed at MLIR pass");
            }
        }

        if (auto undefType = inType.dyn_cast<mlir_ts::UndefinedType>())
        {
            if (auto stringTypeRes = resType.dyn_cast<mlir_ts::StringType>())
            {
                return rewriter.create<mlir_ts::ConstantOp>(loc, stringTypeRes, rewriter.getStringAttr(UNDEFINED_NAME));
            }
        }

        if (auto boolType = resType.dyn_cast<mlir_ts::BooleanType>())
        {
            if (auto tupleType = inType.dyn_cast<mlir_ts::TupleType>())
            {
                auto llvmBoolType = tch.convertType(boolType);
                return rewriter.create<mlir_ts::ConstantOp>(loc, llvmBoolType, rewriter.getBoolAttr(true));
            }

            if (auto constTupleType = inType.dyn_cast<mlir_ts::ConstTupleType>())
            {
                auto llvmBoolType = tch.convertType(boolType);
                return rewriter.create<mlir_ts::ConstantOp>(loc, llvmBoolType, rewriter.getBoolAttr(true));
            }

            if (auto ifaceType = inType.dyn_cast<mlir_ts::InterfaceType>())
            {
                auto ptrValue =
                    rewriter.create<mlir_ts::ExtractInterfaceVTableOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()), in);
                auto inLLVMType = tch.convertType(ptrValue.getType());
                auto llvmBoolType = tch.convertType(boolType);
                return castLLVMTypes(ptrValue, inLLVMType, boolType, llvmBoolType);
            }

            if (auto hybridFuncType = inType.dyn_cast<mlir_ts::HybridFunctionType>())
            {
                auto funcType = mlir_ts::FunctionType::get(rewriter.getContext(), hybridFuncType.getInputs(), hybridFuncType.getResults(), hybridFuncType.isVarArg());
                auto ptrValue = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, in);
                auto inLLVMType = tch.convertType(ptrValue.getType());
                auto llvmBoolType = tch.convertType(boolType);
                return castLLVMTypes(ptrValue, inLLVMType, boolType, llvmBoolType);
            }

            if (auto boundFuncType = inType.dyn_cast<mlir_ts::BoundFunctionType>())
            {
                auto funcType = mlir_ts::FunctionType::get(rewriter.getContext(), boundFuncType.getInputs(), boundFuncType.getResults(), boundFuncType.isVarArg());
                auto ptrValue = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, in);
                auto inLLVMType = tch.convertType(ptrValue.getType());
                auto llvmBoolType = tch.convertType(boolType);
                return castLLVMTypes(ptrValue, inLLVMType, boolType, llvmBoolType);
            }

            if (auto funcType = inType.dyn_cast<mlir_ts::FunctionType>())
            {
                auto llvmBoolType = tch.convertType(boolType);
                return castLLVMTypes(in, inLLVMType, boolType, llvmBoolType);
            }

            if (auto optType = inType.dyn_cast<mlir_ts::OptionalType>())
            {
                // TODO: use cond switch
                auto hasValue = rewriter.create<mlir_ts::HasValueOp>(loc, boolType, in);
                auto val = rewriter.create<mlir_ts::ValueOp>(loc, optType.getElementType(), in);
                auto llvmBoolType = tch.convertType(boolType);
                auto valAsBool = cast(val, val.getType(), tch.convertType(val.getType()), boolType, llvmBoolType);
                if (valAsBool)
                {
                    mlir::Value hasValueAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoolType, hasValue);
                    mlir::Value valAsBoolAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoolType, valAsBool);
                    return rewriter.create<LLVM::AndOp>(loc, llvmBoolType, hasValueAsLLVMType, valAsBoolAsLLVMType);
                }
                else
                {
                    return hasValue;
                }
            }

            if (auto arrayType = inType.dyn_cast<mlir_ts::ArrayType>())
            {
                auto ptrValue = extractArrayPtr(in, arrayType);
                auto inLLVMType = tch.convertType(ptrValue.getType());
                auto llvmBoolType = tch.convertType(boolType);
                return castLLVMTypes(ptrValue, inLLVMType, boolType, llvmBoolType);
            }            

            if (auto unionType = inType.dyn_cast<mlir_ts::UnionType>())
            {
                MLIRTypeHelper mth(unionType.getContext());
                mlir::Type baseType;
                bool needTag = mth.isUnionTypeNeedsTag(unionType, baseType);
                if (!needTag)
                {
                    auto llvmBoolType = tch.convertType(boolType);
                    auto valAsBool = cast(in, baseType, tch.convertType(baseType), boolType, llvmBoolType);
                    return valAsBool;
                }
                else
                {
                    // TODO: finish it, union type has RTTI field, test it first
                    llvm_unreachable("not implemented");
                }
            }
        }

        // cast value value to optional value
        if (auto optType = resType.dyn_cast<mlir_ts::OptionalType>())
        {
            if (inType.isa<mlir_ts::UndefinedType>())
            {
                return rewriter.create<mlir_ts::OptionalUndefOp>(loc, resType);
            }

            auto valCasted = cast(in, inType, inLLVMType, optType.getElementType(), tch.convertType(optType.getElementType()));
            return rewriter.create<mlir_ts::OptionalValueOp>(loc, resType, valCasted);
        }

        if (auto optType = inType.dyn_cast<mlir_ts::OptionalType>())
        {
            auto val = rewriter.create<mlir_ts::ValueOrDefaultOp>(loc, optType.getElementType(), in);
            return cast(val, val.getType(), tch.convertType(val.getType()), resType, resLLVMType);
        }

        if (inType.isa<mlir_ts::UndefinedType>())
        {
            if (auto ifaceType = resType.dyn_cast<mlir_ts::InterfaceType>())
            {
                // create null interface
                auto nullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(ifaceType.getContext()));
                return rewriter.create<mlir_ts::NewInterfaceOp>(loc, ifaceType, nullVal, nullVal);
            }

            if (auto classType = resType.dyn_cast<mlir_ts::ClassType>())
            {
                // create null class
                auto nullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(classType.getContext()));
                return cast(nullVal, nullVal.getType(), tch.convertType(nullVal.getType()), resType, resLLVMType);
            }
        }

        if (auto arrType = resType.dyn_cast<mlir_ts::ArrayType>())
        {
            return castToArrayType(in, inType, resType);
        }

        if (auto opaqueType = resType.dyn_cast<mlir_ts::OpaqueType>())
        {
            if (auto ifaceType = inType.dyn_cast<mlir_ts::InterfaceType>())
            {
                auto ptrValue = rewriter.create<mlir_ts::ExtractInterfaceThisOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()), in);
                return ptrValue;
            }

            if (auto hybridFuncType = inType.dyn_cast<mlir_ts::HybridFunctionType>())
            {
                auto funcType = mlir_ts::FunctionType::get(rewriter.getContext(), hybridFuncType.getInputs(), hybridFuncType.getResults(), hybridFuncType.isVarArg());
                auto ptrValue = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, in);

                auto ptrValueAdapt = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(ptrValue.getType()), ptrValue);
                auto bitcast = rewriter.create<LLVM::BitcastOp>(loc, tch.convertType(opaqueType), ptrValueAdapt);
                return bitcast;
            }

            if (auto boundFuncType = inType.dyn_cast<mlir_ts::BoundFunctionType>())
            {
                auto funcType = mlir_ts::FunctionType::get(rewriter.getContext(), boundFuncType.getInputs(), boundFuncType.getResults(), boundFuncType.isVarArg());
                auto ptrValue = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, in);

                auto ptrValueAdapt = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(ptrValue.getType()), ptrValue);
                auto bitcast = rewriter.create<LLVM::BitcastOp>(loc, tch.convertType(opaqueType), ptrValueAdapt);
                return bitcast;
            }
        }

        /*
        // TODO: we do not need as struct can cast to struct
        if (auto inUnionType = inType.dyn_cast<mlir_ts::UnionType>())
        {
            if (auto resUnionType = resType.dyn_cast<mlir_ts::UnionType>())
            {
                LLVMTypeConverterHelper ltch((LLVMTypeConverter &)tch.typeConverter);
                auto maxStoreType = ltch.findMaxSizeType(inUnionType);
                auto value = rewriter.create<mlir_ts::GetValueFromUnionOp>(loc, maxStoreType, in);
                auto typeOfValue =
                    rewriter.create<mlir_ts::GetTypeInfoFromUnionOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
                auto unionValue = rewriter.create<mlir_ts::CreateUnionInstanceOp>(loc, resType, value, typeOfValue);
                return unionValue;
            }
        }
        */

        if (auto resUnionType = resType.dyn_cast<mlir_ts::UnionType>())
        {
            // TODO: do I need to test income types?
            if (auto inUnionType = inType.dyn_cast<mlir_ts::UnionType>())
            {
                // nothing to do
            }
            else
            {
                MLIRTypeHelper mth(resUnionType.getContext());
                mlir::Type baseType;
                bool needTag = mth.isUnionTypeNeedsTag(resUnionType, baseType);
                if (needTag)
                {
                    auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
                    auto unionValue = rewriter.create<mlir_ts::CreateUnionInstanceOp>(loc, resUnionType, in, typeOfValue);
                    return unionValue;
                }
                else
                {
                    return cast(in, inType, tch.convertType(inType), baseType, tch.convertType(baseType));
                }
            }
        }

        if (auto inUnionType = inType.dyn_cast<mlir_ts::UnionType>())
        {
            MLIRTypeHelper mth(inUnionType.getContext());
            mlir::Type baseType;
            bool needTag = mth.isUnionTypeNeedsTag(inUnionType, baseType);
            if (!needTag)
            {
                return cast(in, baseType, tch.convertType(baseType), resType, resLLVMType);
            }

            // skip to next steps
        }       

        if (auto undefType = inType.dyn_cast<mlir_ts::UndefinedType>())
        {
            in.getDefiningOp()->emitWarning("using casting to undefined value");
            return rewriter.create<mlir_ts::UndefOp>(loc, resType);
        }   

        return mlir::Value();
    }

    bool isBool(mlir::Type type)
    {
        return type.isInteger(1);
    }

    bool isIntOrBool(mlir::Type type)
    {
        return type.isIntOrIndex() && !type.isIndex();
    }

    bool isInt(mlir::Type type)
    {
        return isIntOrBool(type) && !isBool(type);
    }

    bool isFloat(mlir::Type type)
    {
        return type.isIntOrFloat() && !isIntOrBool(type);
    }

    std::pair<mlir::Value, bool> dialectCast(mlir::Value in, mlir::Type inType, mlir::Type resType)
    {
        auto inLLVMType = tch.convertType(inType);
        auto resLLVMType = tch.convertType(resType);
        if (inLLVMType == resLLVMType)
        {
            return {mlir::Value(), false};
        }

        return {castLLVMTypesLogic(in, inLLVMType, resLLVMType, true), true};
    }

    mlir::Value castLLVMTypesLogic(mlir::Value inParam, mlir::Type inLLVMType, mlir::Type resLLVMType, bool skipDialectCast = false)
    {
        if (inLLVMType == resLLVMType)
        {
            return inParam;
        }

        mlir::Value in = inParam;
        if (!skipDialectCast)
        {
            if (inLLVMType != in.getType())
            {
                in = rewriter.create<mlir_ts::DialectCastOp>(loc, inLLVMType, inParam);
            }
        }

        if (isInt(inLLVMType) && isFloat(resLLVMType))
        {
            return rewriter.create<mlir::arith::SIToFPOp>(loc, resLLVMType, in);
        }

        if (isFloat(inLLVMType) && isInt(resLLVMType))
        {
            return rewriter.create<mlir::arith::FPToSIOp>(loc, resLLVMType, in);
        }

        if (isInt(inLLVMType) && isBool(resLLVMType))
        {
            return rewriter.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::ne, in, clh.createIConstantOf(inLLVMType.getIntOrFloatBitWidth(), 0));
        }

        if (isFloat(inLLVMType) && isBool(resLLVMType))
        {
            return rewriter.create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::ONE, in, clh.createFConstantOf(inLLVMType.getIntOrFloatBitWidth(), 0.0));
        }

        if (inLLVMType.isa<LLVM::LLVMPointerType>() && isBool(resLLVMType))
        {            
            auto intVal = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), in);
            return rewriter.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::ne, intVal, clh.createI64ConstantOf(0));
        }

        if (inLLVMType.isa<LLVM::LLVMPointerType>() && isInt(resLLVMType))
        {
            return rewriter.create<LLVM::PtrToIntOp>(loc, resLLVMType, in);
        }

        if (inLLVMType.isa<LLVM::LLVMPointerType>() && isFloat(resLLVMType))
        {
            auto intVal = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), in);
            return rewriter.create<mlir::arith::SIToFPOp>(loc, resLLVMType, intVal);
        }

        if (isInt(inLLVMType) && resLLVMType.isa<LLVM::LLVMPointerType>())
        {
            return rewriter.create<LLVM::IntToPtrOp>(loc, resLLVMType, in);
        }

        if (isIntOrBool(inLLVMType) && isInt(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() < resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<LLVM::ZExtOp>(loc, resLLVMType, in);
        }

        if (isInt(inLLVMType) && isInt(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() > resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<LLVM::TruncOp>(loc, resLLVMType, in);
        }

        if (isFloat(inLLVMType) && isFloat(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() < resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<LLVM::FPExtOp>(loc, resLLVMType, in);
        }

        if (isFloat(inLLVMType) && isFloat(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() > resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<LLVM::FPTruncOp>(loc, resLLVMType, in);
        }

        // ptrs cast
        if (inLLVMType.isa<LLVM::LLVMPointerType>() && resLLVMType.isa<LLVM::LLVMPointerType>())
        {
            return rewriter.create<LLVM::BitcastOp>(loc, resLLVMType, in);
        }

        return mlir::Value();
    }

    mlir::Value castLLVMTypes(mlir::Value in, mlir::Type inLLVMType, mlir::Type resType, mlir::Type resLLVMType)
    {
        if (inLLVMType == resLLVMType)
        {
            return in;
        }

        auto res = castLLVMTypesLogic(in, inLLVMType, resLLVMType);
        if (res)
        {
            return res;
        }

        auto inType = in.getType();

        // review usage of ts.Type here
        // struct to struct. TODO: add validation
        if (inLLVMType.isa<LLVM::LLVMStructType>() && resLLVMType.isa<LLVM::LLVMStructType>())
        {
            LLVMTypeConverterHelper llvmtch((LLVMTypeConverter &)tch.typeConverter);
            auto srcSize = llvmtch.getTypeSizeEstimateInBytes(inLLVMType);
            auto dstSize = llvmtch.getTypeSizeEstimateInBytes(resLLVMType);

            if (srcSize != dstSize)
            {
                op->emitWarning("types have different sizes:\n ")
                    << inLLVMType << " size of #" << srcSize << ",\n " << resLLVMType << " size of #" << dstSize;
            }

            auto srcAddr = rewriter.create<mlir_ts::VariableOp>(
                loc, mlir_ts::RefType::get(inType), in, rewriter.getBoolAttr(false), rewriter.getIndexAttr(0));
            auto dstAddr = rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(resType), 
                mlir::Value(), rewriter.getBoolAttr(false), rewriter.getIndexAttr(0));
            if (srcSize <= 8 && dstSize <= 8)
            {
                rewriter.create<mlir_ts::LoadSaveOp>(loc, dstAddr, srcAddr);
            }
            else
            {
                rewriter.create<mlir_ts::CopyStructOp>(loc, dstAddr, srcAddr);
            }

            auto val = rewriter.create<mlir_ts::LoadOp>(loc, resType, dstAddr);
            return val;
        }

        // value to ref of value
        if (auto destPtr = resLLVMType.dyn_cast<LLVM::LLVMPointerType>())
        {
            if (destPtr.getElementType() == inLLVMType)
            {
                // alloc and return address
                auto valueAddr = rewriter.create<mlir_ts::VariableOp>(
                    loc, mlir_ts::RefType::get(inType), in, rewriter.getBoolAttr(false), rewriter.getIndexAttr(0));
                return valueAddr;
            }
        }

        LLVM_DEBUG(llvm::dbgs() << "invalid cast operator type 1: '" << inLLVMType << "', type 2: '" << resLLVMType << "'\n";);

        // TODO: we return undef bacause if "conditional compiling" we can have non compilable code with "cast" to bypass it we need to retun precompiled value
        emitWarning(loc, "invalid cast from ") << inLLVMType << " to " << resLLVMType;
        return rewriter.create<LLVM::UndefOp>(loc, resLLVMType);
        //emitError(loc, "invalid cast from ") << inLLVMType << " to " << resLLVMType;
        //return mlir::Value();
    }

    mlir::Value castTupleToTuple(mlir::Value in, ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields, mlir_ts::TupleType tupleTypeRes)
    {
        SmallVector<mlir::Type> types;
        for (auto &field : fields)
        {
            types.push_back(field.type);
        }

        auto results = rewriter.create<mlir_ts::DeconstructTupleOp>(loc, types, in);
        auto resultsCount = results.getNumResults();
        mlir::SmallVector<mlir::Value> mappedValues;

        auto error = false;
        auto addByIndex = [&](auto dstIndex, auto destField) {
            if (resultsCount > (unsigned) dstIndex)
            {
                mlir::Value srcValue = results.getResults()[dstIndex];
                if (srcValue.getType() != destField.type)
                {
                    srcValue = cast(srcValue, srcValue.getType(), tch.convertType(srcValue.getType()), destField.type,
                                    tch.convertType(destField.type));
                }

                if (!srcValue)
                {
                    // error
                    error = true;
                }

                mappedValues.push_back(srcValue);
            }
        };

        // map values
        auto dstIndex = -1;
        for (auto destField : tupleTypeRes.getFields())
        {
            dstIndex++;

            if (!destField.id)
            {
                addByIndex(dstIndex, destField);

                if (error)
                {
                    return mlir::Value();
                }

                continue;
            }

            auto found = false;
            auto anyFieldWithName = false;
            for (auto [index, srcField] : enumerate(fields))
            {
                if (!srcField.id)
                {
                    continue;
                }

                anyFieldWithName = true;
                if (srcField.id == destField.id)
                {
                    mlir::Value srcValue = results.getResults()[index];
                    if (srcValue.getType() != destField.type)
                    {
                        srcValue = cast(srcValue, srcValue.getType(), tch.convertType(srcValue.getType()), destField.type,
                                        tch.convertType(destField.type));
                    }

                    if (!srcValue)
                    {
                        // error
                        return mlir::Value();
                    }

                    mappedValues.push_back(srcValue);
                    found = true;
                    break;
                }
            }

            if (!found && !anyFieldWithName)
            {
                // find by index
                addByIndex(dstIndex, destField);
                continue;
            }

            // otherwise undef value
            if (!found)
            {
                auto undefVal = rewriter.create<mlir_ts::UndefOp>(loc, destField.type);
                mappedValues.push_back(undefVal);
            }
        }

        return rewriter.create<mlir_ts::CreateTupleOp>(loc, tupleTypeRes, mappedValues);
    }

    mlir::Value castBoolToString(mlir::Value in)
    {
        mlir::Value valueAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(in.getType()), in);

        return rewriter.create<LLVM::SelectOp>(loc, valueAsLLVMType, ch.getOrCreateGlobalString("__true__", std::string("true")),
                                               ch.getOrCreateGlobalString("__false__", std::string("false")));
    }

    mlir::Value castBoolToNumber(mlir::Value in)
    {
        mlir::Value valueAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(in.getType()), in);

#ifdef NUMBER_F64
        return rewriter.create<LLVM::SelectOp>(loc, valueAsLLVMType, clh.createF64ConstantOf(1),
                                               clh.createF64ConstantOf(0));
#else
        return rewriter.create<LLVM::SelectOp>(loc, valueAsLLVMType, clh.createF32ConstantOf(1),
                                               clh.createF32ConstantOf(0));
#endif
    }    

    mlir::Value castIntToString(mlir::Value in, int width, bool isSigned)
    {
        ConvertLogic cl(op, rewriter, tch, loc, compileOptions);
        return cl.intToString(in, width, isSigned);
    }

    mlir::Value castF64ToString(mlir::Value in)
    {
        ConvertLogic cl(op, rewriter, tch, loc, compileOptions);
        return cl.f64ToString(in);
    }

    mlir::Value castToArrayType(mlir::Value in, mlir::Type type, mlir::Type arrayType)
    {
        mlir::Type srcElementType;
        mlir::Type llvmSrcElementType;
        auto size = 0;
        bool byValue = true;
        bool isUndef = false;
        if (auto constArrayType = type.dyn_cast<mlir_ts::ConstArrayType>())
        {
            size = constArrayType.getSize();
            srcElementType = constArrayType.getElementType();
            llvmSrcElementType = tch.convertType(srcElementType);

            auto dstElementType = arrayType.cast<mlir_ts::ArrayType>().getElementType();
            auto llvmDstElementType = tch.convertType(dstElementType);
            if (size > 0 && llvmDstElementType != llvmSrcElementType)
            {
                emitError(loc) << "source array and destination array have different types, src: " << srcElementType
                               << " dst: " << dstElementType;
                return mlir::Value();
            }
        }
        else if (auto nullType = type.dyn_cast<mlir_ts::NullType>())
        {
            size = 0;
            srcElementType = arrayType.cast<mlir_ts::ArrayType>().getElementType();
            llvmSrcElementType = tch.convertType(srcElementType);         
            byValue = false;               
        }   
        else if (auto undefType = type.dyn_cast<mlir_ts::UndefinedType>())
        {
            size = 0;
            srcElementType = arrayType.cast<mlir_ts::ArrayType>().getElementType();
            llvmSrcElementType = tch.convertType(srcElementType);                  
            isUndef = true;      
        }
        else if (auto ptrValue = type.dyn_cast<LLVM::LLVMPointerType>())
        {
            auto elementType = ptrValue.getElementType();
            if (auto arrayType = elementType.dyn_cast<LLVM::LLVMArrayType>())
            {
                size = arrayType.getNumElements();
                llvmSrcElementType = tch.convertType(arrayType.getElementType());
            }
            else
            {
                LLVM_DEBUG(llvm::dbgs() << "[castToArrayType(2)] from value: " << in << " as type: " << type
                                        << " to type: " << elementType << "\n";);
                llvm_unreachable("not implemented");
            }
        }        
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "[castToArrayType(1)] from value: " << in << " as type: " << type << " to type: " << arrayType
                                    << "\n";);
            llvm_unreachable("not implemented");
        }

        auto sizeValue = clh.createI32ConstantOf(size);
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto destArrayElement = arrayType.cast<mlir_ts::ArrayType>().getElementType();
        auto llvmDestArrayElement = tch.convertType(destArrayElement);
        auto llvmDestArray = LLVM::LLVMPointerType::get(llvmDestArrayElement);

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        if (isUndef)
        {
            return structValue;
        }
        
        auto arrayPtrType = LLVM::LLVMPointerType::get(llvmSrcElementType);
        auto arrayValueSize = LLVM::LLVMArrayType::get(llvmSrcElementType, size);
        auto ptrToArray = LLVM::LLVMPointerType::get(arrayValueSize);

        mlir::Value arrayPtr;
        if (byValue)
        {
            auto bytesSize = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), arrayValueSize);
            // TODO: create MemRef which will store information about memory. stack of heap, to use in array push to realloc
            // auto copyAllocated = ch.Alloca(arrayPtrType, bytesSize);
            auto copyAllocated = ch.MemoryAllocBitcast(arrayPtrType, bytesSize);

            auto ptrToArraySrc = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, in);
            auto ptrToArrayDst = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, copyAllocated);
            rewriter.create<mlir_ts::CopyStructOp>(loc, ptrToArrayDst, ptrToArraySrc);

            arrayPtr = rewriter.create<LLVM::BitcastOp>(loc, llvmDestArray, copyAllocated);
        }
        else
        {
            // copy ptr only (const ptr -> ptr)
            // TODO: here we need to clone body to make it writable (and remove logic from VariableOp)
            arrayPtr = rewriter.create<LLVM::BitcastOp>(loc, arrayPtrType, in);
        }

        auto structValue2 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, arrayPtr, MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));

        auto structValue3 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, sizeValue, MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));

        return structValue3;
    }

    mlir::Value castToAny(mlir::Value in, mlir::Type inType, mlir::Type inLLVMType)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! cast to any: " << inType << " value: " << in << "\n";);

        TypeOfOpHelper toh(rewriter);
        mlir::Value typeOfValue;
        auto valueForBoxing = in;

        if (auto unionType = inType.dyn_cast<mlir_ts::UnionType>())
        {
            MLIRTypeHelper mth(unionType.getContext());
            mlir::Type baseType;
            bool needTag = mth.isUnionTypeNeedsTag(unionType, baseType);
            if (needTag)
            {
                typeOfValue = toh.typeOfLogic(loc, valueForBoxing, unionType);

                LLVMTypeConverterHelper llvmtch(*(LLVMTypeConverter *)&tch.typeConverter);
                // so we need to get biggest value from Union
                auto maxUnionType = llvmtch.findMaxSizeType(unionType);
                LLVM_DEBUG(llvm::dbgs() << "\n!! max size union type: " << maxUnionType << "\n";);
                valueForBoxing = rewriter.create<mlir_ts::GetValueFromUnionOp>(loc, maxUnionType, in);
            }
            else
            {
                typeOfValue = toh.typeOfLogic(loc, inType);    
            }
        }
        else
        {
            typeOfValue = toh.typeOfLogic(loc, inType);
        }

        auto boxedValue = rewriter.create<mlir_ts::BoxOp>(loc, mlir_ts::AnyType::get(rewriter.getContext()),
                                                            valueForBoxing, typeOfValue);
        return boxedValue;
    }

    mlir::Value castFromAny(mlir::Value in, mlir::Type resType)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! cast from any: " << in << " to " << resType << "\n";);

        auto unboxedValue = rewriter.create<mlir_ts::UnboxOp>(loc, resType, in);
        return unboxedValue;
    }

    mlir::Value castToOpaqueType(mlir::Value in, mlir::Type inLLVMType)
    {
        MLIRTypeHelper mth(rewriter.getContext());
        auto variableOp = mth.GetReferenceOfLoadOp(in);
        if (variableOp)
        {
            return clh.castToI8Ptr(variableOp);
        }

        auto valueAddr = rewriter.create<mlir_ts::VariableOp>(
            loc, mlir_ts::RefType::get(in.getType()), in, rewriter.getBoolAttr(false), rewriter.getIndexAttr(0));

        mlir::Value valueAddrAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(valueAddr.getType()), valueAddr);

        return clh.castToI8Ptr(valueAddrAsLLVMType);
    }

    template <typename TupleTy> mlir::Value castTupleToString(mlir::Value in, mlir::Type inType, TupleTy tupleTypeIn)
    {
        // calling method from object
        auto fieldId = MLIRHelper::TupleFieldName(TO_STRING, rewriter.getContext());
        auto fieldIndex = tupleTypeIn.getIndex(fieldId);
        if (fieldIndex < 0)
        {
            // can't cast
            return mlir::Value();
        }

        ::mlir::typescript::FieldInfo fieldInfo = tupleTypeIn.getFieldInfo(fieldIndex);

        auto inCasted = cast(in, inType, mlir_ts::ObjectType::get(tupleTypeIn));

        auto propField = rewriter.create<mlir_ts::PropertyRefOp>(loc, mlir_ts::RefType::get(fieldInfo.type), inCasted,
                                                                 rewriter.getI32IntegerAttr(fieldIndex));

        mlir::Value value = rewriter.create<mlir_ts::LoadOp>(loc, fieldInfo.type, propField);

        auto funcType = fieldInfo.type.cast<mlir_ts::FunctionType>();

        mlir::Value objTypeCasted = cast(inCasted, inCasted.getType(), funcType.getInput(0));

        if (external)
        {
            auto results = rewriter.create<mlir_ts::CallIndirectOp>(
                MLIRHelper::getCallSiteLocation(value, loc),
                value, 
                ValueRange(objTypeCasted));
            return results.getResult(0);
        }

        auto results = rewriter.create<mlir_ts::CallInternalOp>(loc, mlir_ts::StringType::get(rewriter.getContext()),
                                                                ValueRange{value, objTypeCasted});
        return results.getResult(0);
    }

    mlir::Value castBoundRefToRef(mlir::Value in, mlir_ts::BoundRefType boundRefTypeIn, mlir_ts::RefType refTypeOut)
    {
        auto llvmType = tch.convertType(boundRefTypeIn.getElementType());
        auto expectingLlvmType = tch.convertType(refTypeOut);
        auto llvmRefType = LLVM::LLVMPointerType::get(llvmType);

        mlir::Value inAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(in.getType()), in);

        mlir::Value valueRefVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, inAsLLVMType, MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));
        if (expectingLlvmType != llvmRefType)
        {
            valueRefVal = rewriter.create<LLVM::BitcastOp>(loc, expectingLlvmType, valueRefVal);
        }

        return valueRefVal;
    }

    mlir::Value extractArrayPtr(mlir::Value in, mlir_ts::ArrayType arrayType)
    {
        auto llvmType = tch.convertType(arrayType.getElementType());
        auto llvmRefType = LLVM::LLVMPointerType::get(llvmType);

        mlir::Value inAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(in.getType()), in);

        mlir::Value ptrVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, inAsLLVMType, MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
        return ptrVal;
    }

};

template <typename T>
mlir::Value castLogic(mlir::Value size, mlir::Type sizeType, mlir::Operation *op, PatternRewriter &rewriter, TypeConverterHelper tch, CompileOptions &compileOptions)
{
    CastLogicHelper castLogic(op, rewriter, tch, compileOptions);
    return castLogic.cast(size, size.getType(), sizeType);
}

} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CASTLOGICHELPER_H_
