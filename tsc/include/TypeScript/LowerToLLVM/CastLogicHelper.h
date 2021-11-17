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

#include "mlir/Dialect/StandardOps/IR/Ops.h"

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
    bool external;

  public:
    CastLogicHelper(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, &tch.typeConverter), clh(op, rewriter), loc(op->getLoc()),
          external(false)
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

        if (inLLVMType.isInteger(32) && isResString)
        {
            return castI32ToString(in);
        }

        if (inLLVMType.isInteger(64) && isResString)
        {
            return castI64ToString(in);
        }

        if ((inLLVMType.isF32() || inLLVMType.isF64()) && isResString)
        {
            return castF32orF64ToString(in);
        }

        if (auto arrType = resType.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            return castToArrayType(in, inType, resType);
        }

        auto isResAny = resType.isa<mlir_ts::AnyType>();
        if (isResAny)
        {
            return castToAny(in, inType, inLLVMType);
        }

        auto isInAny = inType.isa<mlir_ts::AnyType>();
        if (isInAny)
        {
            return castFromAny(in, resLLVMType);
        }

        if (auto obj = resType.dyn_cast_or_null<mlir_ts::ObjectType>())
        {
            if (obj.getStorageType().isa<mlir_ts::AnyType>())
            {
                return castToOpaqueType(in, inLLVMType);
            }
        }

        if (auto obj = resType.dyn_cast_or_null<mlir_ts::UnknownType>())
        {
            return castToOpaqueType(in, inLLVMType);
        }

        auto isInString = inType.dyn_cast_or_null<mlir_ts::StringType>();
        if (isInString && (resLLVMType.isInteger(32) || resLLVMType.isInteger(64)))
        {
            return rewriter.create<mlir_ts::ParseIntOp>(loc, resLLVMType, in);
        }

        if (isInString && (resLLVMType.isF32() || resLLVMType.isF64()))
        {
            return rewriter.create<mlir_ts::ParseFloatOp>(loc, resLLVMType, in);
        }

        // cast value value to optional value
        if (auto optType = resType.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            return rewriter.create<mlir_ts::CreateOptionalOp>(loc, resType, in);
        }

        if (auto optType = inType.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            if (optType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
            {
                if (auto ifaceType = resType.dyn_cast_or_null<mlir_ts::InterfaceType>())
                {
                    // create null interface
                    auto nullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(ifaceType.getContext()));
                    return rewriter.create<mlir_ts::NewInterfaceOp>(loc, ifaceType, nullVal, nullVal);
                }
            }

            auto val = rewriter.create<mlir_ts::ValueOp>(loc, optType.getElementType(), in);
            return cast(val, val.getType(), tch.convertType(val.getType()), resType, resLLVMType);
        }

        // array to ref of element
        if (auto arrayType = inType.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            if (auto refType = resType.dyn_cast_or_null<mlir_ts::RefType>())
            {
                if (arrayType.getElementType() == refType.getElementType())
                {

                    return rewriter.create<LLVM::ExtractValueOp>(loc, resLLVMType, in,
                                                                 rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));
                }
            }
        }

        if (auto resFuncType = resType.dyn_cast_or_null<mlir::FunctionType>())
        {
            if (auto inBoundFunc = inType.dyn_cast_or_null<mlir_ts::BoundFunctionType>())
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

        if (auto resHybridFunc = resType.dyn_cast_or_null<mlir_ts::HybridFunctionType>())
        {
            if (auto inFuncType = inType.dyn_cast_or_null<mlir::FunctionType>())
            {
                // BoundFunction is the same as HybridFunction
                // null this
                auto thisNullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(rewriter.getContext()));
                auto boundFuncVal = rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, resHybridFunc, thisNullVal, in);
                return boundFuncVal;
            }
        }

        if (auto resRefType = resType.dyn_cast_or_null<mlir_ts::RefType>())
        {
            if (auto inBoundRef = inType.dyn_cast_or_null<mlir_ts::BoundRefType>())
            {
                return castBoundRefToRef(in, inBoundRef, resRefType);
            }
        }

        if (auto tupleTypeIn = inType.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            if (auto tupleTypeRes = resType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                SmallVector<mlir::Type> types;
                for (auto &field : tupleTypeIn.getFields())
                {
                    types.push_back(field.type);
                }

                auto results = rewriter.create<mlir_ts::DeconstructTupleOp>(loc, types, in);
                return rewriter.create<mlir_ts::CreateTupleOp>(loc, tupleTypeRes, results.getResults());
            }
        }

        if (auto nullType = inType.dyn_cast_or_null<mlir_ts::NullType>())
        {
            if (auto ifaceType = resType.dyn_cast_or_null<mlir_ts::InterfaceType>())
            {
                // create null interface
                return rewriter.create<mlir_ts::NewInterfaceOp>(loc, ifaceType, in, in);
            }

            if (auto resHybridFunc = resType.dyn_cast_or_null<mlir_ts::HybridFunctionType>())
            {
                // null this
                auto thisNullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(rewriter.getContext()));
                auto funcType = mlir::FunctionType::get(rewriter.getContext(), resHybridFunc.getInputs(), resHybridFunc.getResults());
                auto castFuncNullVal = cast(in, inType, funcType);
                auto boundFuncVal = rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, resHybridFunc, thisNullVal, castFuncNullVal);
                return boundFuncVal;
            }
        }

        if (auto stringTypeRes = resType.dyn_cast_or_null<mlir_ts::StringType>())
        {
            if (auto tupleTypeIn = inType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
            {
                return castObjectToString<mlir_ts::ConstTupleType>(in, inType, tupleTypeIn);
            }

            if (auto tupleTypeIn = inType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                return castObjectToString<mlir_ts::TupleType>(in, inType, tupleTypeIn);
            }
        }

        if (auto interfaceTypeRes = resType.dyn_cast_or_null<mlir_ts::InterfaceType>())
        {
            if (auto tupleTypeIn = inType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
            {
                llvm_unreachable("not implemented, must be process at MLIR pass");
            }

            if (auto tupleTypeIn = inType.dyn_cast_or_null<mlir_ts::TupleType>())
            {
                llvm_unreachable("not implemented, must be process at MLIR pass");
            }
        }

        if (auto undefPHTypeIn = inType.dyn_cast_or_null<mlir_ts::UndefPlaceHolderType>())
        {
            if (auto stringTypeRes = resType.dyn_cast_or_null<mlir_ts::StringType>())
            {
                return rewriter.create<mlir_ts::ConstantOp>(loc, stringTypeRes, rewriter.getStringAttr("undefined"));
            }
        }

        if (auto boolType = resType.dyn_cast_or_null<mlir_ts::BooleanType>())
        {
            if (auto ifaceType = inType.dyn_cast_or_null<mlir_ts::InterfaceType>())
            {
                auto ptrValue =
                    rewriter.create<mlir_ts::ExtractInterfaceVTableOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()), in);
                auto inLLVMType = tch.convertType(ptrValue.getType());
                auto llvmBoolType = tch.convertType(boolType);
                return castLLVMTypes(ptrValue, inLLVMType, boolType, llvmBoolType);
            }

            if (auto hybridFuncType = inType.dyn_cast_or_null<mlir_ts::HybridFunctionType>())
            {
                auto funcType = mlir::FunctionType::get(rewriter.getContext(), hybridFuncType.getInputs(), hybridFuncType.getResults());
                auto ptrValue = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, in);
                auto inLLVMType = tch.convertType(ptrValue.getType());
                auto llvmBoolType = tch.convertType(boolType);
                return castLLVMTypes(ptrValue, inLLVMType, boolType, llvmBoolType);
            }
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

    mlir::Value castLLVMTypes(mlir::Value in, mlir::Type inLLVMType, mlir::Type resType, mlir::Type resLLVMType)
    {
        if (inLLVMType == resLLVMType)
        {
            return in;
        }

        auto inType = in.getType();
        if (isInt(inLLVMType) && isFloat(resLLVMType))
        {
            return rewriter.create<SIToFPOp>(loc, resLLVMType, in);
        }

        if (isFloat(inLLVMType) && isInt(resLLVMType))
        {
            return rewriter.create<FPToSIOp>(loc, resLLVMType, in);
        }

        if (isInt(inLLVMType) && isBool(resLLVMType))
        {
            return rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, in, clh.createIConstantOf(inLLVMType.getIntOrFloatBitWidth(), 0));
        }

        if (isFloat(inLLVMType) && isBool(resLLVMType))
        {
            return rewriter.create<CmpFOp>(loc, CmpFPredicate::ONE, in, clh.createFConstantOf(inLLVMType.getIntOrFloatBitWidth(), 0.0));
        }

        if (inLLVMType.isa<LLVM::LLVMPointerType>() && isBool(resLLVMType))
        {
            auto intVal = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), in);
            return rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, intVal, clh.createI64ConstantOf(0));
        }

        if (inLLVMType.isa<LLVM::LLVMPointerType>() && isInt(resLLVMType))
        {
            return rewriter.create<LLVM::PtrToIntOp>(loc, resLLVMType, in);
        }

        if (isInt(inLLVMType) && resLLVMType.isa<LLVM::LLVMPointerType>())
        {
            return rewriter.create<LLVM::IntToPtrOp>(loc, resLLVMType, in);
        }

        if (isIntOrBool(inLLVMType) && isInt(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() < resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<ZeroExtendIOp>(loc, in, resLLVMType);
        }

        if (isInt(inLLVMType) && isInt(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() > resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<TruncateIOp>(loc, in, resLLVMType);
        }

        if (isFloat(inLLVMType) && isFloat(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() < resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<FPExtOp>(loc, in, resLLVMType);
        }

        if (isFloat(inLLVMType) && isFloat(resLLVMType) && inLLVMType.getIntOrFloatBitWidth() > resLLVMType.getIntOrFloatBitWidth())
        {
            return rewriter.create<FPTruncOp>(loc, in, resLLVMType);
        }

        // ptrs cast
        if (inLLVMType.isa<LLVM::LLVMPointerType>() && resLLVMType.isa<LLVM::LLVMPointerType>())
        {
            return rewriter.create<LLVM::BitcastOp>(loc, resLLVMType, in);
        }

        // struct to struct. TODO: add validation
        if (inLLVMType.isa<LLVM::LLVMStructType>() && resLLVMType.isa<LLVM::LLVMStructType>())
        {
            LLVMTypeConverterHelper llvmtch((LLVMTypeConverter &)tch.typeConverter);
            auto size1 = llvmtch.getTypeSize(inLLVMType);
            auto size2 = llvmtch.getTypeSize(resLLVMType);

            if (size1 != size2)
            {
                op->emitWarning("types have different sizes: ")
                    << inLLVMType << " size of " << size1 << ", " << resLLVMType << " size of " << size2;
            }

            auto srcAddr = rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(inType), in, rewriter.getBoolAttr(false));
            auto dstAddr =
                rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(resType), mlir::Value(), rewriter.getBoolAttr(false));
            rewriter.create<mlir_ts::MemoryCopyOp>(loc, dstAddr, srcAddr);
            auto val = rewriter.create<mlir_ts::LoadOp>(loc, resType, dstAddr);
            return val;
        }

        // value to ref of value
        if (auto destPtr = resLLVMType.dyn_cast_or_null<LLVM::LLVMPointerType>())
        {
            if (destPtr.getElementType() == inLLVMType)
            {
                // alloc and return address
                auto valueAddr = rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(inType), in, rewriter.getBoolAttr(false));
                return valueAddr;
            }
        }

        emitError(loc, "invalid cast operator type 1: '") << inLLVMType << "', type 2: '" << resLLVMType << "'";
        llvm_unreachable("not implemented");

        return mlir::Value();
    }

    mlir::Value castBoolToString(mlir::Value in)
    {
        return rewriter.create<LLVM::SelectOp>(loc, in, ch.getOrCreateGlobalString("__true__", std::string("true")),
                                               ch.getOrCreateGlobalString("__false__", std::string("false")));
    }

    mlir::Value castI32ToString(mlir::Value in)
    {
        ConvertLogic cl(op, rewriter, tch, loc);
        return cl.intToString(in);
    }

    mlir::Value castI64ToString(mlir::Value in)
    {
        ConvertLogic cl(op, rewriter, tch, loc);
        return cl.int64ToString(in);
    }

    mlir::Value castF32orF64ToString(mlir::Value in)
    {
        ConvertLogic cl(op, rewriter, tch, loc);
        return cl.f32OrF64ToString(in);
    }

    mlir::Value castToArrayType(mlir::Value in, mlir::Type type, mlir::Type arrayType)
    {
        mlir::Type srcElementType;
        mlir::Type llvmSrcElementType;
        auto size = 0;
        if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
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
        else if (auto ptrValue = type.dyn_cast_or_null<LLVM::LLVMPointerType>())
        {
            auto elementType = ptrValue.getElementType();
            if (auto arrayType = elementType.dyn_cast_or_null<LLVM::LLVMArrayType>())
            {
                size = arrayType.getNumElements();
                llvmSrcElementType = tch.convertType(arrayType.getElementType());
            }
            else
            {
                LLVM_DEBUG(llvm::dbgs() << "[castToArrayType(2)] from value: " << in << " type: " << in.getType()
                                        << " to type: " << elementType << "\n";);
                llvm_unreachable("not implemented");
            }
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "[castToArrayType(1)] from value: " << in << " type: " << in.getType() << " to type: " << arrayType
                                    << "\n";);
            llvm_unreachable("not implemented");
        }

        auto sizeValue = clh.createI32ConstantOf(size);
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto destArrayElement = arrayType.cast<mlir_ts::ArrayType>().getElementType();
        auto llvmDestArrayElement = tch.convertType(destArrayElement);
        auto llvmDestArray = LLVM::LLVMPointerType::get(llvmDestArrayElement);

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto arrayPtrType = LLVM::LLVMPointerType::get(llvmSrcElementType);
        auto arrayValueSize = LLVM::LLVMArrayType::get(llvmSrcElementType, size);
        auto ptrToArray = LLVM::LLVMPointerType::get(arrayValueSize);

        mlir::Value arrayPtr;
        bool byValue = true;
        if (byValue)
        {
            auto bytesSize = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), arrayValueSize);
            // TODO: create MemRef which will store information about memory. stack of heap, to use in array push to realloc
            // auto copyAllocated = rewriter.create<LLVM::AllocaOp>(loc, arrayPtrType, bytesSize);
            auto copyAllocated = ch.MemoryAllocBitcast(arrayPtrType, bytesSize);

            auto ptrToArraySrc = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, in);
            auto ptrToArrayDst = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, copyAllocated);
            rewriter.create<mlir_ts::LoadSaveOp>(loc, ptrToArrayDst, ptrToArraySrc);
            // rewriter.create<mlir_ts::MemoryCopyOp>(loc, ptrToArrayDst, ptrToArraySrc);

            arrayPtr = rewriter.create<LLVM::BitcastOp>(loc, llvmDestArray, copyAllocated);
        }
        else
        {
            // copy ptr only (const ptr -> ptr)
            // TODO: here we need to clone body to make it writable (and remove logic from VariableOp)
            arrayPtr = rewriter.create<LLVM::BitcastOp>(loc, arrayPtrType, in);
        }

        auto structValue2 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, arrayPtr, clh.getStructIndexAttr(0));

        auto structValue3 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, sizeValue, clh.getStructIndexAttr(1));

        return structValue3;
    }

    mlir::Value castToAny(mlir::Value in, mlir::Type inType, mlir::Type inLLVMType)
    {
        AnyLogic al(op, rewriter, tch, loc);
        return al.castToAny(in, inType, inLLVMType);
    }

    mlir::Value castFromAny(mlir::Value in, mlir::Type resLLVMType)
    {
        AnyLogic al(op, rewriter, tch, loc);
        return al.castFromAny(in, resLLVMType);
    }

    mlir::Value castToOpaqueType(mlir::Value in, mlir::Type inLLVMType)
    {
        MLIRTypeHelper mth(rewriter.getContext());
        auto variableOp = mth.GetReferenceOfLoadOp(in);
        if (variableOp)
        {
            return clh.castToI8Ptr(variableOp);
        }

        auto valueAddr = rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(in.getType()), in, rewriter.getBoolAttr(false));
        return clh.castToI8Ptr(valueAddr);
    }

    template <typename TupleTy> mlir::Value castObjectToString(mlir::Value in, mlir::Type inType, TupleTy tupleTypeIn)
    {
        // calling method from object
        MLIRTypeHelper mth(rewriter.getContext());
        auto fieldId = mth.TupleFieldName("toString");
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

        auto funcType = fieldInfo.type.cast<mlir::FunctionType>();

        mlir::Value objTypeCasted = cast(inCasted, inCasted.getType(), funcType.getInput(0));

        if (external)
        {
            auto results = rewriter.create<mlir_ts::CallIndirectOp>(loc, value, ValueRange(objTypeCasted));
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

        mlir::Value valueRefVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, in, clh.getStructIndexAttr(DATA_VALUE_INDEX));
        if (expectingLlvmType != llvmRefType)
        {
            valueRefVal = rewriter.create<LLVM::BitcastOp>(loc, expectingLlvmType, valueRefVal);
        }

        return valueRefVal;
    }
};

template <typename T>
mlir::Value castLogic(mlir::Value size, mlir::Type sizeType, mlir::Operation *op, PatternRewriter &rewriter, TypeConverterHelper tch)
{
    CastLogicHelper castLogic(op, rewriter, tch);
    return castLogic.cast(size, size.getType(), sizeType);
}

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_CASTLOGICHELPER_H_
