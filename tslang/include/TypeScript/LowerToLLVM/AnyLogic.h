#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ANYLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ANYLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"
#include "TypeScript/MLIRLogic/TypeOfOpHelper.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class AnyLogic
{
    Operation *op;
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;
    TypeHelper th;
    LLVMCodeHelperBase ch;
    CodeLogicHelper clh;
    Location loc;
    CompileOptions &compileOptions;

  protected:
    mlir::Type indexType;
    mlir::Type llvmIndexType;
    mlir::Type ptrTy;

  public:
    AnyLogic(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, tch.typeConverter, compileOptions), clh(op, rewriter), loc(loc), compileOptions(compileOptions)
    {
        indexType = th.getIndexType();
        llvmIndexType = tch.convertType(indexType);
        ptrTy = th.getPtrType();
    }

    LLVM::LLVMStructType getStorageAnyType(mlir::Type llvmStorageType)
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {llvmIndexType, ptrTy, llvmStorageType}, false);
    }

    mlir::Value castToAny(mlir::Value in, mlir::Type inType, mlir::Type inLLVMType)
    {
        // get typeof value
        // auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
        TypeOfOpHelper toh(rewriter);
        auto typeOfValue = toh.typeOfLogic(loc, in, inType, compileOptions);
        return castToAny(in, typeOfValue, inLLVMType);
    }

    mlir::Value castToAny(mlir::Value in, mlir::Value typeOfValue, mlir::Type inLLVMType)
    {
        // TODO: add type id to track data type
        auto llvmStorageType = inLLVMType;
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto memValue = ch.MemoryAlloc(dataWithSizeType);

        // set value size
        auto sizeMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, indexType, llvmStorageType);
        auto size = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeMLIR);

        auto zero = clh.createI32ConstantOf(ANY_SIZE);
        auto one = clh.createI32ConstantOf(ANY_TYPE);
        auto two = clh.createI32ConstantOf(ANY_DATA);

        auto ptrSize = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, memValue, ValueRange{zero, zero});
        rewriter.create<LLVM::StoreOp>(loc, size, ptrSize);

        auto typeOfStr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, memValue, ValueRange{zero, one});
        rewriter.create<LLVM::StoreOp>(loc, typeOfValue, typeOfStr);

        // set actual value
        auto ptrValue = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, memValue, ValueRange{zero, two});
        rewriter.create<LLVM::StoreOp>(loc, in, ptrValue);

        return memValue;
    }

    mlir::Value UnboxAny(mlir::Value in, mlir::Type resLLVMType, bool rawPointer = false)
    {
        // TODO: add type id to track data type
        // TODO: add data size check
        auto llvmStorageType = resLLVMType;
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto inDataWithSizeTypedValue = rewriter.create<LLVM::BitcastOp>(loc, ptrTy, in);

        auto zero = clh.createI32ConstantOf(ANY_SIZE);
        //auto one = clh.createI32ConstantOf(ANY_TYPE);
        auto two = clh.createI32ConstantOf(ANY_DATA);

        // set actual value
        auto ptrValue =
            rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, inDataWithSizeTypedValue, ValueRange{zero, two});

        // An Opaque read out of a box holding a smaller value (`<Opaque><any>0`: an s32) took the
        // padding after it as the pointer's high bytes, which only a zeroing allocator left at 0. The
        // value is as wide as the box says (ANY_SIZE): its bytes, zero-extended. The data field
        // starts 8-aligned, so the 8 bytes read are inside the box either way.
        if (rawPointer && isa<LLVM::LLVMPointerType>(resLLVMType))
        {
            auto i64Type = th.getI64Type();
            auto ptrSize = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, inDataWithSizeTypedValue,
                                                        ValueRange{zero, zero});
            mlir::Value size = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, ptrSize);
            if (llvmIndexType != i64Type)
            {
                size = rewriter.create<LLVM::ZExtOp>(loc, i64Type, size);
            }

            mlir::Value raw = rewriter.create<LLVM::LoadOp>(loc, i64Type, ptrValue);

            auto eight = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(8));
            auto one64 = rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(1));
            auto isWide = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::uge, size, eight);
            // for a narrower value: (1 << size * 8) - 1, its bytes' mask (a shift by 64 is poison,
            // so the select keeps it out of the wide case)
            auto narrowSize = rewriter.create<LLVM::SelectOp>(loc, isWide, rewriter.create<LLVM::ConstantOp>(loc, i64Type, rewriter.getI64IntegerAttr(0)), size);
            auto bits = rewriter.create<LLVM::MulOp>(loc, narrowSize, eight);
            auto mask = rewriter.create<LLVM::SubOp>(loc, rewriter.create<LLVM::ShlOp>(loc, one64, bits), one64);
            auto narrowed = rewriter.create<LLVM::AndOp>(loc, raw, mask);
            mlir::Value value = rewriter.create<LLVM::SelectOp>(loc, isWide, raw, narrowed);
            return rewriter.create<LLVM::IntToPtrOp>(loc, resLLVMType, value);
        }

        return rewriter.create<LLVM::LoadOp>(loc, resLLVMType, ptrValue);
    }

    mlir::Value getTypeOfAny(mlir::Value in)
    {
        // TODO: add type id to track data type
        // TODO: add data size check
        // any random type
        auto llvmStorageType = th.getI8Type();
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto inDataWithSizeTypedValue = rewriter.create<LLVM::BitcastOp>(loc, ptrTy, in);

        auto zero = clh.createI32ConstantOf(ANY_SIZE);
        auto one = clh.createI32ConstantOf(ANY_TYPE);
        //auto two = clh.createI32ConstantOf(ANY_DATA);

        // set actual value
        auto ptrValue = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, inDataWithSizeTypedValue,
                                                     ValueRange{zero, one});
        return rewriter.create<LLVM::LoadOp>(loc, ptrTy, ptrValue);
    }

    mlir::Value getDataSizeOfAny(mlir::Value in)
    {
        // TODO: add type id to track data type
        // TODO: add data size check
        // any random type
        auto llvmStorageType = th.getI8Type();
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto inDataWithSizeTypedValue = rewriter.create<LLVM::BitcastOp>(loc, ptrTy, in);

        auto zero = clh.createI32ConstantOf(ANY_SIZE);
        //auto one = clh.createI32ConstantOf(ANY_TYPE);
        //auto two = clh.createI32ConstantOf(ANY_DATA);

        // set actual value
        auto ptrValue = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, inDataWithSizeTypedValue,
                                                     ValueRange{zero, zero});
        return rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, ptrValue);
    }    

    mlir::Value getDataPtrOfAny(mlir::Value in)
    {
        // TODO: add type id to track data type
        // TODO: add data size check
        // any random type
        auto llvmStorageType = th.getI8Type();
        auto dataWithSizeType = getStorageAnyType(llvmStorageType);

        auto inDataWithSizeTypedValue = rewriter.create<LLVM::BitcastOp>(loc, ptrTy, in);

        auto zero = clh.createI32ConstantOf(ANY_SIZE);
        //auto one = clh.createI32ConstantOf(ANY_TYPE);
        auto two = clh.createI32ConstantOf(ANY_DATA);

        // set actual value
        auto ptrValue = rewriter.create<LLVM::GEPOp>(loc, ptrTy, dataWithSizeType, inDataWithSizeTypedValue,
                                                     ValueRange{zero, two});
        return ptrValue;
    }    
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_ANYLOGIC_H_
