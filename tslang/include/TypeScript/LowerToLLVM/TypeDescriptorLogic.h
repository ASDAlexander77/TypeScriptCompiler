#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEDESCRIPTORLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEDESCRIPTORLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

// Reads the static per-type descriptor that sits in front of a runtime type tag.
//
// A tag points at the descriptor's trailing name bytes, so the record itself is at
// `tag - sizeof(record)`. See TYPE_DESCR_* in Defines.h for the layout and why it is a
// cross-module contract; TypeDescriptorOpLowering is what emits the records.
class TypeDescriptorLogic
{
    PatternRewriter &rewriter;
    TypeConverterHelper &tch;
    TypeHelper th;
    Location loc;

  public:
    TypeDescriptorLogic(PatternRewriter &rewriter, TypeConverterHelper &tch, Location loc)
        : rewriter(rewriter), tch(tch), th(rewriter), loc(loc)
    {
    }

    // { i32 kind, i32 reserved, ptr release, index blockHeader }
    static LLVM::LLVMStructType getRecordType(mlir::OpBuilder &builder, mlir::Type llvmIndexType)
    {
        auto i32Ty = builder.getI32Type();
        auto ptrTy = LLVM::LLVMPointerType::get(builder.getContext());
        return LLVM::LLVMStructType::getLiteral(builder.getContext(), {i32Ty, i32Ty, ptrTy, llvmIndexType}, false);
    }

    LLVM::LLVMStructType getRecordType()
    {
        return getRecordType(rewriter, tch.convertType(th.getIndexType()));
    }

    // Size of the record, and therefore the distance from a tag back to it. The name is a
    // byte array, which needs no alignment padding in front of it, so the offset of the name
    // within `{ record, [N x i8] }` is exactly the record size regardless of N or target.
    mlir::Value getRecordSize()
    {
        auto ptrTy = th.getPtrType();
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrTy);
        auto endAddr = rewriter.create<LLVM::GEPOp>(loc, ptrTy, getRecordType(), nullPtr, ArrayRef<LLVM::GEPArg>{1});
        return rewriter.create<LLVM::PtrToIntOp>(loc, llvmIndexType, endAddr);
    }

    mlir::Value getRecordPtrFromTag(mlir::Value tagValue)
    {
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto size = getRecordSize();
        auto negatedSize = rewriter.create<LLVM::SubOp>(loc, llvmIndexType,
                                                        rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, rewriter.getIntegerAttr(llvmIndexType, 0)),
                                                        size);
        return rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), th.getI8Type(), tagValue, ValueRange{negatedSize});
    }

    // TYPE_KIND_* for the type this tag names.
    mlir::Value getKindFromTag(mlir::Value tagValue)
    {
        auto recordPtr = getRecordPtrFromTag(tagValue);
        auto kindPtr = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), getRecordType(), recordPtr,
                                                    ArrayRef<LLVM::GEPArg>{0, TYPE_DESCR_KIND});
        return rewriter.create<LLVM::LoadOp>(loc, th.getI32Type(), kindPtr);
    }

    mlir::Value isKind(mlir::Value tagValue, int kind)
    {
        auto kindValue = getKindFromTag(tagValue);
        auto expected = rewriter.create<LLVM::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(kind));
        return rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, kindValue, expected);
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_TYPEDESCRIPTORLOGIC_H_
