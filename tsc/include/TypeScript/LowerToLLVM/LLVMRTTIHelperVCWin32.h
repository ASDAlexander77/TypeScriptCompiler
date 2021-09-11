#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_

#pragma warning(disable : 4834)

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMRTTIHelperVCWin32
{
    Operation *op;
    PatternRewriter &rewriter;
    ModuleOp parentModule;
    TypeHelper th;
    LLVMCodeHelper ch;

  public:
    LLVMRTTIHelperVCWin32(Operation *op, PatternRewriter &rewriter, TypeConverter &typeConverter)
        : op(op), rewriter(rewriter), parentModule(op->getParentOfType<ModuleOp>()), th(rewriter), ch(op, rewriter, &typeConverter)
    {
    }

    LogicalResult setPersonality(mlir::FuncOp newFuncOp)
    {
        auto cxxFrameHandler3 = ch.getOrInsertFunction("__CxxFrameHandler3", th.getFunctionType(th.getI32Type(), {}, true));

        newFuncOp->setAttr(rewriter.getIdentifier("personality"), FlatSymbolRefAttr::get(rewriter.getContext(), "__CxxFrameHandler3"));
        return success();
    }

    LogicalResult typeInfo(mlir::Location loc)
    {
        auto name = "??_7type_info@@6B@";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8PtrType(), true, LLVM::Linkage::External, name, Attribute{});
        return success();
    }

    LogicalResult typeDescriptor2(mlir::Location loc)
    {
        auto name = "??_R0N@8";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        auto rttiTypeDescriptor2Ty = getRttiTypeDescriptor2Ty();
        auto _r0n_Value = rewriter.create<LLVM::GlobalOp>(loc, rttiTypeDescriptor2Ty, false, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_r0n_Value);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, rttiTypeDescriptor2Ty);

            auto itemValue1 = rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrPtrType(),
                                                                FlatSymbolRefAttr::get(rewriter.getContext(), "??_7type_info@@6B@"));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            auto itemValue2 = rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir::ConstantOp>(loc, th.getI8Array(3), ch.getStringAttrWith0(".N"));
            ch.setStructValue(loc, structVal, itemValue3, 2);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_r0n_Value);
        }

        return success();
    }

    LogicalResult imageBase(mlir::Location loc)
    {
        auto name = "__ImageBase";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8Type(), true, LLVM::Linkage::External, name, Attribute{});
        return success();
    }

    LogicalResult catchableType(mlir::Location loc)
    {
        auto name = "_CT??_R0N@88";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        // _CT??_R0N@88
        auto ehCatchableTypeTy = getCatchableTypeTy();
        auto _ct_r0n_Value = rewriter.create<LLVM::GlobalOp>(loc, ehCatchableTypeTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_ct_r0n_Value);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableTypeTy);

            auto itemValue1 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(1));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            // value 2
            auto rttiTypeDescriptor2PtrValue = rewriter.create<mlir::ConstantOp>(loc, getRttiTypeDescriptor2PtrTy(),
                                                                                 FlatSymbolRefAttr::get(rewriter.getContext(), "??_R0N@8"));
            auto rttiTypeDescriptor2IntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiTypeDescriptor2PtrValue);

            auto imageBasePtrValue =
                rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), "__ImageBase"));
            auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

            // sub
            auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiTypeDescriptor2IntValue, imageBaseIntValue);

            // trunc
            auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);

            auto itemValue2 = subRes32Value;
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue3, 2);

            auto itemValue4 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(-1));
            ch.setStructValue(loc, structVal, itemValue4, 3);

            auto itemValue5 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue5, 4);

            auto itemValue6 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(8));
            ch.setStructValue(loc, structVal, itemValue6, 5);

            auto itemValue7 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue7, 6);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_ct_r0n_Value);
        }

        return success();
    }

    LogicalResult catchableArrayType(mlir::Location loc)
    {
        auto name = "_CTA1N";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        // _CT??_R0N@88
        auto ehCatchableArrayTypeTy = getCatchableArrayTypeTy();
        auto _cta1nValue =
            rewriter.create<LLVM::GlobalOp>(loc, ehCatchableArrayTypeTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_cta1nValue);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableArrayTypeTy);

            auto itemValue1 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(1));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            // value 2
            auto rttiCatchableTypePtrValue = rewriter.create<mlir::ConstantOp>(
                loc, getCatchableTypePtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), "_CT??_R0N@88"));
            auto rttiCatchableTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableTypePtrValue);

            auto imageBasePtrValue =
                rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), "__ImageBase"));
            auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

            // sub
            auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiCatchableTypeIntValue, imageBaseIntValue);

            // trunc
            auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);

            // make array
            Value array1Val = rewriter.create<LLVM::UndefOp>(loc, th.getArrayType(th.getI32Type(), 1));
            ch.setStructValue(loc, array1Val, subRes32Value, 0);

            auto itemValue2 = array1Val;
            ch.setStructValue(loc, structVal, itemValue2, 1);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_cta1nValue);
        }

        return success();
    }

    LogicalResult throwInfo(mlir::Location loc)
    {
        auto name = "_TI1N";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        auto throwInfoTy = getThrowInfoTy();
        auto _TI1NValue = rewriter.create<LLVM::GlobalOp>(loc, throwInfoTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        ch.setStructWritingPoint(_TI1NValue);

        Value structValue = ch.getStructFromArrayAttr(
            loc, throwInfoTy,
            rewriter.getArrayAttr({rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0)}));

        // value 3
        auto rttiCatchableArrayTypePtrValue =
            rewriter.create<mlir::ConstantOp>(loc, getCatchableArrayTypePtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), "_CTA1N"));
        auto rttiCatchableArrayTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableArrayTypePtrValue);

        auto imageBasePtrValue =
            rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), "__ImageBase"));
        auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

        // sub
        auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiCatchableArrayTypeIntValue, imageBaseIntValue);

        // trunc
        auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);
        ch.setStructValue(loc, structValue, subRes32Value, 3);

        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structValue});

        rewriter.setInsertionPointAfter(_TI1NValue);

        return success();
    }

    LLVM::LLVMStructType getThrowInfoTy()
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type()},
                                                false);
    }

    LLVM::LLVMPointerType getThrowInfoPtrTy()
    {
        return LLVM::LLVMPointerType::get(getThrowInfoTy());
    }

    LLVM::LLVMStructType getRttiTypeDescriptor2Ty()
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI8PtrPtrType(), th.getI8PtrType(), th.getI8Array(3)}, false);
    }

    LLVM::LLVMPointerType getRttiTypeDescriptor2PtrTy()
    {
        return LLVM::LLVMPointerType::get(getRttiTypeDescriptor2Ty());
    }

    LLVM::LLVMStructType getCatchableTypeTy()
    {
        return LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type()}, false);
    }

    LLVM::LLVMPointerType getCatchableTypePtrTy()
    {
        return LLVM::LLVMPointerType::get(getCatchableTypeTy());
    }

    LLVM::LLVMStructType getCatchableArrayTypeTy()
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI32Type(), th.getI32Array(1)}, false);
    }

    LLVM::LLVMPointerType getCatchableArrayTypePtrTy()
    {
        return LLVM::LLVMPointerType::get(getCatchableArrayTypeTy());
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCWIN32_H_
