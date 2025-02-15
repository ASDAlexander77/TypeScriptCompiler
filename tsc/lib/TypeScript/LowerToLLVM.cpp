#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

#ifdef ENABLE_ASYNC
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#endif
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/DIBuilder.h"

#include "TypeScript/TypeScriptPassContext.h"
#include "TypeScript/LowerToLLVMLogic.h"
#include "TypeScript/LowerToLLVM/LLVMDebugInfo.h"
#include "TypeScript/LowerToLLVM/LLVMDebugInfoFixer.h"

#include "scanner_enums.h"

#define DISABLE_SWITCH_STATE_PASS 1
#define ENABLE_MLIR_INIT

#define DEBUG_TYPE "llvm"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace
{
struct TsLlvmContext
{
    TsLlvmContext(CompileOptions& compileOptions) : compileOptions(compileOptions) {};

    CompileOptions& compileOptions;
};

template <typename OpTy> class TsLlvmPattern : public OpConversionPattern<OpTy>
{
  public:
    using Adaptor = typename OpTy::Adaptor;

    TsLlvmPattern<OpTy>(mlir::LLVMTypeConverter &llvmTypeConverter, MLIRContext *context, TsLlvmContext *tsLlvmContext,
                        PatternBenefit benefit = 1)
        : OpConversionPattern<OpTy>::OpConversionPattern(llvmTypeConverter, context, benefit),
          tsLlvmContext(tsLlvmContext)
    {
    }

  protected:
    TsLlvmContext *tsLlvmContext;
};

class ConvertFOpLowering : public TsLlvmPattern<mlir_ts::ConvertFOp>
{
  public:
    using TsLlvmPattern<mlir_ts::ConvertFOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ConvertFOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        auto i8PtrTy = th.getPtrType();
        auto llvmI32Type = tch.convertType(th.getI32Type());
        auto llvmIndexType = tch.convertType(rewriter.getIndexType());

#ifdef WIN32
        auto sprintfFuncOp = ch.getOrInsertFunction(
            "sprintf_s", th.getFunctionType(rewriter.getI32Type(), {th.getPtrType(), llvmIndexType, th.getPtrType()}, true));
#else
        auto sprintfFuncOp = ch.getOrInsertFunction(
            "snprintf", th.getFunctionType(rewriter.getI32Type(), {th.getPtrType(), llvmIndexType, th.getPtrType()}, true));
#endif

        auto bufferSizeValue = transformed.getBufferSize();
        auto newStringValue = ch.MemoryAlloc(bufferSizeValue, MemoryAllocSet::Atomic);

        auto formatSpecifierValue = transformed.getFormat();

        SmallVector<mlir::Value> values;
        values.push_back(newStringValue);
        values.push_back(bufferSizeValue);
        values.push_back(formatSpecifierValue);
        for (auto inputVal : transformed.getInputs()) values.push_back(inputVal);

        rewriter.create<LLVM::CallOp>(loc, sprintfFuncOp, values);

        // Notify the rewriter that this operation has been removed.
        rewriter.replaceOp(op, newStringValue);

        return success();
    }
};

class AssertOpLowering : public TsLlvmPattern<mlir_ts::AssertOp>
{
  public:
    using TsLlvmPattern<mlir_ts::AssertOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AssertOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        AssertLogic al(op, rewriter, tch, op->getLoc(), tsLlvmContext->compileOptions);
        return al.logic(transformed.getArg(), op.getMsg().str());
    }
};

class PrintOpLowering : public TsLlvmPattern<mlir_ts::PrintOp>
{
  public:
    using TsLlvmPattern<mlir_ts::PrintOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PrintOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        CastLogicHelper castLogic(op, rewriter, tch, tsLlvmContext->compileOptions);

        auto loc = op->getLoc();

        auto i8PtrType = th.getPtrType();
        auto ptrType = th.getPtrType();

        // Get a symbol reference to the printf function, inserting it if necessary.
        auto putsFuncOp = ch.getOrInsertFunction("puts", th.getFunctionType(rewriter.getI32Type(), ptrType, false));

        auto strType = mlir_ts::StringType::get(rewriter.getContext());

        SmallVector<mlir::Value> values;
        mlir::Value spaceString;
        for (auto item : transformed.getInputs())
        {
            assert(item.getType() == i8PtrType);
            if (values.size() > 0)
            {
                if (!spaceString)
                {
                    spaceString = rewriter.create<mlir_ts::ConstantOp>(loc, strType, rewriter.getStringAttr(" "));
                }

                values.push_back(spaceString);
            }

            values.push_back(item);
        }

        if (values.size() > 1)
        {
            auto stack = rewriter.create<LLVM::StackSaveOp>(loc, i8PtrType);

            mlir::Value result =
                rewriter.create<mlir_ts::StringConcatOp>(loc, strType, values, rewriter.getBoolAttr(true));

            mlir::Value valueAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(strType), result);
            mlir::Value valueAsPtr = rewriter.create<LLVM::BitcastOp>(loc, ptrType, valueAsLLVMType);

            rewriter.create<LLVM::CallOp>(loc, putsFuncOp, valueAsPtr);

            rewriter.create<LLVM::StackRestoreOp>(loc, stack);
        }
        else
        {
            mlir::Value valueAsPtr = rewriter.create<LLVM::BitcastOp>(loc, ptrType, values.front());
            rewriter.create<LLVM::CallOp>(loc, putsFuncOp, valueAsPtr);
        }

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(op);

        return success();
    }
};

class ParseIntOpLowering : public TsLlvmPattern<mlir_ts::ParseIntOp>
{
  public:
    using TsLlvmPattern<mlir_ts::ParseIntOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParseIntOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        // Insert the `atoi` declaration if necessary.
        auto i8PtrTy = th.getPtrType();
        LLVM::LLVMFuncOp parseIntFuncOp;
        if (transformed.getBase())
        {
            parseIntFuncOp = ch.getOrInsertFunction(
                "strtol",
                th.getFunctionType(rewriter.getI32Type(), {i8PtrTy, th.getPtrType(), rewriter.getI32Type()}));
            auto nullOp = rewriter.create<LLVM::ZeroOp>(op->getLoc(), th.getPtrType());
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, parseIntFuncOp,
                                                      ValueRange{transformed.getArg(), nullOp, transformed.getBase()});
        }
        else
        {
            parseIntFuncOp = ch.getOrInsertFunction("atoi", th.getFunctionType(rewriter.getI32Type(), {i8PtrTy}));
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, parseIntFuncOp, ValueRange{transformed.getArg()});
        }

        return success();
    }
};

class ParseFloatOpLowering : public TsLlvmPattern<mlir_ts::ParseFloatOp>
{
  public:
    using TsLlvmPattern<mlir_ts::ParseFloatOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParseFloatOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto loc = op->getLoc();

        // Insert the `atof` declaration if necessary.
        auto i8PtrTy = th.getPtrType();
        auto parseFloatFuncOp = ch.getOrInsertFunction("atof", th.getFunctionType(rewriter.getF64Type(), {i8PtrTy}));

#ifdef NUMBER_F64
        auto funcCall = rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, parseFloatFuncOp, ValueRange{transformed.getArg()});
#else
        auto funcCall = rewriter.create<LLVM::CallOp>(loc, parseFloatFuncOp, ValueRange{transformed.getArg()});
        rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, rewriter.getF32Type(), funcCall.getResult());
#endif

        return success();
    }
};

class LoadLibraryPermanentlyOpLowering : public TsLlvmPattern<mlir_ts::LoadLibraryPermanentlyOp>
{
  public:
    using TsLlvmPattern<mlir_ts::LoadLibraryPermanentlyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LoadLibraryPermanentlyOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto i8PtrTy = th.getPtrType();

        auto loadLibraryPermanentlyFuncOp = ch.getOrInsertFunction("LLVMLoadLibraryPermanently", th.getFunctionType(rewriter.getI32Type(), {i8PtrTy}));
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, loadLibraryPermanentlyFuncOp, ValueRange{transformed.getFilename()});

        return success();
    }
};

class SearchForAddressOfSymbolOpLowering : public TsLlvmPattern<mlir_ts::SearchForAddressOfSymbolOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SearchForAddressOfSymbolOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SearchForAddressOfSymbolOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto i8PtrTy = th.getPtrType();

        auto searchForAddressOfSymbolFuncOp = ch.getOrInsertFunction("LLVMSearchForAddressOfSymbol", th.getFunctionType(i8PtrTy, {i8PtrTy}));
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, searchForAddressOfSymbolFuncOp, ValueRange{transformed.getSymbolName()});

        return success();
    }
};

class IsNaNOpLowering : public TsLlvmPattern<mlir_ts::IsNaNOp>
{
  public:
    using TsLlvmPattern<mlir_ts::IsNaNOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::IsNaNOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();

        TypeHelper th(rewriter.getContext());

        // icmp
        auto cmpValue = rewriter.create<LLVM::FCmpOp>(loc, th.getLLVMBoolType(), LLVM::FCmpPredicate::one,
                                                      transformed.getArg(), transformed.getArg());
        rewriter.replaceOp(op, ValueRange{cmpValue});
        return success();
    }
};

class SizeOfOpLowering : public TsLlvmPattern<mlir_ts::SizeOfOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SizeOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SizeOfOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        auto storageType = op.getType();
        // TODO: review usage SizeOf as for example when we allocate Array<Class1>  we use size of element but element is Pointer
        // if (auto classType = dyn_cast<mlir_ts::ClassType>(storageType))
        // {
        //     storageType = classType.getStorageType();
        // }
        // else if (auto valueRefType = dyn_cast<mlir_ts::ValueRefType>(storageType))
        // {
        //     storageType = valueRefType.getElementType();
        // }        
        // else if (auto objectType = dyn_cast<mlir_ts::ObjectType>(storageType))
        // {
        //     storageType = objectType.getStorageType();
        // }

        auto llvmStorageType = tch.convertType(storageType);
        mlir::Type llvmStorageTypePtr = th.getPtrType();
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto nullPtrToTypeValue = rewriter.create<LLVM::ZeroOp>(loc, llvmStorageTypePtr);

        LLVM_DEBUG(llvm::dbgs() << "\n!! size of - storage type: [" << storageType << "] llvm storage type: ["
                                << llvmStorageType << "] llvm ptr: [" << llvmStorageTypePtr << "]\n";);

        auto sizeOfSetAddr =
            rewriter.create<LLVM::GEPOp>(loc, llvmStorageTypePtr, llvmStorageType, nullPtrToTypeValue, ArrayRef<LLVM::GEPArg>{1});

        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, llvmIndexType, sizeOfSetAddr);

        return success();
    }
};

class LengthOfOpLowering : public TsLlvmPattern<mlir_ts::LengthOfOp>
{
  public:
    using TsLlvmPattern<mlir_ts::LengthOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LengthOfOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        auto loc = op->getLoc();

        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(op, typeConverter->convertType(th.getIndexType()), transformed.getOp(),
                                                                MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));

        return success();
    }
};

class SetLengthOfOpLowering : public TsLlvmPattern<mlir_ts::SetLengthOfOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SetLengthOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SetLengthOfOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(op, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = op.getLoc();

        auto ptrType = th.getPtrType();
        auto arrayType = cast<mlir_ts::ArrayType>(cast<mlir_ts::RefType>(op.getOp().getType()).getElementType());
        auto elementType = arrayType.getElementType();

        auto llvmArrayType = tch.convertType(arrayType);
        auto llvmElementType = tch.convertType(elementType);
        auto llvmIndexType = tch.convertType(th.getIndexType());        

        LLVM_DEBUG(llvm::dbgs() << "arrayType: elementType: " << elementType << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "arrayType: llvm: " << tch.convertType(arrayType) << "\n";);

        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                          ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, currentPtrPtr);

        auto countAsIndexTypePtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                              ArrayRef<LLVM::GEPArg>{0, ARRAY_SIZE_INDEX});
        auto countAsIndexType = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, countAsIndexTypePtr);
        auto newCountAsIndexType = op.getNewLength();

        auto sizeOfTypeAsIndexType = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);

        auto multSizeOfTypeValue =
            rewriter.create<mlir::index::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeAsIndexType, newCountAsIndexType});

        auto allocated = ch.MemoryRealloc(currentPtr, multSizeOfTypeValue);

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);

        auto newCountAsLLVMType = rewriter.create<mlir::index::CastUOp>(loc, llvmIndexType, newCountAsIndexType);
        rewriter.create<LLVM::StoreOp>(loc, newCountAsLLVMType, countAsIndexTypePtr);

        rewriter.eraseOp(op);

        return success();
    }
};

class StringLengthOpLowering : public TsLlvmPattern<mlir_ts::StringLengthOp>
{
  public:
    using TsLlvmPattern<mlir_ts::StringLengthOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StringLengthOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();
        auto i8PtrTy = th.getPtrType();
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto strlenFuncOp = ch.getOrInsertFunction("strlen", th.getFunctionType(llvmIndexType, {i8PtrTy}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, strlenFuncOp, transformed.getOp());
        return success();
    }
};

class SetStringLengthOpLowering : public TsLlvmPattern<mlir_ts::SetStringLengthOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SetStringLengthOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SetStringLengthOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        // TODO implement str concat
        auto i8PtrTy = th.getPtrType();

        mlir::Value ptr = transformed.getOp();
        mlir::Value size = transformed.getSize();

        mlir::Value strPtr = rewriter.create<LLVM::LoadOp>(loc, i8PtrTy, ptr);

        mlir::Value newStringValue = ch.MemoryRealloc(strPtr, size);

        rewriter.create<LLVM::StoreOp>(loc, newStringValue, ptr);
        rewriter.eraseOp(op);

        return success();
    }
};

class StringConcatOpLowering : public TsLlvmPattern<mlir_ts::StringConcatOp>
{
  public:
    using TsLlvmPattern<mlir_ts::StringConcatOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StringConcatOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        // TODO implement str concat
        auto i8PtrTy = th.getPtrType();
        auto i8PtrPtrTy = th.getPtrType();
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto strlenFuncOp = ch.getOrInsertFunction("strlen", th.getFunctionType(llvmIndexType, {i8PtrTy}));
        auto strcpyFuncOp = ch.getOrInsertFunction("strcpy", th.getFunctionType(i8PtrTy, {i8PtrTy, i8PtrTy}));
        auto strcatFuncOp = ch.getOrInsertFunction("strcat", th.getFunctionType(i8PtrTy, {i8PtrTy, i8PtrTy}));

        mlir::Value size = clh.createIndexConstantOf(llvmIndexType, 1);
        // calc size
        for (auto oper : transformed.getOps())
        {
            auto size1 = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, oper);
            size = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{size, size1.getResult()});
        }

        auto allocInStack = op.getAllocInStack().has_value() && op.getAllocInStack().value();

        mlir::Value newStringValue = allocInStack ? ch.Alloca(i8PtrTy, size, true)
                                                  : ch.MemoryAlloc(size);

        // copy
        auto concat = false;
        auto result = newStringValue;
        for (auto oper : transformed.getOps())
        {
            if (concat)
            {
                auto callResult = rewriter.create<LLVM::CallOp>(loc, strcatFuncOp, ValueRange{result, oper});
                result = callResult.getResult();
            }
            else
            {
                auto callResult = rewriter.create<LLVM::CallOp>(loc, strcpyFuncOp, ValueRange{result, oper});
                result = callResult.getResult();
            }

            concat = true;
        }

        rewriter.replaceOp(op, ValueRange{newStringValue});

        return success();
    }
};

class StringCompareOpLowering : public TsLlvmPattern<mlir_ts::StringCompareOp>
{
  public:
    using TsLlvmPattern<mlir_ts::StringCompareOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StringCompareOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        LLVMTypeConverterHelper llvmtch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));

        auto loc = op->getLoc();

        auto i8PtrTy = th.getPtrType();

        // compare bodies
        auto strcmpFuncOp = ch.getOrInsertFunction("strcmp", th.getFunctionType(th.getI32Type(), {i8PtrTy, i8PtrTy}));

        // compare ptrs first
        auto intPtrType = llvmtch.getIntPtrType(0);
        auto const0 = clh.createIConstantOf(llvmtch.getPointerBitwidth(0), 0);
        auto leftPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, transformed.getOp1());
        auto rightPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, transformed.getOp2());
        auto ptrCmpResult1 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, leftPtrValue, const0);
        auto ptrCmpResult2 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, rightPtrValue, const0);
        auto cmp32Result1 = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), ptrCmpResult1);
        auto cmp32Result2 = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), ptrCmpResult2);
        auto cmpResult = rewriter.create<LLVM::AndOp>(loc, cmp32Result1, cmp32Result2);
        auto const0I32 = clh.createI32ConstantOf(0);
        auto ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, cmpResult, const0I32);

        auto result = clh.conditionalExpressionLowering(
            loc, th.getBooleanType(), ptrCmpResult,
            [&](OpBuilder &builder, Location loc) {
                // both not null
                auto const0 = clh.createI32ConstantOf(0);
                auto compareResult =
                    rewriter.create<LLVM::CallOp>(loc, strcmpFuncOp, ValueRange{transformed.getOp1(), transformed.getOp2()});

                // else compare body
                mlir::Value bodyCmpResult;
                switch ((SyntaxKind)op.getCode())
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    bodyCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, compareResult.getResult(), const0);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    bodyCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, compareResult.getResult(), const0);
                    break;
                case SyntaxKind::GreaterThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt,
                                                                  compareResult.getResult(), const0);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge,
                                                                  compareResult.getResult(), const0);
                    break;
                case SyntaxKind::LessThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                                  compareResult.getResult(), const0);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle,
                                                                  compareResult.getResult(), const0);
                    break;
                default:
                    llvm_unreachable("not implemented");
                    return mlir::Value();
                }

                return bodyCmpResult;
            },
            [&](OpBuilder &builder, Location loc) {
                // else compare body
                mlir::Value ptrCmpResult;
                switch ((SyntaxKind)op.getCode())
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    ptrCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    ptrCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::GreaterThanToken:
                    ptrCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    ptrCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::LessThanToken:
                    ptrCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    ptrCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, leftPtrValue, rightPtrValue);
                    break;
                default:
                    llvm_unreachable("not implemented");
                    return mlir::Value();
                }

                return ptrCmpResult;
            });

        rewriter.replaceOp(op, result);

        return success();
    }
};

class AnyCompareOpLowering : public TsLlvmPattern<mlir_ts::AnyCompareOp>
{
  public:
    using TsLlvmPattern<mlir_ts::AnyCompareOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AnyCompareOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());
        LLVMTypeConverterHelper llvmtch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));

        auto loc = op->getLoc();

        AnyLogic al(op, rewriter, tch, loc, tsLlvmContext->compileOptions);
        //auto result = al.castToAny(in, transformed.getTypeInfo(), in.getType());

        auto i8PtrTy = th.getPtrType();
        auto llvmIndexType = llvmtch.typeConverter->convertType(th.getIndexType());

        // compare bodies
        auto memcmpFuncOp = ch.getOrInsertFunction("memcmp", th.getFunctionType(th.getI32Type(), {i8PtrTy, i8PtrTy, llvmIndexType}));

        // compare sizes of Any first
        // TODO: finish it

        auto sizeAny1 = al.getDataSizeOfAny(transformed.getOp1());
        auto sizeAny2 = al.getDataSizeOfAny(transformed.getOp2());

        auto ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, sizeAny1, sizeAny2);

        auto dataPtr1 = al.getDataPtrOfAny(transformed.getOp1());
        auto dataPtr2 = al.getDataPtrOfAny(transformed.getOp2());

        auto result = clh.conditionalExpressionLowering(
            loc, th.getBooleanType(), ptrCmpResult,
            [&](OpBuilder &builder, Location loc) {
                auto const0 = clh.createI32ConstantOf(0);
                // sizeAny1 equals sizeAny2
                auto compareResult =
                    rewriter.create<LLVM::CallOp>(loc, memcmpFuncOp, ValueRange{dataPtr1, dataPtr2, sizeAny1});

                // else compare body
                mlir::Value bodyCmpResult;
                switch ((SyntaxKind)op.getCode())
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    bodyCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, compareResult.getResult(), const0);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    bodyCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, compareResult.getResult(), const0);
                    break;
                case SyntaxKind::GreaterThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt,
                                                                  compareResult.getResult(), const0);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge,
                                                                  compareResult.getResult(), const0);
                    break;
                case SyntaxKind::LessThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                                  compareResult.getResult(), const0);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle,
                                                                  compareResult.getResult(), const0);
                    break;
                default:
                    llvm_unreachable("not implemented");
                    return mlir::Value();
                }

                return bodyCmpResult;
            },
            [&](OpBuilder &builder, Location loc) {
                return ptrCmpResult;
            });

        rewriter.replaceOp(op, result);

        return success();
    }
};

class CharToStringOpLowering : public TsLlvmPattern<mlir_ts::CharToStringOp>
{
  public:
    using TsLlvmPattern<mlir_ts::CharToStringOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CharToStringOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        CodeLogicHelper clh(op, rewriter);
        LLVMCodeHelper ch(op, rewriter, typeConverter, tsLlvmContext->compileOptions);

        auto loc = op->getLoc();

        auto charType = mlir_ts::CharType::get(rewriter.getContext());

        auto i8PtrTy = th.getPtrType();

        auto bufferSizeValue = clh.createI64ConstantOf(2);
        // TODO: review it, !! we can't allocate it in stack - otherwise when returned back from function, it will be poisned
        // TODO: maybe you need to add mechanizm to convert stack values to heap when returned from function
        //auto newStringValue = ch.Alloca(i8PtrTy, bufferSizeValue, true);
        auto newStringValue = ch.MemoryAlloc(bufferSizeValue);

        auto index0Value = clh.createI32ConstantOf(0);
        auto index1Value = clh.createI32ConstantOf(1);
        auto nullCharValue = clh.createI8ConstantOf(0);
        auto addr0 = ch.GetAddressOfArrayElement(charType, newStringValue.getType(), newStringValue, index0Value);
        rewriter.create<LLVM::StoreOp>(loc, transformed.getOp(), addr0);
        auto addr1 = ch.GetAddressOfArrayElement(charType, newStringValue.getType(), newStringValue, index1Value);
        rewriter.create<LLVM::StoreOp>(loc, nullCharValue, addr1);

        rewriter.replaceOp(op, ValueRange{newStringValue});

        return success();
    }
};

struct ConstantOpLowering : public TsLlvmPattern<mlir_ts::ConstantOp>
{
    using TsLlvmPattern<mlir_ts::ConstantOp>::TsLlvmPattern;

    template <typename T, typename TOp>
    void getArrayValue(TOp constantOp, T type, ConversionPatternRewriter &rewriter) const
    {
        LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto elementType = cast<T>(type).getElementType();
        auto llvmElementType = tch.convertType(elementType);
        auto arrayAttr = dyn_cast_or_null<ArrayAttr>(constantOp.getValue());

        auto arrayValue =
            ch.getArrayValue(elementType, llvmElementType, arrayAttr.size(), arrayAttr);

        rewriter.replaceOp(constantOp, arrayValue);
    }

    template <typename T, typename TOp>
    void getOrCreateGlobalArray(TOp constantOp, T type, ConversionPatternRewriter &rewriter) const
    {
        LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto elementType = cast<T>(type).getElementType();

        LLVM_DEBUG(llvm::dbgs() << "constArrayType: elementType: "; elementType.dump(); llvm::dbgs() << "\n";);

        auto llvmElementType = tch.convertType(elementType);

        LLVM_DEBUG(llvm::dbgs() << "constArrayType: llvmElementType: "; llvmElementType.dump(); llvm::dbgs() << "\n";);

        auto arrayAttr = dyn_cast_or_null<ArrayAttr>(constantOp.getValue());

        LLVM_DEBUG(llvm::dbgs() << "constArrayType: arrayAttr: "; arrayAttr.dump(); llvm::dbgs() << "\n";);

        auto arrayFirstElementAddrCst =
            ch.getOrCreateGlobalArray(elementType, arrayAttr.size(), arrayAttr);

        rewriter.replaceOp(constantOp, arrayFirstElementAddrCst);
    }

    template <typename TOp>
    void getOrCreateGlobalTuple(TOp constantOp, mlir::Type type, ConversionPatternRewriter &rewriter) const
    {
        auto location = constantOp->getLoc();

        LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto arrayAttr = dyn_cast_or_null<ArrayAttr>(constantOp.getValue());

        auto convertedTupleType = tch.convertType(type);
        /*
        auto tupleConstPtr = ch.getOrCreateGlobalTuple(cast<mlir_ts::ConstTupleType>(type),
                                                       cast<LLVM::LLVMStructType>(convertedTupleType),
        arrayAttr);

        // optimize it and replace it with copy memory. (use canon. pass) check  "EraseRedundantAssertions"
        auto loadedValue = rewriter.create<LLVM::LoadOp>(constantOp->getLoc(), tupleConstPtr);
        */

        auto tupleVal = ch.getTupleFromArrayAttr(location, dyn_cast<mlir_ts::ConstTupleType>(type),
                                                 mlir::cast<LLVM::LLVMStructType>(convertedTupleType), arrayAttr);

        // rewriter.replaceOp(constantOp, ValueRange{loadedValue});
        rewriter.replaceOp(constantOp, ValueRange{tupleVal});
    }

    LogicalResult matchAndRewrite(mlir_ts::ConstantOp constantOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        // load address of const string
        auto type = constantOp.getType();
        if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            type = literalType.getElementType();
        }

        if (isa<mlir_ts::StringType>(type))
        {
            LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

            auto strValue = cast<StringAttr>(constantOp.getValue()).getValue().str();
            auto txtCst = ch.getOrCreateGlobalString(strValue);

            rewriter.replaceOp(constantOp, txtCst);

            return success();
        }

        TypeConverterHelper tch(getTypeConverter());
        if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            LLVM_DEBUG(llvm::dbgs() << "constArrayType: type: "; type.dump(); llvm::dbgs() << "\n";);

            getOrCreateGlobalArray(constantOp, constArrayType, rewriter);
            return success();
        }

        if (auto constArrayValueType = dyn_cast<mlir_ts::ConstArrayValueType>(type))
        {
            if (auto arrayAttr = dyn_cast_or_null<ArrayAttr>(constantOp.getValue()))
            {
                getArrayValue(constantOp, constArrayValueType, rewriter);
                return success();
            }
        }

        if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
        {
            getOrCreateGlobalArray(constantOp, arrayType, rewriter);
            return success();
        }

        if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            getOrCreateGlobalTuple(constantOp, constTupleType, rewriter);
            return success();
        }

        if (auto tupleType = dyn_cast<mlir_ts::TupleType>(type))
        {
            getOrCreateGlobalTuple(constantOp, tupleType, rewriter);
            return success();
        }

        if (auto enumType = dyn_cast<mlir_ts::EnumType>(type))
        {
            rewriter.eraseOp(constantOp);
            return success();
        }

        if (auto valAttr = dyn_cast<mlir::FlatSymbolRefAttr>(constantOp.getValue()))
        {
            rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(constantOp, tch.convertType(type), valAttr);
            return success();
        }

        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(constantOp, tch.convertType(type), constantOp.getValue());
        return success();
    }
};

struct SymbolRefOpLowering : public TsLlvmPattern<mlir_ts::SymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::SymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SymbolRefOp symbolRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(symbolRefOp, tch.convertType(symbolRefOp.getType()),
                                                      symbolRefOp.getIdentifierAttr());
        return success();
    }
};

struct NullOpLowering : public TsLlvmPattern<mlir_ts::NullOp>
{
    using TsLlvmPattern<mlir_ts::NullOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NullOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, tch.convertType(op.getType()));
        return success();
    }
};

class UndefOpLowering : public TsLlvmPattern<mlir_ts::UndefOp>
{
  public:
    using TsLlvmPattern<mlir_ts::UndefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::UndefOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        if (isa<mlir_ts::OptionalType>(op.getType()))
        {
            rewriter.replaceOpWithNewOp<mlir_ts::OptionalUndefOp>(op, op.getType());
            return success();
        }

        TypeConverterHelper tch(getTypeConverter());
        rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, tch.convertType(op.getType()));
        return success();
    }
};

struct ReturnInternalOpLowering : public TsLlvmPattern<mlir_ts::ReturnInternalOp>
{
    using TsLlvmPattern<mlir_ts::ReturnInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ReturnInternalOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{transformed.getRetOperands()});
        return success();
    }
};

struct FuncOpLowering : public TsLlvmPattern<mlir_ts::FuncOp>
{
    using TsLlvmPattern<mlir_ts::FuncOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::FuncOp funcOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto location = funcOp.getLoc();

        auto typeConverter = getTypeConverter();
        auto fnType = funcOp.getFunctionType();

        TypeConverter::SignatureConversion signatureInputsConverter(fnType.getNumInputs());
        for (auto argType : enumerate(funcOp.getFunctionType().getInputs()))
        {
            auto convertedType = typeConverter->convertType(argType.value());
            signatureInputsConverter.addInputs(argType.index(), convertedType);
        }

        TypeConverter::SignatureConversion signatureResultsConverter(fnType.getNumResults());
        for (auto argType : enumerate(funcOp.getFunctionType().getResults()))
        {
            auto convertedType = typeConverter->convertType(argType.value());
            signatureResultsConverter.addInputs(argType.index(), convertedType);
        }

        SmallVector<DictionaryAttr> argDictAttrs;
        if (ArrayAttr argAttrs = funcOp.getAllArgAttrs())
        {
            auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
            argDictAttrs.append(argAttrRange.begin(), argAttrRange.end());
        }

        auto convertedFuncType = rewriter.getFunctionType(signatureInputsConverter.getConvertedTypes(), signatureResultsConverter.getConvertedTypes());
        auto newFuncOp = rewriter.create<mlir::func::FuncOp>(location, funcOp.getName(), convertedFuncType, ArrayRef<NamedAttribute>{}, argDictAttrs);

        for (const auto &namedAttr : funcOp->getAttrs())
        {
            if (namedAttr.getName() == funcOp.getFunctionTypeAttrName())
            {
                continue;
            }

            if (namedAttr.getName() == SymbolTable::getSymbolAttrName())
            {
                //name = dyn_cast<mlir::StringAttr>(namedAttr.getValue()).getValue().str();
                continue;
            }

            newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
        }

        SmallVector<mlir::Attribute> funcAttrs;

        if (funcOp.getPersonality().has_value() && funcOp.getPersonality().value())
        {
            LLVMRTTIHelperVC rttih(funcOp, rewriter, typeConverter, tsLlvmContext->compileOptions);
            rttih.setPersonality(newFuncOp);

            funcAttrs.push_back(ATTR("noinline"));
        }

        // copy attributes over
        auto skipAttrs = mlir::func::FuncOp::getAttributeNames();
        for (auto attr : funcOp->getAttrs())
        {
            auto name = attr.getName();
            if (name == "specialization") {
                auto inlineLinkage = LLVM::linkage::Linkage::LinkonceODR;
                auto linkage = LLVM::LinkageAttr::get(getContext(), inlineLinkage);
                newFuncOp->setAttr("llvm.linkage", linkage);
                // TODO: dso_local somehow linked with -fno-pic
                //newFuncOp->setAttr("dso_local", rewriter.getUnitAttr());
                newFuncOp.setPrivate();

                addComdat(newFuncOp, rewriter);
                continue;
            }

            auto addAttr = 
                std::find(skipAttrs.begin(), skipAttrs.end(), name) == skipAttrs.end();
            if (addAttr) 
                funcAttrs.push_back(name);
        }

#ifdef DISABLE_OPT
        // add LLVM attributes to fix issue with shift >> 32
        funcAttrs.append({
            ATTR("noinline"),
            // ATTR("norecurse"),
            // ATTR("nounwind"),
            ATTR("optnone"),
            // ATTR("uwtable"),
            // NAMED_ATTR("correctly-rounded-divide-sqrt-fp-math","false"),
            // NAMED_ATTR("disable-tail-calls","false"),
            // NAMED_ATTR("frame-pointer","none"),
            // NAMED_ATTR("less-precise-fpmad","false"),
            // NAMED_ATTR("min-legal-vector-width","0"),
            // NAMED_ATTR("no-infs-fp-math","false"),
            // NAMED_ATTR("no-jump-tables","false"),
            // NAMED_ATTR("no-nans-fp-math","false"),
            // NAMED_ATTR("no-signed-zeros-fp-math","false"),
            // NAMED_ATTR("no-trapping-math","true"),
            // NAMED_ATTR("stack-protector-buffer-size","8"),
            // NAMED_ATTR("target-cpu","x86-64"),
            // NAMED_ATTR("target-features","+cx8,+fxsr,+mmx,+sse,+sse2,+x87"),
            // NAMED_ATTR("tune-cpu","generic"),
            // NAMED_ATTR("unsafe-fp-math","false"),
            // NAMED_ATTR("use-soft-float","false"),
        }));
#endif

        if (funcAttrs.size() > 0)
        {
            newFuncOp->setAttr("passthrough", ArrayAttr::get(rewriter.getContext(), funcAttrs));
        }

        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, &signatureInputsConverter)))
        {
            return failure();
        }

        rewriter.eraseOp(funcOp);

        return success();
    }

    static void addComdat(mlir::func::FuncOp &func,
                        mlir::ConversionPatternRewriter &rewriter) {
        auto module = func->getParentOfType<mlir::ModuleOp>();

        const char *comdatName = "__llvm_comdat";
        mlir::LLVM::ComdatOp comdatOp = module.lookupSymbol<mlir::LLVM::ComdatOp>(comdatName);
        if (!comdatOp) 
        {
            comdatOp = rewriter.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
        }

        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&comdatOp.getBody().back());
        auto selectorOp = rewriter.create<mlir::LLVM::ComdatSelectorOp>(
            comdatOp.getLoc(), func.getSymName(),
            mlir::LLVM::comdat::Comdat::Any);
        func->setAttr("comdat", mlir::SymbolRefAttr::get(
            rewriter.getContext(), comdatName,
            mlir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
    }
};

struct SymbolCallInternalOpLowering : public TsLlvmPattern<mlir_ts::SymbolCallInternalOp>
{
    using TsLlvmPattern<mlir_ts::SymbolCallInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SymbolCallInternalOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

        auto funcOp = moduleOp.lookupSymbol(op.getCallee());

        LLVM::LLVMFunctionType llvmFuncType;
        if (auto llvmFuncOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(funcOp))
        {
            llvmFuncType = llvmFuncOp.getFunctionType();
        }
        else if (auto mlirFuncOp = dyn_cast_or_null<func::FuncOp>(funcOp))
        {
            auto funcType = mlirFuncOp.getFunctionType();
            llvmFuncType = tch.convertFunctionSignature(op.getContext(), funcType.getResults(), funcType.getInputs(), false);
        }
        else if (auto tsFuncOp = dyn_cast_or_null<mlir_ts::FuncOp>(funcOp))
        {
            auto tsFuncType = tsFuncOp.getFunctionType();
            llvmFuncType = tch.convertFunctionSignature(op.getContext(), tsFuncType.getResults(), tsFuncType.getInputs(), tsFuncType.isVarArg());
        }

        assert(llvmFuncType);

        auto callRes = rewriter.create<LLVM::CallOp>(
            loc, llvmFuncType, ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), op.getCallee()), transformed.getCallOperands());
        
        auto returns = callRes.getResults();
        if (returns.size() > 0)
        {
            rewriter.replaceOp(op, returns);
        }
        else
        {
            rewriter.eraseOp(op);
        }        

        return success();
    }
};

struct CallInternalOpLowering : public TsLlvmPattern<mlir_ts::CallInternalOp>
{
    using TsLlvmPattern<mlir_ts::CallInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallInternalOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
      

        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());
        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            if (isa<mlir_ts::VoidType>(type))
            {
                continue;
            }

            llvmTypes.push_back(tch.convertType(type));
        }

        // special case for HybridFunctionType
        LLVM_DEBUG(llvm::dbgs() << "\n!! CallInternalOp - arg #0:" << op.getOperand(0) << "\n");
        if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(op.getOperand(0).getType()))
        {
            rewriter.replaceOpWithNewOp<mlir_ts::CallHybridInternalOp>(
                op, hybridFuncType.getResults(), op.getOperand(0),
                OperandRange(op.getOperands().begin() + 1, op.getOperands().end()));
            return success();
        }

        auto callRes = rewriter.create<LLVM::CallOp>(loc, llvmTypes, transformed.getOperands());

        auto returns = callRes.getResults();
        if (returns.size() > 0)
        {
            rewriter.replaceOp(op, returns);
        }
        else
        {
            rewriter.eraseOp(op);
        }      

        return success();
    }
};

struct CallHybridInternalOpLowering : public TsLlvmPattern<mlir_ts::CallHybridInternalOp>
{
    using TsLlvmPattern<mlir_ts::CallHybridInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallHybridInternalOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto hybridFuncType = cast<mlir_ts::HybridFunctionType>(op.getCallee().getType());

        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            if (isa<mlir_ts::VoidType>(type))
            {
                continue;
            }

            llvmTypes.push_back(tch.convertType(type));
        }

        mlir::ValueRange returns;
        mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
        {
            CodeLogicHelper clh(loc, rewriter);

            // test 'this'
            auto thisType = mlir_ts::OpaqueType::get(rewriter.getContext());
            auto thisVal = rewriter.create<mlir_ts::GetThisOp>(loc, thisType, op.getOperand(0));
            auto thisAsBoolVal =
                rewriter.create<mlir_ts::CastOp>(loc, mlir_ts::BooleanType::get(rewriter.getContext()), thisVal);

            SmallVector<mlir::Type, 4> results;
            for (auto &resultType : hybridFuncType.getResults())
            {
                if (isa<mlir_ts::VoidType>(resultType))
                {
                    continue;
                }

                results.push_back(resultType);
            }

            // no value yet.
            returns = clh.conditionalBlocksLowering(
                results, thisAsBoolVal,
                [&](OpBuilder &builder, Location loc) {
                    mlir::SmallVector<mlir::Type> inputs;
                    inputs.push_back(thisType);
                    inputs.append(hybridFuncType.getInputs().begin(), hybridFuncType.getInputs().end());

                    // with this
                    auto funcType =
                        mlir_ts::FunctionType::get(rewriter.getContext(), inputs, hybridFuncType.getResults());
                    auto methodPtr = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, op.getOperand(0));

                    auto methodPtrAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(funcType), methodPtr);
                    auto thisValAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(thisType), thisVal);

                    mlir::SmallVector<mlir::Value> ops;
                    ops.push_back(methodPtrAsLLVMType);
                    ops.push_back(thisValAsLLVMType);
                    ops.append(transformed.getOperands().begin() + 1, transformed.getOperands().end());
                    auto callRes = rewriter.create<LLVM::CallOp>(loc, llvmTypes, ops);
                    return callRes.getResults();
                },
                [&](OpBuilder &builder, Location loc) {
                    // no this
                    auto funcType = mlir_ts::FunctionType::get(rewriter.getContext(), hybridFuncType.getInputs(),
                                                               hybridFuncType.getResults());
                    auto methodPtr = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, op.getOperand(0));

                    auto methodPtrAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(funcType), methodPtr);

                    mlir::SmallVector<mlir::Value> ops;
                    ops.push_back(methodPtrAsLLVMType);
                    ops.append(transformed.getOperands().begin() + 1, transformed.getOperands().end());
                    auto callRes = rewriter.create<LLVM::CallOp>(loc, llvmTypes, ops);
                    return callRes.getResults();
                });
        }

        if (returns.size() > 0)
        {
            rewriter.replaceOp(op, returns);
        }
        else
        {
            rewriter.eraseOp(op);
        }

        return success();
    }
};

struct InvokeOpLowering : public TsLlvmPattern<mlir_ts::InvokeOp>
{
    using TsLlvmPattern<mlir_ts::InvokeOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InvokeOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {

        if (!op.getCallee().has_value())
        {
            // special case for HybridFunctionType
            LLVM_DEBUG(llvm::dbgs() << "\n!! InvokeOp - arg #0:" << op.getOperand(0) << "\n");
            if (auto hybridFuncType = dyn_cast<mlir_ts::HybridFunctionType>(op.getOperand(0).getType()))
            {
                rewriter.replaceOpWithNewOp<mlir_ts::InvokeHybridOp>(
                    op, hybridFuncType.getResults(), op.getOperand(0),
                    OperandRange(op.getOperands().begin() + 1, op.getOperands().end()), op.getNormalDestOperands(),
                    op.getUnwindDestOperands(), op.getNormalDest(), op.getUnwindDest());
                return success();
            }
        }

        TypeConverterHelper tch(getTypeConverter());
        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            llvmTypes.push_back(tch.convertType(type));
        }

        // just replace
        if (op.getCalleeAttr())
        {
            rewriter.replaceOpWithNewOp<LLVM::InvokeOp>(op, llvmTypes, op.getCalleeAttr(), transformed.getCallOperands(),
                op.getNormalDest(), transformed.getNormalDestOperands(), op.getUnwindDest(),
                transformed.getUnwindDestOperands());
        }
        else
        {
            rewriter.replaceOpWithNewOp<LLVM::InvokeOp>(op, llvmTypes, mlir::FlatSymbolRefAttr(), transformed.getOperands(),
                op.getNormalDest(), transformed.getNormalDestOperands(), op.getUnwindDest(),
                transformed.getUnwindDestOperands());
        }
        
        return success();
    }
};

struct InvokeHybridOpLowering : public TsLlvmPattern<mlir_ts::InvokeHybridOp>
{
    using TsLlvmPattern<mlir_ts::InvokeHybridOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InvokeHybridOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {

        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto hybridFuncType = cast<mlir_ts::HybridFunctionType>(op.getCallee().getType());

        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            llvmTypes.push_back(tch.convertType(type));
        }

        mlir::ValueRange returns;
        mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
        {
            CodeLogicHelper clh(loc, rewriter);

            // test 'this'
            auto thisType = mlir_ts::OpaqueType::get(rewriter.getContext());
            auto thisVal = rewriter.create<mlir_ts::GetThisOp>(loc, thisType, op.getOperand(0));
            auto thisAsBoolVal =
                rewriter.create<mlir_ts::CastOp>(loc, mlir_ts::BooleanType::get(rewriter.getContext()), thisVal);

            SmallVector<mlir::Type, 4> results;
            for (auto &resultType : hybridFuncType.getResults())
            {
                if (isa<mlir_ts::VoidType>(resultType))
                {
                    continue;
                }

                results.push_back(resultType);
            }

            // no value yet.
            returns = clh.conditionalBlocksLowering(
                results, thisAsBoolVal,
                [&](OpBuilder &builder, Location loc) {
                    mlir::SmallVector<mlir::Type> inputs;
                    inputs.push_back(thisType);
                    inputs.append(hybridFuncType.getInputs().begin(), hybridFuncType.getInputs().end());

                    // with this
                    auto funcType =
                        mlir_ts::FunctionType::get(rewriter.getContext(), inputs, hybridFuncType.getResults());
                    auto methodPtr = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, op.getOperand(0));

                    mlir::Value methodPtrAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(methodPtr.getType()), methodPtr);
                    mlir::Value thisValAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(thisVal.getType()), thisVal);

                    mlir::SmallVector<mlir::Value> ops;
                    ops.push_back(methodPtrAsLLVMType);
                    ops.push_back(thisValAsLLVMType);
                    ops.append(transformed.getOperands().begin() + 1, transformed.getOperands().end());

                    auto *continuationBlock = clh.CutBlockAndSetInsertPointToEndOfBlock();

                    auto callRes = rewriter.create<LLVM::InvokeOp>(loc, TypeRange(llvmTypes), mlir::FlatSymbolRefAttr(), ops, continuationBlock,
                                                                   transformed.getNormalDestOperands(), 
                                                                   op.getUnwindDest(), transformed.getUnwindDestOperands());

                    rewriter.setInsertionPointToStart(continuationBlock);

                    return callRes.getResults();
                },
                [&](OpBuilder &builder, Location loc) {
                    // no this
                    auto funcType = mlir_ts::FunctionType::get(rewriter.getContext(), hybridFuncType.getInputs(),
                                                               hybridFuncType.getResults());
                    auto methodPtr = rewriter.create<mlir_ts::GetMethodOp>(loc, funcType, op.getOperand(0));

                    mlir::Value methodPtrAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(methodPtr.getType()), methodPtr);

                    mlir::SmallVector<mlir::Value> ops;
                    ops.push_back(methodPtrAsLLVMType);
                    ops.append(transformed.getOperands().begin() + 1, transformed.getOperands().end());

                    auto *continuationBlock = clh.CutBlockAndSetInsertPointToEndOfBlock();

                    auto callRes = rewriter.create<LLVM::InvokeOp>(loc, TypeRange(llvmTypes), mlir::FlatSymbolRefAttr(), ops, continuationBlock,
                                                                   transformed.getNormalDestOperands(), op.getUnwindDest(),
                                                                   transformed.getUnwindDestOperands());

                    rewriter.setInsertionPointToStart(continuationBlock);

                    return callRes.getResults();
                });

            rewriter.create<LLVM::BrOp>(loc, ValueRange{}, op.getNormalDest());
        }

        if (returns.size() > 0)
        {
            rewriter.replaceOp(op, returns);
        }
        else
        {
            rewriter.eraseOp(op);
        }

        return success();
    }
};

struct DialectCastOpLowering : public TsLlvmPattern<mlir_ts::DialectCastOp>
{
    using TsLlvmPattern<mlir_ts::DialectCastOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DialectCastOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();

        

        TypeConverterHelper tch(getTypeConverter());

        auto in = transformed.getIn();
        auto resType = op.getRes().getType();

        CastLogicHelper castLogic(op, rewriter, tch, tsLlvmContext->compileOptions);
        auto [result, converted] = castLogic.dialectCast(in, in.getType(), resType);
        if (!converted)
        {
            rewriter.replaceOp(op, in);
            return success();
        }

        if (!result)
        {
            return failure();
        }

        rewriter.replaceOp(op, result);
        return success();
    }
};

struct CastOpLowering : public TsLlvmPattern<mlir_ts::CastOp>
{
    using TsLlvmPattern<mlir_ts::CastOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CastOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto in = transformed.getIn();
        auto resType = op.getRes().getType();

        CastLogicHelper castLogic(op, rewriter, tch, tsLlvmContext->compileOptions);
        // in case of Union we need mlir_ts::UnionType value
        auto result = castLogic.cast(isa<mlir_ts::UnionType>(op.getIn().getType()) ? op.getIn() : in, op.getIn().getType(), resType);
        if (!result)
        {
            return failure();
        }

        rewriter.replaceOp(op, result);

        return success();
    }
};

struct BoxOpLowering : public TsLlvmPattern<mlir_ts::BoxOp>
{
    using TsLlvmPattern<mlir_ts::BoxOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::BoxOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto in = transformed.getIn();

        AnyLogic al(op, rewriter, tch, loc, tsLlvmContext->compileOptions);
        auto result = al.castToAny(in, transformed.getTypeInfo(), in.getType());

        rewriter.replaceOp(op, result);

        return success();
    }
};

struct UnboxOpLowering : public TsLlvmPattern<mlir_ts::UnboxOp>
{
    using TsLlvmPattern<mlir_ts::UnboxOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::UnboxOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto in = transformed.getIn();
        auto resType = op.getRes().getType();

        AnyLogic al(op, rewriter, tch, loc, tsLlvmContext->compileOptions);
        auto result = al.UnboxAny(in, tch.convertType(resType));

        rewriter.replaceOp(op, result);

        return success();
    }
};

struct CreateUnionInstanceOpLowering : public TsLlvmPattern<mlir_ts::CreateUnionInstanceOp>
{
    using TsLlvmPattern<mlir_ts::CreateUnionInstanceOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateUnionInstanceOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(op, rewriter);
        MLIRTypeHelper mth(rewriter.getContext(), tsLlvmContext->compileOptions);

        CastLogicHelper castLogic(op, rewriter, tch, tsLlvmContext->compileOptions);

        auto in = transformed.getIn();

        auto i8PtrTy = th.getPtrType();
        auto valueType = in.getType();
        auto resType = tch.convertType(op.getRes().getType());

        mlir::SmallVector<mlir::Type> types;
        types.push_back(i8PtrTy);
        types.push_back(valueType);
        auto unionPartialType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types, UNION_TYPE_PACKED);
        if (!mth.isUnionTypeNeedsTag(loc, cast<mlir_ts::UnionType>(op.getType())))
        {
            // this is union of tuples, no need to add Tag to it
            // create tagged union
            auto casted = castLogic.castLLVMTypes(in, in.getType(), op.getType(), resType);
            if (!casted)
            {
                return mlir::failure();
            }

            rewriter.replaceOp(op, ValueRange{casted});
        }
        else
        {
            // create tagged union
            auto udefVal = rewriter.create<LLVM::UndefOp>(loc, unionPartialType);
            auto val0 =
                rewriter.create<LLVM::InsertValueOp>(loc, udefVal, transformed.getTypeInfo(), MLIRHelper::getStructIndex(rewriter, UNION_TAG_INDEX));
            auto val1 = rewriter.create<LLVM::InsertValueOp>(loc, val0, in, MLIRHelper::getStructIndex(rewriter, UNION_VALUE_INDEX));

            auto casted = castLogic.castLLVMTypes(val1, unionPartialType, op.getType(), resType);
            if (!casted)
            {
                return mlir::failure();
            }

            rewriter.replaceOp(op, ValueRange{casted});
        }

        return success();
    }
};

struct GetValueFromUnionOpLowering : public TsLlvmPattern<mlir_ts::GetValueFromUnionOp>
{
    using TsLlvmPattern<mlir_ts::GetValueFromUnionOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GetValueFromUnionOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = op->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(op, rewriter);
        MLIRTypeHelper mth(rewriter.getContext(), tsLlvmContext->compileOptions);

        bool needTag = mth.isUnionTypeNeedsTag(loc, cast<mlir_ts::UnionType>(op.getIn().getType()));
        if (needTag)
        {
            auto in = transformed.getIn();

            auto i8PtrTy = th.getPtrType();
            auto valueType = tch.convertType(op.getType());

            mlir::SmallVector<mlir::Type> types;
            types.push_back(i8PtrTy);
            types.push_back(valueType);
            auto unionPartialType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types, UNION_TYPE_PACKED);

            // TODO: should not cast anything
            CastLogicHelper castLogic(op, rewriter, tch, tsLlvmContext->compileOptions);
            auto casted = castLogic.castLLVMTypes(transformed.getIn(), transformed.getIn().getType(), unionPartialType,
                                                  unionPartialType);
            if (!casted)
            {
                return mlir::failure();
            }

            auto val1 = rewriter.create<LLVM::ExtractValueOp>(loc, valueType, casted, MLIRHelper::getStructIndex(rewriter, UNION_VALUE_INDEX));

            rewriter.replaceOp(op, ValueRange{val1});
        }
        else
        {
            rewriter.replaceOp(op, ValueRange{transformed.getIn()});
        }

        return success();
    }
};

struct GetTypeInfoFromUnionOpLowering : public TsLlvmPattern<mlir_ts::GetTypeInfoFromUnionOp>
{
    using TsLlvmPattern<mlir_ts::GetTypeInfoFromUnionOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GetTypeInfoFromUnionOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(op, rewriter);
        MLIRTypeHelper mth(rewriter.getContext(), tsLlvmContext->compileOptions);

        auto loc = op->getLoc();

        mlir::Type baseType;
        bool needTag = mth.isUnionTypeNeedsTag(loc, cast<mlir_ts::UnionType>(op.getIn().getType()), baseType);
        if (needTag)
        {
            auto val0 = rewriter.create<LLVM::ExtractValueOp>(loc, tch.convertType(op.getType()), transformed.getIn(),
                                                              MLIRHelper::getStructIndex(rewriter, UNION_TAG_INDEX));

            rewriter.replaceOp(op, ValueRange{val0});
        }
        else
        {
            auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), transformed.getIn());

            rewriter.replaceOp(op, ValueRange{typeOfValue});
        }

        return success();
    }
};

struct VariableOpLowering : public TsLlvmPattern<mlir_ts::VariableOp>
{
    using TsLlvmPattern<mlir_ts::VariableOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::VariableOp varOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(varOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(varOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto location = varOp.getLoc();

        auto referenceType = varOp.getType();
        auto storageType = tch.convertType(referenceType.getElementType());

#ifdef ALLOC_ALL_VARS_IN_HEAP
        auto isCaptured = true;
#elif ALLOC_CAPTURED_VARS_IN_HEAP
        auto isCaptured = varOp.getCaptured().has_value() && varOp.getCaptured().value();
#else
        auto isCaptured = false;
#endif

        LLVM_DEBUG(llvm::dbgs() << "\n!! variable op: " << varOp
                                << "\n";);

        LLVM_DEBUG(llvm::dbgs() << "\n!! variable allocation: " << storageType << " is captured: " << isCaptured
                                << "\n";);

        mlir::Value allocated;
        if (!isCaptured)
        {
            auto count = 1;
            if (varOp->hasAttrOfType<mlir::IntegerAttr>(INSTANCES_COUNT_ATTR_NAME))
            {
                auto intAttr = varOp->getAttrOfType<mlir::IntegerAttr>(INSTANCES_COUNT_ATTR_NAME);
                count = intAttr.getValue().getZExtValue();
            }

            allocated = ch.Alloca(storageType, count);
        }
        else
        {

            allocated = ch.MemoryAlloc(storageType);
        }

#ifdef GC_ENABLE
        // register root which is in stack, if you call Malloc - it is not in stack anymore
        if (!isCaptured)
        {
            auto  tsStorageType = referenceType.getElementType();
            if (isa<mlir_ts::ClassType>(tsStorageType) || isa<mlir_ts::StringType>(tsStorageType) ||
 isa<mlir_ts::ArrayType>(tsStorageType) || isa<mlir_ts::ObjectType>(tsStorageType) ||
 isa<mlir_ts::AnyType>(tsStorageType))
            {
                TypeHelper th(rewriter);

                auto gcRootOp = ch.getOrInsertFunction(
                    "llvm.gcroot", th.getFunctionType(th.getVoidType(), {th.getPtrType(), th.getPtrType()}));
                auto nullPtr = rewriter.create<LLVM::ZeroOp>(location, th.getPtrType());
                rewriter.create<LLVM::CallOp>(location, gcRootOp, ValueRange{allocated, nullPtr});
            }
        }
#endif

        LLVM::DILocalVariableAttr varInfo;
        if (tsLlvmContext->compileOptions.generateDebugInfo)
        {
            if (auto localVarAttrFusedLoc = dyn_cast<mlir::FusedLocWith<LLVM::DILocalVariableAttr>>(location))
            {
                varInfo = localVarAttrFusedLoc.getMetadata();
                rewriter.create<LLVM::DbgDeclareOp>(location, allocated, varInfo);

                // TODO: do I need it? does it have an effect?
                allocated.getDefiningOp()->setLoc(mlir::FusedLoc::get(rewriter.getContext(), {allocated.getDefiningOp()->getLoc()}, varInfo));
            }            
        }

        auto value = transformed.getInitializer();
        if (value)
        {
            rewriter.create<LLVM::StoreOp>(location, value, allocated);

#ifdef DBG_INFO_ADD_VALUE_OP            
            if (tsLlvmContext->compileOptions.generateDebugInfo && varInfo)
            {                
                rewriter.create<LLVM::DbgValueOp>(location, value, varInfo);
            }
#endif            
        }

        rewriter.replaceOp(varOp, ValueRange{allocated});
        return success();
    }
};

struct DebugVariableOpLowering : public TsLlvmPattern<mlir_ts::DebugVariableOp>
{
    using TsLlvmPattern<mlir_ts::DebugVariableOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DebugVariableOp debugVarOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(debugVarOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(debugVarOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto location = debugVarOp.getLoc();

        LLVM_DEBUG(llvm::dbgs() << "\n!! debug variable: " << debugVarOp << "\n";);

        //DIScopeAttr scope, StringAttr name, DIFileAttr file, unsigned line, unsigned arg, unsigned alignInBits, DITypeAttr type
        LocationHelper lh(rewriter.getContext());
        if (auto localVarAttrFusedLoc = dyn_cast<mlir::FusedLocWith<LLVM::DILocalVariableAttr>>(location))
        {
            auto value = transformed.getInitializer();

            auto varInfo = localVarAttrFusedLoc.getMetadata();

            auto allocated = ch.Alloca(value.getType(), 1);

            rewriter.create<LLVM::DbgDeclareOp>(location, allocated, varInfo);

            rewriter.create<LLVM::StoreOp>(location, value, allocated);     
#ifdef DBG_INFO_ADD_VALUE_OP                                   
            rewriter.create<LLVM::DbgValueOp>(location, value, varInfo);
#endif
        }

        rewriter.eraseOp(debugVarOp);
        return success();
    }
};

struct AllocaOpLowering : public TsLlvmPattern<mlir_ts::AllocaOp>
{
    using TsLlvmPattern<mlir_ts::AllocaOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AllocaOp varOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(varOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(varOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto location = varOp.getLoc();

        auto referenceType = cast<mlir_ts::RefType>(varOp.getReference().getType());
        auto storageType = referenceType.getElementType();
        auto llvmStorageType = tch.convertType(storageType);

        LLVM_DEBUG(llvm::dbgs() << "\n!! alloca: " << storageType << " llvm: " << llvmStorageType << "\n";);

        mlir::Value count;
        if (transformed.getCount())
        {
            count = transformed.getCount();
        }
        else
        {
            count = clh.createI32ConstantOf(1);
        }

        mlir::Value allocated = rewriter.create<LLVM::AllocaOp>(location, th.getPtrType(), llvmStorageType, count);

        // TODO: call MemSet

        rewriter.replaceOp(varOp, ValueRange{allocated});
        return success();
    }
};

struct NewOpLowering : public TsLlvmPattern<mlir_ts::NewOp>
{
    using TsLlvmPattern<mlir_ts::NewOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewOp newOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(newOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(newOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newOp.getLoc();

        mlir::Type storageType = newOp.getInstance().getType();
        if (auto classType = dyn_cast_or_null<mlir_ts::ClassType>(storageType))
        {
            storageType = classType.getStorageType();
        }
        else if (auto valueRef = dyn_cast<mlir_ts::ValueRefType>(storageType))
        {
            storageType = valueRef.getElementType();
        }

        auto resultType = tch.convertType(newOp.getType());

        mlir::Value value;
        if (newOp.getStackAlloc().has_value() && newOp.getStackAlloc().value())
        {
            value = ch.Alloca(resultType, 1);
        }
        else
        {
            value = ch.MemoryAlloc(storageType, MemoryAllocSet::Zero);
        }

        rewriter.replaceOp(newOp, ValueRange{value});
        return success();
    }
};

struct CreateTupleOpLowering : public TsLlvmPattern<mlir_ts::CreateTupleOp>
{
    using TsLlvmPattern<mlir_ts::CreateTupleOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateTupleOp createTupleOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(createTupleOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(createTupleOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = createTupleOp.getLoc();
        auto tupleType = cast<mlir_ts::TupleType>(createTupleOp.getType());

        auto tupleVar = rewriter.create<mlir_ts::VariableOp>(
            loc, mlir_ts::RefType::get(tupleType), mlir::Value(), rewriter.getBoolAttr(false), rewriter.getIndexAttr(0));

        // set values here
        auto ptrType = th.getPtrType();
        auto llvmTupleType = tch.convertType(tupleType);
        for (auto [index, itemPair] : enumerate(llvm::zip(transformed.getItems(), createTupleOp.getItems())))
        {
            auto item = std::get<0>(itemPair);
            auto itemOrig = std::get<1>(itemPair);

            auto llvmValueType = tch.convertType(itemOrig.getType());

            mlir::Value tupleVarAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, ptrType, tupleVar);

            LLVM_DEBUG(llvm::dbgs() << "\n!! CreateTuple: type - " << tupleType << " llvm: " << llvmTupleType << "\n";);

            auto offset = rewriter.create<LLVM::GEPOp>(
                loc, ptrType, llvmTupleType, tupleVarAsLLVMType, ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(index)});

            LLVM_DEBUG(llvm::dbgs() << "\n!! CreateTuple: op " << offset << "\n";);

            // cast item if needed
            auto destItemType = tupleType.getFields()[index].type;
            auto llvmDestValueType = tch.convertType(destItemType);

            mlir::Value itemValue = item;

            // TODO: Op should ensure that in ops types are equal result types
            if (llvmDestValueType != llvmValueType)
            {
                CastLogicHelper castLogic(createTupleOp, rewriter, tch, tsLlvmContext->compileOptions);
                itemValue =
                    castLogic.cast(itemValue, itemOrig.getType(), llvmValueType, destItemType, llvmDestValueType);
                if (!itemValue)
                {
                    return failure();
                }

                itemValue = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(itemValue.getType()), itemValue);
            }

            rewriter.create<LLVM::StoreOp>(loc, itemValue, offset);
        }

        auto loadedValue = rewriter.create<mlir_ts::LoadOp>(loc, tupleType, tupleVar);

        rewriter.replaceOp(createTupleOp, ValueRange{loadedValue});
        return success();
    }
};

struct DeconstructTupleOpLowering : public TsLlvmPattern<mlir_ts::DeconstructTupleOp>
{
    using TsLlvmPattern<mlir_ts::DeconstructTupleOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DeconstructTupleOp deconstructTupleOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        CodeLogicHelper clh(deconstructTupleOp, rewriter);

        auto loc = deconstructTupleOp.getLoc();
        auto tupleVar = transformed.getInstance();
        auto tupleType = cast<LLVM::LLVMStructType>(tupleVar.getType());

        // values
        SmallVector<mlir::Value> results;

        // set values here
        for (auto [index, item] : enumerate(tupleType.getBody()))
        {
            auto llvmValueType = item;
            auto value =
                rewriter.create<LLVM::ExtractValueOp>(loc, llvmValueType, tupleVar, MLIRHelper::getStructIndex(rewriter, index));

            results.push_back(value);
        }

        rewriter.replaceOp(deconstructTupleOp, ValueRange{results});
        return success();
    }
};

struct CreateArrayOpLowering : public TsLlvmPattern<mlir_ts::CreateArrayOp>
{
    using TsLlvmPattern<mlir_ts::CreateArrayOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateArrayOp createArrayOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(createArrayOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(createArrayOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = createArrayOp.getLoc();

        auto arrayType = createArrayOp.getType();
        auto elementType = arrayType.getElementType();
        auto llvmElementType = tch.convertType(elementType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto newCountAsIndexType = clh.createIndexConstantOf(llvmIndexType, createArrayOp.getItems().size());

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryAlloc(multSizeOfTypeValue);

        mlir::Value index = clh.createIndexConstantOf(llvmIndexType, 0);
        auto next = false;
        mlir::Value value1;
        for (auto item : transformed.getItems())
        {
            if (next)
            {
                if (!value1)
                {
                    value1 = clh.createIndexConstantOf(llvmIndexType, 1);
                }

                index = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{index, value1});
            }

            // save new element
            auto offset = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, allocated, ValueRange{index});

            auto effectiveItem = item;
            if (llvmElementType != item.getType())
            {
                // effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, llvmElementType, item);
                llvm_unreachable("type mismatch");
                return failure();
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated,
                                                                 MLIRHelper::getStructIndex(rewriter, 0));

        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2,
                                                                 newCountAsIndexType, MLIRHelper::getStructIndex(rewriter, 1));

        rewriter.replaceOp(createArrayOp, ValueRange{structValue3});
        return success();
    }
};

struct NewEmptyArrayOpLowering : public TsLlvmPattern<mlir_ts::NewEmptyArrayOp>
{
    using TsLlvmPattern<mlir_ts::NewEmptyArrayOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewEmptyArrayOp newEmptyArrOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(newEmptyArrOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(newEmptyArrOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newEmptyArrOp.getLoc();

        auto arrayType = newEmptyArrOp.getType();
        auto elementType = arrayType.getElementType();

        mlir::Type storageType = elementType;
        // TODO: find out if the following is correct
        // storageType = MLIRHelper::getStorageTypeFrom(elementType);

        auto llvmElementType = tch.convertType(elementType);

        auto allocated = rewriter.create<LLVM::ZeroOp>(loc, th.getPtrType());

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated,
                                                                 MLIRHelper::getStructIndex(rewriter, 0));

        auto size0 = clh.createIndexConstantOf(llvmIndexType, 0);
        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, size0,
                                                                 MLIRHelper::getStructIndex(rewriter, 1));

        rewriter.replaceOp(newEmptyArrOp, ValueRange{structValue3});
        return success();
    }
};

struct NewArrayOpLowering : public TsLlvmPattern<mlir_ts::NewArrayOp>
{
    using TsLlvmPattern<mlir_ts::NewArrayOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewArrayOp newArrOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(newArrOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(newArrOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newArrOp.getLoc();

        auto arrayType = newArrOp.getType();
        auto elementType = arrayType.getElementType();
        auto llvmIndexType = tch.convertType(th.getIndexType());
        auto llvmElementType = tch.convertType(elementType);

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto countAsIndexTypeMLIR = rewriter.create<mlir_ts::CastOp>(loc, th.getIndexType(), transformed.getCount());
        auto countAsIndexType = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, countAsIndexTypeMLIR);

        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{sizeOfTypeValue, countAsIndexType});

        auto allocated = ch.MemoryAlloc(multSizeOfTypeValue);

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated,
                                                                 MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));

        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2,
                                                                 transformed.getCount(), MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));

        rewriter.replaceOp(newArrOp, ValueRange{structValue3});
        return success();
    }
};

struct ArrayPushOpLowering : public TsLlvmPattern<mlir_ts::ArrayPushOp>
{
    using TsLlvmPattern<mlir_ts::ArrayPushOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArrayPushOp pushOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(pushOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(pushOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = pushOp.getLoc();

        auto ptrType = th.getPtrType();
        auto arrayType = cast<mlir_ts::ArrayType>(cast<mlir_ts::RefType>(pushOp.getOp().getType()).getElementType());
        auto elementType = arrayType.getElementType();

        auto llvmArrayType = tch.convertType(arrayType);
        auto llvmElementType = tch.convertType(elementType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        // TODO: use GetAddressOfArrayElement method to sync code
        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                          ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, currentPtrPtr);

        auto countAsIndexTypePtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                              ArrayRef<LLVM::GEPArg>{0, ARRAY_SIZE_INDEX});
        auto countAsIndexType = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, countAsIndexTypePtr);

        auto incSize = clh.createIndexConstantOf(llvmIndexType, transformed.getItems().size());
        auto newCountAsIndexType =
            rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{countAsIndexType, incSize});

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryRealloc(currentPtr, multSizeOfTypeValue);

        mlir::Value index = countAsIndexType;
        auto next = false;
        mlir::Value value1;
        for (auto itemPair : llvm::zip(transformed.getItems(), pushOp.getItems()))
        {
            auto item = std::get<0>(itemPair);
            auto itemOrig = std::get<1>(itemPair);

            if (next)
            {
                if (!value1)
                {
                    value1 = clh.createIndexConstantOf(llvmIndexType, 1);
                }

                index = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{index, value1});
            }

            // save new element
            auto offset = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, allocated, ValueRange{index});

            auto effectiveItem = item;
            if (elementType != itemOrig.getType())
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! push cast: store type: " << elementType
                                        << " value type: " << item.getType() << "\n";);
                llvm_unreachable("cast must happen earlier");
                // effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, elementType, item);
                return failure();
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);
        rewriter.create<LLVM::StoreOp>(loc, newCountAsIndexType, countAsIndexTypePtr);

        rewriter.replaceOp(pushOp, ValueRange{newCountAsIndexType});
        return success();
    }
};

struct ArrayPopOpLowering : public TsLlvmPattern<mlir_ts::ArrayPopOp>
{
    using TsLlvmPattern<mlir_ts::ArrayPopOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArrayPopOp popOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(popOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(popOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = popOp.getLoc();

        auto ptrType = th.getPtrType();
        auto arrayType = cast<mlir_ts::ArrayType>(cast<mlir_ts::RefType>(popOp.getOp().getType()).getElementType());
        auto elementType = arrayType.getElementType();
        
        auto llvmArrayType = tch.convertType(arrayType);
        auto llvmElementType = tch.convertType(elementType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                          ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, currentPtrPtr);

        auto countAsIndexTypePtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                              ArrayRef<LLVM::GEPArg>{0, ARRAY_SIZE_INDEX});
        auto countAsIndexType = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, countAsIndexTypePtr);

        auto incSize = clh.createIndexConstantOf(llvmIndexType, 1);
        auto newCountAsIndexType =
            rewriter.create<LLVM::SubOp>(loc, llvmIndexType, ValueRange{countAsIndexType, incSize});

        // load last element
        auto offset =
            rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, currentPtr, ValueRange{newCountAsIndexType});
        auto loadedElement = rewriter.create<LLVM::LoadOp>(loc, llvmElementType, offset);

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryRealloc(currentPtr, multSizeOfTypeValue);

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);
        rewriter.create<LLVM::StoreOp>(loc, newCountAsIndexType, countAsIndexTypePtr);

        rewriter.replaceOp(popOp, ValueRange{loadedElement});
        return success();
    }
};

struct ArrayUnshiftOpLowering : public TsLlvmPattern<mlir_ts::ArrayUnshiftOp>
{
    using TsLlvmPattern<mlir_ts::ArrayUnshiftOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArrayUnshiftOp unshiftOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(unshiftOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(unshiftOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = unshiftOp.getLoc();

        auto ptrType = th.getPtrType();
        auto arrayType = cast<mlir_ts::ArrayType>(cast<mlir_ts::RefType>(unshiftOp.getOp().getType()).getElementType());
        auto elementType = arrayType.getElementType();

        auto llvmArrayType = tch.convertType(arrayType);
        auto llvmElementType = tch.convertType(elementType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                          ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, currentPtrPtr);

        auto countAsIndexTypePtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                              ArrayRef<LLVM::GEPArg>{0, ARRAY_SIZE_INDEX});
        auto countAsIndexType = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, countAsIndexTypePtr);

        auto incSize = clh.createIndexConstantOf(llvmIndexType, transformed.getItems().size());
        auto newCountAsIndexType =
            rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{countAsIndexType, incSize});

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryRealloc(currentPtr, multSizeOfTypeValue);

        // realloc
        auto offset0 = allocated;
        auto offsetN = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, allocated, ValueRange{incSize});

        auto newCountAsIndexTypeAdapt = rewriter.create<mlir_ts::DialectCastOp>(loc, th.getIndexType(), newCountAsIndexType);
        rewriter.create<mlir_ts::MemoryMoveOp>(loc, offsetN, offset0, newCountAsIndexTypeAdapt);

        mlir::Value index = clh.createIndexConstantOf(llvmIndexType, 0);
        auto next = false;
        mlir::Value value1;
        for (auto itemPair : llvm::zip(transformed.getItems(), unshiftOp.getItems()))
        {
            auto item = std::get<0>(itemPair);
            auto itemOrig = std::get<1>(itemPair);

            if (next)
            {
                if (!value1)
                {
                    value1 = clh.createIndexConstantOf(llvmIndexType, 1);
                }

                index = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{index, value1});
            }

            // save new element
            auto offset = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, allocated, ValueRange{index});

            auto effectiveItem = item;
            if (elementType != itemOrig.getType())
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! push cast: store type: " << elementType
                                        << " value type: " << item.getType() << "\n";);
                llvm_unreachable("cast must happen earlier");
                // effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, elementType, item);
                return failure();
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);
        rewriter.create<LLVM::StoreOp>(loc, newCountAsIndexType, countAsIndexTypePtr);

        rewriter.replaceOp(unshiftOp, ValueRange{newCountAsIndexType});
        return success();
    }
};

struct ArrayShiftOpLowering : public TsLlvmPattern<mlir_ts::ArrayShiftOp>
{
    using TsLlvmPattern<mlir_ts::ArrayShiftOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArrayShiftOp shiftOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(shiftOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(shiftOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = shiftOp.getLoc();

        auto ptrType = th.getPtrType();
        auto arrayType = cast<mlir_ts::ArrayType>(cast<mlir_ts::RefType>(shiftOp.getOp().getType()).getElementType());
        auto elementType = arrayType.getElementType();

        auto llvmArrayType = tch.convertType(arrayType);
        auto llvmElementType = tch.convertType(elementType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        mlir::Type storageType = elementType;
        // storageType = MLIRHelper::getStorageTypeFrom(elementType);

        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                          ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, currentPtrPtr);

        auto countAsIndexTypePtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                              ArrayRef<LLVM::GEPArg>{0, ARRAY_SIZE_INDEX});
        auto countAsIndexType = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, countAsIndexTypePtr);

        auto incSize = clh.createIndexConstantOf(llvmIndexType, 1);
        auto newCountAsIndexType =
            rewriter.create<LLVM::SubOp>(loc, llvmIndexType, ValueRange{countAsIndexType, incSize});

        // load last element
        auto offset0 =
            rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, currentPtr, ValueRange{clh.createIndexConstantOf(llvmIndexType, 0)});
        auto loadedElement = rewriter.create<LLVM::LoadOp>(loc, llvmElementType, offset0);

        auto offset1 =
            rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), th.getI32Type(), currentPtr, ValueRange{incSize});

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto newCountAsIndexTypeAdapt = rewriter.create<mlir_ts::DialectCastOp>(loc, th.getIndexType(), newCountAsIndexType);
        rewriter.create<mlir_ts::MemoryMoveOp>(loc, offset0, offset1, newCountAsIndexTypeAdapt);

        auto allocated = ch.MemoryRealloc(currentPtr, multSizeOfTypeValue);

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);
        rewriter.create<LLVM::StoreOp>(loc, newCountAsIndexType, countAsIndexTypePtr);

        rewriter.replaceOp(shiftOp, ValueRange{loadedElement});
        return success();
    }
};

struct ArraySpliceOpLowering : public TsLlvmPattern<mlir_ts::ArraySpliceOp>
{
    using TsLlvmPattern<mlir_ts::ArraySpliceOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArraySpliceOp spliceOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(spliceOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(spliceOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = spliceOp.getLoc();

        auto ptrType = th.getPtrType();
        auto arrayType = cast<mlir_ts::ArrayType>(cast<mlir_ts::RefType>(spliceOp.getOp().getType()).getElementType());
        auto elementType = arrayType.getElementType();

        auto llvmArrayType = tch.convertType(arrayType);
        auto llvmElementType = tch.convertType(elementType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                          ArrayRef<LLVM::GEPArg>{0, ARRAY_DATA_INDEX});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, currentPtrPtr);

        auto countAsIndexTypePtr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmArrayType, transformed.getOp(),
                                                              ArrayRef<LLVM::GEPArg>{0, ARRAY_SIZE_INDEX});
        auto countAsIndexType = rewriter.create<LLVM::LoadOp>(loc, llvmIndexType, countAsIndexTypePtr);

        auto startIndexAsIndexType = spliceOp.getStart();
        auto decSizeAsIndexType = spliceOp.getDeleteCount();

        auto startIndexAsLLVMType = rewriter.create<mlir::index::CastUOp>(loc, llvmIndexType, startIndexAsIndexType);
        auto decSizeAsLLVMType = rewriter.create<mlir::index::CastUOp>(loc, llvmIndexType, decSizeAsIndexType);            

        auto incSizeAsIndexType = clh.createIndexConstantOf(llvmIndexType, transformed.getItems().size());

        mlir::Value newCountAsIndexType = rewriter.create<mlir::index::SubOp>(loc, llvmIndexType, ValueRange{countAsIndexType, decSizeAsIndexType});

        newCountAsIndexType = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{newCountAsIndexType, incSizeAsIndexType});

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto increaseArrayFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
            auto allocated = ch.MemoryRealloc(currentPtr, multSizeOfTypeValue);

            auto moveCountAsIndexType =
                rewriter.create<mlir::index::SubOp>(loc, llvmIndexType, ValueRange{countAsIndexType, startIndexAsIndexType});
            moveCountAsIndexType =
                rewriter.create<mlir::index::SubOp>(loc, llvmIndexType, ValueRange{moveCountAsIndexType, decSizeAsIndexType});

            // realloc
            auto offsetStart = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, allocated, ValueRange{startIndexAsLLVMType});
            auto offsetFrom = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, offsetStart, ValueRange{decSizeAsLLVMType});
            auto offsetTo = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, offsetStart, ValueRange{incSizeAsIndexType});
            auto moveCountAsIndexTypeAdapt = rewriter.create<mlir_ts::DialectCastOp>(loc, th.getIndexType(), moveCountAsIndexType);
            rewriter.create<mlir_ts::MemoryMoveOp>(loc, offsetTo, offsetFrom, moveCountAsIndexTypeAdapt);

            return allocated;
        };

        auto decreaseArrayFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {

            auto moveCountAsIndexType =
                rewriter.create<mlir::index::SubOp>(loc, llvmIndexType, ValueRange{countAsIndexType, startIndexAsIndexType});
            moveCountAsIndexType =
                rewriter.create<mlir::index::SubOp>(loc, llvmIndexType, ValueRange{moveCountAsIndexType, decSizeAsIndexType});

            // realloc
            auto offsetStart = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, currentPtr, ValueRange{startIndexAsLLVMType});
            auto offsetFrom = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, offsetStart, ValueRange{decSizeAsLLVMType});
            auto offsetTo = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, offsetStart, ValueRange{incSizeAsIndexType});

            auto moveCountAsIndexTypeAdapt = rewriter.create<mlir_ts::DialectCastOp>(loc, th.getIndexType(), moveCountAsIndexType);
            rewriter.create<mlir_ts::MemoryMoveOp>(loc, offsetTo, offsetFrom, moveCountAsIndexTypeAdapt);

            auto allocated = ch.MemoryRealloc(currentPtr, multSizeOfTypeValue);
            return allocated;
        };

        auto cond = rewriter.create<arith::CmpIOp>(loc, th.getLLVMBoolType(), arith::CmpIPredicateAttr::get(rewriter.getContext(), arith::CmpIPredicate::ugt), incSizeAsIndexType, decSizeAsLLVMType);
        auto allocated = clh.conditionalExpressionLowering(loc, th.getPtrType(), cond, increaseArrayFunc, decreaseArrayFunc);

        mlir::Value index = startIndexAsLLVMType;
        auto next = false;
        mlir::Value value1;
        for (auto itemPair : llvm::zip(transformed.getItems(), spliceOp.getItems()))
        {
            auto item = std::get<0>(itemPair);
            auto itemOrig = std::get<1>(itemPair);

            if (next)
            {
                if (!value1)
                {
                    value1 = clh.createIndexConstantOf(llvmIndexType, 1);
                }

                index = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, ValueRange{index, value1});
            }

            // save new element
            auto offset = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, allocated, ValueRange{index});

            auto effectiveItem = item;
            if (elementType != itemOrig.getType())
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! push cast: store type: " << elementType
                                        << " value type: " << item.getType() << "\n";);
                llvm_unreachable("cast must happen earlier");
                // effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, elementType, item);
                return failure();
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);
        rewriter.create<LLVM::StoreOp>(loc, newCountAsIndexType, countAsIndexTypePtr);

        rewriter.replaceOp(spliceOp, ValueRange{newCountAsIndexType});
        return success();
    }
};

struct ArrayViewOpLowering : public TsLlvmPattern<mlir_ts::ArrayViewOp>
{
    using TsLlvmPattern<mlir_ts::ArrayViewOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArrayViewOp arrayViewOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(arrayViewOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(arrayViewOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = arrayViewOp.getLoc();

        auto ptrType = th.getPtrType();
        auto arrayType = arrayViewOp.getOp().getType();
        auto elementType = arrayType.getElementType();

        auto llvmArrayType = tch.convertType(arrayType);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        // TODO: add size check !!!

        auto arrayPtr = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(),
                transformed.getOp(), MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));

        auto arrayOffset = ch.GetAddressOfPointerOffset(elementType, arrayPtr, transformed.getOffset());
        // create array type
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmArrayType);
        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, structValue, arrayOffset,
                                                                 MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));

        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, structValue2,
                                                                 transformed.getCount(), MLIRHelper::getStructIndex(rewriter, ARRAY_SIZE_INDEX));

        rewriter.replaceOp(arrayViewOp, ValueRange{structValue3});
        return success();
    }
};

struct DeleteOpLowering : public TsLlvmPattern<mlir_ts::DeleteOp>
{
    using TsLlvmPattern<mlir_ts::DeleteOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DeleteOp deleteOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(deleteOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        if (mlir::failed(ch.MemoryFree(transformed.getReference())))
        {
            return mlir::failure();
        }

        rewriter.eraseOp(deleteOp);
        return mlir::success();
    }
};

void NegativeOpValue(mlir_ts::ArithmeticUnaryOp &unaryOp, mlir::Value oper, mlir::PatternRewriter &builder)
{
    CodeLogicHelper clh(unaryOp, builder);

    auto type = oper.getType();
    if (type.isIntOrIndex())
    {
        builder.replaceOpWithNewOp<arith::SubIOp>(unaryOp, type, clh.createIConstantOf(type.getIntOrFloatBitWidth(), 0), oper);
    }
    else if (!type.isIntOrIndex() && type.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<arith::SubFOp>(unaryOp, type, clh.createFConstantOf(type.getIntOrFloatBitWidth(), 0.0),
                                           oper);
    }
    else
    {
        llvm_unreachable("not implemented");
    }
}

void NegativeOpBin(mlir_ts::ArithmeticUnaryOp &unaryOp, mlir::Value oper, mlir::PatternRewriter &builder)
{
    CodeLogicHelper clh(unaryOp, builder);

    auto type = oper.getType();
    if (type.isIntOrIndex())
    {
        mlir::Value lhs;
        if (type.isInteger(1))
        {
            lhs = clh.createI1ConstantOf(true);
        }
        else
        {
            // lhs = clh.createI32ConstantOf(-1);
            lhs = clh.createIConstantOf(type.getIntOrFloatBitWidth(), -1);
        }

        builder.replaceOpWithNewOp<LLVM::XOrOp>(unaryOp, type, oper, lhs);
    }
    else if (!type.isIntOrIndex() && type.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<LLVM::XOrOp>(unaryOp, oper);
    }
    else
    {
        llvm_unreachable("not implemented");
    }
}

struct ArithmeticUnaryOpLowering : public TsLlvmPattern<mlir_ts::ArithmeticUnaryOp>
{
    using TsLlvmPattern<mlir_ts::ArithmeticUnaryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArithmeticUnaryOp arithmeticUnaryOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto opCode = (SyntaxKind)arithmeticUnaryOp.getOpCode();
        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:
            NegativeOpBin(arithmeticUnaryOp, transformed.getOperand1(), rewriter);
            return success();
        case SyntaxKind::PlusToken:
            rewriter.replaceOp(arithmeticUnaryOp, transformed.getOperand1());
            return success();
        case SyntaxKind::MinusToken:
            NegativeOpValue(arithmeticUnaryOp, transformed.getOperand1(), rewriter);
            return success();
        case SyntaxKind::TildeToken:
            NegativeOpBin(arithmeticUnaryOp, transformed.getOperand1(), rewriter);
            return success();
        default:
            llvm_unreachable("not implemented");
            return failure();
        }
    }
};

struct ArithmeticBinaryOpLowering : public TsLlvmPattern<mlir_ts::ArithmeticBinaryOp>
{
    using TsLlvmPattern<mlir_ts::ArithmeticBinaryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArithmeticBinaryOp arithmeticBinaryOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto opCode = (SyntaxKind)arithmeticBinaryOp.getOpCode();
        switch (opCode)
        {
        case SyntaxKind::PlusToken:
            if (isa<mlir_ts::StringType>(arithmeticBinaryOp.getOperand1().getType()))
            {
                rewriter.replaceOpWithNewOp<mlir_ts::StringConcatOp>(
                    arithmeticBinaryOp, mlir_ts::StringType::get(rewriter.getContext()),
                    ValueRange{transformed.getOperand1(), transformed.getOperand2()});
                return success();
            }

            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::AddIOp, arith::AddFOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                                transformed.getOperand2(), rewriter);

        case SyntaxKind::MinusToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::SubIOp, arith::SubFOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                               transformed.getOperand2(), rewriter);

        case SyntaxKind::AsteriskToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::MulIOp, arith::MulFOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                               transformed.getOperand2(), rewriter);

        case SyntaxKind::SlashToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::DivSIOp, arith::DivFOp, arith::DivUIOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                               transformed.getOperand2(), rewriter);

        case SyntaxKind::GreaterThanGreaterThanToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::ShRSIOp, arith::ShRSIOp, arith::ShRUIOp>(
                arithmeticBinaryOp, transformed.getOperand1(), transformed.getOperand2(), rewriter);

        case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::ShRUIOp, arith::ShRUIOp>(
                arithmeticBinaryOp, transformed.getOperand1(), transformed.getOperand2(), rewriter);

        case SyntaxKind::LessThanLessThanToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::ShLIOp, arith::ShLIOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                                         transformed.getOperand2(), rewriter);

        case SyntaxKind::AmpersandToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::AndIOp, arith::AndIOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                             transformed.getOperand2(), rewriter);

        case SyntaxKind::BarToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::OrIOp, arith::OrIOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                           transformed.getOperand2(), rewriter);

        case SyntaxKind::CaretToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::XOrIOp, arith::XOrIOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                             transformed.getOperand2(), rewriter);

        case SyntaxKind::PercentToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, arith::RemSIOp, arith::RemFOp, arith::RemUIOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                               transformed.getOperand2(), rewriter);

        case SyntaxKind::AsteriskAsteriskToken:
            return BinOp<mlir_ts::ArithmeticBinaryOp, math::PowFOp, math::PowFOp>(arithmeticBinaryOp, transformed.getOperand1(),
                                                                           transformed.getOperand2(), rewriter);

        default:
            llvm_unreachable("not implemented");
            return failure();
        }
    }
};

struct LogicalBinaryOpLowering : public TsLlvmPattern<mlir_ts::LogicalBinaryOp>
{
    using TsLlvmPattern<mlir_ts::LogicalBinaryOp>::TsLlvmPattern;

    template <arith::CmpIPredicate v1, arith::CmpFPredicate v2>
    mlir::Value logicOp(mlir_ts::LogicalBinaryOp logicalBinaryOp, SyntaxKind op, mlir::Value left,
                        mlir::Type leftTypeOrig, mlir::Value right, mlir::Type rightTypeOrig,
                        PatternRewriter &builder) const
    {
        return LogicOp<arith::CmpIOp, arith::CmpIPredicate, v1, arith::CmpFOp, arith::CmpFPredicate, v2>(
            logicalBinaryOp, op, left, leftTypeOrig, right, rightTypeOrig, builder, *(const LLVMTypeConverter *)getTypeConverter(),
            tsLlvmContext->compileOptions);
    }

    LogicalResult matchAndRewrite(mlir_ts::LogicalBinaryOp logicalBinaryOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto op = (SyntaxKind)logicalBinaryOp.getOpCode();

        auto op1 = transformed.getOperand1();
        auto op2 = transformed.getOperand2();
        auto opType1 = logicalBinaryOp.getOperand1().getType();
        auto opType2 = logicalBinaryOp.getOperand2().getType();

        // int and float
        mlir::Value value;
        switch (op)
        {
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
            value = logicOp<arith::CmpIPredicate::eq, arith::CmpFPredicate::OEQ>(logicalBinaryOp, op, op1, opType1, op2, opType2,
                                                                   rewriter);
            break;
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
            value = logicOp<arith::CmpIPredicate::ne, arith::CmpFPredicate::ONE>(logicalBinaryOp, op, op1, opType1, op2, opType2,
                                                                   rewriter);
            break;
        case SyntaxKind::GreaterThanToken:

            if (opType1.isUnsignedInteger() && opType2.isUnsignedInteger())
            {
                value = logicOp<arith::CmpIPredicate::ugt, arith::CmpFPredicate::OGT>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }
            else
            {
                value = logicOp<arith::CmpIPredicate::sgt, arith::CmpFPredicate::OGT>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }
            break;
        case SyntaxKind::GreaterThanEqualsToken:
            if (opType1.isUnsignedInteger() && opType2.isUnsignedInteger())
            {
                value = logicOp<arith::CmpIPredicate::uge, arith::CmpFPredicate::OGE>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }
            else
            {
                value = logicOp<arith::CmpIPredicate::sge, arith::CmpFPredicate::OGE>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }

            break;
        case SyntaxKind::LessThanToken:
            if (opType1.isUnsignedInteger() && opType2.isUnsignedInteger())
            {
                value = logicOp<arith::CmpIPredicate::ult, arith::CmpFPredicate::OLT>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }
            else
            {
                value = logicOp<arith::CmpIPredicate::slt, arith::CmpFPredicate::OLT>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }

            break;
        case SyntaxKind::LessThanEqualsToken:
            if (opType1.isUnsignedInteger() && opType2.isUnsignedInteger())
            {
                value = logicOp<arith::CmpIPredicate::ule, arith::CmpFPredicate::OLE>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }
            else
            {
                value = logicOp<arith::CmpIPredicate::sle, arith::CmpFPredicate::OLE>(
                    logicalBinaryOp, op, op1, opType1, op2, opType2, rewriter);
            }

            break;
        default:
            llvm_unreachable("not implemented");
            return failure();
        }

        rewriter.replaceOp(logicalBinaryOp, value);
        return success();
    }
};

struct LoadOpLowering : public TsLlvmPattern<mlir_ts::LoadOp>
{
    using TsLlvmPattern<mlir_ts::LoadOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LoadOp loadOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(loadOp, rewriter);

        auto loc = loadOp.getLoc();

        mlir::Type elementType;
        mlir::Type elementTypeConverted;

        auto elementRefType = loadOp.getReference().getType();
        auto resultType = loadOp.getType();

        if (auto refType = dyn_cast<mlir_ts::RefType>(elementRefType))
        {
            elementType = refType.getElementType();
            elementTypeConverted = tch.convertType(elementType);
        }
        else if (auto valueRefType = dyn_cast<mlir_ts::ValueRefType>(elementRefType))
        {
            elementType = valueRefType.getElementType();
            elementTypeConverted = tch.convertType(elementType);
        }

        auto isOptional = false;
        if (auto optType = dyn_cast<mlir_ts::OptionalType>(resultType))
        {
            isOptional = optType.getElementType() == elementType;
        }

        mlir::Value loadedValue;
        auto loadedValueFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
            mlir::Value loadedValue;
            if (elementType)
            {
                auto alignment = 0;
                auto isVolatile = false;
                auto isNonTemporal = false;
                auto isInvariant = false;
                auto ordering = LLVM::AtomicOrdering::not_atomic;
                StringRef syncscope = {};

                if (auto atomicAttr = loadOp->getAttrOfType<mlir::BoolAttr>(ATOMIC_ATTR_NAME))
                {
                    if (atomicAttr.getValue()) 
                    {
                        auto orderingAttr = loadOp->getAttrOfType<mlir::IntegerAttr>(ORDERING_ATTR_NAME);
                        auto syncScopeAttr = loadOp->getAttrOfType<mlir::StringAttr>(SYNCSCOPE_ATTR_NAME);
                        if (orderingAttr && syncScopeAttr)
                        {
                            ordering = (LLVM::AtomicOrdering) orderingAttr.getValue().getZExtValue();
                            if (ordering == LLVM::AtomicOrdering::release || ordering == LLVM::AtomicOrdering::acq_rel)
                            {
                                ordering = LLVM::AtomicOrdering::acquire;
                            }

                            syncscope = syncScopeAttr.getValue();
                            LLVMTypeConverterHelper ltch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
                            alignment = ltch.getTypeAlignSizeInBits(elementTypeConverted);
                        }
                    }
                }

                if (auto volatileAttr = loadOp->getAttrOfType<mlir::BoolAttr>(VOLATILE_ATTR_NAME))
                {
                    isVolatile = volatileAttr.getValue();
                }

                if (auto nonTemporalAttr = loadOp->getAttrOfType<mlir::BoolAttr>(NONTEMPORAL_ATTR_NAME))
                {
                    isNonTemporal = nonTemporalAttr.getValue();
                }    

                if (auto invariantAttr = loadOp->getAttrOfType<mlir::BoolAttr>(INVARIANT_ATTR_NAME))
                {
                    isInvariant = invariantAttr.getValue();
                }    

                loadedValue = rewriter.create<LLVM::LoadOp>(loc, elementTypeConverted, transformed.getReference(), alignment, isVolatile, isNonTemporal, isInvariant, ordering, syncscope);
            }
            else if (auto boundRefType = dyn_cast<mlir_ts::BoundRefType>(elementRefType))
            {
                auto loadRoundRef = rewriter.create<mlir_ts::LoadBoundRefOp>(loc, resultType, loadOp.getReference());

                // copy attrs over
                if (auto atomicAttr = loadOp->getAttrOfType<mlir::BoolAttr>(ATOMIC_ATTR_NAME))
                {
                    auto orderingAttr = loadOp->getAttrOfType<mlir::IntegerAttr>(ORDERING_ATTR_NAME);
                    auto syncScopeAttr = loadOp->getAttrOfType<mlir::StringAttr>(SYNCSCOPE_ATTR_NAME);
                    loadRoundRef->setAttr(ATOMIC_ATTR_NAME, atomicAttr);
                    loadRoundRef->setAttr(ORDERING_ATTR_NAME, orderingAttr);
                    loadRoundRef->setAttr(SYNCSCOPE_ATTR_NAME, syncScopeAttr);
                }

                if (auto volatileAttr = loadOp->getAttrOfType<mlir::BoolAttr>(VOLATILE_ATTR_NAME))
                {
                    loadRoundRef->setAttr(VOLATILE_ATTR_NAME, volatileAttr);
                }    

                if (auto nonTemporalAttr = loadOp->getAttrOfType<mlir::BoolAttr>(NONTEMPORAL_ATTR_NAME))
                {
                    loadRoundRef->setAttr(NONTEMPORAL_ATTR_NAME, nonTemporalAttr);
                }    

                if (auto invariantAttr = loadOp->getAttrOfType<mlir::BoolAttr>(INVARIANT_ATTR_NAME))
                {
                    loadRoundRef->setAttr(INVARIANT_ATTR_NAME, invariantAttr);
                }    

                loadedValue = loadRoundRef;            
            }

            return loadedValue;
        };

        if (isOptional)
        {
            auto resultTypeLlvm = tch.convertType(resultType);

            auto undefOptionalFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
                mlir::Value val = rewriter.create<mlir_ts::OptionalUndefOp>(loc, resultType);
                mlir::Value valAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, resultTypeLlvm, val);
                return valAsLLVMType;
            };

            auto createOptionalFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
                auto dataValue = loadedValueFunc(builder, location);
                mlir::Value val = rewriter.create<mlir_ts::OptionalValueOp>(loc, resultType, dataValue);
                mlir::Value valAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, resultTypeLlvm, val);
                return valAsLLVMType;
            };

            LLVMTypeConverterHelper llvmtch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));

            auto intPtrType = llvmtch.getIntPtrType(0);

            // not null condition
            auto dataIntPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, transformed.getReference());
            auto const0 = clh.createIConstantOf(llvmtch.getPointerBitwidth(0), 0);
            auto ptrCmpResult1 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, dataIntPtrValue, const0);

            loadedValue =
                clh.conditionalExpressionLowering(loc, resultTypeLlvm, ptrCmpResult1, createOptionalFunc, undefOptionalFunc);
        }
        else
        {
            loadedValue = loadedValueFunc(rewriter, loc);
        }

        if (!loadedValue)
        {
            llvm_unreachable("not implemented");
            return failure();
        }

        rewriter.replaceOp(loadOp, ValueRange{loadedValue});

        //LLVM_DEBUG(llvm::dbgs() << "\n!! LoadOp Ref value: \n" << transformed.getReference() << "\n";);
        //LLVM_DEBUG(llvm::dbgs() << "\n!! LoadOp DUMP: \n" << *loadOp->getParentOp() << "\n";);

        return success();
    }
};

struct StoreOpLowering : public TsLlvmPattern<mlir_ts::StoreOp>
{
    using TsLlvmPattern<mlir_ts::StoreOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StoreOp storeOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        if (auto boundRefType = dyn_cast<mlir_ts::BoundRefType>(storeOp.getReference().getType()))
        {
            auto storeBoundRefOp = rewriter.create<mlir_ts::StoreBoundRefOp>(storeOp->getLoc(), storeOp.getValue(), storeOp.getReference());

            // copy attrs over
            if (auto atomicAttr = storeOp->getAttrOfType<mlir::BoolAttr>(ATOMIC_ATTR_NAME))
            {
                auto orderingAttr = storeOp->getAttrOfType<mlir::IntegerAttr>(ORDERING_ATTR_NAME);
                auto syncScopeAttr = storeOp->getAttrOfType<mlir::StringAttr>(SYNCSCOPE_ATTR_NAME);
                storeBoundRefOp->setAttr(ATOMIC_ATTR_NAME, atomicAttr);
                storeBoundRefOp->setAttr(ORDERING_ATTR_NAME, orderingAttr);
                storeBoundRefOp->setAttr(SYNCSCOPE_ATTR_NAME, syncScopeAttr);
            }

            if (auto volatileAttr = storeOp->getAttrOfType<mlir::BoolAttr>(VOLATILE_ATTR_NAME))
            {
                storeBoundRefOp->setAttr(VOLATILE_ATTR_NAME, volatileAttr);
            }    

            if (auto nonTemporalAttr = storeOp->getAttrOfType<mlir::BoolAttr>(NONTEMPORAL_ATTR_NAME))
            {
                storeBoundRefOp->setAttr(NONTEMPORAL_ATTR_NAME, nonTemporalAttr);
            }    

            // if (auto invariantAttr = storeOp->getAttrOfType<mlir::BoolAttr>(INVARIANT_ATTR_NAME))
            // {
            //     storeBoundRefOp->setAttr(INVARIANT_ATTR_NAME, invariantAttr);
            // }   

            rewriter.eraseOp(storeOp);
            return success();
        }

        // atomic attributes
        auto alignment = 0;
        auto isVolatile = false;
        auto isNonTemporal = false;
        auto ordering = LLVM::AtomicOrdering::not_atomic;
        StringRef syncscope = {};

        if (auto atomicAttr = storeOp->getAttrOfType<mlir::BoolAttr>(ATOMIC_ATTR_NAME))
        {
            if (atomicAttr.getValue()) 
            {
                auto orderingAttr = storeOp->getAttrOfType<mlir::IntegerAttr>(ORDERING_ATTR_NAME);
                auto syncScopeAttr = storeOp->getAttrOfType<mlir::StringAttr>(SYNCSCOPE_ATTR_NAME);
                if (orderingAttr && syncScopeAttr)
                {
                    ordering = (LLVM::AtomicOrdering) orderingAttr.getValue().getZExtValue();
                    if (ordering == LLVM::AtomicOrdering::acquire || ordering == LLVM::AtomicOrdering::acq_rel)
                    {
                        ordering = LLVM::AtomicOrdering::release;
                    }

                    syncscope = syncScopeAttr.getValue();
                    LLVMTypeConverterHelper ltch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
                    alignment = ltch.getTypeAlignSizeInBits(transformed.getValue().getType());
                }
            }
        }

        if (auto volatileAttr = storeOp->getAttrOfType<mlir::BoolAttr>(VOLATILE_ATTR_NAME))
        {
            isVolatile = volatileAttr.getValue();
        }        

        if (auto nonTemporalAttr = storeOp->getAttrOfType<mlir::BoolAttr>(NONTEMPORAL_ATTR_NAME))
        {
            isNonTemporal = nonTemporalAttr.getValue();
        }    

        // if (auto invariantAttr = storeOp->getAttrOfType<mlir::BoolAttr>(INVARIANT_ATTR_NAME))
        // {
        //     isInvariant = invariantAttr.getValue();
        // }           

        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, transformed.getValue(), transformed.getReference(), alignment, isVolatile, isNonTemporal, ordering, syncscope);
#ifdef DBG_INFO_ADD_VALUE_OP        
        if (tsLlvmContext->compileOptions.generateDebugInfo)
        {
            if (auto varInfo = dyn_cast<mlir::FusedLocWith<LLVM::DILocalVariableAttr>>(transformed.getReference().getDefiningOp()->getLoc()))
            {
                rewriter.create<LLVM::DbgValueOp>(storeOp->getLoc(), transformed.getValue(), varInfo.getMetadata());
            }
        }
#endif        

        return success();
    }
};

struct ElementRefOpLowering : public TsLlvmPattern<mlir_ts::ElementRefOp>
{
    using TsLlvmPattern<mlir_ts::ElementRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ElementRefOp elementOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(elementOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto elementType = MLIRHelper::getElementTypeOrSelf(elementOp.getArray().getType());

        auto addr = ch.GetAddressOfArrayElement(elementType, elementOp.getArray().getType(),
                                                transformed.getArray(), transformed.getIndex());
        rewriter.replaceOp(elementOp, addr);
        return success();
    }
};

struct PointerOffsetRefOpLowering : public TsLlvmPattern<mlir_ts::PointerOffsetRefOp>
{
    using TsLlvmPattern<mlir_ts::PointerOffsetRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PointerOffsetRefOp elementOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(elementOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto addr = ch.GetAddressOfPointerOffset(elementOp.getResult().getType().getElementType(), 
            transformed.getRef(), transformed.getIndex());
        rewriter.replaceOp(elementOp, addr);
        return success();
    }
};

struct ExtractPropertyOpLowering : public TsLlvmPattern<mlir_ts::ExtractPropertyOp>
{
    using TsLlvmPattern<mlir_ts::ExtractPropertyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ExtractPropertyOp extractPropertyOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(extractPropertyOp,
                                                          tch.convertType(extractPropertyOp.getType()),
                                                          transformed.getObject(), extractPropertyOp.getPosition());
        return success();
    }
};

struct InsertPropertyOpLowering : public TsLlvmPattern<mlir_ts::InsertPropertyOp>
{
    using TsLlvmPattern<mlir_ts::InsertPropertyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InsertPropertyOp insertPropertyOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        auto loc = insertPropertyOp->getLoc();

        rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
            insertPropertyOp, tch.convertType(insertPropertyOp.getObject().getType()), transformed.getObject(),
            transformed.getValue(), insertPropertyOp.getPosition());

        return success();
    }
};

struct PropertyRefOpLowering : public TsLlvmPattern<mlir_ts::PropertyRefOp>
{
    using TsLlvmPattern<mlir_ts::PropertyRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PropertyRefOp propertyRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        assert(propertyRefOp.getPosition() != -1);

        LLVMCodeHelper ch(propertyRefOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto addr =
            ch.GetAddressOfStructElement(propertyRefOp.getObjectRef().getType(), transformed.getObjectRef(), propertyRefOp.getPosition());

        if (auto boundRefType = dyn_cast<mlir_ts::BoundRefType>(propertyRefOp.getType()))
        {
            auto boundRef = rewriter.create<mlir_ts::CreateBoundRefOp>(propertyRefOp->getLoc(), boundRefType,
                                                                       propertyRefOp.getObjectRef(), addr);
            addr = boundRef;
        }

        rewriter.replaceOp(propertyRefOp, addr);

        return success();
    }
};

struct GlobalOpLowering : public TsLlvmPattern<mlir_ts::GlobalOp>
{
    using TsLlvmPattern<mlir_ts::GlobalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GlobalOp globalOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = globalOp->getLoc();

        LLVMCodeHelper lch(globalOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto createAsGlobalConstructor = false;
        // TODO: we need to write correct attributes to Ops and detect which ops should be in GlobalConstructor
        auto visitorAllOps = [&](Operation *op) {
            if (isa<mlir_ts::NewOp>(op) || isa<mlir_ts::NewInterfaceOp>(op) || isa<mlir_ts::NewArrayOp>(op) ||
                isa<mlir_ts::SymbolCallInternalOp>(op) || isa<mlir_ts::CallInternalOp>(op) ||
                isa<mlir_ts::CallHybridInternalOp>(op) || isa<mlir_ts::VariableOp>(op) || isa<mlir_ts::AllocaOp>(op) ||
                isa<mlir_ts::CreateArrayOp>(op) || isa<mlir_ts::NewEmptyArrayOp>(op) || isa<mlir_ts::CreateTupleOp>(op) ||
                isa<mlir_ts::LoadOp>(op) || isa<mlir_ts::StoreOp>(op) || 
                isa<mlir_ts::LoadLibraryPermanentlyOp>(op) || isa<mlir_ts::SearchForAddressOfSymbolOp>(op))
            {
                createAsGlobalConstructor = true;
            }
            else if (auto castOp = dyn_cast<mlir_ts::CastOp>(op))
            {
                auto castType = castOp.getRes().getType();
                if (isa<mlir_ts::ArrayType>(castType) || isa<mlir_ts::TupleType>(castType)) 
                {
                   createAsGlobalConstructor = true; 
                }
            }

            // TODO: error in try-catch
            // if it has memory side effect - can't be in global init.
            // auto iface = dyn_cast<MemoryEffectOpInterface>(op);
            // if (!iface || !iface.hasNoEffect())
            // {
            //     createAsGlobalConstructor = true;
            // }
        };

        globalOp.getInitializerRegion().walk(visitorAllOps);

        auto linkage = globalOp.getLinkage();
        LLVM::GlobalOp llvmGlobalOp;
        if (createAsGlobalConstructor)
        {
            // we can't have constant here as we need to initialize it in global construct
            llvmGlobalOp = lch.createUndefGlobalVarIfNew(
                globalOp.getSymName(), getTypeConverter()->convertType(globalOp.getType()),
                globalOp.getValueAttr(), false /*globalOp.getConstant()*/, linkage);

            auto name = globalOp.getSymName().str();
            name.append("__cctor");
            lch.createFunctionFromRegion(loc, name, globalOp.getInitializerRegion(), globalOp.getSymName());

            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                
                auto parentModule = globalOp->getParentOfType<ModuleOp>();
                rewriter.setInsertionPointToStart(parentModule.getBody());

                lch.seekLastOp<mlir_ts::GlobalConstructorOp>(parentModule.getBody());

                rewriter.create<mlir_ts::GlobalConstructorOp>(loc, 
                    FlatSymbolRefAttr::get(rewriter.getContext(), StringRef(name)), rewriter.getIndexAttr(1000));
            }
        }
        else
        {
            llvmGlobalOp = lch.createGlobalVarIfNew(globalOp.getSymName(), getTypeConverter()->convertType(globalOp.getType()),
                                     globalOp.getValueAttr(), globalOp.getConstant(), globalOp.getInitializerRegion(),
                                     linkage);
        }

        // copy attributes over
        if (llvmGlobalOp)
        {
            auto attrs = globalOp->getAttrs();

            for (auto &attr : attrs)
            {
                if (attr.getName() == "export")
                {
                    llvmGlobalOp.setSection("export");
                    llvmGlobalOp.setPublic();
                }

                if (attr.getName() == "import")
                {
                    llvmGlobalOp.setSection("import");
                }

                if (attr.getName() == DLL_EXPORT)
                {
                    llvmGlobalOp.setSection(DLL_EXPORT);
                }

                if (attr.getName() == DLL_IMPORT)
                {
                    llvmGlobalOp.setSection(DLL_IMPORT);
                }
            }

            if (globalOp.getLinkage() == LLVM::Linkage::LinkonceODR || globalOp.getComdat().has_value())
            {
                auto comdat = globalOp.getComdat().has_value() 
                    ? static_cast<mlir::LLVM::comdat::Comdat>(globalOp.getComdat().value()) 
                    : mlir::LLVM::comdat::Comdat::Any;
                addComdat(llvmGlobalOp, rewriter, comdat);
            }

            if (tsLlvmContext->compileOptions.generateDebugInfo)
            {
                LocationHelper lh(rewriter.getContext());
                if (auto globalVarAttrFusedLoc = dyn_cast<mlir::FusedLocWith<LLVM::DIGlobalVariableAttr>>(loc))
                {
                    auto varInfo = globalVarAttrFusedLoc.getMetadata();
                    LLVM::DIExpressionAttr exprAttr;
                    llvmGlobalOp.setDbgExprAttr(LLVM::DIGlobalVariableExpressionAttr::get(rewriter.getContext(), varInfo, exprAttr));
                }
            }
        }

        rewriter.eraseOp(globalOp);
        return success();
    }

    static void addComdat(mlir::LLVM::GlobalOp &global,
                        mlir::ConversionPatternRewriter &rewriter, mlir::LLVM::comdat::Comdat comdat = mlir::LLVM::comdat::Comdat::Any) {
        auto module = global->getParentOfType<mlir::ModuleOp>();

        const char *comdatName = "__llvm_comdat";
        mlir::LLVM::ComdatOp comdatOp =
            module.lookupSymbol<mlir::LLVM::ComdatOp>(comdatName);
        if (!comdatOp) 
        {
            comdatOp = rewriter.create<mlir::LLVM::ComdatOp>(module.getLoc(), comdatName);
        }

        // TODO: add check if name already added
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(&comdatOp.getBody().back());
        auto selectorOp = rewriter.create<mlir::LLVM::ComdatSelectorOp>(
            comdatOp.getLoc(), global.getSymName(), comdat);
        global.setComdatAttr(mlir::SymbolRefAttr::get(
            rewriter.getContext(), comdatName,
            mlir::FlatSymbolRefAttr::get(selectorOp.getSymNameAttr())));
    }
};

struct GlobalResultOpLowering : public TsLlvmPattern<mlir_ts::GlobalResultOp>
{
    using TsLlvmPattern<mlir_ts::GlobalResultOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GlobalResultOp globalResultOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(globalResultOp, transformed.getResults());
        return success();
    }
};

struct AddressOfOpLowering : public TsLlvmPattern<mlir_ts::AddressOfOp>
{
    using TsLlvmPattern<mlir_ts::AddressOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AddressOfOp addressOfOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper lch(addressOfOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto actualType = addressOfOp.getType();
        if (isa<mlir_ts::OpaqueType>(actualType))
        {
            // load type from symbol
            auto module = addressOfOp->getParentOfType<mlir::ModuleOp>();
            assert(module);
            auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(addressOfOp.getGlobalName());
            if (!globalOp)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! NOT found symbol: " << addressOfOp.getGlobalName() << "\n";);
                assert(globalOp);
                return mlir::failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! found symbol: " << globalOp << "\n";);
            actualType = mlir_ts::RefType::get(globalOp.getType());

            auto value = lch.getAddressOfGlobalVar(addressOfOp.getGlobalName(), tch.convertType(actualType),
                                                   addressOfOp.getOffset() ? addressOfOp.getOffset().value() : 0);

            mlir::Value castedValue =
                rewriter.create<LLVM::BitcastOp>(addressOfOp->getLoc(), tch.convertType(addressOfOp.getType()), value);

            rewriter.replaceOp(addressOfOp, castedValue);
        }
        else
        {
            auto value = lch.getAddressOfGlobalVar(addressOfOp.getGlobalName(), tch.convertType(actualType),
                                                   addressOfOp.getOffset() ? addressOfOp.getOffset().value() : 0);
            rewriter.replaceOp(addressOfOp, value);
        }
        return success();
    }
};

struct DefaultOpLowering : public TsLlvmPattern<mlir_ts::DefaultOp>
{
    using TsLlvmPattern<mlir_ts::DefaultOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DefaultOp defaultOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = defaultOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        DefaultLogic dl(defaultOp, rewriter, tch, loc, tsLlvmContext->compileOptions);

        auto valueType = defaultOp.getRes().getType();
        auto llvmValueType = tch.convertType(valueType);

        mlir::Value defaultValue = dl.getDefaultValueForOrUndef(llvmValueType);

        rewriter.replaceOp(defaultOp, {defaultValue});

        return success();
    }
};

struct OptionalOpLowering : public TsLlvmPattern<mlir_ts::OptionalOp>
{
    using TsLlvmPattern<mlir_ts::OptionalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::OptionalOp optionalOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = optionalOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(optionalOp, rewriter);

        auto boxedType = cast<mlir_ts::OptionalType>(optionalOp.getRes().getType()).getElementType();
        auto llvmBoxedType = tch.convertType(boxedType);
        auto llvmOptType = tch.convertType(optionalOp.getRes().getType());

        auto valueOrigType = optionalOp.getIn().getType();

        LLVM_DEBUG(llvm::dbgs() << "\n!! Optional : " << optionalOp.getIn() << " flag: " << optionalOp.getFlag() << "\n";);

        auto value = transformed.getIn();
        auto valueLLVMType = value.getType();

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmOptType);

        // TODO: it should be tested by OP that value is equal to value in optional type
        if (valueLLVMType != llvmBoxedType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! Optional value types : " << valueLLVMType
                                    << " optional type: " << llvmBoxedType << "\n";);

            // cast value to box
            CastLogicHelper castLogic(optionalOp, rewriter, tch, tsLlvmContext->compileOptions);
            value = castLogic.cast(value, valueOrigType, valueLLVMType, boxedType, llvmBoxedType);
            if (!value)
            {
                return failure();
            }

            value = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoxedType, value);
        }

        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmOptType, structValue, value,
                                                                 MLIRHelper::getStructIndex(rewriter, OPTIONAL_VALUE_INDEX));

        rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(optionalOp, llvmOptType, structValue2, transformed.getFlag(),
                                                         MLIRHelper::getStructIndex(rewriter, OPTIONAL_HASVALUE_INDEX));

        return success();
    }
};

struct ValueOptionalOpLowering : public TsLlvmPattern<mlir_ts::OptionalValueOp>
{
    using TsLlvmPattern<mlir_ts::OptionalValueOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::OptionalValueOp createOptionalOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = createOptionalOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(createOptionalOp, rewriter);

        auto boxedType = cast<mlir_ts::OptionalType>(createOptionalOp.getRes().getType()).getElementType();
        auto llvmBoxedType = tch.convertType(boxedType);
        auto llvmOptType = tch.convertType(createOptionalOp.getRes().getType());

        auto valueOrigType = createOptionalOp.getIn().getType();

        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateOptional : " << createOptionalOp.getIn() << "\n";);

        auto value = transformed.getIn();
        auto valueLLVMType = value.getType();

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmOptType);

        // TODO: it should be tested by OP that value is equal to value in optional type
        if (valueLLVMType != llvmBoxedType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! CreateOptional value types : " << valueLLVMType
                                    << " optional type: " << llvmBoxedType << "\n";);

            // cast value to box
            CastLogicHelper castLogic(createOptionalOp, rewriter, tch, tsLlvmContext->compileOptions);
            value = castLogic.cast(value, valueOrigType, valueLLVMType, boxedType, llvmBoxedType);
            if (!value)
            {
                return failure();
            }

            value = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoxedType, value);
        }

        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmOptType, structValue, value,
                                                                 MLIRHelper::getStructIndex(rewriter, OPTIONAL_VALUE_INDEX));

        auto trueValue = clh.createI1ConstantOf(true);
        rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(createOptionalOp, llvmOptType, structValue2, trueValue,
                                                         MLIRHelper::getStructIndex(rewriter, OPTIONAL_HASVALUE_INDEX));

        return success();
    }
};

struct UndefOptionalOpLowering : public TsLlvmPattern<mlir_ts::OptionalUndefOp>
{
    using TsLlvmPattern<mlir_ts::OptionalUndefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::OptionalUndefOp undefOptionalOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = undefOptionalOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(undefOptionalOp, rewriter);
        DefaultLogic dl(undefOptionalOp, rewriter, tch, loc, tsLlvmContext->compileOptions);

        auto boxedType = cast<mlir_ts::OptionalType>(undefOptionalOp.getRes().getType()).getElementType();
        auto llvmBoxedType = tch.convertType(boxedType);
        auto llvmOptType = tch.convertType(undefOptionalOp.getRes().getType());

        mlir::Value structValue = rewriter.create<LLVM::UndefOp>(loc, llvmOptType);
        auto structValue2 = structValue;

        // default value
        mlir::Value defaultValue = dl.getDefaultValueForOrUndef(llvmBoxedType);
        if (defaultValue)
        {
            structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmOptType, structValue, defaultValue,
                                                                MLIRHelper::getStructIndex(rewriter, OPTIONAL_VALUE_INDEX));
        }

        auto falseValue = clh.createI1ConstantOf(false);
        rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(undefOptionalOp, llvmOptType, structValue2, falseValue,
                                                         MLIRHelper::getStructIndex(rewriter, OPTIONAL_HASVALUE_INDEX));

        return success();
    }
};

struct HasValueOpLowering : public TsLlvmPattern<mlir_ts::HasValueOp>
{
    using TsLlvmPattern<mlir_ts::HasValueOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::HasValueOp hasValueOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = hasValueOp->getLoc();

        TypeHelper th(rewriter);

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(hasValueOp, th.getLLVMBoolType(), transformed.getIn(),
                                                          MLIRHelper::getStructIndex(rewriter, OPTIONAL_HASVALUE_INDEX));

        return success();
    }
};

struct ValueOpLowering : public TsLlvmPattern<mlir_ts::ValueOp>
{
    using TsLlvmPattern<mlir_ts::ValueOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ValueOp valueOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = valueOp->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto valueType = valueOp.getRes().getType();
        auto llvmValueType = tch.convertType(valueType);

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(valueOp, llvmValueType, transformed.getIn(),
                                                          MLIRHelper::getStructIndex(rewriter, OPTIONAL_VALUE_INDEX));

        return success();
    }
};

struct ValueOrDefaultOpLowering : public TsLlvmPattern<mlir_ts::ValueOrDefaultOp>
{
    using TsLlvmPattern<mlir_ts::ValueOrDefaultOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ValueOrDefaultOp valueOrDefaultOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = valueOrDefaultOp->getLoc();

        valueOrDefaultOp->emitWarning("casting from optional (undefined) value");

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        DefaultLogic dl(valueOrDefaultOp, rewriter, tch, loc, tsLlvmContext->compileOptions);

        auto valueType = valueOrDefaultOp.getRes().getType();
        auto llvmValueType = tch.convertType(valueType);

        auto hasValue = rewriter.create<LLVM::ExtractValueOp>(loc, th.getLLVMBoolType(), transformed.getIn(),
                                                          MLIRHelper::getStructIndex(rewriter, OPTIONAL_HASVALUE_INDEX));
        auto value = rewriter.create<LLVM::ExtractValueOp>(loc, llvmValueType, transformed.getIn(),
                                                          MLIRHelper::getStructIndex(rewriter, OPTIONAL_VALUE_INDEX));

        mlir::Value defaultValue = dl.getDefaultValueForOrUndef(llvmValueType);

        rewriter.replaceOpWithNewOp<LLVM::SelectOp>(valueOrDefaultOp, hasValue, value, defaultValue);

        return success();
    }
};

struct LoadSaveValueLowering : public TsLlvmPattern<mlir_ts::LoadSaveOp>
{
    using TsLlvmPattern<mlir_ts::LoadSaveOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LoadSaveOp loadSaveOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = loadSaveOp->getLoc();

        auto value = rewriter.create<LLVM::LoadOp>(loc, transformed.getSrc().getType(), transformed.getSrc());
        rewriter.create<LLVM::StoreOp>(loc, value, transformed.getDst());

        rewriter.eraseOp(loadSaveOp);

        return success();
    }
};

struct CopyStructOpLowering : public TsLlvmPattern<mlir_ts::CopyStructOp>
{
    using TsLlvmPattern<mlir_ts::CopyStructOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CopyStructOp memoryCopyOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = memoryCopyOp->getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(memoryCopyOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(memoryCopyOp, rewriter);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto copyMemFuncOp = ch.getOrInsertFunction(
            llvmIndexType.getIntOrFloatBitWidth() == 32 
                ? "llvm.memcpy.p0.p0.i32" 
                : "llvm.memcpy.p0.p0.i64", 
            th.getFunctionType(th.getVoidType(), {th.getPtrType(), th.getPtrType(), llvmIndexType, th.getLLVMBoolType()}));

        mlir::SmallVector<mlir::Value, 4> values;
        values.push_back(transformed.getDst());
        values.push_back(transformed.getSrc());

        LLVM_DEBUG(llvm::dbgs() << "[CopyStructOp(1)] from type: " << memoryCopyOp.getSrc().getType() << " to type: " << memoryCopyOp.getDst().getType()
                                << "\n";);        

        assert(isa<LLVM::LLVMPointerType>(transformed.getSrc().getType()));
        auto srcStorageType = MLIRHelper::getElementTypeOrSelf(memoryCopyOp.getSrc().getType());
        auto srcSizeMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), srcStorageType);
        auto srcSize = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, srcSizeMLIR);

        assert(isa<LLVM::LLVMPointerType>(transformed.getDst().getType()));
        auto dstStorageType = MLIRHelper::getElementTypeOrSelf(memoryCopyOp.getDst().getType());
        auto dstSizeMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), dstStorageType);
        auto dstSize = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, dstSizeMLIR);

        auto cmpVal = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ult, srcSize, dstSize);
        auto minSize = rewriter.create<LLVM::SelectOp>(loc, cmpVal, srcSize, dstSize);

        values.push_back(minSize);

        auto immarg = clh.createI1ConstantOf(false);
        values.push_back(immarg);

        rewriter.create<LLVM::CallOp>(loc, copyMemFuncOp, values);

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(memoryCopyOp);

        return success();
    }
};

struct MemoryCopyOpLowering : public TsLlvmPattern<mlir_ts::MemoryCopyOp>
{
    using TsLlvmPattern<mlir_ts::MemoryCopyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::MemoryCopyOp memoryCopyOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = memoryCopyOp->getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(memoryCopyOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(memoryCopyOp, rewriter);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto copyMemFuncOp = ch.getOrInsertFunction(
            llvmIndexType.getIntOrFloatBitWidth() == 32 
                ? "llvm.memcpy.p0.p0.i32" 
                : "llvm.memcpy.p0.p0.i64", 
            th.getFunctionType(th.getVoidType(), {th.getPtrType(), th.getPtrType(), llvmIndexType, th.getLLVMBoolType()}));

        mlir::SmallVector<mlir::Value, 4> values;
        values.push_back(transformed.getDst());
        values.push_back(transformed.getSrc());

        auto countIndexType = transformed.getCount();

        values.push_back(countIndexType);

        auto immarg = clh.createI1ConstantOf(false);
        values.push_back(immarg);

        rewriter.create<LLVM::CallOp>(loc, copyMemFuncOp, values);

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(memoryCopyOp);

        return success();
    }
};

struct MemoryMoveOpLowering : public TsLlvmPattern<mlir_ts::MemoryMoveOp>
{
    using TsLlvmPattern<mlir_ts::MemoryMoveOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::MemoryMoveOp memoryMoveOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = memoryMoveOp->getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(memoryMoveOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(memoryMoveOp, rewriter);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto moveMemFuncOp = ch.getOrInsertFunction(
            llvmIndexType.getIntOrFloatBitWidth() == 32 
                ? "llvm.memmove.p0.p0.i32" 
                : "llvm.memmove.p0.p0.i64", 
            th.getFunctionType(th.getVoidType(), {th.getPtrType(), th.getPtrType(), llvmIndexType, th.getLLVMBoolType()}));

        mlir::SmallVector<mlir::Value, 4> values;
        values.push_back(transformed.getDst());
        values.push_back(transformed.getSrc());

        auto countAsIndexType = memoryMoveOp.getCount();

        auto llvmSrcType = tch.convertType(memoryMoveOp.getSrc().getType());
        auto srcSizeMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), transformed.getSrc().getType());
        auto srcSize = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, srcSizeMLIR);
        auto countAsIndexLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, countAsIndexType);
        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, llvmIndexType, ValueRange{srcSize, countAsIndexLLVMType});

        values.push_back(multSizeOfTypeValue);

        auto immarg = clh.createI1ConstantOf(false);
        values.push_back(immarg);
        
        rewriter.create<LLVM::CallOp>(loc, moveMemFuncOp, values);

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(memoryMoveOp);

        return success();
    }
};

struct UnreachableOpLowering : public TsLlvmPattern<mlir_ts::UnreachableOp>
{
    using TsLlvmPattern<mlir_ts::UnreachableOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::UnreachableOp unreachableOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = unreachableOp.getLoc();
        CodeLogicHelper clh(unreachableOp, rewriter);

        auto unreachable = clh.FindUnreachableBlockOrCreate();

        rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(unreachableOp, unreachable);

        // no need for cut if this is last op in block
        auto terminator = rewriter.getInsertionBlock()->getTerminator();
        if (terminator != unreachableOp && terminator != unreachableOp->getNextNode())
        {
            clh.CutBlock();
        }

        return success();
    }
};

struct ThrowCallOpLowering : public TsLlvmPattern<mlir_ts::ThrowCallOp>
{
    using TsLlvmPattern<mlir_ts::ThrowCallOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThrowCallOp throwCallOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());

        ThrowLogic tl(throwCallOp, rewriter, tch, throwCallOp.getLoc(), tsLlvmContext->compileOptions);
        tl.logic(transformed.getException(), throwCallOp.getException().getType(), nullptr);

        rewriter.eraseOp(throwCallOp);

        return success();
    }
};

struct ThrowUnwindOpLowering : public TsLlvmPattern<mlir_ts::ThrowUnwindOp>
{
    using TsLlvmPattern<mlir_ts::ThrowUnwindOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThrowUnwindOp throwUnwindOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        ThrowLogic tl(throwUnwindOp, rewriter, tch, throwUnwindOp.getLoc(), tsLlvmContext->compileOptions);
        tl.logic(transformed.getException(), throwUnwindOp.getException().getType(), throwUnwindOp.getUnwindDest());

        rewriter.eraseOp(throwUnwindOp);

        return success();
    }
};

namespace windows {

struct LandingPadOpLowering : public TsLlvmPattern<mlir_ts::LandingPadOp>
{
    using TsLlvmPattern<mlir_ts::LandingPadOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LandingPadOp landingPadOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = landingPadOp.getLoc();
        if (!landingPadOp.getCleanup())
        {
            auto catch1 = transformed.getCatches().front();
            mlir::Type llvmLandingPadTy = getTypeConverter()->convertType(landingPadOp.getType());
            rewriter.replaceOpWithNewOp<LLVM::LandingpadOp>(landingPadOp, llvmLandingPadTy, false, ValueRange{catch1});
        }
        else
        {
            // BUG: in LLVM landing pad is not fully implemented
            // so lets create filter with undef value to mark cleanup landing
            auto catch1Fake = transformed.getCatches().front();

            mlir::Type llvmLandingPadTy = getTypeConverter()->convertType(landingPadOp.getType());
            rewriter.replaceOpWithNewOp<LLVM::LandingpadOp>(landingPadOp, llvmLandingPadTy, true,
                                                            ValueRange{catch1Fake});
        }

        return success();
    }
};

struct CompareCatchTypeOpLowering : public TsLlvmPattern<mlir_ts::CompareCatchTypeOp>
{
    using TsLlvmPattern<mlir_ts::CompareCatchTypeOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CompareCatchTypeOp compareCatchTypeOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        CodeLogicHelper clh(compareCatchTypeOp, rewriter);

        auto trueVal = clh.createI1ConstantOf(true);
        rewriter.replaceOp(compareCatchTypeOp, ValueRange{trueVal});

        return success();
    }
};

struct BeginCatchOpLowering : public TsLlvmPattern<mlir_ts::BeginCatchOp>
{
    using TsLlvmPattern<mlir_ts::BeginCatchOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::BeginCatchOp beginCatchOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = beginCatchOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(beginCatchOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(beginCatchOp, rewriter);

        auto i8PtrTy = th.getPtrType();

        // catches:extract
        auto loadedI8PtrValue = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(), transformed.getLandingPad(),
                                                                      MLIRHelper::getStructIndex(rewriter, 0));

        auto beginCatchFuncName = "__cxa_begin_catch";
        auto beginCatchFunc = ch.getOrInsertFunction(beginCatchFuncName, th.getFunctionType(i8PtrTy, {i8PtrTy}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(beginCatchOp, beginCatchFunc, ValueRange{loadedI8PtrValue});

        return success();
    }
};

struct SaveCatchVarOpLowering : public TsLlvmPattern<mlir_ts::SaveCatchVarOp>
{
    using TsLlvmPattern<mlir_ts::SaveCatchVarOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SaveCatchVarOp saveCatchVarOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = saveCatchVarOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(saveCatchVarOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto ptr = rewriter.create<mlir_ts::CastOp>(loc, th.getPtrType(), transformed.getVarStore());

        auto saveCatchFuncName = "ts.internal.save_catch_var";
        auto saveCatchFunc = ch.getOrInsertFunction(
            saveCatchFuncName,
            th.getFunctionType(th.getVoidType(), ArrayRef<mlir::Type>{getTypeConverter()->convertType(
                                                                          /*saveCatchVarOp*/transformed.getExceptionInfo().getType()),
                                                                      ptr.getType()}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(saveCatchVarOp, saveCatchFunc,
                                                  ValueRange{/*saveCatchVarOp*/transformed.getExceptionInfo(), ptr});

        return success();
    }
};

struct EndCatchOpLowering : public TsLlvmPattern<mlir_ts::EndCatchOp>
{
    using TsLlvmPattern<mlir_ts::EndCatchOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::EndCatchOp endCatchOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = endCatchOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(endCatchOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto endCatchFuncName = "__cxa_end_catch";
        auto endCatchFunc =
            ch.getOrInsertFunction(endCatchFuncName, th.getFunctionType(th.getVoidType(), ArrayRef<mlir::Type>{}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(endCatchOp, endCatchFunc, ValueRange{});

        return success();
    }
};

struct BeginCleanupOpLowering : public TsLlvmPattern<mlir_ts::BeginCleanupOp>
{
    using TsLlvmPattern<mlir_ts::BeginCleanupOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::BeginCleanupOp beginCleanupOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.eraseOp(beginCleanupOp);

        return success();
    }
};

struct EndCleanupOpLowering : public TsLlvmPattern<mlir_ts::EndCleanupOp>
{
    using TsLlvmPattern<mlir_ts::EndCleanupOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::EndCleanupOp endCleanupOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = endCleanupOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(endCleanupOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(endCleanupOp, rewriter);

        auto endCatchFuncName = "__cxa_end_catch";
        if (!endCleanupOp.getUnwindDest().empty())
        {
            clh.Invoke(loc, [&](mlir::Block *continueBlock) {
                rewriter.replaceOpWithNewOp<LLVM::InvokeOp>(
                    endCleanupOp, mlir::TypeRange{},
                    ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), endCatchFuncName), ValueRange{},
                    continueBlock, ValueRange{}, endCleanupOp.getUnwindDest().front(), ValueRange{});
            });
            rewriter.setInsertionPointAfter(endCleanupOp);
        }
        else
        {
            auto endCatchFunc =
                ch.getOrInsertFunction(endCatchFuncName, th.getFunctionType(th.getVoidType(), ArrayRef<mlir::Type>{}));
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(endCleanupOp, endCatchFunc, ValueRange{});
            rewriter.setInsertionPointAfter(endCleanupOp);
        }

        rewriter.create<LLVM::ResumeOp>(loc, transformed.getLandingPad());

        // no need for cut if this is last op in block
        auto terminator = rewriter.getInsertionBlock()->getTerminator();
        if (terminator != endCleanupOp && terminator != endCleanupOp->getNextNode())
        {
            clh.CutBlock();
        }

        // add resume

        return success();
    }
};

} // namespace windows

namespace linux {
struct LandingPadOpLowering : public TsLlvmPattern<mlir_ts::LandingPadOp>
{
    using TsLlvmPattern<mlir_ts::LandingPadOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LandingPadOp landingPadOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = landingPadOp.getLoc();

        TypeHelper th(rewriter);

        auto catch1 = transformed.getCatches().front();

        mlir::Type llvmLandingPadTy = getTypeConverter()->convertType(landingPadOp.getType());
        rewriter.replaceOpWithNewOp<LLVM::LandingpadOp>(landingPadOp, llvmLandingPadTy, false, ValueRange{catch1});

        return success();
    }
};

struct CompareCatchTypeOpLowering : public TsLlvmPattern<mlir_ts::CompareCatchTypeOp>
{
    using TsLlvmPattern<mlir_ts::CompareCatchTypeOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CompareCatchTypeOp compareCatchTypeOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = compareCatchTypeOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(compareCatchTypeOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(compareCatchTypeOp, rewriter);

        auto i8PtrTy = th.getPtrType();

        auto loadedI32Value = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI32Type(), transformed.getLandingPad(),
                                                                    MLIRHelper::getStructIndex(rewriter, 1));

        auto typeIdFuncName = "llvm.eh.typeid.for.p0";
        auto typeIdFunc = ch.getOrInsertFunction(typeIdFuncName, th.getFunctionType(th.getI32Type(), {i8PtrTy}));

        auto callInfo =
            rewriter.create<LLVM::CallOp>(loc, typeIdFunc, ValueRange{transformed.getThrowTypeInfo()});
        auto typeIdValue = callInfo.getResult();

        // icmp
        auto cmpValue = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, loadedI32Value, typeIdValue);
        rewriter.replaceOp(compareCatchTypeOp, ValueRange{cmpValue});

        return success();
    }
};

struct BeginCatchOpLowering : public TsLlvmPattern<mlir_ts::BeginCatchOp>
{
    using TsLlvmPattern<mlir_ts::BeginCatchOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::BeginCatchOp beginCatchOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = beginCatchOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(beginCatchOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(beginCatchOp, rewriter);

        auto i8PtrTy = th.getPtrType();

        // catches:extract
        auto loadedI8PtrValue = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(), transformed.getLandingPad(),
                                                                      MLIRHelper::getStructIndex(rewriter, 0));

        auto beginCatchFuncName = "__cxa_begin_catch";
        auto beginCatchFunc = ch.getOrInsertFunction(beginCatchFuncName, th.getFunctionType(i8PtrTy, {i8PtrTy}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(beginCatchOp, beginCatchFunc, ValueRange{loadedI8PtrValue});

        return success();
    }
};

struct SaveCatchVarOpLowering : public TsLlvmPattern<mlir_ts::SaveCatchVarOp>
{
    using TsLlvmPattern<mlir_ts::SaveCatchVarOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SaveCatchVarOp saveCatchVarOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = saveCatchVarOp.getLoc();

        TypeHelper th(rewriter);

        auto catchRefType = cast<mlir_ts::RefType>(saveCatchVarOp.getVarStore().getType());
        auto catchType = catchRefType.getElementType();
        auto llvmCatchType = getTypeConverter()->convertType(catchType);

        mlir::Value catchVal;
        if (!isa<LLVM::LLVMPointerType>(llvmCatchType))
        {
            auto ptrVal =
                rewriter.create<LLVM::BitcastOp>(loc, th.getPtrType(), transformed.getExceptionInfo());
            catchVal = rewriter.create<LLVM::LoadOp>(loc, llvmCatchType, ptrVal);
        }
        else
        {
            catchVal = rewriter.create<LLVM::BitcastOp>(loc, llvmCatchType, transformed.getExceptionInfo());
        }

        rewriter.replaceOpWithNewOp<mlir_ts::StoreOp>(saveCatchVarOp, catchVal, transformed.getVarStore());

        return success();
    }
};

struct EndCatchOpLowering : public TsLlvmPattern<mlir_ts::EndCatchOp>
{
    using TsLlvmPattern<mlir_ts::EndCatchOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::EndCatchOp endCatchOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = endCatchOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(endCatchOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto endCatchFuncName = "__cxa_end_catch";
        auto endCatchFunc =
            ch.getOrInsertFunction(endCatchFuncName, th.getFunctionType(th.getVoidType(), ArrayRef<mlir::Type>{}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(endCatchOp, endCatchFunc, ValueRange{});

        return success();
    }
};

struct BeginCleanupOpLowering : public TsLlvmPattern<mlir_ts::BeginCleanupOp>
{
    using TsLlvmPattern<mlir_ts::BeginCleanupOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::BeginCleanupOp beginCleanupOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.eraseOp(beginCleanupOp);

        return success();
    }
};

struct EndCleanupOpLowering : public TsLlvmPattern<mlir_ts::EndCleanupOp>
{
    using TsLlvmPattern<mlir_ts::EndCleanupOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::EndCleanupOp endCleanupOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = endCleanupOp.getLoc();

        CodeLogicHelper clh(endCleanupOp, rewriter);

        rewriter.replaceOpWithNewOp<LLVM::ResumeOp>(endCleanupOp, transformed.getLandingPad());

        auto terminator = rewriter.getInsertionBlock()->getTerminator();
        if (terminator != endCleanupOp && terminator != endCleanupOp->getNextNode())
        {
            clh.CutBlock();
        }

        // add resume

        return success();
    }
};

} // namespace linux

struct VTableOffsetRefOpLowering : public TsLlvmPattern<mlir_ts::VTableOffsetRefOp>
{
    using TsLlvmPattern<mlir_ts::VTableOffsetRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::VTableOffsetRefOp vtableOffsetRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = vtableOffsetRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(vtableOffsetRefOp, rewriter);

        auto ptrToArrOfPtrs = rewriter.create<mlir_ts::CastOp>(loc, th.getPtrType(), transformed.getVtable());

        auto index = clh.createI32ConstantOf(vtableOffsetRefOp.getIndex());
        auto methodOrInterfacePtrPtr =
            rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), th.getPtrType(), ptrToArrOfPtrs, ValueRange{index});
        auto methodOrInterfacePtr = rewriter.create<LLVM::LoadOp>(loc, th.getPtrType(), methodOrInterfacePtrPtr);

        rewriter.replaceOp(vtableOffsetRefOp, ValueRange{methodOrInterfacePtr});

        return success();
    }
};

struct VirtualSymbolRefOpLowering : public TsLlvmPattern<mlir_ts::VirtualSymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::VirtualSymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::VirtualSymbolRefOp virtualSymbolRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        assert(virtualSymbolRefOp.getIndex() != -1);

        Location loc = virtualSymbolRefOp.getLoc();

        TypeHelper th(rewriter);

        auto methodOrFieldPtr = rewriter.create<mlir_ts::VTableOffsetRefOp>(
            loc, th.getPtrType(), transformed.getVtable(), virtualSymbolRefOp.getIndex());

        if (auto funcType = dyn_cast<mlir_ts::FunctionType>(virtualSymbolRefOp.getType()))
        {
            auto methodTyped = rewriter.create<mlir_ts::CastOp>(loc, funcType, methodOrFieldPtr);
            rewriter.replaceOp(virtualSymbolRefOp, ValueRange{methodTyped});
        }
        else if (auto fieldType = dyn_cast<mlir_ts::RefType>(virtualSymbolRefOp.getType()))
        {
            auto fieldTyped = rewriter.create<mlir_ts::CastOp>(loc, fieldType, methodOrFieldPtr);
            rewriter.replaceOp(virtualSymbolRefOp, ValueRange{fieldTyped});
        }
        else
        {
            llvm_unreachable("not implemented");
            return failure();
        }

        return success();
    }
};

struct ThisVirtualSymbolRefOpLowering : public TsLlvmPattern<mlir_ts::ThisVirtualSymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::ThisVirtualSymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThisVirtualSymbolRefOp thisVirtualSymbolRefOp,
                                  Adaptor transformed, ConversionPatternRewriter &rewriter) const final
    {
        

        assert(thisVirtualSymbolRefOp.getIndex() != -1);

        Location loc = thisVirtualSymbolRefOp.getLoc();

        TypeHelper th(rewriter);

        auto methodPtr = rewriter.create<mlir_ts::VTableOffsetRefOp>(loc, th.getPtrType(), transformed.getVtable(),
                                                                     thisVirtualSymbolRefOp.getIndex());
        // auto methodTyped = rewriter.create<mlir_ts::CastOp>(loc, thisVirtualSymbolRefOp.getType(), methodPtr);

        if (auto boundFunc = dyn_cast<mlir_ts::BoundFunctionType>(thisVirtualSymbolRefOp.getType()))
        {
            auto thisOpaque = rewriter.create<mlir_ts::CastOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()),
                                                               transformed.getThisVal());
            auto methodTyped = rewriter.create<mlir_ts::CastOp>(
                loc, mlir_ts::FunctionType::get(rewriter.getContext(), boundFunc.getInputs(), boundFunc.getResults()),
                methodPtr);
            auto boundFuncVal =
                rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, boundFunc, thisOpaque, methodTyped);
            rewriter.replaceOp(thisVirtualSymbolRefOp, ValueRange{boundFuncVal});
        }
        else
        {
            llvm_unreachable("not implemented");
            return failure();
        }

        return success();
    }
};

struct InterfaceSymbolRefOpLowering : public TsLlvmPattern<mlir_ts::InterfaceSymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::InterfaceSymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InterfaceSymbolRefOp interfaceSymbolRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        assert(interfaceSymbolRefOp.getIndex() != -1);

        Location loc = interfaceSymbolRefOp.getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(interfaceSymbolRefOp, rewriter);
        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto fieldLLVMTypeRef = tch.convertType(interfaceSymbolRefOp.getType());

        auto isOptional = interfaceSymbolRefOp.getOptional().has_value() && interfaceSymbolRefOp.getOptional().value();

        auto vtable = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(), transformed.getInterfaceVal(),
                                                            MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));
        auto thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(), transformed.getInterfaceVal(),
                                                             MLIRHelper::getStructIndex(rewriter, THIS_VALUE_INDEX));

        auto methodOrFieldPtr =
            rewriter.create<mlir_ts::VTableOffsetRefOp>(loc, th.getPtrType(), vtable, interfaceSymbolRefOp.getIndex());

        if (auto boundFunc = dyn_cast<mlir_ts::BoundFunctionType>(interfaceSymbolRefOp.getType()))
        {
            auto thisOpaque =
                rewriter.create<mlir_ts::CastOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()), thisVal);
            auto methodTypedPtr = rewriter.create<mlir_ts::CastOp>(
                loc, mlir_ts::FunctionType::get(rewriter.getContext(), boundFunc.getInputs(), boundFunc.getResults()),
                methodOrFieldPtr);
            auto boundFuncVal =
                rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, boundFunc, thisOpaque, methodTypedPtr);
            rewriter.replaceOp(interfaceSymbolRefOp, ValueRange{boundFuncVal});
        }
        else
        {
            auto calcFieldTotalAddrFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
                // BoundRef
                auto p1 = rewriter.create<LLVM::PtrToIntOp>(loc, llvmIndexType, thisVal);
                auto p2 = rewriter.create<LLVM::PtrToIntOp>(loc, llvmIndexType, methodOrFieldPtr);
                auto padded = rewriter.create<LLVM::AddOp>(loc, llvmIndexType, p1, p2);
                auto typedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, fieldLLVMTypeRef, padded);

                // no need to BoundRef
                // auto boundRefVal = rewriter.create<mlir_ts::CreateBoundRefOp>(loc, thisVal, typedPtr);
                return typedPtr;
            };

            mlir::Value fieldAddr;
            if (isOptional)
            {
                auto nullAddrFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
                    auto typedPtr = rewriter.create<LLVM::ZeroOp>(loc, fieldLLVMTypeRef);
                    return typedPtr;
                };

                LLVMTypeConverterHelper llvmtch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));

                auto negative1 = tsLlvmContext->compileOptions.sizeBits == 32 
                    ? clh.createI32ConstantOf(-1) 
                    : clh.createI64ConstantOf(-1);
                auto intPtrType = llvmtch.getIntPtrType(0);
                auto methodOrFieldIntPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, methodOrFieldPtr);
                auto condVal =
                    rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, methodOrFieldIntPtrValue, negative1);

                auto result =
                    clh.conditionalExpressionLowering(loc, fieldLLVMTypeRef, condVal, nullAddrFunc, calcFieldTotalAddrFunc);
                fieldAddr = result;
            }
            else
            {
                auto typedPtr = calcFieldTotalAddrFunc(rewriter, loc);
                fieldAddr = typedPtr;
            }

            rewriter.replaceOp(interfaceSymbolRefOp, ValueRange{fieldAddr});
        }

        return success();
    }
};

struct NewInterfaceOpLowering : public TsLlvmPattern<mlir_ts::NewInterfaceOp>
{
    using TsLlvmPattern<mlir_ts::NewInterfaceOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewInterfaceOp newInterfaceOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = newInterfaceOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(newInterfaceOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto llvmInterfaceType = tch.convertType(newInterfaceOp.getType());

        auto structVal = rewriter.create<LLVM::UndefOp>(loc, llvmInterfaceType);
        auto structVal2 = rewriter.create<LLVM::InsertValueOp>(
            loc, structVal, transformed.getInterfaceVTable(), MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, transformed.getThisVal(),
                                                               MLIRHelper::getStructIndex(rewriter, THIS_VALUE_INDEX));

        rewriter.replaceOp(newInterfaceOp, ValueRange{structVal3});

        return success();
    }
};

struct ExtractInterfaceThisOpLowering : public TsLlvmPattern<mlir_ts::ExtractInterfaceThisOp>
{
    using TsLlvmPattern<mlir_ts::ExtractInterfaceThisOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ExtractInterfaceThisOp extractInterfaceThisOp,
                                  Adaptor transformed, ConversionPatternRewriter &rewriter) const final
    {
        

        // TODO: hack, if NullOp, return null

        Location loc = extractInterfaceThisOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(extractInterfaceThisOp, rewriter);

        LLVM_DEBUG(llvm::dbgs() << "\n!! ExtractInterfaceThis from: " << extractInterfaceThisOp.getInterfaceVal() << "\n");

        auto vtable = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(), transformed.getInterfaceVal(),
                                                            MLIRHelper::getStructIndex(rewriter, THIS_VALUE_INDEX));

        rewriter.replaceOp(extractInterfaceThisOp, ValueRange{vtable});

        return success();
    }
};

struct ExtractInterfaceVTableOpLowering : public TsLlvmPattern<mlir_ts::ExtractInterfaceVTableOp>
{
    using TsLlvmPattern<mlir_ts::ExtractInterfaceVTableOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ExtractInterfaceVTableOp extractInterfaceVTableOp,
                                  Adaptor transformed, ConversionPatternRewriter &rewriter) const final
    {
        

        // TODO: hack, if NullOp, return null

        Location loc = extractInterfaceVTableOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(extractInterfaceVTableOp, rewriter);

        LLVM_DEBUG(llvm::dbgs() << "\n!! ExtractInterfaceVTable from: " << transformed.getInterfaceVal()
                                << "\n");

        auto vtable = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(), transformed.getInterfaceVal(),
                                                            MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));

        rewriter.replaceOp(extractInterfaceVTableOp, ValueRange{vtable});

        return success();
    }
};

struct LoadBoundRefOpLowering : public TsLlvmPattern<mlir_ts::LoadBoundRefOp>
{
    using TsLlvmPattern<mlir_ts::LoadBoundRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LoadBoundRefOp loadBoundRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = loadBoundRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(loadBoundRefOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto boundRefType = cast<mlir_ts::BoundRefType>(loadBoundRefOp.getReference().getType());

        auto llvmType = tch.convertType(boundRefType.getElementType());
        auto ptrType = th.getPtrType();

        auto thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, transformed.getReference(),
                                                             MLIRHelper::getStructIndex(rewriter, THIS_VALUE_INDEX));
        auto valueRefVal = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, transformed.getReference(),
                                                                 MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));

        // atomic access attrs
        auto alignment = 0;
        auto isVolatile = false;
        auto isNonTemporal = false;
        auto isInvariant = false;
        auto ordering = LLVM::AtomicOrdering::not_atomic;
        StringRef syncscope = {};

        if (auto atomicAttr = loadBoundRefOp->getAttrOfType<mlir::BoolAttr>(ATOMIC_ATTR_NAME))
        {
            if (atomicAttr.getValue()) 
            {
                auto orderingAttr = loadBoundRefOp->getAttrOfType<mlir::IntegerAttr>(ORDERING_ATTR_NAME);
                auto syncScopeAttr = loadBoundRefOp->getAttrOfType<mlir::StringAttr>(SYNCSCOPE_ATTR_NAME);
                if (orderingAttr && syncScopeAttr)
                {
                    ordering = (LLVM::AtomicOrdering) orderingAttr.getValue().getZExtValue();
                    if (ordering == LLVM::AtomicOrdering::release || ordering == LLVM::AtomicOrdering::acq_rel)
                    {
                        ordering = LLVM::AtomicOrdering::acquire;
                    }

                    syncscope = syncScopeAttr.getValue();
                    LLVMTypeConverterHelper ltch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
                    alignment = ltch.getTypeAlignSizeInBits(llvmType);
                }
            }
        }

        if (auto volatileAttr = loadBoundRefOp->getAttrOfType<mlir::BoolAttr>(VOLATILE_ATTR_NAME))
        {
            isVolatile = volatileAttr.getValue();
        }

        mlir::Value loadedValue = rewriter.create<LLVM::LoadOp>(loc, llvmType, valueRefVal, alignment, isVolatile, isNonTemporal, isInvariant, ordering, syncscope);

        if (auto funcType = dyn_cast<mlir_ts::FunctionType>(boundRefType.getElementType()))
        {
            mlir::Value boundMethodValue =
                rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, loadBoundRefOp.getType(), thisVal, loadedValue);

            LLVM_DEBUG(llvm::dbgs() << "\n!! LoadOp Bound Ref: LLVM Type :" << tch.convertType(loadBoundRefOp.getType())
                                    << "\n";);

            rewriter.replaceOp(loadBoundRefOp, boundMethodValue);
        }
        else
        {
            rewriter.replaceOp(loadBoundRefOp, loadedValue);
        }

        return success();
    }
};

struct StoreBoundRefOpLowering : public TsLlvmPattern<mlir_ts::StoreBoundRefOp>
{
    using TsLlvmPattern<mlir_ts::StoreBoundRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StoreBoundRefOp storeBoundRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = storeBoundRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(storeBoundRefOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto boundRefType = cast<mlir_ts::BoundRefType>(storeBoundRefOp.getReference().getType());

        auto llvmType = tch.convertType(boundRefType.getElementType());
        auto llvmRefType = th.getPtrType();

        auto valueRefVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, transformed.getReference(),
                                                                 MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));

        // atomic attributes
        auto alignment = 0;
        auto isVolatile = false;
        auto isNonTemporal = false;
        auto ordering = LLVM::AtomicOrdering::not_atomic;
        StringRef syncscope = {};

        if (auto atomicAttr = storeBoundRefOp->getAttrOfType<mlir::BoolAttr>(ATOMIC_ATTR_NAME))
        {
            if (atomicAttr.getValue()) 
            {
                auto orderingAttr = storeBoundRefOp->getAttrOfType<mlir::IntegerAttr>(ORDERING_ATTR_NAME);
                auto syncScopeAttr = storeBoundRefOp->getAttrOfType<mlir::StringAttr>(SYNCSCOPE_ATTR_NAME);
                if (orderingAttr && syncScopeAttr)
                {
                    ordering = (LLVM::AtomicOrdering) orderingAttr.getValue().getZExtValue();
                    if (ordering == LLVM::AtomicOrdering::acquire || ordering == LLVM::AtomicOrdering::acq_rel)
                    {
                        ordering = LLVM::AtomicOrdering::release;
                    }

                    syncscope = syncScopeAttr.getValue();
                    LLVMTypeConverterHelper ltch(static_cast<const LLVMTypeConverter *>(getTypeConverter()));
                    alignment = ltch.getTypeAlignSizeInBits(transformed.getValue().getType());
                }
            }
        }

        if (auto volatileAttr = storeBoundRefOp->getAttrOfType<mlir::BoolAttr>(VOLATILE_ATTR_NAME))
        {
            isVolatile = volatileAttr.getValue();
        } 

        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeBoundRefOp, transformed.getValue(), valueRefVal, alignment, isVolatile, isNonTemporal, ordering, syncscope);
        return success();
    }
};

struct CreateBoundRefOpLowering : public TsLlvmPattern<mlir_ts::CreateBoundRefOp>
{
    using TsLlvmPattern<mlir_ts::CreateBoundRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateBoundRefOp createBoundRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = createBoundRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(createBoundRefOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto llvmBoundRefType = tch.convertType(createBoundRefOp.getType());

        auto structVal = rewriter.create<mlir_ts::UndefOp>(loc, llvmBoundRefType);
        auto structVal2 = rewriter.create<LLVM::InsertValueOp>(loc, structVal, transformed.getValueRef(),
                                                               MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, transformed.getThisVal(),
                                                               MLIRHelper::getStructIndex(rewriter, THIS_VALUE_INDEX));

        rewriter.replaceOp(createBoundRefOp, ValueRange{structVal3});

        return success();
    }
};

struct CreateBoundFunctionOpLowering : public TsLlvmPattern<mlir_ts::CreateBoundFunctionOp>
{
    using TsLlvmPattern<mlir_ts::CreateBoundFunctionOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateBoundFunctionOp createBoundFunctionOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = createBoundFunctionOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(createBoundFunctionOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        assert(createBoundFunctionOp.getType());
        assert(isa<mlir_ts::BoundFunctionType>(createBoundFunctionOp.getType()) ||
               isa<mlir_ts::HybridFunctionType>(createBoundFunctionOp.getType()));

        auto llvmBoundFunctionType = tch.convertType(createBoundFunctionOp.getType());

        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateBoundFunction: LLVM Type :" << llvmBoundFunctionType << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateBoundFunction: Func Type :"
                                << tch.convertType(createBoundFunctionOp.getFunc().getType()) << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateBoundFunction: This Type :" << createBoundFunctionOp.getThisVal().getType()
                                << "\n";);

        auto structVal = rewriter.create<mlir_ts::UndefOp>(loc, llvmBoundFunctionType);
        auto structVal2 = rewriter.create<LLVM::InsertValueOp>(loc, structVal, transformed.getFunc(),
                                                               MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, transformed.getThisVal(),
                                                               MLIRHelper::getStructIndex(rewriter, THIS_VALUE_INDEX));

        rewriter.replaceOp(createBoundFunctionOp, ValueRange{structVal3});

        return success();
    }
};

struct GetThisOpLowering : public TsLlvmPattern<mlir_ts::GetThisOp>
{
    using TsLlvmPattern<mlir_ts::GetThisOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GetThisOp getThisOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = getThisOp.getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(getThisOp, rewriter);

        auto llvmThisType = tch.convertType(getThisOp.getType());

        mlir::Value thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, th.getPtrType(), transformed.getBoundFunc(),
                                                                    MLIRHelper::getStructIndex(rewriter, THIS_VALUE_INDEX));

        auto thisValCasted = rewriter.create<LLVM::BitcastOp>(loc, llvmThisType, thisVal);

        rewriter.replaceOp(getThisOp, thisValCasted);

        return success();
    }
};

struct GetMethodOpLowering : public TsLlvmPattern<mlir_ts::GetMethodOp>
{
    using TsLlvmPattern<mlir_ts::GetMethodOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GetMethodOp getMethodOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = getMethodOp.getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(getMethodOp, rewriter);
        CastLogicHelper castLogic(getMethodOp, rewriter, tch, tsLlvmContext->compileOptions);

        auto origType = getMethodOp.getBoundFunc().getType();

        mlir_ts::FunctionType funcType;
        mlir::Type llvmMethodType;
        if (auto boundType = dyn_cast<mlir_ts::BoundFunctionType>(origType))
        {
            funcType = mlir_ts::FunctionType::get(rewriter.getContext(), boundType.getInputs(), boundType.getResults());
            llvmMethodType = tch.convertType(funcType);
        }
        else if (auto hybridType = dyn_cast<mlir_ts::HybridFunctionType>(origType))
        {
            funcType =
                mlir_ts::FunctionType::get(rewriter.getContext(), hybridType.getInputs(), hybridType.getResults());
            llvmMethodType = tch.convertType(funcType);
        }
        else if (auto structType = dyn_cast<LLVM::LLVMStructType>(origType))
        {
            auto ptrType = cast<LLVM::LLVMPointerType>(structType.getBody().front());
            //assert(isa<LLVM::LLVMFunctionType>(ptrType.getElementType()));
            llvmMethodType = ptrType;
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! GetMethodOp: " << getMethodOp << " result type: " << getMethodOp.getType()
                                    << "\n");
            llvm_unreachable("not implemented");
            return failure();
        }

        mlir::Value methodVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmMethodType, transformed.getBoundFunc(),
                                                                      MLIRHelper::getStructIndex(rewriter, DATA_VALUE_INDEX));

        if (methodVal.getType() != getMethodOp.getType())
        {
            methodVal = castLogic.cast(methodVal, methodVal.getType(), getMethodOp.getType());
        }

        rewriter.replaceOp(getMethodOp, ValueRange{methodVal});
        return success();
    }
};

// TODO: review it, i need it for Union type
struct TypeOfOpLowering : public TsLlvmPattern<mlir_ts::TypeOfOp>
{
    using TsLlvmPattern<mlir_ts::TypeOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TypeOfOp typeOfOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeOfOpHelper toh(rewriter);
        auto typeOfValue = toh.typeOfLogic(typeOfOp->getLoc(), transformed.getValue(), typeOfOp.getValue().getType(), tsLlvmContext->compileOptions);

        rewriter.replaceOp(typeOfOp, ValueRange{typeOfValue});
        return success();
    }
};

struct TypeOfAnyOpLowering : public TsLlvmPattern<mlir_ts::TypeOfAnyOp>
{
    using TsLlvmPattern<mlir_ts::TypeOfAnyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TypeOfAnyOp typeOfAnyOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = typeOfAnyOp.getLoc();

        TypeConverterHelper tch(getTypeConverter());

        LLVM_DEBUG(llvm::dbgs() << "\n!! TypeOf: " << typeOfAnyOp.getValue() << "\n";);

        AnyLogic al(typeOfAnyOp, rewriter, tch, loc, tsLlvmContext->compileOptions);
        auto typeOfValue = al.getTypeOfAny(transformed.getValue());

        rewriter.replaceOp(typeOfAnyOp, ValueRange{typeOfValue});
        return success();
    }
};

class DebuggerOpLowering : public TsLlvmPattern<mlir_ts::DebuggerOp>
{
  public:
    using TsLlvmPattern<mlir_ts::DebuggerOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DebuggerOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        // Get a symbol reference to the printf function, inserting it if necessary.
        auto debugtrapFuncOp = ch.getOrInsertFunction("llvm.debugtrap", th.getFunctionType({}));

        rewriter.create<LLVM::CallOp>(loc, debugtrapFuncOp, ValueRange{});

        rewriter.eraseOp(op);

        return success();
    }
};

#ifndef DISABLE_SWITCH_STATE_PASS

struct StateLabelOpLowering : public TsLlvmPattern<mlir_ts::StateLabelOp>
{
    using TsLlvmPattern<mlir_ts::StateLabelOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StateLabelOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(op, rewriter);

        clh.BeginBlock(op.getLoc());

        rewriter.eraseOp(op);

        return success();
    }
};

#define MLIR_SWITCH 1
class SwitchStateOpLowering : public TsLlvmPattern<mlir_ts::SwitchStateOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SwitchStateOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SwitchStateOp switchStateOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        CodeLogicHelper clh(switchStateOp, rewriter);

        auto loc = switchStateOp->getLoc();

        auto returnBlock = clh.FindReturnBlock(true);

        assert(returnBlock);

        LLVM_DEBUG(llvm::dbgs() << "\n!! return block: "; returnBlock->dump(); llvm::dbgs() << "\n";);

        auto defaultBlock = returnBlock;

        assert(defaultBlock != nullptr);

        SmallVector<int32_t> caseValues;
        SmallVector<mlir::Block *> caseDestinations;

        SmallPtrSet<Operation *, 16> stateLabels;

        auto index = 1;

        // select all states
        auto visitorAllStateLabels = [&](Operation *op) {
            if (auto stateLabelOp = dyn_cast_or_null<mlir_ts::StateLabelOp>(op))
            {
                stateLabels.insert(op);
            }
        };

        switchStateOp->getParentOp()->walk(visitorAllStateLabels);

        {
            mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
            for (auto [index, op] : enumerate(stateLabels))
            {
                auto stateLabelOp = dyn_cast_or_null<mlir_ts::StateLabelOp>(op);
                rewriter.setInsertionPoint(stateLabelOp);

                auto *continuationBlock = clh.BeginBlock(loc);

                rewriter.eraseOp(stateLabelOp);

                // add switch
                caseValues.push_back(index);
                caseDestinations.push_back(continuationBlock);
            }
        }

        // make switch to be terminator
        rewriter.setInsertionPointAfter(switchStateOp);
        auto *continuationBlock = clh.CutBlockAndSetInsertPointToEndOfBlock();

        // insert 0 state label
        caseValues.insert(caseValues.begin(), 0);
        caseDestinations.insert(caseDestinations.begin(), continuationBlock);

        rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(switchStateOp, transformed.state(),
                                                    defaultBlock ? defaultBlock : continuationBlock, ValueRange{},
                                                    caseValues, caseDestinations);

        LLVM_DEBUG(llvm::dbgs() << "\n!! SWITCH DUMP: \n" << *switchStateOp->getParentOp() << "\n";);

        return success();
    }
};

struct YieldReturnValOpLowering : public TsLlvmPattern<mlir_ts::YieldReturnValOp>
{
    using TsLlvmPattern<mlir_ts::YieldReturnValOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::YieldReturnValOp yieldReturnValOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        CodeLogicHelper clh(yieldReturnValOp, rewriter);

        auto returnBlock = clh.FindReturnBlock();

        assert(returnBlock);

        LLVM_DEBUG(llvm::dbgs() << "\n!! return block: "; returnBlock->dump(); llvm::dbgs() << "\n";);

        auto retBlock = returnBlock;

        rewriter.replaceOpWithNewOp<mlir_ts::StoreOp>(yieldReturnValOp, transformed.operand(), transformed.getReference());

        rewriter.setInsertionPointAfter(yieldReturnValOp);
        clh.JumpTo(yieldReturnValOp.getLoc(), retBlock);

        LLVM_DEBUG(llvm::dbgs() << "\n!! YIELD DUMP: \n" << *yieldReturnValOp->getParentOp() << "\n";);

        return success();
    }
};

#endif

class SwitchStateInternalOpLowering : public TsLlvmPattern<mlir_ts::SwitchStateInternalOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SwitchStateInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SwitchStateInternalOp switchStateOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        CodeLogicHelper clh(switchStateOp, rewriter);

        auto loc = switchStateOp->getLoc();

        SmallVector<int32_t> caseValues;
        SmallVector<mlir::Block *> caseDestinations;
        SmallVector<ValueRange> caseOperands;

        for (auto [index, case1] : enumerate(switchStateOp.getCases()))
        {
            caseValues.push_back(index);
            caseDestinations.push_back(case1);
            caseOperands.push_back(ValueRange());
        }

        rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(switchStateOp, transformed.getState(), switchStateOp.getDefaultDest(),
                                                    ValueRange{}, caseValues, caseDestinations, caseOperands);

        LLVM_DEBUG(llvm::dbgs() << "\n!! SWITCH DUMP: \n" << *switchStateOp->getParentOp() << "\n";);

        return success();
    }
};

struct GlobalConstructorOpLowering : public TsLlvmPattern<mlir_ts::GlobalConstructorOp>
{
    using TsLlvmPattern<mlir_ts::GlobalConstructorOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GlobalConstructorOp globalConstructorOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        if (tsLlvmContext->compileOptions.appendGCtorsToMethod)
        {
            auto loc = globalConstructorOp->getLoc();

            auto priority = globalConstructorOp.getPriority().getLimitedValue();

            auto parentModule = globalConstructorOp->getParentOfType<ModuleOp>();
            //auto mlirGCtors = parentModule.lookupSymbol<func::FuncOp>(MLIR_GCTORS);
            auto mlirGCtors = parentModule.lookupSymbol(MLIR_GCTORS);
            if (!mlirGCtors)
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);

                // create dummy __mlir_gctors for JIT
                rewriter.setInsertionPointToEnd(parentModule.getBody());
                auto llvmFnType = mlir::FunctionType::get(rewriter.getContext(), {}, {});
                auto initFunc = rewriter.create<func::FuncOp>(loc, MLIR_GCTORS, llvmFnType);
                initFunc.setPublic();                
                auto linkage = LLVM::LinkageAttr::get(rewriter.getContext(), LLVM::Linkage::External);
                initFunc->setAttr("llvm.linkage", linkage);
                if (true || tsLlvmContext->compileOptions.isDLL)
                {
                    SmallVector<mlir::Attribute> funcAttrs;
                    funcAttrs.push_back(ATTR("export"));
                    initFunc->setAttr("passthrough", ArrayAttr::get(rewriter.getContext(), funcAttrs));
                }

                auto &entryBlock = *initFunc.addEntryBlock();
                rewriter.setInsertionPointToEnd(&entryBlock);

                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, globalConstructorOp.getGlobalNameAttr(), ValueRange{});
                rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
            }
            else if (auto llvmGCtors = dyn_cast_or_null<LLVM::LLVMFuncOp>(mlirGCtors))
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                if (priority <= 1000)
                {
                    rewriter.setInsertionPointToStart(&llvmGCtors.getBody().front());
                }
                else
                {
                    rewriter.setInsertionPoint(llvmGCtors.getBody().back().getTerminator());
                }

                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, globalConstructorOp.getGlobalNameAttr(), ValueRange{});
            }
            else if (auto tsFuncGCtors = dyn_cast_or_null<mlir_ts::FuncOp>(mlirGCtors))
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                if (priority <= 1000)
                {
                    rewriter.setInsertionPointToStart(&tsFuncGCtors.getBody().front());
                }
                else
                {
                    rewriter.setInsertionPoint(tsFuncGCtors.getBody().back().getTerminator());
                }

                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, globalConstructorOp.getGlobalNameAttr(), ValueRange{});
            }
            else if (auto funcGCtors = dyn_cast_or_null<func::FuncOp>(mlirGCtors))
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                if (priority <= 1000)
                {
                    rewriter.setInsertionPointToStart(&funcGCtors.getBody().front());
                }
                else
                {
                    rewriter.setInsertionPoint(funcGCtors.getBody().back().getTerminator());
                }

                rewriter.create<LLVM::CallOp>(loc, TypeRange{}, globalConstructorOp.getGlobalNameAttr(), ValueRange{});
            }            
            else 
            {
                assert(false);
                return mlir::failure();
            }

            rewriter.eraseOp(globalConstructorOp);
        }
        else
        {
            rewriter.replaceOpWithNewOp<LLVM::GlobalCtorsOp>(
                globalConstructorOp, 
                rewriter.getArrayAttr({ globalConstructorOp.getGlobalNameAttr() }), 
                rewriter.getArrayAttr({ rewriter.getI32IntegerAttr(globalConstructorOp.getPriority().getLimitedValue()) }));
        }

        return success();
    }
};

struct AppendToUsedOpLowering : public TsLlvmPattern<mlir_ts::AppendToUsedOp>
{
    using TsLlvmPattern<mlir_ts::AppendToUsedOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AppendToUsedOp appendToUsedOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = appendToUsedOp->getLoc();

        LLVMCodeHelper lch(appendToUsedOp, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        
        lch.createGlobalVarRegionWithAppendingSymbolRef("llvm.used", appendToUsedOp.getGlobalNameAttr(), "llvm.metadata");

        rewriter.eraseOp(appendToUsedOp);
        return success();
    }
};

struct BodyInternalOpLowering : public TsLlvmPattern<mlir_ts::BodyInternalOp>
{
    using TsLlvmPattern<mlir_ts::BodyInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::BodyInternalOp bodyInternalOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        CodeLogicHelper clh(bodyInternalOp, rewriter);

        auto location = bodyInternalOp.getLoc();

        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        mlir::Block *beforeBody = &bodyInternalOp.getBody().front();
        mlir::Block *afterBody = &bodyInternalOp.getBody().back();
        rewriter.inlineRegionBefore(bodyInternalOp.getBody(), continuationBlock);

        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::BrOp>(location, ValueRange(), beforeBody);

        rewriter.setInsertionPointToEnd(afterBody);
        auto bodyResultInternalOp = cast<mlir_ts::BodyResultInternalOp>(afterBody->getTerminator());
        auto branchOp = rewriter.replaceOpWithNewOp<LLVM::BrOp>(bodyResultInternalOp, bodyResultInternalOp.getResults(),
                                                                continuationBlock);

        rewriter.setInsertionPoint(branchOp);

        rewriter.replaceOp(bodyInternalOp, continuationBlock->getArguments());

        return success();
    }
};

struct BodyResultInternalOpLowering : public TsLlvmPattern<mlir_ts::BodyResultInternalOp>
{
    using TsLlvmPattern<mlir_ts::BodyResultInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::BodyResultInternalOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOp(op, op.getResults());
        return success();
    }
};

struct NoOpLowering : public TsLlvmPattern<mlir_ts::NoOp>
{
    using TsLlvmPattern<mlir_ts::NoOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NoOp noOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.eraseOp(noOp);
        return success();
    }
};

#ifdef ENABLE_TYPED_GC
class GCMakeDescriptorOpLowering : public TsLlvmPattern<mlir_ts::GCMakeDescriptorOp>
{
  public:
    using TsLlvmPattern<mlir_ts::GCMakeDescriptorOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GCMakeDescriptorOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);

        auto i64PtrTy = th.getPtrType();

        auto gcMakeDescriptorFunc = ch.getOrInsertFunction("GC_make_descriptor", th.getFunctionType(rewriter.getI64Type(), {i64PtrTy, rewriter.getI64Type()}));
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, gcMakeDescriptorFunc, ValueRange{transformed.getTypeBitmap(), transformed.getSizeOfBitmapInElements()});

        return success();
    }
};

class GCNewExplicitlyTypedOpLowering : public TsLlvmPattern<mlir_ts::GCNewExplicitlyTypedOp>
{
  public:
    using TsLlvmPattern<mlir_ts::GCNewExplicitlyTypedOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GCNewExplicitlyTypedOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(op, rewriter, getTypeConverter(), tsLlvmContext->compileOptions);
        CodeLogicHelper clh(op, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto llvmIndexType = tch.convertType(th.getIndexType());

        auto loc = op.getLoc();

        mlir::Type storageType = op.getInstance().getType();
        if (auto classType = dyn_cast_or_null<mlir_ts::ClassType>(storageType))
        {
            storageType = classType.getStorageType();
        }
        else if (auto valueRef = dyn_cast<mlir_ts::ValueRefType>(storageType))
        {
            storageType = valueRef.getElementType();
        }

        auto resultType = tch.convertType(op.getType());

        auto sizeOfTypeValueMLIR = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto sizeOfTypeValue = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmIndexType, sizeOfTypeValueMLIR);

        auto i8PtrTy = th.getPtrType();

        auto gcMallocExplicitlyTypedFunc = ch.getOrInsertFunction("GC_malloc_explicitly_typed", th.getFunctionType(i8PtrTy, {rewriter.getI64Type(), rewriter.getI64Type()}));
        auto value = rewriter.create<LLVM::CallOp>(loc, gcMallocExplicitlyTypedFunc, ValueRange{sizeOfTypeValue, transformed.getTypeDescr()});

        rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultType, value.getResult());

        return success();
    }
};

#endif

struct UnrealizedConversionCastOpLowering : public ConvertOpToLLVMPattern<UnrealizedConversionCastOp>
{
    using ConvertOpToLLVMPattern<UnrealizedConversionCastOp>::ConvertOpToLLVMPattern;

    LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override
    {
        SmallVector<mlir::Type> convertedTypes;
        if (succeeded(typeConverter->convertTypes(op.getOutputs().getTypes(), convertedTypes)) &&
            convertedTypes == adaptor.getInputs().getTypes())
        {
            rewriter.replaceOp(op, adaptor.getInputs());
            return success();
        }

        convertedTypes.clear();
        if (succeeded(typeConverter->convertTypes(adaptor.getInputs().getTypes(), convertedTypes)) &&
            convertedTypes == op.getOutputs().getType())
        {
            rewriter.replaceOp(op, adaptor.getInputs());
            return success();
        }
        return failure();
    }
};

// internals
struct AtomicRMWOpLowering : public TsLlvmPattern<mlir_ts::AtomicRMWOp>
{
    using TsLlvmPattern<mlir_ts::AtomicRMWOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AtomicRMWOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<LLVM::AtomicRMWOp>(
            op, 
            (LLVM::AtomicBinOp)transformed.getBinOp(), 
            transformed.getPtr(), 
            transformed.getValue(), 
            (LLVM::AtomicOrdering)transformed.getOrdering(),
            transformed.getSyncscope().value_or(StringRef()),
            transformed.getAlignment().value_or(0),
            transformed.getVolatile_());
        return success();
    }
};

struct AtomicCmpXchgOpLowering : public TsLlvmPattern<mlir_ts::AtomicCmpXchgOp>
{
    using TsLlvmPattern<mlir_ts::AtomicCmpXchgOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AtomicCmpXchgOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<LLVM::AtomicCmpXchgOp>(
            op, 
            transformed.getPtr(), 
            transformed.getCmp(), 
            transformed.getValue(), 
            (LLVM::AtomicOrdering)transformed.getSuccessOrdering(),
            (LLVM::AtomicOrdering)transformed.getFailureOrdering(),
            transformed.getSyncscope().value_or(StringRef()),
            transformed.getAlignment().value_or(0),
            transformed.getWeak(),
            transformed.getVolatile_());
        return success();
    }
};

struct FenceOpLowering : public TsLlvmPattern<mlir_ts::FenceOp>
{
    using TsLlvmPattern<mlir_ts::FenceOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::FenceOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<LLVM::FenceOp>(
            op, 
            (LLVM::AtomicOrdering)transformed.getOrdering(),
            transformed.getSyncscope().value_or(StringRef())
        );
        return success();
    }
};

struct InlineAsmOpLowering : public TsLlvmPattern<mlir_ts::InlineAsmOp>
{
    using TsLlvmPattern<mlir_ts::InlineAsmOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InlineAsmOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        SmallVector<mlir::Type> convertedTypes;
        if (failed(typeConverter->convertTypes(op->getResultTypes(), convertedTypes)))
        {
            return failure();
        }

        rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
            op, 
            TypeRange(convertedTypes), 
            transformed.getOperands(),
            transformed.getAsmString(), 
            transformed.getConstraints(), 
            transformed.getHasSideEffects(), 
            transformed.getIsAlignStack(),
            LLVM::AsmDialectAttr::get(rewriter.getContext(), transformed.getAsmDialect().value_or(0) == 0 
                ? LLVM::AsmDialect::AD_ATT : LLVM::AsmDialect::AD_Intel), 
            transformed.getOperandAttrsAttr());

        return success();
    }
};

struct CallIntrinsicOpLowering : public TsLlvmPattern<mlir_ts::CallIntrinsicOp>
{
    using TsLlvmPattern<mlir_ts::CallIntrinsicOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallIntrinsicOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        SmallVector<mlir::Type> convertedTypes;
        if (failed(typeConverter->convertTypes(op->getResultTypes(), convertedTypes)))
        {
            return failure();
        }

        rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
            op, 
            TypeRange(convertedTypes), 
            transformed.getIntrin(), 
            transformed.getOperands(),
            (LLVM::FastmathFlags)transformed.getFastmathFlags());

        return success();
    }
};

struct LinkerOptionsOpLowering : public TsLlvmPattern<mlir_ts::LinkerOptionsOp>
{
    using TsLlvmPattern<mlir_ts::LinkerOptionsOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LinkerOptionsOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto module = op->getParentOfType<mlir::ModuleOp>();
        auto sp = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LinkerOptionsOp>(op->getLoc(), transformed.getOptions());
        rewriter.restoreInsertionPoint(sp);

        rewriter.eraseOp(op);
        return success();
    }
};

static void populateTypeScriptConversionPatterns(LLVMTypeConverter &converter, mlir::ModuleOp &m,
                                                 mlir::SmallPtrSet<mlir::Type, 32> &usedTypes, CompileOptions& compileOptions)
{
    converter.addConversion(
        [&](mlir_ts::AnyType type) { return LLVM::LLVMPointerType::get(m.getContext()); });

    converter.addConversion(
        [&](mlir_ts::NullType type) { return LLVM::LLVMPointerType::get(m.getContext()); });

    converter.addConversion([&](mlir_ts::OpaqueType type) { return LLVM::LLVMPointerType::get(m.getContext()); });

    converter.addConversion([&](mlir_ts::VoidType type) { return LLVM::LLVMVoidType::get(m.getContext()); });

    converter.addConversion([&](mlir_ts::BooleanType type) {
        TypeHelper th(m.getContext());
        return th.getLLVMBoolType();
    });

    converter.addConversion([&](mlir_ts::TypePredicateType type) {
        TypeHelper th(m.getContext());
        return th.getLLVMBoolType();
    });

    converter.addConversion([&](mlir_ts::CharType type) {
        return mlir::IntegerType::get(m.getContext(), 8 /*, mlir::IntegerType::SignednessSemantics::Unsigned*/);
    });

    converter.addConversion([&](mlir_ts::ByteType type) {
        return mlir::IntegerType::get(m.getContext(), 8 /*, mlir::IntegerType::SignednessSemantics::Unsigned*/);
    });

    converter.addConversion([&](mlir_ts::NumberType type) {
#ifdef NUMBER_F64
        return Float64Type::get(m.getContext());
#else
        return Float32Type::get(m.getContext());
#endif
    });

    converter.addConversion([&](mlir_ts::BigIntType type) {
        return mlir::IntegerType::get(m.getContext(), 64 /*, mlir::IntegerType::SignednessSemantics::Signed*/);
    });

    converter.addConversion([&](mlir_ts::StringType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::EnumType type) { return converter.convertType(type.getElementType()); });

    converter.addConversion([&](mlir_ts::ConstArrayType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::ConstArrayValueType type) {
        return LLVM::LLVMArrayType::get(converter.convertType(type.getElementType()), type.getSize());
    });

    converter.addConversion([&](mlir_ts::ArrayType type) {
        TypeHelper th(m.getContext());

        SmallVector<mlir::Type> rtArrayType;
        // pointer to data type
        rtArrayType.push_back(th.getPtrType());
        // field which store length of array
        rtArrayType.push_back(converter.convertType(th.getIndexType()));

        return LLVM::LLVMStructType::getLiteral(type.getContext(), rtArrayType, false);
    });

    converter.addConversion([&](mlir_ts::RefType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::ValueRefType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::ConstTupleType type) {
        SmallVector<mlir::Type> convertedTypes;
        for (auto subType : type.getFields())
        {
            convertedTypes.push_back(converter.convertType(subType.type));
        }

        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, TUPLE_TYPE_PACKED);
    });

    converter.addConversion([&](mlir_ts::TupleType type) {
        SmallVector<mlir::Type> convertedTypes;
        for (auto subType : type.getFields())
        {
            convertedTypes.push_back(converter.convertType(subType.type));
        }

        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, TUPLE_TYPE_PACKED);
    });

    converter.addConversion([&](mlir_ts::BoundRefType type) {
        SmallVector<mlir::Type> llvmStructType;
        llvmStructType.push_back(LLVM::LLVMPointerType::get(m.getContext()));
        llvmStructType.push_back(LLVM::LLVMPointerType::get(m.getContext()));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
    });

    converter.addConversion([&](mlir_ts::FunctionType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::ConstructFunctionType type) {
        return LLVM::LLVMPointerType::get(m.getContext());    
    });

    converter.addConversion([&](mlir_ts::BoundFunctionType type) {
        SmallVector<mlir::Type> llvmStructType;
        llvmStructType.push_back(LLVM::LLVMPointerType::get(m.getContext()));
        llvmStructType.push_back(LLVM::LLVMPointerType::get(m.getContext()));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
    });

    converter.addConversion([&](mlir_ts::HybridFunctionType type) {
        SmallVector<mlir::Type> llvmStructType;
        llvmStructType.push_back(LLVM::LLVMPointerType::get(m.getContext()));
        llvmStructType.push_back(LLVM::LLVMPointerType::get(m.getContext()));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
    });

    converter.addConversion([&](mlir_ts::ObjectType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::ObjectStorageType type) {
        auto identStruct = LLVM::LLVMStructType::getIdentified(type.getContext(), type.getName().getValue());
        if (!usedTypes.contains(identStruct))
        {
            usedTypes.insert(identStruct);
            SmallVector<mlir::Type> convertedTypes;
            for (auto subType : type.getFields())
            {
                convertedTypes.push_back(converter.convertType(subType.type));
            }

            identStruct.setBody(convertedTypes, false);
        }

        return identStruct;
    });    

    converter.addConversion([&](mlir_ts::UnknownType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::SymbolType type) { 
        return  LLVM::LLVMStructType::getOpaque("Symbol", type.getContext()); 
     });

    converter.addConversion([&](mlir_ts::UndefinedType type) { 
        auto identStruct = LLVM::LLVMStructType::getIdentified(type.getContext(), UNDEFINED_NAME);
        SmallVector<mlir::Type> undefBodyTypes;
        // TODO: review size of int here, should it be 8?
        undefBodyTypes.push_back(mlir::IntegerType::get(m.getContext(), 1));
        identStruct.setBody(undefBodyTypes, false);
        return identStruct;
    });

    converter.addConversion([&](mlir_ts::ClassStorageType type) {
        auto identStruct = LLVM::LLVMStructType::getIdentified(type.getContext(), type.getName().getValue());
        if (!usedTypes.contains(identStruct))
        {
            usedTypes.insert(identStruct);
            SmallVector<mlir::Type> convertedTypes;
            for (auto subType : type.getFields())
            {
                convertedTypes.push_back(converter.convertType(subType.type));
            }

            identStruct.setBody(convertedTypes, false);
        }

        return identStruct;
    });

    converter.addConversion([&](mlir_ts::ClassType type) {
        return LLVM::LLVMPointerType::get(m.getContext());
    });

    converter.addConversion([&](mlir_ts::InterfaceType type) {
        TypeHelper th(m.getContext());

        SmallVector<mlir::Type> rtInterfaceType;
        // vtable
        rtInterfaceType.push_back(th.getPtrType());
        // this
        rtInterfaceType.push_back(th.getPtrType());

        return LLVM::LLVMStructType::getLiteral(type.getContext(), rtInterfaceType, false);
    });

    converter.addConversion([&](mlir_ts::OptionalType type) {
        SmallVector<mlir::Type> convertedTypes;

        TypeHelper th(m.getContext());

        // wrapped type
        convertedTypes.push_back(converter.convertType(type.getElementType()));
        // field which shows undefined state
        convertedTypes.push_back(th.getLLVMBoolType());

        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, false);
    });

    converter.addConversion([&](mlir_ts::UnionType type) {
        TypeHelper th(m.getContext());
        LLVMTypeConverterHelper ltch(&converter);
        MLIRTypeHelper mth(m.getContext(), compileOptions);

        mlir::Type selectedType = ltch.findMaxSizeType(type);
        bool needTag = mth.isUnionTypeNeedsTag(mlir::UnknownLoc::get(type.getContext()), type);

        LLVM_DEBUG(llvm::dbgs() << "\n!! max size type in union: " << selectedType
                                << "\n size: " << ltch.getTypeAllocSizeInBytes(selectedType) << "\n Tag: " << (needTag ? "yes" : "no")
                                << "\n union type: " << type << "\n";);

        SmallVector<mlir::Type> convertedTypes;
        if (needTag)
        {
            convertedTypes.push_back(th.getPtrType());
        }

        convertedTypes.push_back(selectedType);
        if (convertedTypes.size() == 1)
        {
            return convertedTypes.front();
        }

        mlir::Type structType = LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, UNION_TYPE_PACKED);
        return structType;
    });

    converter.addConversion([&](mlir_ts::NeverType type) { return LLVM::LLVMVoidType::get(type.getContext()); });

    converter.addConversion([&](mlir_ts::LiteralType type) { return converter.convertType(type.getElementType()); });

    converter.addConversion([&](mlir_ts::IntersectionType type) {
        llvm_unreachable("type usage (IntersectionType) is not implemented");
        return mlir::Type();
    }); 

    /*
    converter.addSourceMaterialization(
        [&](OpBuilder &builder, mlir::Type resultType, ValueRange inputs, Location loc) -> std::optional<mlir::Value> {
            if (inputs.size() != 1)
                return std::nullopt;

            LLVM_DEBUG(llvm::dbgs() << "\n!! Materialization (Source): " << loc << " result type: " << resultType; for (auto inputType : inputs) llvm::dbgs() << "\n <- input: " << inputType;);

            mlir::Value val = builder.create<mlir_ts::DialectCastOp>(loc, resultType, inputs[0]);
            return val;
            //return inputs[0];
        });
    converter.addTargetMaterialization(
        [&](OpBuilder &builder, mlir::Type resultType, ValueRange inputs, Location loc) -> std::optional<mlir::Value> {
            if (inputs.size() != 1)
                return std::nullopt;

            LLVM_DEBUG(llvm::dbgs() << "\n!! Materialization (Target): " << loc << " result type: " << resultType; for (auto inputType : inputs) llvm::dbgs() << "\n <- input: " << inputType;);

            mlir::Value val = builder.create<mlir_ts::DialectCastOp>(loc, resultType, inputs[0]);
            return val;
            //return inputs[0];
        });
    */
}

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TypeScriptToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace
{

struct TypeScriptToLLVMLoweringPass : public PassWrapper<TypeScriptToLLVMLoweringPass, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeScriptToLLVMLoweringPass)

    TSContext tsContext;    

    TypeScriptToLLVMLoweringPass(CompileOptions &compileOptions) : tsContext(compileOptions)
    {
    }

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<
            LLVM::LLVMDialect, 
            mlir::math::MathDialect, 
            mlir::arith::ArithDialect, 
            mlir::cf::ControlFlowDialect, 
            mlir::func::FuncDialect,
            mlir::index::IndexDialect>();            
    }

    void runOnOperation() final;
};

} // end anonymous namespace

static LogicalResult verifyTerminatorSuccessors(Operation *op)
{
    auto *parent = op->getParentRegion();

    // Verify that the operands lines up with the BB arguments in the successor.
    for (mlir::Block *succ : op->getSuccessors())
        if (succ->getParent() != parent)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! reference to block defined in another region: "; op->dump();
                       llvm::dbgs() << "\n";);
            assert(false);
            return op->emitError("DEBUG TEST: reference to block defined in another region");
        }

    return success();
}

static LogicalResult verifyAlloca(mlir::Block *block)
{
    auto beginAlloca = true;
    auto failed = false;
    auto withingStackSaveRestore = false;
    auto visitorAllOps = [&](Operation *op) {
        if (failed)
        {
            return;
        }

        if (isa<LLVM::ConstantOp>(op))
        {
            // ignore it
            return;
        }

        if (auto allocaOp = dyn_cast<LLVM::AllocaOp>(op))
        {
            if (beginAlloca)
            {
                return;
            }

            // check only alloca with const size
            auto sizeOp = allocaOp.getArraySize().getDefiningOp();
            if (!isa<mlir_ts::ConstantOp>(sizeOp) && !isa<mlir::arith::ConstantOp>(sizeOp))
            {
                return;
            }

            if (!withingStackSaveRestore)
            {
                failed = true;
                LLVM_DEBUG(llvm::dbgs() << "\n!! operator change stack without restoring it: "; op->dump();
                           llvm::dbgs() << "\n";);
                LLVM_DEBUG(llvm::dbgs() << "\n!! in func: \n"; op->getParentOfType<LLVM::LLVMFuncOp>()->dump();
                           llvm::dbgs() << "\n";);
                assert(false);
                op->emitError("DEBUG TEST: operator change stack without restoring it");
                return;
            }
        }

        if (isa<LLVM::StackSaveOp>(op))
        {
            withingStackSaveRestore = true;
        }

        if (isa<LLVM::StackRestoreOp>(op))
        {
            withingStackSaveRestore = false;
        }

        beginAlloca = false;
    };

    block->walk(visitorAllOps);

    if (failed)
    {
        return failure();
    }

    return success();
}

static LogicalResult verifyModule(mlir::ModuleOp &module)
{
    for (auto &block : module.getBodyRegion())
    {
        for (auto &op : block.getOperations())
        {
            if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op))
            {
                for (auto &region : funcOp->getRegions())
                {
                    for (auto &regionBlock : region)
                    {
                        if (failed(verifyTerminatorSuccessors(regionBlock.getTerminator())))
                        {
                            assert(false);
                            return failure();
                        }

                        if (failed(verifyAlloca(&regionBlock)))
                        {
                            assert(false);
                            return failure();
                        }
                    }
                }
            }
        }
    }

    return success();
}

static void selectAllVariablesAndDebugVariables(mlir::ModuleOp &module, SmallPtrSet<Operation *, 16> &workSet)
{
    auto visitorVariablesAndDebugVariablesOp = [&](Operation *op) {
        if (auto variableOp = dyn_cast_or_null<VariableOp>(op))
        {
            if (variableOp->getParentOfType<mlir_ts::GlobalOp>())
            {
                return;
            }

            workSet.insert(variableOp);
        }
        else if (auto debugVariableOp = dyn_cast_or_null<DebugVariableOp>(op))
        {
            workSet.insert(debugVariableOp);
        }        
        else if (auto globalOp = dyn_cast_or_null<GlobalOp>(op))
        {
            workSet.insert(globalOp);
        }        
    };

    module.walk(visitorVariablesAndDebugVariablesOp);
}

static void selectAllFuncOp(mlir::ModuleOp &module, SmallPtrSet<Operation *, 16> &workSet)
{
    auto visitorFuncOp = [&](Operation *op) {
        if (auto funcOp = dyn_cast_or_null<mlir_ts::FuncOp>(op))
        {
            workSet.insert(op);
        }
    };

    module.walk(visitorFuncOp);
}

static LogicalResult preserveTypesForDebugInfo(mlir::ModuleOp &module, LLVMTypeConverter &llvmTypeConverter, CompileOptions& compileOptions)
{
    SmallPtrSet<Operation *, 16> workSet;
    selectAllVariablesAndDebugVariables(module, workSet);

    for (auto op : workSet)
    {
        LLVM_DEBUG(llvm::dbgs() << "\n!! workSet DI: " << *op << "\n");

        auto location = op->getLoc();
        //DIScopeAttr scope, StringAttr name, DIFileAttr file, unsigned line, unsigned arg, unsigned alignInBits, DITypeAttr type
        if (auto scopeFusedLoc = dyn_cast<mlir::FusedLocWith<LLVM::DIScopeAttr>>(location))
        {
            if (auto namedLoc = dyn_cast_or_null<mlir::NameLoc>(scopeFusedLoc.getLocations().front()))
            {
                LocationHelper lh(location.getContext());
                // we don't need TypeConverter here
                LLVMTypeConverterHelper llvmtch(&llvmTypeConverter);
                LLVMDebugInfoHelper di(location.getContext(), llvmtch, compileOptions);

                auto [file, lineAndColumn] = lh.getLineAndColumnAndFile(namedLoc);
                auto [line, column] = lineAndColumn;

                if (!file)
                {
                    continue;
                }

                mlir::Type dataType;
                auto argIndex = 0;
                auto isGlobal = false;
                mlir::StringAttr linkageNameAttr;
                if (auto variableOp = dyn_cast<mlir_ts::VariableOp>(op))
                {
                    dataType = variableOp.getType().getElementType();
                    auto argVal = variableOp.getDiArgNumber();
                    argIndex = argVal.has_value() ? argVal.value().getLimitedValue() : 0;
                }
                else if (auto debugVariableOp = dyn_cast<mlir_ts::DebugVariableOp>(op))
                {
                    dataType = debugVariableOp.getInitializer().getType();
                }
                else if (auto globalOp = dyn_cast<mlir_ts::GlobalOp>(op))
                {
                    dataType = globalOp.getType();
                    linkageNameAttr = globalOp.getSymNameAttr();
                    isGlobal = true;
                }

                // TODO: finish the DI logic
                unsigned alignInBits = llvmTypeConverter.getPointerBitwidth();
                auto diType = di.getDIType(location, mlir::Type(), dataType, file, line, file);

                // MLIRTypeHelper mth(module.getContext());
                // if ((mth.isAnyFunctionType(dataType) || isa<mlir_ts::TupleType>(dataType)) && argIndex > 0) {
                //     diType = di.getDIPointerType(diType, file, line);
                // }

                auto name = namedLoc.getName();
                auto scope = scopeFusedLoc.getMetadata();
                
                LLVM_DEBUG(llvm::dbgs() << "\n!! name: " << name << " scope: " << scope << "\n");

                if (isGlobal)
                {
                    // recreate globalVar later to set correct LinkageAttr and isDefined
                    auto varInfo = LLVM::DIGlobalVariableAttr::get(
                        location.getContext(), scope, name, linkageNameAttr, 
                        file, line, diType, false, true, alignInBits);
                    op->setLoc(mlir::FusedLoc::get(location.getContext(), {location}, varInfo));
                }
                else
                {
                    auto varInfo = LLVM::DILocalVariableAttr::get(
                        location.getContext(), scope, name, file, line, argIndex, alignInBits, diType);
                    op->setLoc(mlir::FusedLoc::get(location.getContext(), {location}, varInfo));
                }
            }
        }
    }    

    return success();
}

static LogicalResult setDISubProgramTypesToFormOp(mlir::ModuleOp &module, LLVMTypeConverter &llvmTypeConverter, CompileOptions& compileOptions)
{
    // fixes for FuncOps, and it should be first
    SmallPtrSet<Operation *, 16> workSetFuncOps;

    selectAllFuncOp(module, workSetFuncOps);

    for (auto op : workSetFuncOps)
    {
        if (auto funcOp = dyn_cast<mlir_ts::FuncOp>(op)) {
            // debug info - adding return type
            auto hasBody = !funcOp.getBody().empty();
            if ((funcOp.getResultTypes().size() > 0 || funcOp.getArgumentTypes().size() > 0) && hasBody)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! function fix: " << funcOp.getName() << "\n");

                LLVMDebugInfoHelperFixer ldif(funcOp, &llvmTypeConverter, compileOptions);
                ldif.fix();
            } else if (!hasBody) {
                // func declaration, we do not need Debug info for it
                funcOp->setLoc(mlir::UnknownLoc::get(funcOp->getContext()));
            }
        }
    }

    return success();
}

static void selectAllUnrealizedConversionCast(mlir::ModuleOp &module, SmallPtrSet<Operation *, 16> &workSet)
{
    auto visitorUnrealizedConversionCast = [&](Operation *op) {
        if (auto unrealizedConversionCastOp = dyn_cast_or_null<UnrealizedConversionCastOp>(op))
        {
            workSet.insert(unrealizedConversionCastOp);
        }
    };

    module.walk(visitorUnrealizedConversionCast);
}

static LogicalResult cleanupUnrealizedConversionCast(mlir::ModuleOp &module)
{
    SmallPtrSet<Operation *, 16> workSet;

    selectAllUnrealizedConversionCast(module, workSet);

    SmallPtrSet<Operation *, 16> removed;

    for (auto op : workSet)
    {
        if (removed.find(op) != removed.end())
        {
            continue;
        }

        auto unrealizedConversionCastOp = cast<UnrealizedConversionCastOp>(op);

        LLVM_DEBUG(llvm::dbgs() << "\nUnrealizedConversionCastOp to analyze: \n" << unrealizedConversionCastOp << "\n";);

        auto hasAnyUse = false;
        for (auto user : unrealizedConversionCastOp.getResult(0).getUsers())
        {
            hasAnyUse = true;
            auto nextUnrealizedConversionCastOp = dyn_cast_or_null<UnrealizedConversionCastOp>(user);
            if (nextUnrealizedConversionCastOp)
            {
                LLVM_DEBUG(llvm::dbgs() 
                    << "\n -> Next UnrealizedConversionCastOp: \n" 
                    << nextUnrealizedConversionCastOp 
                    << " <- result type: " 
                    << nextUnrealizedConversionCastOp.getResult(0).getType() 
                    << " -> input: " << unrealizedConversionCastOp.getOperand(0).getType() 
                    << "\n";);

                if (nextUnrealizedConversionCastOp.getResult(0).getType() == unrealizedConversionCastOp.getOperand(0).getType())
                {
                    // remove both
                    nextUnrealizedConversionCastOp->getResult(0).replaceAllUsesWith(unrealizedConversionCastOp.getOperand(0));
                    
                    removed.insert(nextUnrealizedConversionCastOp);
                    if (removed.find(unrealizedConversionCastOp) == removed.end())
                    {
                        removed.insert(unrealizedConversionCastOp);
                    }                    
                }
            }
        }

        if (!hasAnyUse)
        {
            removed.insert(unrealizedConversionCastOp);
        }
    }    

    for (auto removedOp : removed)
    {
        removedOp->erase();
    }

    return success();
}

void TypeScriptToLLVMLoweringPass::runOnOperation()
{
    auto m = getOperation();

    // The first thing to define is the conversion target. This will define the
    // final target for this lowering. For this lowering, we are only targeting
    // the LLVM dialect.
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    // During this lowering, we will also be lowering the MemRef types, that are
    // currently being operated on, to a representation in LLVM. To perform this
    // conversion we use a TypeConverter as part of the lowering. This converter
    // details how one type maps to another. This is necessary now that we will be
    // doing more complicated lowerings, involving loop region arguments.
    mlir::DataLayout dl(m);
    LowerToLLVMOptions options(&getContext(), dl);
    if (tsContext.compileOptions.isWasm && tsContext.compileOptions.sizeBits == 32)
    {
        options.dataLayout.reset("e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20");

        m->setAttr(
            mlir::LLVM::LLVMDialect::getDataLayoutAttrName(), 
            mlir::StringAttr::get(&getContext(), "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-f128:64-n32:64-S128-ni:1:10:20"));        
    }

    options.allocLowering = LowerToLLVMOptions::AllocLowering::AlignedAlloc;
    DataLayoutAnalysis analysis(m);
    LLVMTypeConverter typeConverter(&getContext(), options, &analysis);

    // Now that the conversion target has been defined, we need to provide the
    // patterns used for lowering. At this point of the compilation process, we
    // have a combination of `typescript`, `affine`, and `std` operations. Luckily, there
    // are already exists a set of patterns to transform `affine` and `std`
    // dialects. These patterns lowering in multiple stages, relying on transitive
    // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
    // patterns must be applied to fully transform an illegal operation into a
    // set of legal ones.
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

#ifdef ENABLE_ASYNC
    populateAsyncStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
#endif

    // The only remaining operation to lower from the `typescript` dialect, is the PrintOp.
    TsLlvmContext tsLlvmContext{tsContext.compileOptions};
    patterns.insert<
        AddressOfOpLowering, ArithmeticUnaryOpLowering, ArithmeticBinaryOpLowering,
        AssertOpLowering, CastOpLowering, ConstantOpLowering, DefaultOpLowering, 
        OptionalOpLowering, ValueOptionalOpLowering, UndefOptionalOpLowering, HasValueOpLowering, ValueOpLowering, 
        ValueOrDefaultOpLowering, SymbolRefOpLowering, GlobalOpLowering, GlobalResultOpLowering,
        FuncOpLowering, LoadOpLowering, ElementRefOpLowering, PropertyRefOpLowering, ExtractPropertyOpLowering,
        PointerOffsetRefOpLowering, LogicalBinaryOpLowering, NullOpLowering, NewOpLowering, CreateTupleOpLowering,
        DeconstructTupleOpLowering, CreateArrayOpLowering, NewEmptyArrayOpLowering, NewArrayOpLowering, ArrayPushOpLowering,
        ArrayPopOpLowering, ArrayUnshiftOpLowering, ArrayShiftOpLowering, ArraySpliceOpLowering, ArrayViewOpLowering, DeleteOpLowering, 
        ParseFloatOpLowering, ParseIntOpLowering, IsNaNOpLowering, PrintOpLowering, ConvertFOpLowering, StoreOpLowering, SizeOfOpLowering, 
        InsertPropertyOpLowering, LengthOfOpLowering, SetLengthOfOpLowering, StringLengthOpLowering, SetStringLengthOpLowering, StringConcatOpLowering, 
        StringCompareOpLowering, AnyCompareOpLowering, CharToStringOpLowering, UndefOpLowering, CopyStructOpLowering, MemoryCopyOpLowering, MemoryMoveOpLowering, 
        LoadSaveValueLowering, ThrowUnwindOpLowering, ThrowCallOpLowering, VariableOpLowering, DebugVariableOpLowering, AllocaOpLowering, InvokeOpLowering, 
        InvokeHybridOpLowering, VirtualSymbolRefOpLowering, ThisVirtualSymbolRefOpLowering, InterfaceSymbolRefOpLowering, 
        NewInterfaceOpLowering, VTableOffsetRefOpLowering, LoadBoundRefOpLowering, StoreBoundRefOpLowering, CreateBoundRefOpLowering, 
        CreateBoundFunctionOpLowering, GetThisOpLowering, GetMethodOpLowering, TypeOfOpLowering, TypeOfAnyOpLowering, DebuggerOpLowering,
        UnreachableOpLowering, SymbolCallInternalOpLowering, CallInternalOpLowering, CallHybridInternalOpLowering, 
        ReturnInternalOpLowering, NoOpLowering, GlobalConstructorOpLowering, AppendToUsedOpLowering, ExtractInterfaceThisOpLowering, 
        ExtractInterfaceVTableOpLowering, BoxOpLowering, UnboxOpLowering, DialectCastOpLowering, CreateUnionInstanceOpLowering,
        GetValueFromUnionOpLowering, GetTypeInfoFromUnionOpLowering, BodyInternalOpLowering, BodyResultInternalOpLowering
#ifndef DISABLE_SWITCH_STATE_PASS
        ,
        SwitchStateOpLowering, StateLabelOpLowering, YieldReturnValOpLowering
#endif
        ,
        SwitchStateInternalOpLowering, LoadLibraryPermanentlyOpLowering, SearchForAddressOfSymbolOpLowering,
        AtomicRMWOpLowering, AtomicCmpXchgOpLowering, FenceOpLowering, InlineAsmOpLowering, CallIntrinsicOpLowering, LinkerOptionsOpLowering>(
            typeConverter, &getContext(), &tsLlvmContext);

    if (tsLlvmContext.compileOptions.isWindows)
    {
        using namespace windows;

        patterns.insert<
            LandingPadOpLowering, CompareCatchTypeOpLowering, BeginCatchOpLowering,
            SaveCatchVarOpLowering, EndCatchOpLowering, BeginCleanupOpLowering, EndCleanupOpLowering>(
                typeConverter, &getContext(), &tsLlvmContext);        
    }
    else
    {
        using namespace linux;

        patterns.insert<
            LandingPadOpLowering, CompareCatchTypeOpLowering, BeginCatchOpLowering,
            SaveCatchVarOpLowering, EndCatchOpLowering, BeginCleanupOpLowering, EndCleanupOpLowering>(
                typeConverter, &getContext(), &tsLlvmContext);        
    }

#ifdef ENABLE_TYPED_GC
    patterns.insert<
        GCMakeDescriptorOpLowering, GCNewExplicitlyTypedOpLowering>(typeConverter, &getContext(), &tsLlvmContext);
#endif        

    mlir::SmallPtrSet<mlir::Type, 32> usedTypes;
    populateTypeScriptConversionPatterns(typeConverter, m, usedTypes, tsContext.compileOptions);

    // in processing ops types will be changed by LLVM versions overtime, we need to have actual information about types 
    // when generate Debug Info
    if (tsLlvmContext.compileOptions.generateDebugInfo)
    {
        setDISubProgramTypesToFormOp(m, typeConverter, tsContext.compileOptions);
        preserveTypesForDebugInfo(m, typeConverter, tsContext.compileOptions);
    }

    LLVM_DEBUG(llvm::dbgs() << "\n!! BEFORE DUMP: \n" << m << "\n";);

    if (failed(applyFullConversion(m, target, std::move(patterns))))
    {
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "\n!! AFTER DUMP - BEFORE CLEANUP: \n" << m << "\n";);

    cleanupUnrealizedConversionCast(m);

    LLVM_DEBUG(llvm::dbgs() << "\n!! AFTER DUMP: \n" << m << "\n";);

    LLVM_DEBUG(verifyModule(m););
}

/// Create a pass for lowering operations the remaining `TypeScript` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::typescript::createLowerToLLVMPass(CompileOptions &compileOptions)
{
    return std::make_unique<TypeScriptToLLVMLoweringPass>(compileOptions);
}
