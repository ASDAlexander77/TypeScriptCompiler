#define DEBUG_TYPE "llvm"

#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"

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

#include "TypeScript/LowerToLLVMLogic.h"

#include "scanner_enums.h"

#define DISABLE_SWITCH_STATE_PASS 1
#define ENABLE_MLIR_INIT

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
    TsLlvmContext() = default;
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

#ifdef PRINTF_SUPPORT
class PrintOpLowering : public TsLlvmPattern<mlir_ts::PrintOp>
{
  public:
    using TsLlvmPattern<mlir_ts::PrintOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PrintOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        // Get a symbol reference to the printf function, inserting it if necessary.
        auto printfFuncOp =
            ch.getOrInsertFunction("printf", th.getFunctionType(rewriter.getI32Type(), th.getI8PtrType(), true));

        std::stringstream format;
        auto count = 0;

        std::function<void(mlir::Type)> processFormatForType = [&](mlir::Type type) {
            auto llvmType = tch.convertType(type);

            if (auto s = type.dyn_cast_or_null<mlir_ts::StringType>())
            {
                format << "%s";
            }
            else if (auto c = type.dyn_cast_or_null<mlir_ts::CharType>())
            {
                format << "%c";
            }
            else if (llvmType.isIntOrIndexOrFloat() && !llvmType.isIntOrIndex())
            {
                format << "%f";
            }
            else if (auto o = type.dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                format << "%s:";
                processFormatForType(o.getElementType());
            }
            else if (llvmType.isIntOrIndex())
            {
                if (llvmType.isInteger(1))
                {
                    format << "%s";
                }
                else
                {
                    format << "%d";
                }
            }
            else
            {
                format << "%d";
            }
        };

        for (auto item : op.inputs())
        {
            auto type = item.getType();

            if (count++ > 0)
            {
                format << " ";
            }

            processFormatForType(type);
        }

        format << "\n";

        auto opHash = std::hash<std::string>{}(format.str());

        std::stringstream formatVarName;
        formatVarName << "frmt_" << opHash;

        auto formatSpecifierCst = ch.getOrCreateGlobalString(formatVarName.str(), format.str());

        auto i8PtrTy = th.getI8PtrType();

        mlir::SmallVector<mlir::Value, 4> values;
        values.push_back(formatSpecifierCst);

        std::function<void(mlir::Type, mlir::Value)> fval;
        fval = [&](mlir::Type type, mlir::Value item) {
            auto llvmType = tch.convertType(type);

            if (llvmType.isIntOrIndexOrFloat() && !llvmType.isIntOrIndex())
            {
                values.push_back(rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), item));
            }
            else if (llvmType.isInteger(1))
            {
                values.push_back(rewriter.create<LLVM::SelectOp>(
                    item.getLoc(), item, ch.getOrCreateGlobalString("__true__", std::string("true")),
                    ch.getOrCreateGlobalString("__false__", std::string("false"))));
            }
            else if (auto o = type.dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                auto boolPart = rewriter.create<mlir_ts::HasValueOp>(item.getLoc(), th.getBooleanType(), item);
                values.push_back(rewriter.create<LLVM::SelectOp>(
                    item.getLoc(), boolPart, ch.getOrCreateGlobalString("__true__", std::string("true")),
                    ch.getOrCreateGlobalString("__false__", std::string("false"))));
                auto optVal = rewriter.create<mlir_ts::ValueOp>(item.getLoc(), o.getElementType(), item);
                fval(optVal.getType(), optVal);
            }
            else
            {
                values.push_back(item);
            }
        };

        for (auto item : transformed.inputs())
        {
            fval(item.getType(), item);
        }

        // print new line
        rewriter.create<LLVM::CallOp>(loc, printfFuncOp, values);

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(op);

        return success();
    }
};
#else

class AssertOpLowering : public TsLlvmPattern<mlir_ts::AssertOp>
{
  public:
    using TsLlvmPattern<mlir_ts::AssertOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AssertOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeConverterHelper tch(getTypeConverter());
        AssertLogic al(op, rewriter, tch, op->getLoc());
        return al.logic(transformed.arg(), op.msg().str());
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
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        CastLogicHelper castLogic(op, rewriter, tch);

        auto loc = op->getLoc();

        auto i8PtrType = th.getI8PtrType();

        // Get a symbol reference to the printf function, inserting it if necessary.
        auto putsFuncOp = ch.getOrInsertFunction("puts", th.getFunctionType(rewriter.getI32Type(), i8PtrType, false));

        auto strType = mlir_ts::StringType::get(rewriter.getContext());

        SmallVector<mlir::Value> values;
        mlir::Value spaceString;
        for (auto item : transformed.inputs())
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

            rewriter.create<LLVM::CallOp>(loc, putsFuncOp, valueAsLLVMType);

            rewriter.create<LLVM::StackRestoreOp>(loc, stack);
        }
        else
        {
            rewriter.create<LLVM::CallOp>(loc, putsFuncOp, values.front());
        }

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(op);

        return success();
    }
};
#endif

class ParseIntOpLowering : public TsLlvmPattern<mlir_ts::ParseIntOp>
{
  public:
    using TsLlvmPattern<mlir_ts::ParseIntOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParseIntOp op, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        // Insert the `atoi` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        LLVM::LLVMFuncOp parseIntFuncOp;
        if (transformed.base())
        {
            parseIntFuncOp = ch.getOrInsertFunction(
                "strtol",
                th.getFunctionType(rewriter.getI32Type(), {i8PtrTy, th.getI8PtrPtrType(), rewriter.getI32Type()}));
            auto nullOp = rewriter.create<LLVM::NullOp>(op->getLoc(), th.getI8PtrPtrType());
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, parseIntFuncOp,
                                                      ValueRange{transformed.arg(), nullOp, transformed.base()});
        }
        else
        {
            parseIntFuncOp = ch.getOrInsertFunction("atoi", th.getFunctionType(rewriter.getI32Type(), {i8PtrTy}));
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, parseIntFuncOp, ValueRange{transformed.arg()});
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
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto loc = op->getLoc();

        // Insert the `atof` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        auto parseFloatFuncOp = ch.getOrInsertFunction("atof", th.getFunctionType(rewriter.getF64Type(), {i8PtrTy}));

#ifdef NUMBER_F64
        auto funcCall = rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, parseFloatFuncOp, ValueRange{transformed.arg()});
#else
        auto funcCall = rewriter.create<LLVM::CallOp>(loc, parseFloatFuncOp, ValueRange{transformed.arg()});
        rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, rewriter.getF32Type(), funcCall.getResult(0));
#endif

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
                                                      transformed.arg(), transformed.arg());
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

        auto storageType = op.type();

        auto stripPtr = false;
        mlir::TypeSwitch<mlir::Type>(storageType)
            .Case<mlir_ts::ClassType>([&](auto classType) { stripPtr = true; })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { stripPtr = true; })
            .Default([&](auto type) { });

        mlir::Type llvmStorageType = tch.convertType(storageType);
        mlir::Type llvmStorageTypePtr = LLVM::LLVMPointerType::get(llvmStorageType);
        if (stripPtr)
        {
            llvmStorageTypePtr = llvmStorageType;
        }

        auto nullPtrToTypeValue = rewriter.create<LLVM::NullOp>(loc, llvmStorageTypePtr);

        LLVM_DEBUG(llvm::dbgs() << "\n!! size of - storage type: [" << storageType << "] llvm storage type: ["
                                << llvmStorageType << "] llvm ptr: [" << llvmStorageTypePtr << "]\n";);

        auto cst1 = rewriter.create<LLVM::ConstantOp>(loc, th.getI64Type(), th.getIndexAttrValue(1));
        auto sizeOfSetAddr =
            rewriter.create<LLVM::GEPOp>(loc, llvmStorageTypePtr, nullPtrToTypeValue, ArrayRef<mlir::Value>({cst1}));

        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, th.getIndexType(), sizeOfSetAddr);

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

        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(op, th.getI32Type(), transformed.op(),
                                                                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

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
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto loc = op->getLoc();
        auto i8PtrTy = th.getI8PtrType();

        auto strlenFuncOp = ch.getOrInsertFunction("strlen", th.getFunctionType(th.getI64Type(), {i8PtrTy}));

        // calc size
        auto size = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, ValueRange{transformed.op()});
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, th.getI32Type(), size.getResult(0));

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
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto loc = op->getLoc();

        // TODO implement str concat
        auto i8PtrTy = th.getI8PtrType();
        auto i8PtrPtrTy = th.getI8PtrPtrType();

        auto strlenFuncOp = ch.getOrInsertFunction("strlen", th.getFunctionType(rewriter.getI64Type(), {i8PtrTy}));
        auto strcpyFuncOp = ch.getOrInsertFunction("strcpy", th.getFunctionType(i8PtrTy, {i8PtrTy, i8PtrTy}));
        auto strcatFuncOp = ch.getOrInsertFunction("strcat", th.getFunctionType(i8PtrTy, {i8PtrTy, i8PtrTy}));

        mlir::Value size = clh.createI64ConstantOf(1);
        // calc size
        for (auto oper : transformed.ops())
        {
            auto size1 = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, oper);
            size = rewriter.create<LLVM::AddOp>(loc, rewriter.getI64Type(), ValueRange{size, size1.getResult(0)});
        }

        auto allocInStack = op.allocInStack().hasValue() && op.allocInStack().getValue();

        mlir::Value newStringValue = allocInStack ? rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, size, true)
                                                  : ch.MemoryAllocBitcast(i8PtrTy, size);

        // copy
        auto concat = false;
        auto result = newStringValue;
        for (auto oper : transformed.ops())
        {
            if (concat)
            {
                auto callResult = rewriter.create<LLVM::CallOp>(loc, strcatFuncOp, ValueRange{result, oper});
                result = callResult.getResult(0);
            }
            else
            {
                auto callResult = rewriter.create<LLVM::CallOp>(loc, strcpyFuncOp, ValueRange{result, oper});
                result = callResult.getResult(0);
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
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());
        LLVMTypeConverterHelper llvmtch(*(LLVMTypeConverter *)getTypeConverter());

        auto loc = op->getLoc();

        auto i8PtrTy = th.getI8PtrType();

        // compare bodies
        auto strcmpFuncOp = ch.getOrInsertFunction("strcmp", th.getFunctionType(th.getI32Type(), {i8PtrTy, i8PtrTy}));

        // compare ptrs first
        auto intPtrType = llvmtch.getIntPtrType(0);
        auto const0 = clh.createIConstantOf(llvmtch.getPointerBitwidth(0), 0);
        auto leftPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, transformed.op1());
        auto rightPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, transformed.op2());
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
                    rewriter.create<LLVM::CallOp>(loc, strcmpFuncOp, ValueRange{transformed.op1(), transformed.op2()});

                // else compare body
                mlir::Value bodyCmpResult;
                switch ((SyntaxKind)op.code())
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    bodyCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    bodyCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::GreaterThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt,
                                                                  compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge,
                                                                  compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::LessThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt,
                                                                  compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle,
                                                                  compareResult.getResult(0), const0);
                    break;
                default:
                    llvm_unreachable("not implemented");
                }

                return bodyCmpResult;
            },
            [&](OpBuilder &builder, Location loc) {
                // else compare body
                mlir::Value ptrCmpResult;
                switch ((SyntaxKind)op.code())
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
                }

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
        LLVMCodeHelper ch(op, rewriter, typeConverter);

        auto loc = op->getLoc();

        auto charType = mlir_ts::CharType::get(rewriter.getContext());
        auto charRefType = mlir_ts::RefType::get(charType);
        auto i8PtrTy = th.getI8PtrType();

        auto bufferSizeValue = clh.createI64ConstantOf(2);
        // TODO: review it
        auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);

        auto index0Value = clh.createI32ConstantOf(0);
        auto index1Value = clh.createI32ConstantOf(1);
        auto nullCharValue = clh.createI8ConstantOf(0);
        auto addr0 = ch.GetAddressOfArrayElement(charRefType, newStringValue.getType(), newStringValue, index0Value);
        rewriter.create<LLVM::StoreOp>(loc, transformed.op(), addr0);
        auto addr1 = ch.GetAddressOfArrayElement(charRefType, newStringValue.getType(), newStringValue, index1Value);
        rewriter.create<LLVM::StoreOp>(loc, nullCharValue, addr1);

        rewriter.replaceOp(op, ValueRange{newStringValue});

        return success();
    }
};

struct ConstantOpLowering : public TsLlvmPattern<mlir_ts::ConstantOp>
{
    using TsLlvmPattern<mlir_ts::ConstantOp>::TsLlvmPattern;

    template <typename T, typename TOp>
    void getOrCreateGlobalArray(TOp constantOp, T type, ConversionPatternRewriter &rewriter) const
    {
        LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto elementType = type.template cast<T>().getElementType();
        auto llvmElementType = tch.convertType(elementType);
        auto arrayAttr = constantOp.value().template dyn_cast_or_null<ArrayAttr>();

        auto arrayFirstElementAddrCst =
            ch.getOrCreateGlobalArray(elementType, llvmElementType, arrayAttr.size(), arrayAttr);

        rewriter.replaceOp(constantOp, arrayFirstElementAddrCst);
    }

    template <typename TOp>
    void getOrCreateGlobalTuple(TOp constantOp, mlir::Type type, ConversionPatternRewriter &rewriter) const
    {
        auto location = constantOp->getLoc();

        LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto arrayAttr = constantOp.value().template dyn_cast_or_null<ArrayAttr>();

        auto convertedTupleType = tch.convertType(type);
        /*
        auto tupleConstPtr = ch.getOrCreateGlobalTuple(type.template cast<mlir_ts::ConstTupleType>(),
                                                       convertedTupleType.template cast<LLVM::LLVMStructType>(),
        arrayAttr);

        // optimize it and replace it with copy memory. (use canon. pass) check  "EraseRedundantAssertions"
        auto loadedValue = rewriter.create<LLVM::LoadOp>(constantOp->getLoc(), tupleConstPtr);
        */

        auto tupleVal = ch.getTupleFromArrayAttr(location, type.dyn_cast_or_null<mlir_ts::ConstTupleType>(),
                                                 convertedTupleType.cast<LLVM::LLVMStructType>(), arrayAttr);

        // rewriter.replaceOp(constantOp, ValueRange{loadedValue});
        rewriter.replaceOp(constantOp, ValueRange{tupleVal});
    }

    LogicalResult matchAndRewrite(mlir_ts::ConstantOp constantOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        // load address of const string
        auto type = constantOp.getType();
        if (auto literalType = type.dyn_cast<mlir_ts::LiteralType>())
        {
            type = literalType.getElementType();
        }

        if (type.isa<mlir_ts::StringType>())
        {
            LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter());

            auto strValue = constantOp.value().cast<StringAttr>().getValue().str();
            auto txtCst = ch.getOrCreateGlobalString(strValue);

            rewriter.replaceOp(constantOp, txtCst);

            return success();
        }

        TypeConverterHelper tch(getTypeConverter());
        if (auto constArrayType = type.dyn_cast<mlir_ts::ConstArrayType>())
        {
            getOrCreateGlobalArray(constantOp, constArrayType, rewriter);
            return success();
        }

        if (auto arrayType = type.dyn_cast<mlir_ts::ArrayType>())
        {
            getOrCreateGlobalArray(constantOp, arrayType, rewriter);
            return success();
        }

        if (auto constTupleType = type.dyn_cast<mlir_ts::ConstTupleType>())
        {
            getOrCreateGlobalTuple(constantOp, constTupleType, rewriter);
            return success();
        }

        if (auto tupleType = type.dyn_cast<mlir_ts::TupleType>())
        {
            getOrCreateGlobalTuple(constantOp, tupleType, rewriter);
            return success();
        }

        if (auto enumType = type.dyn_cast<mlir_ts::EnumType>())
        {
            rewriter.eraseOp(constantOp);
            return success();
        }

        if (auto valAttr = constantOp.value().dyn_cast<mlir::FlatSymbolRefAttr>())
        {
            rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(constantOp, tch.convertType(type), valAttr);
            return success();
        }

        rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(constantOp, tch.convertType(type), constantOp.value());
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
                                                      symbolRefOp.identifierAttr());
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
        rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, tch.convertType(op.getType()));
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
        

        if (op.getType().isa<mlir_ts::OptionalType>())
        {
            rewriter.replaceOpWithNewOp<mlir_ts::UndefOptionalOp>(op, op.getType());
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
        

        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, ValueRange{transformed.operands()});
        return success();
    }
};

struct FuncOpLowering : public TsLlvmPattern<mlir_ts::FuncOp>
{
    using TsLlvmPattern<mlir_ts::FuncOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::FuncOp funcOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto location = funcOp->getLoc();

        auto &typeConverter = *getTypeConverter();
        auto fnType = funcOp.getFunctionType();

        TypeConverter::SignatureConversion signatureInputsConverter(fnType.getNumInputs());
        for (auto argType : enumerate(funcOp.getFunctionType().getInputs()))
        {
            auto convertedType = typeConverter.convertType(argType.value());
            signatureInputsConverter.addInputs(argType.index(), convertedType);
        }

        TypeConverter::SignatureConversion signatureResultsConverter(fnType.getNumResults());
        for (auto argType : enumerate(funcOp.getFunctionType().getResults()))
        {
            auto convertedType = typeConverter.convertType(argType.value());
            signatureResultsConverter.addInputs(argType.index(), convertedType);
        }

        SmallVector<DictionaryAttr> argDictAttrs;
        if (ArrayAttr argAttrs = funcOp.getAllArgAttrs())
        {
            auto argAttrRange = argAttrs.template getAsRange<DictionaryAttr>();
            argDictAttrs.append(argAttrRange.begin(), argAttrRange.end());
        }

        std::string name;
        auto newFuncOp =
            rewriter.create<mlir::func::FuncOp>(funcOp.getLoc(), funcOp.getName(),
                                          rewriter.getFunctionType(signatureInputsConverter.getConvertedTypes(),
                                                                   signatureResultsConverter.getConvertedTypes()),
                                          ArrayRef<NamedAttribute>{}, argDictAttrs);
        for (const auto &namedAttr : funcOp->getAttrs())
        {
            if (namedAttr.getName() == function_interface_impl::getTypeAttrName())
            {
                continue;
            }

            if (namedAttr.getName() == SymbolTable::getSymbolAttrName())
            {
                name = namedAttr.getValue().dyn_cast_or_null<mlir::StringAttr>().getValue().str();
                continue;
            }

            newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
        }

        SmallVector<mlir::Attribute> funcAttrs;

        if (funcOp.personality().hasValue() && funcOp.personality().getValue())
        {
#if WIN_EXCEPTION
            LLVMRTTIHelperVCWin32 rttih(funcOp, rewriter, typeConverter);
#else
            LLVMRTTIHelperVCLinux rttih(funcOp, rewriter, typeConverter);
#endif
            rttih.setPersonality(newFuncOp);

            funcAttrs.push_back(ATTR("noinline"));
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
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter, &signatureInputsConverter)))
        {
            return failure();
        }

        rewriter.eraseOp(funcOp);

        return success();
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
        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            if (type.isa<mlir_ts::VoidType>())
            {
                continue;
            }

            llvmTypes.push_back(tch.convertType(type));
        }

        auto callRes = rewriter.create<LLVM::CallOp>(
            loc, llvmTypes, ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), op.callee()), transformed.operands());
        
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
            if (type.isa<mlir_ts::VoidType>())
            {
                continue;
            }

            llvmTypes.push_back(tch.convertType(type));
        }

        // special case for HybridFunctionType
        LLVM_DEBUG(llvm::dbgs() << "\n!! CallInternalOp - arg #0:" << op.getOperand(0) << "\n");
        if (auto hybridFuncType = op.getOperand(0).getType().dyn_cast<mlir_ts::HybridFunctionType>())
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

        auto hybridFuncType = op.callee().getType().cast<mlir_ts::HybridFunctionType>();

        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            if (type.isa<mlir_ts::VoidType>())
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
                if (resultType.isa<mlir_ts::VoidType>())
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

        if (!op.callee().hasValue())
        {
            // special case for HybridFunctionType
            LLVM_DEBUG(llvm::dbgs() << "\n!! InvokeOp - arg #0:" << op.getOperand(0) << "\n");
            if (auto hybridFuncType = op.getOperand(0).getType().dyn_cast<mlir_ts::HybridFunctionType>())
            {
                rewriter.replaceOpWithNewOp<mlir_ts::InvokeHybridOp>(
                    op, hybridFuncType.getResults(), op.getOperand(0),
                    OperandRange(op.getOperands().begin() + 1, op.getOperands().end()), op.normalDestOperands(),
                    op.unwindDestOperands(), op.normalDest(), op.unwindDest());
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
        rewriter.replaceOpWithNewOp<LLVM::InvokeOp>(op, llvmTypes, op.calleeAttr(), transformed.operands(),
                                                    op.normalDest(), transformed.normalDestOperands(), op.unwindDest(),
                                                    transformed.unwindDestOperands());
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

        auto hybridFuncType = op.callee().getType().cast<mlir_ts::HybridFunctionType>();

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
                if (resultType.isa<mlir_ts::VoidType>())
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

                    auto callRes = rewriter.create<LLVM::InvokeOp>(loc, llvmTypes, ops, continuationBlock,
                                                                   transformed.normalDestOperands(), op.unwindDest(),
                                                                   transformed.unwindDestOperands());

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

                    auto callRes = rewriter.create<LLVM::InvokeOp>(loc, llvmTypes, ops, continuationBlock,
                                                                   transformed.normalDestOperands(), op.unwindDest(),
                                                                   transformed.unwindDestOperands());

                    rewriter.setInsertionPointToStart(continuationBlock);

                    return callRes.getResults();
                });

            rewriter.create<LLVM::BrOp>(loc, ValueRange{}, op.normalDest());
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

        auto in = transformed.in();
        auto resType = op.res().getType();

        CastLogicHelper castLogic(op, rewriter, tch);
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

        

        // TODO: review usage of CastOp from LLVM level
        if (op.getType().isa<mlir_ts::AnyType>())
        {
            // TODO: boxing, finish it, need to send TypeOf
            TypeOfOpHelper toh(rewriter);
            auto typeOfValue = toh.typeOfLogic(loc, op.in().getType());
            // auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc,
            // mlir_ts::StringType::get(rewriter.getContext()), in);
            auto boxedValue = rewriter.create<mlir_ts::BoxOp>(loc, mlir_ts::AnyType::get(rewriter.getContext()),
                                                              transformed.in(), typeOfValue);
            rewriter.replaceOp(op, ValueRange{boxedValue});
            return success();
        }

        if (op.in().getType().isa<mlir_ts::AnyType>())
        {
            auto unboxedValue = rewriter.create<mlir_ts::UnboxOp>(loc, op.getType(), transformed.in());
            rewriter.replaceOp(op, ValueRange{unboxedValue});
            return success();
        }
        // end of hack

        TypeConverterHelper tch(getTypeConverter());

        auto in = transformed.in();
        auto resType = op.res().getType();

        CastLogicHelper castLogic(op, rewriter, tch);
        auto result = castLogic.cast(in, op.in().getType(), resType);
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

        auto in = transformed.in();

        AnyLogic al(op, rewriter, tch, loc);
        auto result = al.castToAny(in, transformed.typeInfo(), in.getType());

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

        auto in = transformed.in();
        auto resType = op.res().getType();

        AnyLogic al(op, rewriter, tch, loc);
        auto result = al.castFromAny(in, tch.convertType(resType));

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
        MLIRTypeHelper mth(rewriter.getContext());

        CastLogicHelper castLogic(op, rewriter, tch);

        auto in = transformed.in();

        auto i8PtrTy = th.getI8PtrType();
        auto valueType = transformed.in().getType();
        auto resType = tch.convertType(op.getType());

        mlir::SmallVector<mlir::Type> types;
        types.push_back(i8PtrTy);
        types.push_back(valueType);
        auto unionPartialType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types, false);
        if (!mth.isUnionTypeNeedsTag(op.getType().cast<mlir_ts::UnionType>()))
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
                rewriter.create<LLVM::InsertValueOp>(loc, udefVal, transformed.typeInfo(), clh.getStructIndexAttr(0));
            auto val1 = rewriter.create<LLVM::InsertValueOp>(loc, val0, in, clh.getStructIndexAttr(1));

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
        MLIRTypeHelper mth(rewriter.getContext());

        bool needTag = mth.isUnionTypeNeedsTag(op.in().getType().cast<mlir_ts::UnionType>());
        if (needTag)
        {
            auto in = transformed.in();

            auto i8PtrTy = th.getI8PtrType();
            auto valueType = tch.convertType(op.getType());

            mlir::SmallVector<mlir::Type> types;
            types.push_back(i8PtrTy);
            types.push_back(valueType);
            auto unionPartialType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types, false);

            CastLogicHelper castLogic(op, rewriter, tch);
            auto casted = castLogic.castLLVMTypes(transformed.in(), transformed.in().getType(), unionPartialType,
                                                  unionPartialType);
            if (!casted)
            {
                return mlir::failure();
            }

            auto val0 = rewriter.create<LLVM::ExtractValueOp>(loc, valueType, casted, clh.getStructIndexAttr(1));

            rewriter.replaceOp(op, ValueRange{val0});
        }
        else
        {
            rewriter.replaceOp(op, ValueRange{transformed.in()});
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
        MLIRTypeHelper mth(rewriter.getContext());

        auto loc = op->getLoc();

        mlir::Type baseType;
        bool needTag = mth.isUnionTypeNeedsTag(op.in().getType().cast<mlir_ts::UnionType>(), baseType);
        if (needTag)
        {
            auto val0 = rewriter.create<LLVM::ExtractValueOp>(loc, tch.convertType(op.getType()), transformed.in(),
                                                              clh.getStructIndexAttr(0));

            rewriter.replaceOp(op, ValueRange{val0});
        }
        else
        {
            auto typeOfValue = rewriter.create<mlir_ts::TypeOfOp>(loc, baseType, transformed.in());

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
        

        LLVMCodeHelper ch(varOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(varOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto location = varOp.getLoc();

        auto referenceType = varOp.reference().getType().cast<mlir_ts::RefType>();
        auto storageType = referenceType.getElementType();
        auto llvmReferenceType = tch.convertType(referenceType);

#ifdef ALLOC_ALL_VARS_IN_HEAP
        auto isCaptured = true;
#elif ALLOC_CAPTURED_VARS_IN_HEAP
        auto isCaptured = varOp.captured().hasValue() && varOp.captured().getValue();
#else
        auto isCaptured = false;
#endif

        LLVM_DEBUG(llvm::dbgs() << "\n!! variable allocation: " << storageType << " is captured: " << isCaptured
                                << "\n";);

        mlir::Value allocated;
        if (!isCaptured)
        {
            auto count = 1;
            if (varOp->hasAttrOfType<mlir::IntegerAttr>(INSTANCES_COUNT_ATTR_NAME))
            {
                auto intAttr = varOp->getAttrOfType<mlir::IntegerAttr>(INSTANCES_COUNT_ATTR_NAME);
                count = intAttr.getInt();
            }

            // put all allocs at 'func' top
            auto parentFuncOp = varOp->getParentOfType<LLVM::LLVMFuncOp>();
            if (parentFuncOp)
            {
                // if inside function (not in global op)
                mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(&parentFuncOp.getBody().front().front());
                allocated = rewriter.create<LLVM::AllocaOp>(location, llvmReferenceType, clh.createI32ConstantOf(count));
            }
            else
            {
                allocated = rewriter.create<LLVM::AllocaOp>(location, llvmReferenceType, clh.createI32ConstantOf(count));
            }
        }
        else
        {

            allocated = ch.MemoryAllocBitcast(llvmReferenceType, storageType);
        }

#ifdef GC_ENABLE
        // register root which is in stack, if you call Malloc - it is not in stack anymore
        if (!isCaptured)
        {
            if (storageType.isa<mlir_ts::ClassType>() || storageType.isa<mlir_ts::StringType>() ||
                storageType.isa<mlir_ts::ArrayType>() || storageType.isa<mlir_ts::ObjectType>() ||
                storageType.isa<mlir_ts::AnyType>())
            {
                if (auto ptrType = llvmReferenceType.dyn_cast_or_null<LLVM::LLVMPointerType>())
                {
                    if (ptrType.getElementType().isa<LLVM::LLVMPointerType>())
                    {
                        TypeHelper th(rewriter);

                        auto i8PtrPtrTy = th.getI8PtrPtrType();
                        auto i8PtrTy = th.getI8PtrType();
                        auto gcRootOp = ch.getOrInsertFunction(
                            "llvm.gcroot", th.getFunctionType(th.getVoidType(), {i8PtrPtrTy, i8PtrTy}));
                        auto nullPtr = rewriter.create<LLVM::NullOp>(location, i8PtrTy);
                        rewriter.create<LLVM::CallOp>(location, gcRootOp,
                                                      ValueRange{clh.castToI8PtrPtr(allocated), nullPtr});
                    }
                }
            }
        }
#endif

        auto value = transformed.initializer();
        if (value)
        {
            rewriter.create<LLVM::StoreOp>(location, value, allocated);
        }

        rewriter.replaceOp(varOp, ValueRange{allocated});
        return success();
    }
};

struct AllocaOpLowering : public TsLlvmPattern<mlir_ts::AllocaOp>
{
    using TsLlvmPattern<mlir_ts::AllocaOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AllocaOp varOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(varOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(varOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto location = varOp.getLoc();

        auto referenceType = varOp.reference().getType().cast<mlir_ts::RefType>();
        auto storageType = referenceType.getElementType();
        auto llvmReferenceType = tch.convertType(referenceType);

        LLVM_DEBUG(llvm::dbgs() << "\n!! alloca: " << storageType << "\n";);

        mlir::Value count;
        if (transformed.count())
        {
            count = transformed.count();
        }
        else
        {
            count = clh.createI32ConstantOf(1);
        }

        mlir::Value allocated = rewriter.create<LLVM::AllocaOp>(location, llvmReferenceType, count);

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
        

        LLVMCodeHelper ch(newOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(newOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newOp.getLoc();

        mlir::Type storageType = newOp.getType();

        auto resultType = tch.convertType(newOp.getType());

        mlir::Value value;
        if (newOp.stackAlloc().hasValue() && newOp.stackAlloc().getValue())
        {
            value = rewriter.create<LLVM::AllocaOp>(loc, resultType, clh.createI32ConstantOf(1));
        }
        else
        {
            value = ch.MemoryAllocBitcast(resultType, storageType, MemoryAllocSet::Zero);
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
        

        LLVMCodeHelper ch(createTupleOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(createTupleOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = createTupleOp.getLoc();
        auto tupleType = createTupleOp.getType().cast<mlir_ts::TupleType>();

        auto tupleVar = rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(tupleType), mlir::Value(),
                                                             rewriter.getBoolAttr(false));

        // set values here
        mlir::Value zero = clh.createIndexConstantOf(0);
        auto index = 0;
        for (auto itemPair : llvm::zip(transformed.items(), createTupleOp.items()))
        {
            auto item = std::get<0>(itemPair);
            auto itemOrig = std::get<1>(itemPair);

            mlir::Value fieldIndex = clh.createStructIndexConstantOf(index);
            auto llvmValueType = tch.convertType(itemOrig.getType());
            auto llvmValuePtrType = LLVM::LLVMPointerType::get(llvmValueType);

            mlir::Value tupleVarAsLLVMType = rewriter.create<mlir_ts::DialectCastOp>(loc, tch.convertType(tupleVar.getType()), tupleVar);

            auto offset = rewriter.create<LLVM::GEPOp>(loc, llvmValuePtrType, tupleVarAsLLVMType, ValueRange{zero, fieldIndex});

            // cast item if needed
            auto destItemType = tupleType.getFields()[index].type;
            auto llvmDestValueType = tch.convertType(destItemType);

            mlir::Value itemValue = item;

            // TODO: Op should ensure that in ops types are equal result types
            if (llvmDestValueType != llvmValueType)
            {
                CastLogicHelper castLogic(createTupleOp, rewriter, tch);
                itemValue =
                    castLogic.cast(itemValue, itemOrig.getType(), llvmValueType, destItemType, llvmDestValueType);
                if (!itemValue)
                {
                    return failure();
                }
            }

            rewriter.create<LLVM::StoreOp>(loc, itemValue, offset);

            index++;
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
        auto tupleVar = transformed.instance();
        auto tupleType = tupleVar.getType().cast<LLVM::LLVMStructType>();

        // values
        SmallVector<mlir::Value> results;

        // set values here
        auto index = 0;
        for (auto &item : tupleType.getBody())
        {
            auto llvmValueType = item;
            auto value =
                rewriter.create<LLVM::ExtractValueOp>(loc, llvmValueType, tupleVar, clh.getStructIndexAttr(index));

            results.push_back(value);

            index++;
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
        

        LLVMCodeHelper ch(createArrayOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(createArrayOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = createArrayOp.getLoc();

        auto arrayType = createArrayOp.getType();
        auto elementType = arrayType.getElementType();

        mlir::Type storageType;
        mlir::TypeSwitch<mlir::Type>(elementType)
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        auto newCountAsIndexType = clh.createIndexConstantOf(createArrayOp.items().size());

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryAllocBitcast(llvmPtrElementType, multSizeOfTypeValue);

        mlir::Value index = clh.createIndexConstantOf(0);
        auto next = false;
        mlir::Value value1;
        for (auto item : transformed.items())
        {
            if (next)
            {
                if (!value1)
                {
                    value1 = clh.createIndexConstantOf(1);
                }

                index = rewriter.create<LLVM::AddOp>(loc, th.getIndexType(), ValueRange{index, value1});
            }

            // save new element
            auto offset = rewriter.create<LLVM::GEPOp>(loc, llvmPtrElementType, allocated, ValueRange{index});

            auto effectiveItem = item;
            if (llvmElementType != item.getType())
            {
                // effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, llvmElementType, item);
                llvm_unreachable("type mismatch");
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated,
                                                                 clh.getStructIndexAttr(0));

        auto newCountAsI32Type = clh.createI32ConstantOf(createArrayOp.items().size());
        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2,
                                                                 newCountAsI32Type, clh.getStructIndexAttr(1));

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
        

        LLVMCodeHelper ch(newEmptyArrOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(newEmptyArrOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newEmptyArrOp.getLoc();

        auto arrayType = newEmptyArrOp.getType();
        auto elementType = arrayType.getElementType();

        mlir::Type storageType;
        mlir::TypeSwitch<mlir::Type>(elementType)
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        auto allocated = rewriter.create<LLVM::NullOp>(loc, llvmPtrElementType);

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated,
                                                                 clh.getStructIndexAttr(0));

        auto size0 = clh.createI32ConstantOf(0);
        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, size0,
                                                                 clh.getStructIndexAttr(1));

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
        

        LLVMCodeHelper ch(newArrOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(newArrOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newArrOp.getLoc();

        auto arrayType = newArrOp.getType();
        auto elementType = arrayType.getElementType();

        mlir::Type storageType;
        mlir::TypeSwitch<mlir::Type>(elementType)
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto countAsIndexType = rewriter.create<mlir_ts::CastOp>(loc, th.getIndexType(), transformed.count());
        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, countAsIndexType});

        auto allocated = ch.MemoryAllocBitcast(llvmPtrElementType, multSizeOfTypeValue);

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated,
                                                                 clh.getStructIndexAttr(0));

        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2,
                                                                 transformed.count(), clh.getStructIndexAttr(1));

        rewriter.replaceOp(newArrOp, ValueRange{structValue3});
        return success();
    }
};

struct PushOpLowering : public TsLlvmPattern<mlir_ts::PushOp>
{
    using TsLlvmPattern<mlir_ts::PushOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PushOp pushOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(pushOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(pushOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = pushOp.getLoc();

        auto arrayType = pushOp.op().getType().cast<mlir_ts::RefType>().getElementType().cast<mlir_ts::ArrayType>();
        auto elementType = arrayType.getElementType();
        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        auto ind0 = clh.createI32ConstantOf(0);
        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(llvmPtrElementType), transformed.op(),
                                                          ValueRange{ind0, ind0});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, llvmPtrElementType, currentPtrPtr);

        auto ind1 = clh.createI32ConstantOf(1);
        auto countAsI32TypePtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(th.getI32Type()), transformed.op(),
                                                              ValueRange{ind0, ind1});
        auto countAsI32Type = rewriter.create<LLVM::LoadOp>(loc, th.getI32Type(), countAsI32TypePtr);

        auto countAsIndexType = rewriter.create<LLVM::ZExtOp>(loc, th.getIndexType(), countAsI32Type);

        auto incSize = clh.createIndexConstantOf(transformed.items().size());
        auto newCountAsIndexType =
            rewriter.create<LLVM::AddOp>(loc, th.getIndexType(), ValueRange{countAsIndexType, incSize});

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryReallocBitcast(llvmPtrElementType, currentPtr, multSizeOfTypeValue);

        mlir::Value index = countAsIndexType;
        auto next = false;
        mlir::Value value1;
        for (auto itemPair : llvm::zip(transformed.items(), pushOp.items()))
        {
            auto item = std::get<0>(itemPair);
            auto itemOrig = std::get<1>(itemPair);

            if (next)
            {
                if (!value1)
                {
                    value1 = clh.createIndexConstantOf(1);
                }

                index = rewriter.create<LLVM::AddOp>(loc, th.getIndexType(), ValueRange{index, value1});
            }

            // save new element
            auto offset = rewriter.create<LLVM::GEPOp>(loc, llvmPtrElementType, allocated, ValueRange{index});

            auto effectiveItem = item;
            if (elementType != itemOrig.getType())
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! push cast: store type: " << elementType
                                        << " value type: " << item.getType() << "\n";);
                llvm_unreachable("cast must happen earlier");
                // effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, elementType, item);
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);

        auto newCountAsI32Type = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), newCountAsIndexType);

        rewriter.create<LLVM::StoreOp>(loc, newCountAsI32Type, countAsI32TypePtr);

        rewriter.replaceOp(pushOp, ValueRange{newCountAsIndexType});
        return success();
    }
};

struct PopOpLowering : public TsLlvmPattern<mlir_ts::PopOp>
{
    using TsLlvmPattern<mlir_ts::PopOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PopOp popOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(popOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(popOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = popOp.getLoc();

        auto arrayType = popOp.op().getType().cast<mlir_ts::RefType>().getElementType().cast<mlir_ts::ArrayType>();
        auto elementType = arrayType.getElementType();
        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        mlir::Type storageType;
        mlir::TypeSwitch<mlir::Type>(popOp.op().getType())
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto ind0 = clh.createI32ConstantOf(0);
        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(llvmPtrElementType), transformed.op(),
                                                          ValueRange{ind0, ind0});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, llvmPtrElementType, currentPtrPtr);

        auto ind1 = clh.createI32ConstantOf(1);
        auto countAsI32TypePtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(th.getI32Type()), transformed.op(),
                                                              ValueRange{ind0, ind1});
        auto countAsI32Type = rewriter.create<LLVM::LoadOp>(loc, th.getI32Type(), countAsI32TypePtr);

        auto countAsIndexType = rewriter.create<LLVM::ZExtOp>(loc, th.getIndexType(), countAsI32Type);

        auto incSize = clh.createIndexConstantOf(1);
        auto newCountAsIndexType =
            rewriter.create<LLVM::SubOp>(loc, th.getIndexType(), ValueRange{countAsIndexType, incSize});

        // load last element
        auto offset =
            rewriter.create<LLVM::GEPOp>(loc, llvmPtrElementType, currentPtr, ValueRange{newCountAsIndexType});
        auto loadedElement = rewriter.create<LLVM::LoadOp>(loc, llvmElementType, offset);

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto multSizeOfTypeValue =
            rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryReallocBitcast(llvmPtrElementType, currentPtr, multSizeOfTypeValue);

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);

        auto newCountAsI32Type = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), newCountAsIndexType);

        rewriter.create<LLVM::StoreOp>(loc, newCountAsI32Type, countAsI32TypePtr);

        rewriter.replaceOp(popOp, ValueRange{loadedElement});
        return success();
    }
};

struct DeleteOpLowering : public TsLlvmPattern<mlir_ts::DeleteOp>
{
    using TsLlvmPattern<mlir_ts::DeleteOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DeleteOp deleteOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(deleteOp, rewriter, getTypeConverter());

        if (mlir::failed(ch.MemoryFree(transformed.reference())))
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
        

        auto opCode = (SyntaxKind)arithmeticUnaryOp.opCode();
        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:
            NegativeOpBin(arithmeticUnaryOp, transformed.operand1(), rewriter);
            return success();
        case SyntaxKind::PlusToken:
            rewriter.replaceOp(arithmeticUnaryOp, transformed.operand1());
            return success();
        case SyntaxKind::MinusToken:
            NegativeOpValue(arithmeticUnaryOp, transformed.operand1(), rewriter);
            return success();
        case SyntaxKind::TildeToken:
            NegativeOpBin(arithmeticUnaryOp, transformed.operand1(), rewriter);
            return success();
        default:
            llvm_unreachable("not implemented");
        }
    }
};

struct ArithmeticBinaryOpLowering : public TsLlvmPattern<mlir_ts::ArithmeticBinaryOp>
{
    using TsLlvmPattern<mlir_ts::ArithmeticBinaryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArithmeticBinaryOp arithmeticBinaryOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto opCode = (SyntaxKind)arithmeticBinaryOp.opCode();
        switch (opCode)
        {
        case SyntaxKind::PlusToken:
            if (arithmeticBinaryOp.operand1().getType().isa<mlir_ts::StringType>())
            {
                rewriter.replaceOpWithNewOp<mlir_ts::StringConcatOp>(
                    arithmeticBinaryOp, mlir_ts::StringType::get(rewriter.getContext()),
                    ValueRange{transformed.operand1(), transformed.operand2()});
            }
            else
            {
                BinOp<mlir_ts::ArithmeticBinaryOp, arith::AddIOp, arith::AddFOp>(arithmeticBinaryOp, transformed.operand1(),
                                                                   transformed.operand2(), rewriter);
            }

            return success();

        case SyntaxKind::MinusToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::SubIOp, arith::SubFOp>(arithmeticBinaryOp, transformed.operand1(),
                                                               transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::AsteriskToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::MulIOp, arith::MulFOp>(arithmeticBinaryOp, transformed.operand1(),
                                                               transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::SlashToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::DivSIOp, arith::DivFOp, arith::DivUIOp>(arithmeticBinaryOp, transformed.operand1(),
                                                               transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::GreaterThanGreaterThanToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::ShRSIOp, arith::ShRSIOp, arith::ShRUIOp>(
                arithmeticBinaryOp, transformed.operand1(), transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::ShRUIOp, arith::ShRUIOp>(
                arithmeticBinaryOp, transformed.operand1(), transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::LessThanLessThanToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::ShLIOp, arith::ShLIOp>(arithmeticBinaryOp, transformed.operand1(),
                                                                         transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::AmpersandToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::AndIOp, arith::AndIOp>(arithmeticBinaryOp, transformed.operand1(),
                                                             transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::BarToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::OrIOp, arith::OrIOp>(arithmeticBinaryOp, transformed.operand1(),
                                                           transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::CaretToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::XOrIOp, arith::XOrIOp>(arithmeticBinaryOp, transformed.operand1(),
                                                             transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::PercentToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, arith::RemSIOp, arith::RemFOp, arith::RemUIOp>(arithmeticBinaryOp, transformed.operand1(),
                                                               transformed.operand2(), rewriter);
            return success();

        case SyntaxKind::AsteriskAsteriskToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, math::PowFOp, math::PowFOp>(arithmeticBinaryOp, transformed.operand1(),
                                                                           transformed.operand2(), rewriter);
            return success();

        default:
            llvm_unreachable("not implemented");
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
        return LogicOp<arith::CmpIOp, arith::CmpIPredicate, v1, arith::CmpFOp, arith::CmpFPredicate, v2>(logicalBinaryOp, op, left, leftTypeOrig,
                                                                             right, rightTypeOrig, builder,
                                                                             *(LLVMTypeConverter *)getTypeConverter());
    }

    LogicalResult matchAndRewrite(mlir_ts::LogicalBinaryOp logicalBinaryOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto op = (SyntaxKind)logicalBinaryOp.opCode();

        auto op1 = transformed.operand1();
        auto op2 = transformed.operand2();
        auto opType1 = logicalBinaryOp.operand1().getType();
        auto opType2 = logicalBinaryOp.operand2().getType();

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
            value = logicOp<arith::CmpIPredicate::sgt, arith::CmpFPredicate::OGT>(logicalBinaryOp, op, op1, opType1, op2, opType2,
                                                                    rewriter);
            break;
        case SyntaxKind::GreaterThanEqualsToken:
            value = logicOp<arith::CmpIPredicate::sge, arith::CmpFPredicate::OGE>(logicalBinaryOp, op, op1, opType1, op2, opType2,
                                                                    rewriter);
            break;
        case SyntaxKind::LessThanToken:
            value = logicOp<arith::CmpIPredicate::slt, arith::CmpFPredicate::OLT>(logicalBinaryOp, op, op1, opType1, op2, opType2,
                                                                    rewriter);
            break;
        case SyntaxKind::LessThanEqualsToken:
            value = logicOp<arith::CmpIPredicate::sle, arith::CmpFPredicate::OLE>(logicalBinaryOp, op, op1, opType1, op2, opType2,
                                                                    rewriter);
            break;
        default:
            llvm_unreachable("not implemented");
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

        auto elementRefType = loadOp.reference().getType();
        auto resultType = loadOp.getType();

        if (auto refType = elementRefType.dyn_cast_or_null<mlir_ts::RefType>())
        {
            elementType = refType.getElementType();
            elementTypeConverted = tch.convertType(elementType);
        }
        else if (auto valueRefType = elementRefType.dyn_cast_or_null<mlir_ts::ValueRefType>())
        {
            elementType = valueRefType.getElementType();
            elementTypeConverted = tch.convertType(elementType);
        }

        auto isOptional = false;
        if (auto optType = resultType.dyn_cast<mlir_ts::OptionalType>())
        {
            isOptional = optType.getElementType() == elementType;
        }

        mlir::Value loadedValue;
        auto loadedValueFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
            mlir::Value loadedValue;
            if (elementType)
            {
                loadedValue = rewriter.create<LLVM::LoadOp>(loc, elementTypeConverted, transformed.reference());
            }
            else if (auto boundRefType = elementRefType.dyn_cast_or_null<mlir_ts::BoundRefType>())
            {
                loadedValue = rewriter.create<mlir_ts::LoadBoundRefOp>(loc, resultType, loadOp.reference());
            }

            return loadedValue;
        };

        if (isOptional)
        {
            auto resultTypeLlvm = tch.convertType(resultType);

            auto undefOptionalFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
                mlir::Value val = rewriter.create<mlir_ts::UndefOptionalOp>(loc, resultType);
                mlir::Value valAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, resultTypeLlvm, val);
                return valAsLLVMType;
            };

            auto createOptionalFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
                auto dataValue = loadedValueFunc(builder, location);
                mlir::Value val = rewriter.create<mlir_ts::CreateOptionalOp>(loc, resultType, dataValue);
                mlir::Value valAsLLVMType = builder.create<mlir_ts::DialectCastOp>(loc, resultTypeLlvm, val);
                return valAsLLVMType;
            };

            LLVMTypeConverterHelper llvmtch(*(LLVMTypeConverter *)getTypeConverter());

            auto intPtrType = llvmtch.getIntPtrType(0);

            // not null condition
            auto dataIntPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, transformed.reference());
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

        //LLVM_DEBUG(llvm::dbgs() << "\n!! LoadOp Ref value: \n" << transformed.reference() << "\n";);
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
        

        if (auto boundRefType = storeOp.reference().getType().dyn_cast_or_null<mlir_ts::BoundRefType>())
        {
            rewriter.replaceOpWithNewOp<mlir_ts::StoreBoundRefOp>(storeOp, storeOp.value(), storeOp.reference());
            return success();
        }

        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, transformed.value(), transformed.reference());
        return success();
    }
};

struct ElementRefOpLowering : public TsLlvmPattern<mlir_ts::ElementRefOp>
{
    using TsLlvmPattern<mlir_ts::ElementRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ElementRefOp elementOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper ch(elementOp, rewriter, getTypeConverter());

        auto addr = ch.GetAddressOfArrayElement(elementOp.getResult().getType(), elementOp.array().getType(),
                                                transformed.array(), transformed.index());
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
        

        LLVMCodeHelper ch(elementOp, rewriter, getTypeConverter());

        auto addr = ch.GetAddressOfPointerOffset(elementOp.getResult().getType(), elementOp.ref().getType(),
                                                transformed.ref(), transformed.index());
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
                                                          transformed.object(), extractPropertyOp.position());

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
            insertPropertyOp, tch.convertType(insertPropertyOp.object().getType()), transformed.object(),
            transformed.value(), insertPropertyOp.position());

        return success();
    }
};

struct PropertyRefOpLowering : public TsLlvmPattern<mlir_ts::PropertyRefOp>
{
    using TsLlvmPattern<mlir_ts::PropertyRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PropertyRefOp propertyRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        assert(propertyRefOp.position() != -1);

        LLVMCodeHelper ch(propertyRefOp, rewriter, getTypeConverter());

        auto addr =
            ch.GetAddressOfStructElement(propertyRefOp.getType(), transformed.objectRef(), propertyRefOp.position());

        if (auto boundRefType = propertyRefOp.getType().dyn_cast_or_null<mlir_ts::BoundRefType>())
        {
            auto boundRef = rewriter.create<mlir_ts::CreateBoundRefOp>(propertyRefOp->getLoc(), boundRefType,
                                                                       propertyRefOp.objectRef(), addr);
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

        LLVMCodeHelper lch(globalOp, rewriter, getTypeConverter());

        auto createAsGlobalConstructor = false;
        auto visitorAllOps = [&](Operation *op) {
            if (isa<mlir_ts::NewOp>(op) || isa<mlir_ts::NewInterfaceOp>(op) || isa<mlir_ts::NewArrayOp>(op) ||
                isa<mlir_ts::SymbolCallInternalOp>(op) || isa<mlir_ts::CallInternalOp>(op) ||
                isa<mlir_ts::CallHybridInternalOp>(op) || isa<mlir_ts::VariableOp>(op) || isa<mlir_ts::AllocaOp>(op))
            {
                createAsGlobalConstructor = true;
            }
        };

        globalOp.getInitializerRegion().walk(visitorAllOps);

        auto linkage = lch.getLinkage(globalOp);

        if (createAsGlobalConstructor)
        {
            // TODO: create function and call GlobalConstructor
            lch.createUndefGlobalVarIfNew(globalOp.sym_name(), getTypeConverter()->convertType(globalOp.type()),
                                          globalOp.valueAttr(), globalOp.constant(), linkage);

            auto name = globalOp.sym_name().str();
            name.append("__cctor");
            lch.createFunctionFromRegion(loc, name, globalOp.getInitializerRegion(), globalOp.sym_name());
            rewriter.create<mlir_ts::GlobalConstructorOp>(loc, name);
        }
        else
        {
            lch.createGlobalVarIfNew(globalOp.sym_name(), getTypeConverter()->convertType(globalOp.type()),
                                     globalOp.valueAttr(), globalOp.constant(), globalOp.getInitializerRegion(),
                                     linkage);
        }

        rewriter.eraseOp(globalOp);
        return success();
    }
};

struct GlobalResultOpLowering : public TsLlvmPattern<mlir_ts::GlobalResultOp>
{
    using TsLlvmPattern<mlir_ts::GlobalResultOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GlobalResultOp globalResultOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(globalResultOp, transformed.results());
        return success();
    }
};

struct AddressOfOpLowering : public TsLlvmPattern<mlir_ts::AddressOfOp>
{
    using TsLlvmPattern<mlir_ts::AddressOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AddressOfOp addressOfOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        LLVMCodeHelper lch(addressOfOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto actualType = addressOfOp.getType();
        if (actualType.isa<mlir_ts::OpaqueType>())
        {
            // load type from symbol
            auto module = addressOfOp->getParentOfType<mlir::ModuleOp>();
            assert(module);
            auto globalOp = module.lookupSymbol<LLVM::GlobalOp>(addressOfOp.global_name());
            if (!globalOp)
            {
                LLVM_DEBUG(llvm::dbgs() << "\n!! NOT found symbol: " << addressOfOp.global_name() << "\n";);
                assert(globalOp);
                return mlir::failure();
            }

            LLVM_DEBUG(llvm::dbgs() << "\n!! found symbol: " << globalOp << "\n";);
            actualType = mlir_ts::RefType::get(globalOp.getType());

            auto value = lch.getAddressOfGlobalVar(addressOfOp.global_name(), tch.convertType(actualType),
                                                   addressOfOp.offset() ? addressOfOp.offset().getValue() : 0);

            mlir::Value castedValue =
                rewriter.create<LLVM::BitcastOp>(addressOfOp->getLoc(), tch.convertType(addressOfOp.getType()), value);

            rewriter.replaceOp(addressOfOp, castedValue);
        }
        else
        {
            auto value = lch.getAddressOfGlobalVar(addressOfOp.global_name(), tch.convertType(actualType),
                                                   addressOfOp.offset() ? addressOfOp.offset().getValue() : 0);
            rewriter.replaceOp(addressOfOp, value);
        }
        return success();
    }
};

struct AddressOfConstStringOpLowering : public TsLlvmPattern<mlir_ts::AddressOfConstStringOp>
{
    using TsLlvmPattern<mlir_ts::AddressOfConstStringOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AddressOfConstStringOp addressOfConstStringOp,
                                  Adaptor transformed, ConversionPatternRewriter &rewriter) const final
    {
        

        TypeHelper th(rewriter);

        auto loc = addressOfConstStringOp->getLoc();
        auto globalPtr =
            rewriter.create<LLVM::AddressOfOp>(loc, th.getI8PtrType(), addressOfConstStringOp.global_name());
        auto cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getI64Type(), th.getIndexAttrValue(0));
        rewriter.replaceOpWithNewOp<LLVM::GEPOp>(addressOfConstStringOp, th.getI8PtrType(), globalPtr,
                                                 ArrayRef<mlir::Value>({cst0, cst0}));

        return success();
    }
};

struct CreateOptionalOpLowering : public TsLlvmPattern<mlir_ts::CreateOptionalOp>
{
    using TsLlvmPattern<mlir_ts::CreateOptionalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateOptionalOp createOptionalOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = createOptionalOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(createOptionalOp, rewriter);

        auto boxedType = createOptionalOp.res().getType().cast<mlir_ts::OptionalType>().getElementType();
        auto llvmBoxedType = tch.convertType(boxedType);
        auto llvmOptType = tch.convertType(createOptionalOp.res().getType());

        auto valueOrigType = createOptionalOp.in().getType();

        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateOptional : " << createOptionalOp.in() << "\n";);

        auto value = transformed.in();
        auto valueLLVMType = value.getType();

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmOptType);

        // TODO: it should be tested by OP that value is equal to value in optional type
        if (valueLLVMType != llvmBoxedType)
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! CreateOptional value types : " << valueLLVMType
                                    << " optional type: " << llvmBoxedType << "\n";);

            // cast value to box
            CastLogicHelper castLogic(createOptionalOp, rewriter, tch);
            value = castLogic.cast(value, valueOrigType, valueLLVMType, boxedType, llvmBoxedType);
            if (!value)
            {
                return failure();
            }

            value = rewriter.create<mlir_ts::DialectCastOp>(loc, llvmBoxedType, value);
        }

        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmOptType, structValue, value,
                                                                 rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));

        auto trueValue = clh.createI1ConstantOf(true);
        rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(createOptionalOp, llvmOptType, structValue2, trueValue,
                                                         rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

        return success();
    }
};

struct UndefOptionalOpLowering : public TsLlvmPattern<mlir_ts::UndefOptionalOp>
{
    using TsLlvmPattern<mlir_ts::UndefOptionalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::UndefOptionalOp undefOptionalOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        auto loc = undefOptionalOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(undefOptionalOp, rewriter);

        auto boxedType = undefOptionalOp.res().getType().cast<mlir_ts::OptionalType>().getElementType();
        auto llvmBoxedType = tch.convertType(boxedType);
        auto llvmOptType = tch.convertType(undefOptionalOp.res().getType());

        mlir::Value structValue = rewriter.create<LLVM::UndefOp>(loc, llvmOptType);
        auto structValue2 = structValue;

        // default value
        mlir::Value defaultValue;

        if (llvmBoxedType.isa<LLVM::LLVMPointerType>())
        {
            defaultValue = rewriter.create<LLVM::NullOp>(loc, llvmBoxedType);
        }
        else if (llvmBoxedType.isa<mlir::IntegerType>())
        {
            llvmBoxedType.cast<mlir::IntegerType>().getWidth();
            defaultValue = clh.createIConstantOf(llvmBoxedType.cast<mlir::IntegerType>().getWidth(), 0);
        }
        else if (llvmBoxedType.isa<mlir::FloatType>())
        {
            llvmBoxedType.cast<mlir::FloatType>().getWidth();
            defaultValue = clh.createFConstantOf(llvmBoxedType.cast<mlir::FloatType>().getWidth(), 0.0);
        }

        if (defaultValue)
        {
            structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmOptType, structValue, defaultValue,
                                                                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));
        }

        auto falseValue = clh.createI1ConstantOf(false);
        rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(undefOptionalOp, llvmOptType, structValue2, falseValue,
                                                         rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

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

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(hasValueOp, th.getLLVMBoolType(), transformed.in(),
                                                          rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

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

        auto valueType = valueOp.res().getType();
        auto llvmValueType = tch.convertType(valueType);

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(valueOp, llvmValueType, transformed.in(),
                                                          rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));

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

        auto value = rewriter.create<LLVM::LoadOp>(loc, transformed.src());
        rewriter.create<LLVM::StoreOp>(loc, value, transformed.dst());

        rewriter.eraseOp(loadSaveOp);

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
        LLVMCodeHelper ch(memoryCopyOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(memoryCopyOp, rewriter);

        auto copyMemFuncOp = ch.getOrInsertFunction(
            "llvm.memcpy.p0.p0.i64", th.getFunctionType(th.getVoidType(), {th.getI8PtrType(), th.getI8PtrType(),
                                                                               th.getI64Type(), th.getLLVMBoolType()}));

        mlir::SmallVector<mlir::Value, 4> values;
        values.push_back(clh.castToI8Ptr(transformed.dst()));
        values.push_back(clh.castToI8Ptr(transformed.src()));

        auto llvmSrcType = tch.convertType(memoryCopyOp.src().getType());
        auto srcValueType = llvmSrcType.cast<LLVM::LLVMPointerType>().getElementType();
        auto srcSize = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), srcValueType);

        auto llvmDstType = tch.convertType(memoryCopyOp.dst().getType());
        auto dstValueType = llvmDstType.cast<LLVM::LLVMPointerType>().getElementType();
        auto dstSize = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), dstValueType);

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

        ThrowLogic tl(throwCallOp, rewriter, tch, throwCallOp.getLoc());
        tl.logic(transformed.exception(), throwCallOp.exception().getType(), nullptr);

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
        ThrowLogic tl(throwUnwindOp, rewriter, tch, throwUnwindOp.getLoc());
        tl.logic(transformed.exception(), throwUnwindOp.exception().getType(), throwUnwindOp.unwindDest());

        rewriter.eraseOp(throwUnwindOp);

        return success();
    }
};

#ifdef WIN_EXCEPTION

struct LandingPadOpLowering : public TsLlvmPattern<mlir_ts::LandingPadOp>
{
    using TsLlvmPattern<mlir_ts::LandingPadOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LandingPadOp landingPadOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = landingPadOp.getLoc();
        if (!landingPadOp.cleanup())
        {
            auto catch1 = transformed.catches().front();
            mlir::Type llvmLandingPadTy = getTypeConverter()->convertType(landingPadOp.getType());
            rewriter.replaceOpWithNewOp<LLVM::LandingpadOp>(landingPadOp, llvmLandingPadTy, false, ValueRange{catch1});
        }
        else
        {
            // BUG: in LLVM landing pad is not fully implemented
            // so lets create filter with undef value to mark cleanup landing
            auto catch1Fake = transformed.catches().front();

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
        LLVMCodeHelper ch(beginCatchOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(beginCatchOp, rewriter);

        auto i8PtrTy = th.getI8PtrType();

        // catches:extract
        auto loadedI8PtrValue = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.landingPad(),
                                                                      clh.getStructIndexAttr(0));

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
        LLVMCodeHelper ch(saveCatchVarOp, rewriter, getTypeConverter());

        auto ptr = rewriter.create<mlir_ts::CastOp>(loc, th.getI8PtrType(), transformed.varStore());

        auto saveCatchFuncName = "ts.internal.save_catch_var";
        auto saveCatchFunc = ch.getOrInsertFunction(
            saveCatchFuncName,
            th.getFunctionType(th.getVoidType(), ArrayRef<mlir::Type>{getTypeConverter()->convertType(
                                                                          /*saveCatchVarOp*/transformed.exceptionInfo().getType()),
                                                                      ptr.getType()}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(saveCatchVarOp, saveCatchFunc,
                                                  ValueRange{/*saveCatchVarOp*/transformed.exceptionInfo(), ptr});

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
        LLVMCodeHelper ch(endCatchOp, rewriter, getTypeConverter());

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
        LLVMCodeHelper ch(endCleanupOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(endCleanupOp, rewriter);

        auto endCatchFuncName = "__cxa_end_catch";
        if (!endCleanupOp.unwindDest().empty())
        {
            clh.Invoke(loc, [&](mlir::Block *continueBlock) {
                rewriter.replaceOpWithNewOp<LLVM::InvokeOp>(
                    endCleanupOp, mlir::TypeRange{},
                    ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), endCatchFuncName), ValueRange{},
                    continueBlock, ValueRange{}, endCleanupOp.unwindDest().front(), ValueRange{});
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

        rewriter.create<LLVM::ResumeOp>(loc, transformed.landingPad());

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

#else

struct LandingPadOpLowering : public TsLlvmPattern<mlir_ts::LandingPadOp>
{
    using TsLlvmPattern<mlir_ts::LandingPadOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LandingPadOp landingPadOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = landingPadOp.getLoc();

        TypeHelper th(rewriter);

        auto catch1 = transformed.catches().front();

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
        LLVMCodeHelper ch(compareCatchTypeOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(compareCatchTypeOp, rewriter);

        auto i8PtrTy = th.getI8PtrType();

        auto loadedI32Value = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI32Type(), transformed.landingPad(),
                                                                    clh.getStructIndexAttr(1));

        auto typeIdFuncName = "llvm.eh.typeid.for";
        auto typeIdFunc = ch.getOrInsertFunction(typeIdFuncName, th.getFunctionType(th.getI32Type(), {i8PtrTy}));

        auto callInfo =
            rewriter.create<LLVM::CallOp>(loc, typeIdFunc, ValueRange{clh.castToI8Ptr(transformed.throwTypeInfo())});
        auto typeIdValue = callInfo.getResult(0);

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
        LLVMCodeHelper ch(beginCatchOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(beginCatchOp, rewriter);

        auto i8PtrTy = th.getI8PtrType();

        // catches:extract
        auto loadedI8PtrValue = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.landingPad(),
                                                                      clh.getStructIndexAttr(0));

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

        auto catchRefType = saveCatchVarOp.varStore().getType().cast<mlir_ts::RefType>();
        auto catchType = catchRefType.getElementType();
        auto llvmCatchType = getTypeConverter()->convertType(catchType);

        mlir::Value catchVal;
        if (!llvmCatchType.isa<LLVM::LLVMPointerType>())
        {
            auto ptrVal =
                rewriter.create<LLVM::BitcastOp>(loc, th.getPointerType(llvmCatchType), transformed.exceptionInfo());
            catchVal = rewriter.create<LLVM::LoadOp>(loc, llvmCatchType, ptrVal);
        }
        else
        {
            catchVal = rewriter.create<LLVM::BitcastOp>(loc, llvmCatchType, transformed.exceptionInfo());
        }

        rewriter.replaceOpWithNewOp<mlir_ts::StoreOp>(saveCatchVarOp, catchVal, transformed.varStore());

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
        LLVMCodeHelper ch(endCatchOp, rewriter, getTypeConverter());

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

        rewriter.replaceOpWithNewOp<LLVM::ResumeOp>(endCleanupOp, transformed.landingPad());

        auto terminator = rewriter.getInsertionBlock()->getTerminator();
        if (terminator != endCleanupOp && terminator != endCleanupOp->getNextNode())
        {
            clh.CutBlock();
        }

        // add resume

        return success();
    }
};

#endif
struct TrampolineOpLowering : public TsLlvmPattern<mlir_ts::TrampolineOp>
{
    using TsLlvmPattern<mlir_ts::TrampolineOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TrampolineOp trampolineOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        // TODO: missing attribute "nest" on parameter
        auto location = trampolineOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(trampolineOp, rewriter);
        LLVMCodeHelper ch(trampolineOp, rewriter, getTypeConverter());
        CastLogicHelper castLogic(trampolineOp, rewriter, tch);

        auto i8PtrTy = th.getI8PtrType();

        auto initTrampolineFuncOp = ch.getOrInsertFunction(
            "llvm.init.trampoline", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, i8PtrTy}));
        auto adjustTrampolineFuncOp =
            ch.getOrInsertFunction("llvm.adjust.trampoline", th.getFunctionType(i8PtrTy, {i8PtrTy}));
        // Win32 specifics
        auto enableExecuteStackFuncOp =
            ch.getOrInsertFunction("__enable_execute_stack", th.getFunctionType(th.getVoidType(), {i8PtrTy}));

        // allocate temp trampoline
        auto bufferType = th.getPointerType(th.getI8Array(TRAMPOLINE_SIZE));

#ifdef ALLOC_TRAMPOLINE_IN_HEAP
        auto allocInHeap = true;
#else
        auto allocInHeap = trampolineOp.allocInHeap().hasValue() && trampolineOp.allocInHeap().getValue();
#endif

        mlir::Value trampolinePtr;
        if (allocInHeap)
        {
            trampolinePtr = ch.MemoryAlloc(bufferType);
        }
        else
        {
            mlir::Value trampoline;
            {
                // we can't reallocate alloc for trampolines
                /*
                // put all allocs at 'func' top
                auto parentFuncOp = trampolineOp->getParentOfType<LLVM::LLVMFuncOp>();
                assert(parentFuncOp);
                mlir::OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPoint(&parentFuncOp.getBody().front().front());
                */

                trampoline = rewriter.create<LLVM::AllocaOp>(location, bufferType, clh.createI32ConstantOf(1));
            }

            auto const0 = clh.createI32ConstantOf(0);
            trampolinePtr = rewriter.create<LLVM::GEPOp>(location, i8PtrTy, trampoline, ValueRange{const0, const0});
        }

        // init trampoline
        rewriter.create<LLVM::CallOp>(location, initTrampolineFuncOp,
                                      ValueRange{trampolinePtr, clh.castToI8Ptr(transformed.callee()),
                                                 clh.castToI8Ptr(transformed.data_reference())});

        auto callAdjustedTrampoline =
            rewriter.create<LLVM::CallOp>(location, adjustTrampolineFuncOp, ValueRange{trampolinePtr});
        auto adjustedTrampolinePtr = callAdjustedTrampoline.getResult(0);

        rewriter.create<LLVM::CallOp>(location, enableExecuteStackFuncOp, ValueRange{adjustedTrampolinePtr});

        // mlir::Value castFunc = rewriter.create<mlir_ts::CastOp>(location, trampolineOp.getType(),
        // adjustedTrampolinePtr); replacement
        auto castFunc = castLogic.cast(adjustedTrampolinePtr, adjustedTrampolinePtr.getType(), trampolineOp.getType());

        rewriter.replaceOp(trampolineOp, castFunc);

        return success();
    }
};

struct VTableOffsetRefOpLowering : public TsLlvmPattern<mlir_ts::VTableOffsetRefOp>
{
    using TsLlvmPattern<mlir_ts::VTableOffsetRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::VTableOffsetRefOp vtableOffsetRefOp, Adaptor transformed,
                                  ConversionPatternRewriter &rewriter) const final
    {
        

        Location loc = vtableOffsetRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(vtableOffsetRefOp, rewriter);

        auto ptrToArrOfPtrs = rewriter.create<mlir_ts::CastOp>(loc, th.getI8PtrPtrType(), transformed.vtable());

        auto index = clh.createI32ConstantOf(vtableOffsetRefOp.index());
        auto methodOrInterfacePtrPtr =
            rewriter.create<LLVM::GEPOp>(loc, th.getI8PtrPtrType(), ptrToArrOfPtrs, ValueRange{index});
        auto methodOrInterfacePtr = rewriter.create<LLVM::LoadOp>(loc, methodOrInterfacePtrPtr);

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
        

        assert(virtualSymbolRefOp.index() != -1);

        Location loc = virtualSymbolRefOp.getLoc();

        TypeHelper th(rewriter);

        auto methodOrFieldPtr = rewriter.create<mlir_ts::VTableOffsetRefOp>(
            loc, th.getI8PtrType(), transformed.vtable(), virtualSymbolRefOp.index());

        if (auto funcType = virtualSymbolRefOp.getType().dyn_cast<mlir_ts::FunctionType>())
        {
            auto methodTyped = rewriter.create<mlir_ts::CastOp>(loc, funcType, methodOrFieldPtr);
            rewriter.replaceOp(virtualSymbolRefOp, ValueRange{methodTyped});
        }
        else if (auto fieldType = virtualSymbolRefOp.getType().dyn_cast<mlir_ts::RefType>())
        {
            auto fieldTyped = rewriter.create<mlir_ts::CastOp>(loc, fieldType, methodOrFieldPtr);
            rewriter.replaceOp(virtualSymbolRefOp, ValueRange{fieldTyped});
        }
        else
        {
            llvm_unreachable("not implemented");
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
        

        assert(thisVirtualSymbolRefOp.index() != -1);

        Location loc = thisVirtualSymbolRefOp.getLoc();

        TypeHelper th(rewriter);

        auto methodPtr = rewriter.create<mlir_ts::VTableOffsetRefOp>(loc, th.getI8PtrType(), transformed.vtable(),
                                                                     thisVirtualSymbolRefOp.index());
        // auto methodTyped = rewriter.create<mlir_ts::CastOp>(loc, thisVirtualSymbolRefOp.getType(), methodPtr);

        if (auto boundFunc = thisVirtualSymbolRefOp.getType().dyn_cast<mlir_ts::BoundFunctionType>())
        {
            auto thisOpaque = rewriter.create<mlir_ts::CastOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()),
                                                               transformed.thisVal());
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
        

        assert(interfaceSymbolRefOp.index() != -1);

        Location loc = interfaceSymbolRefOp.getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(interfaceSymbolRefOp, rewriter);

        auto fieldLLVMTypeRef = tch.convertType(interfaceSymbolRefOp.getType());

        auto isOptional = interfaceSymbolRefOp.optional().hasValue() && interfaceSymbolRefOp.optional().getValue();

        auto vtable = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.interfaceVal(),
                                                            clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.interfaceVal(),
                                                             clh.getStructIndexAttr(THIS_VALUE_INDEX));

        auto methodOrFieldPtr =
            rewriter.create<mlir_ts::VTableOffsetRefOp>(loc, th.getI8PtrType(), vtable, interfaceSymbolRefOp.index());

        if (auto boundFunc = interfaceSymbolRefOp.getType().dyn_cast<mlir_ts::BoundFunctionType>())
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
                auto p1 = rewriter.create<LLVM::PtrToIntOp>(loc, th.getIndexType(), thisVal);
                auto p2 = rewriter.create<LLVM::PtrToIntOp>(loc, th.getIndexType(), methodOrFieldPtr);
                auto padded = rewriter.create<LLVM::AddOp>(loc, th.getIndexType(), p1, p2);
                auto typedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, fieldLLVMTypeRef, padded);

                // no need to BoundRef
                // auto boundRefVal = rewriter.create<mlir_ts::CreateBoundRefOp>(loc, thisVal, typedPtr);
                return typedPtr;
            };

            mlir::Value fieldAddr;
            if (isOptional)
            {
                auto nullAddrFunc = [&](OpBuilder &builder, Location location) -> mlir::Value {
                    auto typedPtr = rewriter.create<LLVM::NullOp>(loc, fieldLLVMTypeRef);
                    return typedPtr;
                };

                LLVMTypeConverterHelper llvmtch(*(LLVMTypeConverter *)getTypeConverter());

                auto negative1 = clh.createI64ConstantOf(-1);
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
            loc, structVal, clh.castToI8Ptr(transformed.interfaceVTable()), clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, clh.castToI8Ptr(transformed.thisVal()),
                                                               clh.getStructIndexAttr(THIS_VALUE_INDEX));

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

        LLVM_DEBUG(llvm::dbgs() << "\n!! ExtractInterfaceThis from: " << extractInterfaceThisOp.interfaceVal() << "\n");

        auto vtable = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.interfaceVal(),
                                                            clh.getStructIndexAttr(THIS_VALUE_INDEX));

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

        LLVM_DEBUG(llvm::dbgs() << "\n!! ExtractInterfaceVTable from: " << extractInterfaceVTableOp.interfaceVal()
                                << "\n");

        auto vtable = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.interfaceVal(),
                                                            clh.getStructIndexAttr(DATA_VALUE_INDEX));

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

        auto boundRefType = loadBoundRefOp.reference().getType().cast<mlir_ts::BoundRefType>();

        auto llvmType = tch.convertType(boundRefType.getElementType());
        auto llvmRefType = LLVM::LLVMPointerType::get(llvmType);

        auto thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.reference(),
                                                             clh.getStructIndexAttr(THIS_VALUE_INDEX));
        auto valueRefVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, transformed.reference(),
                                                                 clh.getStructIndexAttr(DATA_VALUE_INDEX));

        mlir::Value loadedValue = rewriter.create<LLVM::LoadOp>(loc, valueRefVal);

        if (auto funcType = boundRefType.getElementType().dyn_cast_or_null<mlir_ts::FunctionType>())
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

        auto boundRefType = storeBoundRefOp.reference().getType().cast<mlir_ts::BoundRefType>();

        auto llvmType = tch.convertType(boundRefType.getElementType());
        auto llvmRefType = LLVM::LLVMPointerType::get(llvmType);

        auto valueRefVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, transformed.reference(),
                                                                 clh.getStructIndexAttr(DATA_VALUE_INDEX));

        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeBoundRefOp, transformed.value(), valueRefVal);
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
        auto structVal2 = rewriter.create<LLVM::InsertValueOp>(loc, structVal, transformed.valueRef(),
                                                               clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, clh.castToI8Ptr(transformed.thisVal()),
                                                               clh.getStructIndexAttr(THIS_VALUE_INDEX));

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
        assert(createBoundFunctionOp.getType().isa<mlir_ts::BoundFunctionType>() ||
               createBoundFunctionOp.getType().isa<mlir_ts::HybridFunctionType>());

        auto llvmBoundFunctionType = tch.convertType(createBoundFunctionOp.getType());

        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateBoundFunction: LLVM Type :" << llvmBoundFunctionType << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateBoundFunction: Func Type :"
                                << tch.convertType(createBoundFunctionOp.func().getType()) << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "\n!! CreateBoundFunction: This Type :" << createBoundFunctionOp.thisVal().getType()
                                << "\n";);

        auto structVal = rewriter.create<mlir_ts::UndefOp>(loc, llvmBoundFunctionType);
        auto structVal2 = rewriter.create<LLVM::InsertValueOp>(loc, structVal, transformed.func(),
                                                               clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, transformed.thisVal(),
                                                               clh.getStructIndexAttr(THIS_VALUE_INDEX));

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

        mlir::Value thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), transformed.boundFunc(),
                                                                    clh.getStructIndexAttr(THIS_VALUE_INDEX));

        auto thisValCasted = rewriter.create<LLVM::BitcastOp>(loc, llvmThisType, thisVal);

        rewriter.replaceOp(getThisOp, {thisValCasted});

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
        CastLogicHelper castLogic(getMethodOp, rewriter, tch);

        auto origType = getMethodOp.boundFunc().getType();

        mlir_ts::FunctionType funcType;
        mlir::Type llvmMethodType;
        if (auto boundType = origType.dyn_cast<mlir_ts::BoundFunctionType>())
        {
            funcType = mlir_ts::FunctionType::get(rewriter.getContext(), boundType.getInputs(), boundType.getResults());
            llvmMethodType = tch.convertType(funcType);
        }
        else if (auto hybridType = origType.dyn_cast<mlir_ts::HybridFunctionType>())
        {
            funcType =
                mlir_ts::FunctionType::get(rewriter.getContext(), hybridType.getInputs(), hybridType.getResults());
            llvmMethodType = tch.convertType(funcType);
        }
        else if (auto structType = origType.dyn_cast<LLVM::LLVMStructType>())
        {
            auto ptrType = structType.getBody().front().cast<LLVM::LLVMPointerType>();
            assert(ptrType.getElementType().isa<LLVM::LLVMFunctionType>());
            llvmMethodType = ptrType;
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "\n!! GetMethodOp: " << getMethodOp << " result type: " << getMethodOp.getType()
                                    << "\n");
            llvm_unreachable("not implemented");
        }

        mlir::Value methodVal = rewriter.create<LLVM::ExtractValueOp>(loc, llvmMethodType, transformed.boundFunc(),
                                                                      clh.getStructIndexAttr(DATA_VALUE_INDEX));

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
        auto typeOfValue = toh.typeOfLogic(typeOfOp->getLoc(), transformed.value(), typeOfOp.value().getType());

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

        LLVM_DEBUG(llvm::dbgs() << "\n!! TypeOf: " << typeOfAnyOp.value() << "\n";);

        AnyLogic al(typeOfAnyOp, rewriter, tch, loc);
        auto typeOfValue = al.typeOfFromAny(transformed.value());

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
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());
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
            for (auto op : stateLabels)
            {
                auto stateLabelOp = dyn_cast_or_null<mlir_ts::StateLabelOp>(op);
                rewriter.setInsertionPoint(stateLabelOp);

                auto *continuationBlock = clh.BeginBlock(loc);

                rewriter.eraseOp(stateLabelOp);

                // add switch
                caseValues.push_back(index++);
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

        rewriter.replaceOpWithNewOp<mlir_ts::StoreOp>(yieldReturnValOp, transformed.operand(), transformed.reference());

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

        auto index = 0;
        for (auto case1 : switchStateOp.cases())
        {
            caseValues.push_back(index++);
            caseDestinations.push_back(case1);
            caseOperands.push_back(ValueRange());
        }

        rewriter.replaceOpWithNewOp<LLVM::SwitchOp>(switchStateOp, transformed.state(), switchStateOp.defaultDest(),
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
        

        TypeHelper th(rewriter.getContext());
        LLVMCodeHelper lch(globalConstructorOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(globalConstructorOp, rewriter);

        mlir::Location loc = globalConstructorOp->getLoc();

        auto parentModule = globalConstructorOp->getParentOfType<ModuleOp>();
        if (!parentModule.lookupSymbol<LLVM::GlobalOp>(GLOBAL_CONSTUCTIONS_NAME))
        {
            SmallVector<mlir_ts::GlobalConstructorOp, 4> globalConstructs;
            auto visitorAllGlobalConstructs = [&](Operation *op) {
                if (auto globalConstrOp = dyn_cast_or_null<mlir_ts::GlobalConstructorOp>(op))
                {
                    globalConstructs.push_back(globalConstrOp);
                }
            };

            globalConstructorOp->getParentOp()->walk(visitorAllGlobalConstructs);

            auto funcType = th.getPointerType(th.getFunctionType(ArrayRef<mlir::Type>{}));

            mlir::SmallVector<mlir::Type, 4> llvmTypes;
            llvmTypes.push_back(th.getI32Type());
            llvmTypes.push_back(funcType);
            llvmTypes.push_back(th.getI8PtrType());
            auto elementType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), llvmTypes, false);

#ifndef ENABLE_MLIR_INIT
            auto size = globalConstructs.size();
#else
            auto size = 1;
#endif
            auto arrayConstType = th.getArrayType(elementType, size);

            // TODO: include initialize block
            lch.createGlobalConstructorIfNew(
                GLOBAL_CONSTUCTIONS_NAME, arrayConstType, LLVM::Linkage::Appending, [&](LLVMCodeHelper *ch) {
                    mlir::Value arrayInstance = rewriter.create<LLVM::UndefOp>(loc, arrayConstType);

#ifndef ENABLE_MLIR_INIT
                    auto index = 0;
                    for (auto globalConstr : llvm::reverse(globalConstructs))
                    {
                        mlir::Value instanceVal = rewriter.create<LLVM::UndefOp>(loc, elementType);

                        auto orderNumber = clh.createI32ConstantOf(65535);

                        ch->setStructValue(loc, instanceVal, orderNumber, 0);

                        auto addrVal = ch->getAddressOfGlobalVar(globalConstr.global_name(), funcType, 0);

                        ch->setStructValue(loc, instanceVal, addrVal, 1);

                        auto nullVal = rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());
                        ch->setStructValue(loc, instanceVal, nullVal, 2);

                        // set array value
                        ch->setStructValue(loc, arrayInstance, instanceVal, index++);
                    }
#else                
                    mlir::Value instanceVal = rewriter.create<LLVM::UndefOp>(loc, elementType);

                    auto orderNumber = clh.createI32ConstantOf(65535);

                    ch->setStructValue(loc, instanceVal, orderNumber, 0);

                    auto addrVal = ch->getAddressOfGlobalVar("__mlir_gctors", funcType, 0);

                    ch->setStructValue(loc, instanceVal, addrVal, 1);

                    auto nullVal = rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());
                    ch->setStructValue(loc, instanceVal, nullVal, 2);

                    // set array value
                    ch->setStructValue(loc, arrayInstance, instanceVal, 0);
#endif

                    auto retVal = rewriter.create<LLVM::ReturnOp>(loc, mlir::ValueRange{arrayInstance});
                });

#ifdef ENABLE_MLIR_INIT
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);

                // create __mlir_runner_init for JIT
                rewriter.setInsertionPointToEnd(parentModule.getBody());
                auto llvmFnType = LLVM::LLVMFunctionType::get(th.getVoidType(), {}, /*isVarArg=*/false);
                auto initFunc = rewriter.create<LLVM::LLVMFuncOp>(loc, "__mlir_gctors", llvmFnType);
                auto &entryBlock = *initFunc.addEntryBlock();
                rewriter.setInsertionPointToEnd(&entryBlock);

                for (auto gctor : llvm::reverse(globalConstructs))
                {
                    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, gctor.global_nameAttr(), ValueRange{});
                }

                rewriter.create<LLVM::ReturnOp>(loc, ValueRange{});
            }
#endif
        }

        rewriter.eraseOp(globalConstructorOp);
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

        mlir::Block *beforeBody = &bodyInternalOp.body().front();
        mlir::Block *afterBody = &bodyInternalOp.body().back();
        rewriter.inlineRegionBefore(bodyInternalOp.body(), continuationBlock);

        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::BrOp>(location, ValueRange(), beforeBody);

        rewriter.setInsertionPointToEnd(afterBody);
        auto bodyResultInternalOp = cast<mlir_ts::BodyResultInternalOp>(afterBody->getTerminator());
        auto branchOp = rewriter.replaceOpWithNewOp<LLVM::BrOp>(bodyResultInternalOp, bodyResultInternalOp.results(),
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
        rewriter.replaceOp(op, op.results());
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
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto i64PtrTy = th.getPointerType(th.getI64Type());

        auto gcMakeDescriptorFunc = ch.getOrInsertFunction("GC_make_descriptor", th.getFunctionType(rewriter.getI64Type(), {i64PtrTy, rewriter.getI64Type()}));
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, gcMakeDescriptorFunc, ValueRange{transformed.typeBitmap(), transformed.sizeOfBitmapInElements()});

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
        

        LLVMCodeHelper ch(op, rewriter, getTypeConverter());
        CodeLogicHelper clh(op, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = op.getLoc();

        mlir::Type storageType = op.instance().getType();

        auto resultType = tch.convertType(op.getType());

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);

        auto i8PtrTy = th.getI8PtrType();

        auto gcMallocExplicitlyTypedFunc = ch.getOrInsertFunction("GC_malloc_explicitly_typed", th.getFunctionType(i8PtrTy, {rewriter.getI64Type(), rewriter.getI64Type()}));
        auto value = rewriter.create<LLVM::CallOp>(loc, gcMallocExplicitlyTypedFunc, ValueRange{sizeOfTypeValue, transformed.typeDescr()});

        rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, resultType, value.getResult(0));

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

static void populateTypeScriptConversionPatterns(LLVMTypeConverter &converter, mlir::ModuleOp &m,
                                                 mlir::SetVector<mlir::Type> &stack)
{
    converter.addConversion(
        [&](mlir_ts::AnyType type) { return LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8)); });

    converter.addConversion(
        [&](mlir_ts::NullType type) { return LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8)); });

    converter.addConversion([&](mlir_ts::OpaqueType type) {
        return LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8));
    });

    converter.addConversion([&](mlir_ts::VoidType type) { return LLVM::LLVMVoidType::get(m.getContext()); });

    converter.addConversion([&](mlir_ts::BooleanType type) {
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
        return LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8));
    });

    converter.addConversion([&](mlir_ts::EnumType type) { return converter.convertType(type.getElementType()); });

    converter.addConversion([&](mlir_ts::ConstArrayType type) {
        return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType()));
    });

    converter.addConversion([&](mlir_ts::ConstArrayValueType type) {
        return LLVM::LLVMArrayType::get(converter.convertType(type.getElementType()), type.getSize());
    });

    converter.addConversion([&](mlir_ts::ArrayType type) {
        TypeHelper th(m.getContext());

        SmallVector<mlir::Type> rtArrayType;
        // pointer to data type
        rtArrayType.push_back(LLVM::LLVMPointerType::get(converter.convertType(type.getElementType())));
        // field which store length of array
        rtArrayType.push_back(th.getI32Type());

        return LLVM::LLVMStructType::getLiteral(type.getContext(), rtArrayType, false);
    });

    converter.addConversion([&](mlir_ts::RefType type) {
        return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType()));
    });

    converter.addConversion([&](mlir_ts::ValueRefType type) {
        return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType()));
    });

    converter.addConversion([&](mlir_ts::ConstTupleType type) {
        SmallVector<mlir::Type> convertedTypes;
        for (auto subType : type.getFields())
        {
            convertedTypes.push_back(converter.convertType(subType.type));
        }

        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, false);
    });

    converter.addConversion([&](mlir_ts::TupleType type) {
        SmallVector<mlir::Type> convertedTypes;
        for (auto subType : type.getFields())
        {
            convertedTypes.push_back(converter.convertType(subType.type));
        }

        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, false);
    });

    converter.addConversion([&](mlir_ts::BoundRefType type) {
        SmallVector<mlir::Type> llvmStructType;
        llvmStructType.push_back(converter.convertType(mlir_ts::RefType::get(type.getElementType())));
        llvmStructType.push_back(LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8)));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
    });

    converter.addConversion([&](mlir_ts::FunctionType type) {
        SmallVector<mlir::Type> convertedInputs;
        for (auto subType : type.getInputs())
        {
            convertedInputs.push_back(converter.convertType(subType));
        }

        SmallVector<mlir::Type> convertedResults;
        for (auto subType : type.getResults())
        {
            convertedResults.push_back(converter.convertType(subType));
        }

        auto funcType = mlir::FunctionType::get(type.getContext(), convertedInputs, convertedResults);

        LLVMTypeConverter::SignatureConversion result(convertedInputs.size());
        auto llvmFuncType = converter.convertFunctionSignature(funcType, false, result);
        auto llvmPtrType = LLVM::LLVMPointerType::get(llvmFuncType);
        return llvmPtrType;
    });

    converter.addConversion([&](mlir_ts::BoundFunctionType type) {
        SmallVector<mlir::Type> convertedInputs;
        for (auto subType : type.getInputs())
        {
            convertedInputs.push_back(converter.convertType(subType));
        }

        SmallVector<mlir::Type> convertedResults;
        for (auto subType : type.getResults())
        {
            convertedResults.push_back(converter.convertType(subType));
        }

        auto funcType = mlir::FunctionType::get(type.getContext(), convertedInputs, convertedResults);

        LLVMTypeConverter::SignatureConversion result(convertedInputs.size());
        auto llvmFuncType = converter.convertFunctionSignature(funcType, false, result);
        auto llvmPtrType = LLVM::LLVMPointerType::get(llvmFuncType);
        // return llvmPtrType;

        SmallVector<mlir::Type> llvmStructType;
        llvmStructType.push_back(llvmPtrType);
        llvmStructType.push_back(LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8)));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
    });

    converter.addConversion([&](mlir_ts::HybridFunctionType type) {
        SmallVector<mlir::Type> convertedInputs;
        for (auto subType : type.getInputs())
        {
            convertedInputs.push_back(converter.convertType(subType));
        }

        SmallVector<mlir::Type> convertedResults;
        for (auto subType : type.getResults())
        {
            convertedResults.push_back(converter.convertType(subType));
        }

        auto funcType = mlir::FunctionType::get(type.getContext(), convertedInputs, convertedResults);

        LLVMTypeConverter::SignatureConversion result(convertedInputs.size());
        auto llvmFuncType = converter.convertFunctionSignature(funcType, false, result);
        auto llvmPtrType = LLVM::LLVMPointerType::get(llvmFuncType);
        // return llvmPtrType;

        SmallVector<mlir::Type> llvmStructType;
        llvmStructType.push_back(llvmPtrType);
        llvmStructType.push_back(LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8)));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
    });

    converter.addConversion([&](mlir_ts::ObjectType type) {
        if (type.getStorageType() == mlir_ts::AnyType::get(type.getContext()))
        {
            return LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8));
        }

        return LLVM::LLVMPointerType::get(converter.convertType(type.getStorageType()));
    });

    converter.addConversion([&](mlir_ts::ObjectStorageType type) {
        auto identStruct = LLVM::LLVMStructType::getIdentified(type.getContext(), type.getName().getValue());
        if (!stack.contains(identStruct))
        {
            stack.insert(identStruct);
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
        return LLVM::LLVMPointerType::get(mlir::IntegerType::get(m.getContext(), 8));
    });

    converter.addConversion([&](mlir_ts::SymbolType type) { return mlir::IntegerType::get(m.getContext(), 32); });

    converter.addConversion([&](mlir_ts::UndefinedType type) { return mlir::IntegerType::get(m.getContext(), 1); });

    converter.addConversion([&](mlir_ts::ClassStorageType type) {
        auto identStruct = LLVM::LLVMStructType::getIdentified(type.getContext(), type.getName().getValue());
        if (!stack.contains(identStruct))
        {
            stack.insert(identStruct);
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
        return LLVM::LLVMPointerType::get(converter.convertType(type.getStorageType()));
    });

    converter.addConversion([&](mlir_ts::InterfaceType type) {
        TypeHelper th(m.getContext());

        SmallVector<mlir::Type> rtInterfaceType;
        // vtable
        rtInterfaceType.push_back(th.getI8PtrType());
        // this
        rtInterfaceType.push_back(th.getI8PtrType());

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

    converter.addConversion([&](mlir_ts::UndefPlaceHolderType type) {
        return mlir::IntegerType::get(m.getContext(), 8 /*, mlir::IntegerType::SignednessSemantics::Unsigned*/);
    });

    converter.addConversion([&](mlir_ts::UnionType type) {
        TypeHelper th(m.getContext());
        LLVMTypeConverterHelper ltch(converter);
        MLIRTypeHelper mth(m.getContext());

        mlir::Type selectedType = ltch.findMaxSizeType(type);
        bool needTag = mth.isUnionTypeNeedsTag(type);

        LLVM_DEBUG(llvm::dbgs() << "\n!! max size type in union: " << selectedType
                                << " size: " << ltch.getTypeSize(selectedType) << " Tag: " << (needTag ? "yes" : "no")
                                << " union type: " << type << "\n";);

        SmallVector<mlir::Type> convertedTypes;
        if (needTag)
        {
            convertedTypes.push_back(th.getI8PtrType());
        }

        convertedTypes.push_back(selectedType);
        if (convertedTypes.size() == 1)
        {
            return convertedTypes.front();
        }

        mlir::Type structType = LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, false);
        return structType;
    });

    converter.addConversion([&](mlir_ts::NeverType type) { return LLVM::LLVMVoidType::get(type.getContext()); });

    converter.addConversion([&](mlir_ts::LiteralType type) { return converter.convertType(type.getElementType()); });

    /*
    converter.addSourceMaterialization(
        [&](OpBuilder &builder, mlir::Type resultType, ValueRange inputs, Location loc) -> Optional<mlir::Value> {
            if (inputs.size() != 1)
                return llvm::None;

            LLVM_DEBUG(llvm::dbgs() << "\n!! Materialization (Source): " << loc << " result type: " << resultType; for (auto inputType : inputs) llvm::dbgs() << "\n <- input: " << inputType;);

            mlir::Value val = builder.create<mlir_ts::DialectCastOp>(loc, resultType, inputs[0]);
            return val;
            //return inputs[0];
        });
    converter.addTargetMaterialization(
        [&](OpBuilder &builder, mlir::Type resultType, ValueRange inputs, Location loc) -> Optional<mlir::Value> {
            if (inputs.size() != 1)
                return llvm::None;

            LLVM_DEBUG(llvm::dbgs() << "\n!! Materialization (Target): " << loc << " result type: " << resultType; for (auto inputType : inputs) llvm::dbgs() << "\n <- input: " << inputType;);

            mlir::Value val = builder.create<mlir_ts::DialectCastOp>(loc, resultType, inputs[0]);
            return val;
            //return inputs[0];
        });
    */
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TypeScriptToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace
{

struct TypeScriptToLLVMLoweringPass : public PassWrapper<TypeScriptToLLVMLoweringPass, OperationPass<ModuleOp>>
{
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeScriptToLLVMLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<
            LLVM::LLVMDialect, 
            mlir::math::MathDialect, 
            mlir::arith::ArithmeticDialect, 
            mlir::cf::ControlFlowDialect, 
            mlir::func::FuncDialect>();            
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
    target.addLegalOp<mlir_ts::GlobalConstructorOp>();

    // During this lowering, we will also be lowering the MemRef types, that are
    // currently being operated on, to a representation in LLVM. To perform this
    // conversion we use a TypeConverter as part of the lowering. This converter
    // details how one type maps to another. This is necessary now that we will be
    // doing more complicated lowerings, involving loop region arguments.
    LLVMTypeConverter typeConverter(&getContext());

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
    arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

#ifdef ENABLE_ASYNC
    populateAsyncStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
#endif

    // The only remaining operation to lower from the `typescript` dialect, is the PrintOp.
    TsLlvmContext tsLlvmContext{};
    patterns.insert<
        AddressOfOpLowering, AddressOfConstStringOpLowering, ArithmeticUnaryOpLowering, ArithmeticBinaryOpLowering,
        AssertOpLowering, CastOpLowering, ConstantOpLowering, CreateOptionalOpLowering, UndefOptionalOpLowering,
        HasValueOpLowering, ValueOpLowering, SymbolRefOpLowering, GlobalOpLowering, GlobalResultOpLowering,
        FuncOpLowering, LoadOpLowering, ElementRefOpLowering, PropertyRefOpLowering, ExtractPropertyOpLowering,
        PointerOffsetRefOpLowering, LogicalBinaryOpLowering, NullOpLowering, NewOpLowering, CreateTupleOpLowering,
        DeconstructTupleOpLowering, CreateArrayOpLowering, NewEmptyArrayOpLowering, NewArrayOpLowering, PushOpLowering,
        PopOpLowering, DeleteOpLowering, ParseFloatOpLowering, ParseIntOpLowering, IsNaNOpLowering, PrintOpLowering,
        StoreOpLowering, SizeOfOpLowering, InsertPropertyOpLowering, LengthOfOpLowering, StringLengthOpLowering,
        StringConcatOpLowering, StringCompareOpLowering, CharToStringOpLowering, UndefOpLowering, MemoryCopyOpLowering,
        LoadSaveValueLowering, ThrowUnwindOpLowering, ThrowCallOpLowering, TrampolineOpLowering, VariableOpLowering,
        AllocaOpLowering, InvokeOpLowering, InvokeHybridOpLowering, VirtualSymbolRefOpLowering,
        ThisVirtualSymbolRefOpLowering, InterfaceSymbolRefOpLowering, NewInterfaceOpLowering, VTableOffsetRefOpLowering,
        LoadBoundRefOpLowering, StoreBoundRefOpLowering, CreateBoundRefOpLowering, CreateBoundFunctionOpLowering,
        GetThisOpLowering, GetMethodOpLowering, TypeOfOpLowering, TypeOfAnyOpLowering, DebuggerOpLowering,
        UnreachableOpLowering, LandingPadOpLowering, CompareCatchTypeOpLowering, BeginCatchOpLowering,
        SaveCatchVarOpLowering, EndCatchOpLowering, BeginCleanupOpLowering, EndCleanupOpLowering,
        SymbolCallInternalOpLowering, CallInternalOpLowering, CallHybridInternalOpLowering, ReturnInternalOpLowering,
        NoOpLowering,
        /*GlobalConstructorOpLowering,*/ ExtractInterfaceThisOpLowering, ExtractInterfaceVTableOpLowering,
        BoxOpLowering, UnboxOpLowering, DialectCastOpLowering, CreateUnionInstanceOpLowering,
        GetValueFromUnionOpLowering, GetTypeInfoFromUnionOpLowering, BodyInternalOpLowering,
        BodyResultInternalOpLowering
#ifndef DISABLE_SWITCH_STATE_PASS
        ,
        SwitchStateOpLowering, StateLabelOpLowering, YieldReturnValOpLowering
#endif
        ,
        SwitchStateInternalOpLowering>(typeConverter, &getContext(), &tsLlvmContext);

#ifdef ENABLE_TYPED_GC
    patterns.insert<
        GCMakeDescriptorOpLowering, GCNewExplicitlyTypedOpLowering>(typeConverter, &getContext(), &tsLlvmContext);
#endif        

    mlir::SetVector<mlir::Type> stack;
    populateTypeScriptConversionPatterns(typeConverter, m, stack);

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "\n!! BEFORE DUMP: \n" << module << "\n";);

    if (failed(applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
    }

    LLVMConversionTarget target2(getContext());
    target2.addLegalOp<ModuleOp>();

    RewritePatternSet patterns2(&getContext());
    patterns2.insert<GlobalConstructorOpLowering, DialectCastOpLowering>(typeConverter, &getContext(), &tsLlvmContext);

    if (failed(applyFullConversion(module, target2, std::move(patterns2))))
    {
        signalPassFailure();
    }

    /*
    LLVM_DEBUG(llvm::dbgs() << "\n!! AFTER DUMP - BEFORE CLEANUP: \n" << module << "\n";);
    */

    cleanupUnrealizedConversionCast(module);

    LLVM_DEBUG(llvm::dbgs() << "\n!! AFTER DUMP: \n" << module << "\n";);

    LLVM_DEBUG(verifyModule(module););
}

/// Create a pass for lowering operations the remaining `TypeScript` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::typescript::createLowerToLLVMPass()
{
    return std::make_unique<TypeScriptToLLVMLoweringPass>();
}
