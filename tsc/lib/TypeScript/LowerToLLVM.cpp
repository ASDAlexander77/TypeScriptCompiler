//#define ALLOC_ALL_VARS_IN_HEAP 1
//#define ALLOC_CAPTURED_VARS_IN_HEAP 1
//#define ALLOC_CAPTURE_IN_HEAP 1
// TODO: if I uncomment it, it will create errors in capture vars. calls. find out why? (wrong size of buffers?)
//#define ALLOC_TRAMPOLINE_IN_HEAP 1
#define DEBUG_TYPE "llvm"

#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
    // invoke normal, unwind
    DenseMap<Operation *, mlir::Value> catchOpData;
};

template <typename OpTy> class TsLlvmPattern : public OpConversionPattern<OpTy>
{
  public:
    TsLlvmPattern<OpTy>(mlir::LLVMTypeConverter &llvmTypeConverter, MLIRContext *context, TsLlvmContext *tsLlvmContext,
                        PatternBenefit benefit = 1)
        : OpConversionPattern<OpTy>::OpConversionPattern(llvmTypeConverter, context, benefit), tsLlvmContext(tsLlvmContext)
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

    LogicalResult matchAndRewrite(mlir_ts::PrintOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        // Get a symbol reference to the printf function, inserting it if necessary.
        auto printfFuncOp = ch.getOrInsertFunction("printf", th.getFunctionType(rewriter.getI32Type(), th.getI8PtrType(), true));

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

        for (auto item : op->getOperands())
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
                values.push_back(rewriter.create<LLVM::SelectOp>(item.getLoc(), item,
                                                                 ch.getOrCreateGlobalString("__true__", std::string("true")),
                                                                 ch.getOrCreateGlobalString("__false__", std::string("false"))));
            }
            else if (auto o = type.dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                auto boolPart = rewriter.create<mlir_ts::HasValueOp>(item.getLoc(), th.getBooleanType(), item);
                values.push_back(rewriter.create<LLVM::SelectOp>(item.getLoc(), boolPart,
                                                                 ch.getOrCreateGlobalString("__true__", std::string("true")),
                                                                 ch.getOrCreateGlobalString("__false__", std::string("false"))));
                auto optVal = rewriter.create<mlir_ts::ValueOp>(item.getLoc(), o.getElementType(), item);
                fval(optVal.getType(), optVal);
            }
            else
            {
                values.push_back(item);
            }
        };

        for (auto item : op->getOperands())
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

    LogicalResult matchAndRewrite(mlir_ts::AssertOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        AssertLogic al(op, rewriter, tch, op->getLoc());
        return al.logic(op.arg(), op.msg().str());
    }
};
class PrintOpLowering : public TsLlvmPattern<mlir_ts::PrintOp>
{
  public:
    using TsLlvmPattern<mlir_ts::PrintOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PrintOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        CastLogicHelper castLogic(op, rewriter, tch);

        auto loc = op->getLoc();

        // Get a symbol reference to the printf function, inserting it if necessary.
        auto putsFuncOp = ch.getOrInsertFunction("puts", th.getFunctionType(rewriter.getI32Type(), th.getI8PtrType(), false));

        auto strType = mlir_ts::StringType::get(rewriter.getContext());

        SmallVector<mlir::Value> values;
        mlir::Value spaceString;
        for (auto item : op->getOperands())
        {
            auto result = castLogic.cast(item, strType);
            if (!result)
            {
                return failure();
            }

            if (values.size() > 0)
            {
                if (!spaceString)
                {
                    spaceString = rewriter.create<mlir_ts::ConstantOp>(loc, strType, rewriter.getStringAttr(" "));
                }

                values.push_back(spaceString);
            }

            values.push_back(result);
        }

        if (values.size() > 1)
        {
            mlir::Value result = rewriter.create<mlir_ts::StringConcatOp>(loc, strType, values, rewriter.getBoolAttr(true));

            rewriter.create<LLVM::CallOp>(loc, putsFuncOp, result);
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

    LogicalResult matchAndRewrite(mlir_ts::ParseIntOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        // Insert the `atoi` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        auto parseIntFuncOp = ch.getOrInsertFunction("atoi", th.getFunctionType(rewriter.getI32Type(), {i8PtrTy}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, parseIntFuncOp, op->getOperands());

        return success();
    }
};

class ParseFloatOpLowering : public TsLlvmPattern<mlir_ts::ParseFloatOp>
{
  public:
    using TsLlvmPattern<mlir_ts::ParseFloatOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ParseFloatOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto loc = op->getLoc();

        // Insert the `atof` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        auto parseFloatFuncOp = ch.getOrInsertFunction("atof", th.getFunctionType(rewriter.getF64Type(), {i8PtrTy}));

        auto funcCall = rewriter.create<LLVM::CallOp>(loc, parseFloatFuncOp, operands);

        rewriter.replaceOpWithNewOp<LLVM::FPTruncOp>(op, rewriter.getF32Type(), funcCall.getResult(0));

        return success();
    }
};

class SizeOfOpLowering : public TsLlvmPattern<mlir_ts::SizeOfOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SizeOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SizeOfOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto loc = op->getLoc();

        auto storageType = op.type();
        auto llvmStorageType = tch.convertType(op.type());
        auto llvmStorageTypePtr = LLVM::LLVMPointerType::get(llvmStorageType);
        auto nullPtrToTypeValue = rewriter.create<LLVM::NullOp>(loc, llvmStorageTypePtr);

        LLVM_DEBUG(llvm::dbgs() << "size of - storage type: [" << storageType << "] llvm storage type: [" << llvmStorageType
                                << "] llvm ptr: [" << llvmStorageTypePtr << "] value: [" << nullPtrToTypeValue << "]\n";);

        auto cst1 = rewriter.create<LLVM::ConstantOp>(loc, th.getI64Type(), th.getIndexAttrValue(1));
        auto sizeOfSetAddr = rewriter.create<LLVM::GEPOp>(loc, llvmStorageTypePtr, nullPtrToTypeValue, ArrayRef<Value>({cst1}));

        rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, th.getIndexType(), sizeOfSetAddr);

        return success();
    }
};

class LengthOfOpLowering : public TsLlvmPattern<mlir_ts::LengthOfOp>
{
  public:
    using TsLlvmPattern<mlir_ts::LengthOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LengthOfOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        auto loc = op->getLoc();

        rewriter.replaceOpWithNewOp<mlir::LLVM::ExtractValueOp>(op, th.getI32Type(), op.op(),
                                                                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

        return success();
    }
};

class StringLengthOpLowering : public TsLlvmPattern<mlir_ts::StringLengthOp>
{
  public:
    using TsLlvmPattern<mlir_ts::StringLengthOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StringLengthOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto loc = op->getLoc();
        auto i8PtrTy = th.getI8PtrType();

        auto strlenFuncOp = ch.getOrInsertFunction("strlen", th.getFunctionType(th.getI64Type(), {i8PtrTy}));

        // calc size
        auto size = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, ValueRange{op.op()});
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, th.getI32Type(), size.getResult(0));

        return success();
    }
};

class StringConcatOpLowering : public TsLlvmPattern<mlir_ts::StringConcatOp>
{
  public:
    using TsLlvmPattern<mlir_ts::StringConcatOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StringConcatOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
        for (auto op : op.ops())
        {
            auto size1 = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, op);
            size = rewriter.create<LLVM::AddOp>(loc, rewriter.getI64Type(), ValueRange{size, size1.getResult(0)});
        }

        auto allocInStack = op.allocInStack().hasValue() && op.allocInStack().getValue();

        mlir::Value newStringValue =
            allocInStack ? rewriter.create<LLVM::AllocaOp>(op->getLoc(), i8PtrTy, size, true) : ch.MemoryAllocBitcast(i8PtrTy, size);

        // copy
        auto concat = false;
        auto result = newStringValue;
        for (auto op : op.ops())
        {
            if (concat)
            {
                auto callResult = rewriter.create<LLVM::CallOp>(loc, strcatFuncOp, ValueRange{result, op});
                result = callResult.getResult(0);
            }
            else
            {
                auto callResult = rewriter.create<LLVM::CallOp>(loc, strcpyFuncOp, ValueRange{result, op});
                result = callResult.getResult(0);
            }

            concat = true;
        }

        rewriter.replaceOp(op, ValueRange{result});

        return success();
    }
};

class StringCompareOpLowering : public TsLlvmPattern<mlir_ts::StringCompareOp>
{
  public:
    using TsLlvmPattern<mlir_ts::StringCompareOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StringCompareOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
        auto leftPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, op.op1());
        auto rightPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, op.op2());
        auto ptrCmpResult1 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, leftPtrValue, const0);
        auto ptrCmpResult2 = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, rightPtrValue, const0);
        auto cmp32Result1 = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), ptrCmpResult1);
        auto cmp32Result2 = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), ptrCmpResult2);
        auto cmpResult = rewriter.create<LLVM::AndOp>(loc, cmp32Result1, cmp32Result2);
        auto const0I32 = clh.createI32ConstantOf(0);
        auto ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, cmpResult, const0I32);

        auto result = clh.conditionalExpressionLowering(
            th.getBooleanType(), ptrCmpResult,
            [&](OpBuilder &builder, Location loc) {
                // both not null
                auto const0 = clh.createI32ConstantOf(0);
                auto compareResult = rewriter.create<LLVM::CallOp>(loc, strcmpFuncOp, ValueRange{op.op1(), op.op2()});

                // else compare body
                mlir::Value bodyCmpResult;
                switch ((SyntaxKind)op.code())
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::GreaterThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::LessThanToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, compareResult.getResult(0), const0);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    bodyCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, compareResult.getResult(0), const0);
                    break;
                default:
                    llvm_unreachable("not implemented");
                }

                return bodyCmpResult;
            },
            [&](OpBuilder &builder, Location loc) {
                // any 1 null
                auto leftPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, op.op1());
                auto rightPtrValue = rewriter.create<LLVM::PtrToIntOp>(loc, intPtrType, op.op2());

                // else compare body
                mlir::Value ptrCmpResult;
                switch ((SyntaxKind)op.code())
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::GreaterThanToken:
                    ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::LessThanToken:
                    ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, leftPtrValue, rightPtrValue);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    ptrCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, leftPtrValue, rightPtrValue);
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

    LogicalResult matchAndRewrite(mlir_ts::CharToStringOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
        auto addr0 = ch.GetAddressOfArrayElement(charRefType, newStringValue, index0Value);
        rewriter.create<LLVM::StoreOp>(loc, op.op(), addr0);
        auto addr1 = ch.GetAddressOfArrayElement(charRefType, newStringValue, index1Value);
        rewriter.create<LLVM::StoreOp>(loc, nullCharValue, addr1);

        rewriter.replaceOp(op, ValueRange{newStringValue});

        return success();
    }
};

struct ConstantOpLowering : public TsLlvmPattern<mlir_ts::ConstantOp>
{
    using TsLlvmPattern<mlir_ts::ConstantOp>::TsLlvmPattern;

    template <typename T, typename TOp> void getOrCreateGlobalArray(TOp constantOp, T type, ConversionPatternRewriter &rewriter) const
    {
        LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto elementType = type.template cast<T>().getElementType();
        auto llvmElementType = tch.convertType(elementType);
        auto arrayAttr = constantOp.value().template dyn_cast_or_null<ArrayAttr>();

        auto arrayFirstElementAddrCst = ch.getOrCreateGlobalArray(elementType, llvmElementType, arrayAttr.size(), arrayAttr);

        rewriter.replaceOp(constantOp, arrayFirstElementAddrCst);
    }

    template <typename T, typename TOp> void getOrCreateGlobalTuple(TOp constantOp, T type, ConversionPatternRewriter &rewriter) const
    {
        LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto arrayAttr = constantOp.value().template dyn_cast_or_null<ArrayAttr>();

        auto convertedTupleType = tch.convertType(type);
        auto tupleConstPtr = ch.getOrCreateGlobalTuple(type.template cast<mlir_ts::ConstTupleType>(),
                                                       convertedTupleType.template cast<LLVM::LLVMStructType>(), arrayAttr);

        // optimize it and replace it with copy memory. (use canon. pass) check  "EraseRedundantAssertions"
        auto loadedValue = rewriter.create<LLVM::LoadOp>(constantOp->getLoc(), tupleConstPtr);

        rewriter.replaceOp(constantOp, ValueRange{loadedValue});
    }

    LogicalResult matchAndRewrite(mlir_ts::ConstantOp constantOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        // load address of const string
        auto type = constantOp.getType();
        if (type.isa<mlir_ts::StringType>())
        {
            LLVMCodeHelper ch(constantOp, rewriter, getTypeConverter());

            auto strValue = constantOp.value().cast<StringAttr>().getValue().str();
            auto txtCst = ch.getOrCreateGlobalString(strValue);

            rewriter.replaceOp(constantOp, txtCst);

            return success();
        }

        TypeConverterHelper tch(getTypeConverter());
        if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            getOrCreateGlobalArray(constantOp, constArrayType, rewriter);
            return success();
        }

        if (auto arrayType = type.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            getOrCreateGlobalArray(constantOp, arrayType, rewriter);
            return success();
        }

        if (auto constTupleType = type.dyn_cast_or_null<mlir_ts::ConstTupleType>())
        {
            getOrCreateGlobalTuple(constantOp, constTupleType, rewriter);
            return success();
        }

        if (auto tupleType = type.dyn_cast_or_null<mlir_ts::TupleType>())
        {
            getOrCreateGlobalTuple(constantOp, tupleType, rewriter);
            return success();
        }

        if (auto enumType = type.dyn_cast_or_null<mlir_ts::EnumType>())
        {
            rewriter.eraseOp(constantOp);
            return success();
        }

        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(constantOp, tch.convertType(type), constantOp.getValue());
        return success();
    }
};

struct SymbolRefOpLowering : public TsLlvmPattern<mlir_ts::SymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::SymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SymbolRefOp symbolRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        rewriter.replaceOpWithNewOp<mlir::ConstantOp>(symbolRefOp, tch.convertType(symbolRefOp.getType()), symbolRefOp.getValue());
        return success();
    }
};

struct NullOpLowering : public TsLlvmPattern<mlir_ts::NullOp>
{
    using TsLlvmPattern<mlir_ts::NullOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NullOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

    LogicalResult matchAndRewrite(mlir_ts::UndefOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

struct ReturnOpLowering : public TsLlvmPattern<mlir_ts::ReturnOp>
{
    using TsLlvmPattern<mlir_ts::ReturnOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ReturnOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(op, rewriter);

        auto retBlock = clh.FindReturnBlock();

        // Split block at `assert` operation.
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.create<mlir::BranchOp>(op.getLoc(), retBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct ReturnValOpLowering : public TsLlvmPattern<mlir_ts::ReturnValOp>
{
    using TsLlvmPattern<mlir_ts::ReturnValOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ReturnValOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(op, rewriter);

        auto retBlock = clh.FindReturnBlock();

        rewriter.create<LLVM::StoreOp>(op.getLoc(), op.operand(), op.reference());

        // Split block at `assert` operation.
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        // save value into return

        rewriter.create<mlir::BranchOp>(op.getLoc(), retBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct FuncOpLowering : public TsLlvmPattern<mlir_ts::FuncOp>
{
    using TsLlvmPattern<mlir_ts::FuncOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::FuncOp funcOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto location = funcOp->getLoc();

        auto &typeConverter = *getTypeConverter();
        auto fnType = funcOp.getType();

        TypeConverter::SignatureConversion signatureInputsConverter(fnType.getNumInputs());
        for (auto argType : enumerate(funcOp.getType().getInputs()))
        {
            auto convertedType = typeConverter.convertType(argType.value());
            signatureInputsConverter.addInputs(argType.index(), convertedType);
        }

        TypeConverter::SignatureConversion signatureResultsConverter(fnType.getNumResults());
        for (auto argType : enumerate(funcOp.getType().getResults()))
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
        auto newFuncOp = rewriter.create<mlir::FuncOp>(
            funcOp.getLoc(), funcOp.getName(),
            rewriter.getFunctionType(signatureInputsConverter.getConvertedTypes(), signatureResultsConverter.getConvertedTypes()),
            ArrayRef<NamedAttribute>{}, argDictAttrs);
        for (const auto &namedAttr : funcOp->getAttrs())
        {
            if (namedAttr.first == function_like_impl::getTypeAttrName())
            {
                continue;
            }

            if (namedAttr.first == SymbolTable::getSymbolAttrName())
            {
                name = namedAttr.second.dyn_cast_or_null<mlir::StringAttr>().getValue().str();
                continue;
            }

            newFuncOp->setAttr(namedAttr.first, namedAttr.second);
        }

        if (funcOp.personality().hasValue() && funcOp.personality().getValue())
        {
#if WIN_EXCEPTION
            LLVMRTTIHelperVCWin32 rttih(funcOp, rewriter, typeConverter);
#else
            LLVMRTTIHelperVCLinux rttih(funcOp, rewriter, typeConverter);
#endif
            rttih.setPersonality(newFuncOp);
        }

#ifdef DISABLE_OPT
        // add LLVM attributes to fix issue with shift >> 32
        newFuncOp->setAttr("passthrough",
                           ArrayAttr::get(rewriter.getContext(), {
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

        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter, &signatureInputsConverter)))
        {
            return failure();
        }

        /*
                if (name == "main")
                {
                    rewriter.setInsertionPointToStart(&newFuncOp.getBody().front());

                    TypeHelper th(rewriter);
                    LLVMCodeHelper ch(funcOp, rewriter, getTypeConverter());
                    auto i8PtrTy = th.getI8PtrType();
                    auto gcInitFuncOp = ch.getOrInsertFunction("GC_init", th.getFunctionType(th.getVoidType(),
           mlir::ArrayRef<mlir::Type>{})); rewriter.create<LLVM::CallOp>(location, gcInitFuncOp, ValueRange{});
                }
        */

        rewriter.eraseOp(funcOp);

        return success();
    }
};

struct CallInternalOpLowering : public TsLlvmPattern<mlir_ts::CallInternalOp>
{
    using TsLlvmPattern<mlir_ts::CallInternalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallInternalOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            llvmTypes.push_back(tch.convertType(type));
        }

        // just replace
        if (op.callee().hasValue())
        {
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, llvmTypes, op.calleeAttr(), op.getArgOperands());
        }
        else
        {
            rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, llvmTypes, op.getOperands());
        }

        return success();
    }
};

struct InvokeOpLowering : public TsLlvmPattern<mlir_ts::InvokeOp>
{
    using TsLlvmPattern<mlir_ts::InvokeOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InvokeOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        SmallVector<mlir::Type> llvmTypes;
        for (auto type : op.getResultTypes())
        {
            llvmTypes.push_back(tch.convertType(type));
        }

        // just replace
        rewriter.replaceOpWithNewOp<LLVM::InvokeOp>(op, llvmTypes, op.calleeAttr(), op.getOperands(), op.normalDest(),
                                                    op.normalDestOperands(), op.unwindDest(), op.unwindDestOperands());
        return success();
    }
};

struct CastOpLowering : public TsLlvmPattern<mlir_ts::CastOp>
{
    using TsLlvmPattern<mlir_ts::CastOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CastOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto loc = op->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto in = op.in();
        auto resType = op.res().getType();

        CastLogicHelper castLogic(op, rewriter, tch);
        auto result = castLogic.cast(in, resType);
        if (!result)
        {
            return failure();
        }

        rewriter.replaceOp(op, result);

        return success();
    }
};

struct VariableOpLowering : public TsLlvmPattern<mlir_ts::VariableOp>
{
    using TsLlvmPattern<mlir_ts::VariableOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::VariableOp varOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(varOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(varOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto location = varOp.getLoc();

        auto referenceType = varOp.reference().getType().dyn_cast_or_null<mlir_ts::RefType>();
        auto storageType = referenceType.getElementType();
        auto llvmReferenceType = tch.convertType(referenceType);

#ifdef ALLOC_ALL_VARS_IN_HEAP
        auto isCaptured = true;
#elif ALLOC_CAPTURED_VARS_IN_HEAP
        auto isCaptured = varOp.captured().hasValue() && varOp.captured().getValue();
#else
        auto isCaptured = false;
#endif

        LLVM_DEBUG(llvm::dbgs() << ">>> variable allocation: " << storageType << " is captured: " << isCaptured << "\n";);

        auto allocated = isCaptured ? ch.MemoryAllocBitcast(llvmReferenceType, storageType)
                                    : rewriter.create<LLVM::AllocaOp>(location, llvmReferenceType, clh.createI32ConstantOf(1));

#ifdef GC_ENABLE
        // register root which is in stack, if you call Malloc - it is not in stack anymore
        if (!isCaptured)
        {
            if (storageType.isa<mlir_ts::ClassType>() || storageType.isa<mlir_ts::StringType>() || storageType.isa<mlir_ts::ArrayType>() ||
                storageType.isa<mlir_ts::ObjectType>() || storageType.isa<mlir_ts::AnyType>())
            {
                if (auto ptrType = llvmReferenceType.dyn_cast_or_null<LLVM::LLVMPointerType>())
                {
                    if (ptrType.getElementType().isa<LLVM::LLVMPointerType>())
                    {
                        TypeHelper th(rewriter);

                        auto i8PtrPtrTy = th.getI8PtrPtrType();
                        auto i8PtrTy = th.getI8PtrType();
                        auto gcRootOp = ch.getOrInsertFunction("llvm.gcroot", th.getFunctionType(th.getVoidType(), {i8PtrPtrTy, i8PtrTy}));
                        auto nullPtr = rewriter.create<LLVM::NullOp>(location, i8PtrTy);
                        rewriter.create<LLVM::CallOp>(location, gcRootOp, ValueRange{clh.castToI8PtrPtr(allocated), nullPtr});
                    }
                }
            }
        }
#endif

        auto value = varOp.initializer();
        if (value)
        {
            rewriter.create<LLVM::StoreOp>(location, value, allocated);
        }

        rewriter.replaceOp(varOp, ValueRange{allocated});
        return success();
    }
};

struct NewOpLowering : public TsLlvmPattern<mlir_ts::NewOp>
{
    using TsLlvmPattern<mlir_ts::NewOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewOp newOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(newOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(newOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newOp.getLoc();

        mlir::Type storageType;
        TypeSwitch<Type>(newOp.getType())
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

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

    LogicalResult matchAndRewrite(mlir_ts::CreateTupleOp createTupleOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(createTupleOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(createTupleOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = createTupleOp.getLoc();
        auto tupleType = createTupleOp.getType().cast<mlir_ts::TupleType>();

        auto tupleVar =
            rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(tupleType), mlir::Value(), rewriter.getBoolAttr(false));

        // set values here
        mlir::Value zero = clh.createIndexConstantOf(0);
        auto index = 0;
        for (auto item : createTupleOp.items())
        {
            mlir::Value fieldIndex = clh.createStructIndexConstantOf(index);
            auto llvmValueType = tch.convertType(item.getType());
            auto llvmValuePtrType = LLVM::LLVMPointerType::get(llvmValueType);
            auto offset = rewriter.create<LLVM::GEPOp>(loc, llvmValuePtrType, tupleVar, ValueRange{zero, fieldIndex});

            // cast item if needed
            auto destItemType = tupleType.getFields()[index].type;
            auto llvmDestValueType = tch.convertType(destItemType);
            if (llvmDestValueType != llvmValueType)
            {
                CastLogicHelper castLogic(createTupleOp, rewriter, tch);
                item = castLogic.cast(item, llvmValueType, destItemType, llvmDestValueType);
                if (!item)
                {
                    return failure();
                }
            }

            rewriter.create<LLVM::StoreOp>(loc, item, offset);

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

    LogicalResult matchAndRewrite(mlir_ts::DeconstructTupleOp deconstructTupleOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(deconstructTupleOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(deconstructTupleOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = deconstructTupleOp.getLoc();
        auto tupleVar = deconstructTupleOp.instance();
        auto tupleType = tupleVar.getType().cast<mlir_ts::TupleType>();

        // values
        SmallVector<mlir::Value> results;

        // set values here
        auto index = 0;
        for (auto &item : tupleType.getFields())
        {
            auto llvmValueType = tch.convertType(item.type);
            auto value = rewriter.create<LLVM::ExtractValueOp>(loc, llvmValueType, tupleVar, clh.getStructIndexAttr(index));

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

    LogicalResult matchAndRewrite(mlir_ts::CreateArrayOp createArrayOp, ArrayRef<Value> operands,
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
        TypeSwitch<Type>(elementType)
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        auto newCountAsIndexType = clh.createIndexConstantOf(createArrayOp.items().size());

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto multSizeOfTypeValue = rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryAllocBitcast(llvmPtrElementType, multSizeOfTypeValue);

        mlir::Value index = clh.createIndexConstantOf(0);
        auto next = false;
        mlir::Value value1;
        for (auto item : createArrayOp.items())
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
            if (elementType != item.getType())
            {
                effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, elementType, item);
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated, clh.getStructIndexAttr(0));

        auto newCountAsI32Type = clh.createI32ConstantOf(createArrayOp.items().size());
        auto structValue3 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, newCountAsI32Type, clh.getStructIndexAttr(1));

        rewriter.replaceOp(createArrayOp, ValueRange{structValue3});
        return success();
    }
};

struct NewEmptyArrayOpLowering : public TsLlvmPattern<mlir_ts::NewEmptyArrayOp>
{
    using TsLlvmPattern<mlir_ts::NewEmptyArrayOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewEmptyArrayOp newEmptyArrOp, ArrayRef<Value> operands,
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
        TypeSwitch<Type>(elementType)
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        auto allocated = rewriter.create<LLVM::NullOp>(loc, llvmPtrElementType);

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated, clh.getStructIndexAttr(0));

        auto size0 = clh.createI32ConstantOf(0);
        auto structValue3 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, size0, clh.getStructIndexAttr(1));

        rewriter.replaceOp(newEmptyArrOp, ValueRange{structValue3});
        return success();
    }
};

struct NewArrayOpLowering : public TsLlvmPattern<mlir_ts::NewArrayOp>
{
    using TsLlvmPattern<mlir_ts::NewArrayOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewArrayOp newArrOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(newArrOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(newArrOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = newArrOp.getLoc();

        auto arrayType = newArrOp.getType();
        auto elementType = arrayType.getElementType();

        mlir::Type storageType;
        TypeSwitch<Type>(elementType)
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto llvmElementType = tch.convertType(elementType);
        auto llvmPtrElementType = th.getPointerType(llvmElementType);

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto countAsIndexType = rewriter.create<mlir_ts::CastOp>(loc, th.getIndexType(), newArrOp.count());
        auto multSizeOfTypeValue = rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, countAsIndexType});

        auto allocated = ch.MemoryAllocBitcast(llvmPtrElementType, multSizeOfTypeValue);

        // create array type
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto structValue2 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, allocated, clh.getStructIndexAttr(0));

        auto structValue3 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, newArrOp.count(), clh.getStructIndexAttr(1));

        rewriter.replaceOp(newArrOp, ValueRange{structValue3});
        return success();
    }
};

struct PushOpLowering : public TsLlvmPattern<mlir_ts::PushOp>
{
    using TsLlvmPattern<mlir_ts::PushOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PushOp pushOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(llvmPtrElementType), pushOp.op(), ValueRange{ind0, ind0});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, llvmPtrElementType, currentPtrPtr);

        auto ind1 = clh.createI32ConstantOf(1);
        auto countAsI32TypePtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(th.getI32Type()), pushOp.op(), ValueRange{ind0, ind1});
        auto countAsI32Type = rewriter.create<LLVM::LoadOp>(loc, th.getI32Type(), countAsI32TypePtr);

        auto countAsIndexType = rewriter.create<ZeroExtendIOp>(loc, countAsI32Type, th.getIndexType());

        auto incSize = clh.createIndexConstantOf(pushOp.items().size());
        auto newCountAsIndexType = rewriter.create<LLVM::AddOp>(loc, th.getIndexType(), ValueRange{countAsIndexType, incSize});

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), elementType);
        auto multSizeOfTypeValue = rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryReallocBitcast(llvmPtrElementType, currentPtr, multSizeOfTypeValue);

        mlir::Value index = countAsIndexType;
        auto next = false;
        mlir::Value value1;
        for (auto item : pushOp.items())
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
            if (elementType != item.getType())
            {
                effectiveItem = rewriter.create<mlir_ts::CastOp>(loc, elementType, item);
            }

            auto save = rewriter.create<LLVM::StoreOp>(loc, effectiveItem, offset);
            next = true;
        }

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);

        auto newCountAsI32Type = rewriter.create<TruncateIOp>(loc, newCountAsIndexType, th.getI32Type());

        rewriter.create<LLVM::StoreOp>(loc, newCountAsI32Type, countAsI32TypePtr);

        rewriter.replaceOp(pushOp, ValueRange{newCountAsIndexType});
        return success();
    }
};

struct PopOpLowering : public TsLlvmPattern<mlir_ts::PopOp>
{
    using TsLlvmPattern<mlir_ts::PopOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PopOp popOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
        TypeSwitch<Type>(popOp.op().getType())
            .Case<mlir_ts::ClassType>([&](auto classType) { storageType = classType.getStorageType(); })
            .Case<mlir_ts::ValueRefType>([&](auto valueRefType) { storageType = valueRefType.getElementType(); })
            .Default([&](auto type) { storageType = type; });

        auto ind0 = clh.createI32ConstantOf(0);
        auto currentPtrPtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(llvmPtrElementType), popOp.op(), ValueRange{ind0, ind0});
        auto currentPtr = rewriter.create<LLVM::LoadOp>(loc, llvmPtrElementType, currentPtrPtr);

        auto ind1 = clh.createI32ConstantOf(1);
        auto countAsI32TypePtr = rewriter.create<LLVM::GEPOp>(loc, th.getPointerType(th.getI32Type()), popOp.op(), ValueRange{ind0, ind1});
        auto countAsI32Type = rewriter.create<LLVM::LoadOp>(loc, th.getI32Type(), countAsI32TypePtr);

        auto countAsIndexType = rewriter.create<ZeroExtendIOp>(loc, countAsI32Type, th.getIndexType());

        auto incSize = clh.createIndexConstantOf(1);
        auto newCountAsIndexType = rewriter.create<LLVM::SubOp>(loc, th.getIndexType(), ValueRange{countAsIndexType, incSize});

        // load last element
        auto offset = rewriter.create<LLVM::GEPOp>(loc, llvmPtrElementType, currentPtr, ValueRange{newCountAsIndexType});
        auto loadedElement = rewriter.create<LLVM::LoadOp>(loc, llvmElementType, offset);

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        auto multSizeOfTypeValue = rewriter.create<LLVM::MulOp>(loc, th.getIndexType(), ValueRange{sizeOfTypeValue, newCountAsIndexType});

        auto allocated = ch.MemoryReallocBitcast(llvmPtrElementType, currentPtr, multSizeOfTypeValue);

        rewriter.create<LLVM::StoreOp>(loc, allocated, currentPtrPtr);

        auto newCountAsI32Type = rewriter.create<TruncateIOp>(loc, newCountAsIndexType, th.getI32Type());

        rewriter.create<LLVM::StoreOp>(loc, newCountAsI32Type, countAsI32TypePtr);

        rewriter.replaceOp(popOp, ValueRange{loadedElement});
        return success();
    }
};

struct DeleteOpLowering : public TsLlvmPattern<mlir_ts::DeleteOp>
{
    using TsLlvmPattern<mlir_ts::DeleteOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DeleteOp deleteOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(deleteOp, rewriter, getTypeConverter());

        if (mlir::failed(ch.MemoryFree(deleteOp.reference())))
        {
            return mlir::failure();
        }

        rewriter.eraseOp(deleteOp);
        return mlir::success();
    }
};

void NegativeOpValue(mlir_ts::ArithmeticUnaryOp &unaryOp, mlir::PatternRewriter &builder)
{
    CodeLogicHelper clh(unaryOp, builder);

    auto oper = unaryOp.operand1();
    auto type = oper.getType();
    if (type.isIntOrIndex())
    {
        builder.replaceOpWithNewOp<SubIOp>(unaryOp, type, clh.createIConstantOf(type.getIntOrFloatBitWidth(), 0), oper);
    }
    else if (!type.isIntOrIndex() && type.isIntOrIndexOrFloat())
    {
        builder.replaceOpWithNewOp<SubFOp>(unaryOp, type, clh.createFConstantOf(type.getIntOrFloatBitWidth(), 0.0), oper);
    }
    else
    {
        llvm_unreachable("not implemented");
    }
}

void NegativeOpBin(mlir_ts::ArithmeticUnaryOp &unaryOp, mlir::PatternRewriter &builder, TypeConverter *typeConverter)
{
    CodeLogicHelper clh(unaryOp, builder);
    TypeConverterHelper tch(typeConverter);

    auto oper = unaryOp.operand1();
    auto type = tch.convertType(oper.getType());
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

    LogicalResult matchAndRewrite(mlir_ts::ArithmeticUnaryOp arithmeticUnaryOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto opCode = (SyntaxKind)arithmeticUnaryOp.opCode();
        switch (opCode)
        {
        case SyntaxKind::ExclamationToken:
            NegativeOpBin(arithmeticUnaryOp, rewriter, getTypeConverter());
            return success();
        case SyntaxKind::PlusToken:
            rewriter.replaceOp(arithmeticUnaryOp, arithmeticUnaryOp.operand1());
            return success();
        case SyntaxKind::MinusToken:
            NegativeOpValue(arithmeticUnaryOp, rewriter);
            return success();
        case SyntaxKind::TildeToken:
            NegativeOpBin(arithmeticUnaryOp, rewriter, getTypeConverter());
            return success();
        default:
            llvm_unreachable("not implemented");
        }
    }
};

struct ArithmeticBinaryOpLowering : public TsLlvmPattern<mlir_ts::ArithmeticBinaryOp>
{
    using TsLlvmPattern<mlir_ts::ArithmeticBinaryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ArithmeticBinaryOp arithmeticBinaryOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto opCode = (SyntaxKind)arithmeticBinaryOp.opCode();
        switch (opCode)
        {
        case SyntaxKind::PlusToken:
            if (arithmeticBinaryOp->getOperand(0).getType().isa<mlir_ts::StringType>())
            {
                rewriter.replaceOpWithNewOp<mlir_ts::StringConcatOp>(
                    arithmeticBinaryOp, mlir_ts::StringType::get(rewriter.getContext()),
                    ValueRange{arithmeticBinaryOp.getOperand(0), arithmeticBinaryOp.getOperand(1)});
            }
            else
            {
                BinOp<mlir_ts::ArithmeticBinaryOp, AddIOp, AddFOp>(arithmeticBinaryOp, rewriter);
            }

            return success();

        case SyntaxKind::MinusToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, SubIOp, SubFOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::AsteriskToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, MulIOp, MulFOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::SlashToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, DivFOp, DivFOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::GreaterThanGreaterThanToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, SignedShiftRightOp, SignedShiftRightOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::GreaterThanGreaterThanGreaterThanToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, UnsignedShiftRightOp, UnsignedShiftRightOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::LessThanLessThanToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, ShiftLeftOp, ShiftLeftOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::AmpersandToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, AndOp, AndOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::BarToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, OrOp, OrOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::CaretToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, XOrOp, XOrOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::PercentToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, RemFOp, RemFOp>(arithmeticBinaryOp, rewriter);
            return success();

        case SyntaxKind::AsteriskAsteriskToken:
            BinOp<mlir_ts::ArithmeticBinaryOp, math::PowFOp, math::PowFOp>(arithmeticBinaryOp, rewriter);
            return success();

        default:
            llvm_unreachable("not implemented");
        }
    }
};

struct LogicalBinaryOpLowering : public TsLlvmPattern<mlir_ts::LogicalBinaryOp>
{
    using TsLlvmPattern<mlir_ts::LogicalBinaryOp>::TsLlvmPattern;

    template <CmpIPredicate v1, CmpFPredicate v2>
    mlir::Value logicOp(mlir_ts::LogicalBinaryOp logicalBinaryOp, SyntaxKind op, PatternRewriter &builder) const
    {
        return LogicOp<CmpIOp, CmpIPredicate, v1, CmpFOp, CmpFPredicate, v2>(
            logicalBinaryOp, op, logicalBinaryOp.operand1(), logicalBinaryOp.operand2(), builder, *(LLVMTypeConverter *)getTypeConverter());
    }

    LogicalResult matchAndRewrite(mlir_ts::LogicalBinaryOp logicalBinaryOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto op = (SyntaxKind)logicalBinaryOp.opCode();

        // int and float
        mlir::Value value;
        switch (op)
        {
        case SyntaxKind::EqualsEqualsToken:
        case SyntaxKind::EqualsEqualsEqualsToken:
            value = logicOp<CmpIPredicate::eq, CmpFPredicate::OEQ>(logicalBinaryOp, op, rewriter);
            break;
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
            value = logicOp<CmpIPredicate::ne, CmpFPredicate::ONE>(logicalBinaryOp, op, rewriter);
            break;
        case SyntaxKind::GreaterThanToken:
            value = logicOp<CmpIPredicate::sgt, CmpFPredicate::OGT>(logicalBinaryOp, op, rewriter);
            break;
        case SyntaxKind::GreaterThanEqualsToken:
            value = logicOp<CmpIPredicate::sge, CmpFPredicate::OGE>(logicalBinaryOp, op, rewriter);
            break;
        case SyntaxKind::LessThanToken:
            value = logicOp<CmpIPredicate::slt, CmpFPredicate::OLT>(logicalBinaryOp, op, rewriter);
            break;
        case SyntaxKind::LessThanEqualsToken:
            value = logicOp<CmpIPredicate::sle, CmpFPredicate::OLE>(logicalBinaryOp, op, rewriter);
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

    LogicalResult matchAndRewrite(mlir_ts::LoadOp loadOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(loadOp, rewriter);

        if (auto refType = loadOp.reference().getType().dyn_cast_or_null<mlir_ts::RefType>())
        {
            auto elementType = refType.getElementType();
            auto elementTypeConverted = tch.convertType(elementType);

            rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, elementTypeConverted, loadOp.reference());
            return success();
        }

        if (auto boundRefType = loadOp.reference().getType().dyn_cast_or_null<mlir_ts::BoundRefType>())
        {
            rewriter.replaceOpWithNewOp<mlir_ts::LoadBoundRefOp>(loadOp, loadOp.getType(), loadOp.reference());
            return success();
        }

        return failure();
    }
};

struct StoreOpLowering : public TsLlvmPattern<mlir_ts::StoreOp>
{
    using TsLlvmPattern<mlir_ts::StoreOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StoreOp storeOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        if (auto boundRefType = storeOp.reference().getType().dyn_cast_or_null<mlir_ts::BoundRefType>())
        {
            rewriter.replaceOpWithNewOp<mlir_ts::StoreBoundRefOp>(storeOp, storeOp.value(), storeOp.reference());
            return success();
        }

        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, storeOp.value(), storeOp.reference());
        return success();
    }
};

struct ElementRefOpLowering : public TsLlvmPattern<mlir_ts::ElementRefOp>
{
    using TsLlvmPattern<mlir_ts::ElementRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ElementRefOp elementOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(elementOp, rewriter, getTypeConverter());

        auto addr = ch.GetAddressOfArrayElement(elementOp.getResult().getType(), elementOp.array(), elementOp.index());
        rewriter.replaceOp(elementOp, addr);
        return success();
    }
};

struct ExtractPropertyOpLowering : public TsLlvmPattern<mlir_ts::ExtractPropertyOp>
{
    using TsLlvmPattern<mlir_ts::ExtractPropertyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ExtractPropertyOp extractPropertyOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(extractPropertyOp, tch.convertType(extractPropertyOp.getType()),
                                                          extractPropertyOp.object(), extractPropertyOp.position());

        return success();
    }
};

struct InsertPropertyOpLowering : public TsLlvmPattern<mlir_ts::InsertPropertyOp>
{
    using TsLlvmPattern<mlir_ts::InsertPropertyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InsertPropertyOp insertPropertyOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        auto loc = insertPropertyOp->getLoc();

        rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(insertPropertyOp, tch.convertType(insertPropertyOp.object().getType()),
                                                         insertPropertyOp.object(), insertPropertyOp.value(), insertPropertyOp.position());

        return success();
    }
};

struct PropertyRefOpLowering : public TsLlvmPattern<mlir_ts::PropertyRefOp>
{
    using TsLlvmPattern<mlir_ts::PropertyRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::PropertyRefOp propertyRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(propertyRefOp, rewriter, getTypeConverter());

        auto addr = ch.GetAddressOfStructElement(propertyRefOp.getType(), propertyRefOp.objectRef(), propertyRefOp.position());

        if (auto boundRefType = propertyRefOp.getType().dyn_cast_or_null<mlir_ts::BoundRefType>())
        {
            auto boundRef =
                rewriter.create<mlir_ts::CreateBoundRefOp>(propertyRefOp->getLoc(), boundRefType, propertyRefOp.objectRef(), addr);
            addr = boundRef;
        }

        rewriter.replaceOp(propertyRefOp, addr);

        return success();
    }
};

struct GlobalOpLowering : public TsLlvmPattern<mlir_ts::GlobalOp>
{
    using TsLlvmPattern<mlir_ts::GlobalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GlobalOp globalOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper lch(globalOp, rewriter, getTypeConverter());

        auto linkage = LLVM::Linkage::Internal;
        if (auto linkageAttr = globalOp->getAttrOfType<StringAttr>("Linkage"))
        {
            auto val = linkageAttr.getValue();
            if (val == "External")
            {
                linkage = LLVM::Linkage::External;
            }
            else if (val == "Linkonce")
            {
                linkage = LLVM::Linkage::Linkonce;
            }
            else if (val == "LinkonceODR")
            {
                linkage = LLVM::Linkage::LinkonceODR;
            }
        }

        // TODO: include initialize block
        lch.createGlobalVarIfNew(globalOp.sym_name(), getTypeConverter()->convertType(globalOp.type()), globalOp.valueAttr(),
                                 globalOp.constant(), globalOp.getInitializerRegion(), linkage);
        rewriter.eraseOp(globalOp);
        return success();
    }
};

struct GlobalResultOpLowering : public TsLlvmPattern<mlir_ts::GlobalResultOp>
{
    using TsLlvmPattern<mlir_ts::GlobalResultOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GlobalResultOp globalResultOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(globalResultOp, globalResultOp.results());
        return success();
    }
};

struct AddressOfOpLowering : public TsLlvmPattern<mlir_ts::AddressOfOp>
{
    using TsLlvmPattern<mlir_ts::AddressOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AddressOfOp addressOfOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper lch(addressOfOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());

        auto value = lch.getAddressOfGlobalVar(addressOfOp.global_name(), tch.convertType(addressOfOp.getType()),
                                               addressOfOp.offset() ? addressOfOp.offset().getValue() : 0);
        rewriter.replaceOp(addressOfOp, value);
        return success();
    }
};

struct AddressOfConstStringOpLowering : public TsLlvmPattern<mlir_ts::AddressOfConstStringOp>
{
    using TsLlvmPattern<mlir_ts::AddressOfConstStringOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AddressOfConstStringOp addressOfConstStringOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        auto loc = addressOfConstStringOp->getLoc();
        auto globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, th.getI8PtrType(), addressOfConstStringOp.global_name());
        auto cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getI64Type(), th.getIndexAttrValue(0));
        rewriter.replaceOpWithNewOp<LLVM::GEPOp>(addressOfConstStringOp, th.getI8PtrType(), globalPtr, ArrayRef<Value>({cst0, cst0}));

        return success();
    }
};

struct CreateOptionalOpLowering : public TsLlvmPattern<mlir_ts::CreateOptionalOp>
{
    using TsLlvmPattern<mlir_ts::CreateOptionalOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateOptionalOp createOptionalOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = createOptionalOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(createOptionalOp, rewriter);

        auto boxedType = createOptionalOp.res().getType().cast<mlir_ts::OptionalType>().getElementType();
        auto llvmBoxedType = tch.convertType(boxedType);
        auto llvmOptType = tch.convertType(createOptionalOp.res().getType());

        auto value = createOptionalOp.in();
        auto valueLLVMType = tch.convertType(value.getType());

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmOptType);

        if (valueLLVMType != llvmBoxedType)
        {
            // cast value to box
            CastLogicHelper castLogic(createOptionalOp, rewriter, tch);
            value = castLogic.cast(value, valueLLVMType, boxedType, llvmBoxedType);
            if (!value)
            {
                return failure();
            }
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

    LogicalResult matchAndRewrite(mlir_ts::UndefOptionalOp undefOptionalOp, ArrayRef<Value> operands,
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

    LogicalResult matchAndRewrite(mlir_ts::HasValueOp hasValueOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto loc = hasValueOp->getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(hasValueOp, th.getLLVMBoolType(), hasValueOp.in(),
                                                          rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

        return success();
    }
};

struct ValueOpLowering : public TsLlvmPattern<mlir_ts::ValueOp>
{
    using TsLlvmPattern<mlir_ts::ValueOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ValueOp valueOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto loc = valueOp->getLoc();

        TypeConverterHelper tch(getTypeConverter());

        auto valueType = valueOp.res().getType();
        auto llvmValueType = tch.convertType(valueType);

        rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(valueOp, llvmValueType, valueOp.in(),
                                                          rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));

        return success();
    }
};

struct LoadSaveValueLowering : public TsLlvmPattern<mlir_ts::LoadSaveOp>
{
    using TsLlvmPattern<mlir_ts::LoadSaveOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LoadSaveOp loadSaveOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto loc = loadSaveOp->getLoc();

        auto value = rewriter.create<LLVM::LoadOp>(loc, loadSaveOp.src());
        rewriter.create<LLVM::StoreOp>(loc, value, loadSaveOp.dst());

        rewriter.eraseOp(loadSaveOp);

        return success();
    }
};

struct MemoryCopyOpLowering : public TsLlvmPattern<mlir_ts::MemoryCopyOp>
{
    using TsLlvmPattern<mlir_ts::MemoryCopyOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::MemoryCopyOp memoryCopyOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = memoryCopyOp->getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(memoryCopyOp, rewriter, getTypeConverter());
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(memoryCopyOp, rewriter);

        auto copyMemFuncOp = ch.getOrInsertFunction(
            "llvm.memcpy.p0i8.p0i8.i64",
            th.getFunctionType(th.getVoidType(), {th.getI8PtrType(), th.getI8PtrType(), th.getI64Type(), th.getLLVMBoolType()}));

        mlir::SmallVector<mlir::Value, 4> values;
        values.push_back(clh.castToI8Ptr(memoryCopyOp.dst()));
        values.push_back(clh.castToI8Ptr(memoryCopyOp.src()));

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

    LogicalResult matchAndRewrite(mlir_ts::UnreachableOp unreachableOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = unreachableOp.getLoc();
        CodeLogicHelper clh(unreachableOp, rewriter);

        auto unreachable = clh.FindUnreachableBlockOrCreate();

        rewriter.replaceOpWithNewOp<mlir::BranchOp>(unreachableOp, unreachable);

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

    LogicalResult matchAndRewrite(mlir_ts::ThrowCallOp throwCallOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());

        ThrowLogic tl(throwCallOp, rewriter, tch, throwCallOp.getLoc());
        tl.logic(throwCallOp.exception(), nullptr);

        rewriter.eraseOp(throwCallOp);

        return success();
    }
};

struct ThrowUnwindOpLowering : public TsLlvmPattern<mlir_ts::ThrowUnwindOp>
{
    using TsLlvmPattern<mlir_ts::ThrowUnwindOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThrowUnwindOp throwUnwindOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        TypeConverterHelper tch(getTypeConverter());
        ThrowLogic tl(throwUnwindOp, rewriter, tch, throwUnwindOp.getLoc());
        tl.logic(throwUnwindOp.exception(), throwUnwindOp.unwindDest());

        rewriter.eraseOp(throwUnwindOp);

        return success();
    }
};

#ifdef WIN_EXCEPTION

struct LandingPadOpLowering : public TsLlvmPattern<mlir_ts::LandingPadOp>
{
    using TsLlvmPattern<mlir_ts::LandingPadOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LandingPadOp landingPadOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = landingPadOp.getLoc();

        TypeHelper th(rewriter);

        auto catch1 = landingPadOp.catches().front();

        mlir::Type llvmLandingPadTy = getTypeConverter()->convertType(landingPadOp.getType());
        rewriter.replaceOpWithNewOp<LLVM::LandingpadOp>(landingPadOp, llvmLandingPadTy, false, ValueRange{catch1});

        return success();
    }
};

struct CompareCatchTypeOpLowering : public TsLlvmPattern<mlir_ts::CompareCatchTypeOp>
{
    using TsLlvmPattern<mlir_ts::CompareCatchTypeOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CompareCatchTypeOp compareCatchTypeOp, ArrayRef<Value> operands,
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

    LogicalResult matchAndRewrite(mlir_ts::BeginCatchOp beginCatchOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = beginCatchOp.getLoc();

        auto nullVal = rewriter.create<mlir_ts::NullOp>(loc, mlir_ts::NullType::get(rewriter.getContext()));
        auto opaqueValue = rewriter.create<mlir_ts::CastOp>(loc, mlir_ts::OpaqueType::get(rewriter.getContext()), nullVal);
        rewriter.replaceOp(beginCatchOp, ValueRange{opaqueValue});

        return success();
    }
};

struct SaveCatchVarOpLowering : public TsLlvmPattern<mlir_ts::SaveCatchVarOp>
{
    using TsLlvmPattern<mlir_ts::SaveCatchVarOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SaveCatchVarOp saveCatchVarOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = saveCatchVarOp.getLoc();

        auto catchRefType = saveCatchVarOp.varStore().getType().cast<mlir_ts::RefType>();
        auto catchType = catchRefType.getElementType();

        // this is hook call to finish later in Win32 exception pass
        auto catchVal = rewriter.create<mlir_ts::UndefOp>(loc, catchType);
        rewriter.replaceOpWithNewOp<mlir_ts::StoreOp>(saveCatchVarOp, catchVal, saveCatchVarOp.varStore());

        return success();
    }
};

struct EndCatchOpLowering : public TsLlvmPattern<mlir_ts::EndCatchOp>
{
    using TsLlvmPattern<mlir_ts::EndCatchOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::EndCatchOp endCatchOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        Location loc = endCatchOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(endCatchOp, rewriter, getTypeConverter());

        auto endCatchFuncName = "__cxa_end_catch";
        auto endCatchFunc = ch.getOrInsertFunction(endCatchFuncName, th.getFunctionType(th.getVoidType(), ArrayRef<Type>{}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(endCatchOp, endCatchFunc, ValueRange{});

        return success();
    }
};

struct TryOpLowering : public TsLlvmPattern<mlir_ts::TryOp>
{
    using TsLlvmPattern<mlir_ts::TryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TryOp tryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        Location loc = tryOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(tryOp, rewriter, getTypeConverter());
        LLVMRTTIHelperVCWin32 rttih(tryOp, rewriter, *getTypeConverter());

        auto visitorCatchContinue = [&](Operation *op) {
            if (auto catchOp = dyn_cast_or_null<mlir_ts::CatchOp>(op))
            {
                // rttih.setRTTIForType(loc, catchOp.catchArg().getType().cast<mlir_ts::RefType>().getElementType());
                rttih.setType(catchOp.catchArg().getType().cast<mlir_ts::RefType>().getElementType());
            }
        };
        tryOp.catches().walk(visitorCatchContinue);

        OpBuilder::InsertionGuard guard(rewriter);
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *bodyRegion = &tryOp.body().front();
        auto *bodyRegionLast = &tryOp.body().back();
        auto *catchesRegion = &tryOp.catches().front();
        auto *catchesRegionLast = &tryOp.catches().back();
        auto *finallyBlockRegion = &tryOp.finallyBlock().front();
        auto *finallyBlockRegionLast = &tryOp.finallyBlock().back();

        // logic to set Invoke attribute CallOp
        // auto visitorCallOpContinue = [&](Operation *op) {
        //     if (auto callOp = dyn_cast_or_null<mlir_ts::CallOp>(op))
        //     {
        //         tsLlvmContext->unwind[op] = catchesRegion;
        //     }
        //     else if (auto callIndirectOp = dyn_cast_or_null<mlir_ts::CallIndirectOp>(op))
        //     {
        //         tsLlvmContext->unwind[op] = catchesRegion;
        //     }
        //     else if (auto throwOp = dyn_cast_or_null<mlir_ts::ThrowOp>(op))
        //     {
        //         tsLlvmContext->unwind[op] = catchesRegion;
        //     }
        // };
        // tryOp.body().walk(visitorCallOpContinue);

        // Branch to the "body" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(loc, bodyRegion, ValueRange{});

        rewriter.inlineRegionBefore(tryOp.body(), continuation);

        rewriter.inlineRegionBefore(tryOp.catches(), continuation);

        rewriter.inlineRegionBefore(tryOp.finallyBlock(), continuation);

        // Body:catch vars
        rewriter.setInsertionPointToStart(bodyRegion);
        auto catch1 = rttih.hasType() ? (mlir::Value)rttih.typeInfoPtrValue(loc)
                                      : /*catch all*/ (mlir::Value)rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());

        // filter means - allow all in = catch (...)
        // auto filter = rewriter.create<LLVM::UndefOp>(loc, th.getI8PtrPtrType());

        // Body:exit -> replace ResultOp with br
        rewriter.setInsertionPointToEnd(bodyRegionLast);

        auto resultOp = cast<mlir_ts::ResultOp>(bodyRegionLast->getTerminator());
        // rewriter.replaceOpWithNewOp<BranchOp>(resultOp, continuation, ValueRange{});
        rewriter.replaceOpWithNewOp<BranchOp>(resultOp, finallyBlockRegion, ValueRange{});

        // catches;landingpad
        rewriter.setInsertionPointToStart(catchesRegion);

        auto landingPadTypeWin32 =
            LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI8PtrType(), th.getI32Type(), th.getI8PtrType()}, false);
        auto landingPadOp = rewriter.create<LLVM::LandingpadOp>(loc, landingPadTypeWin32, false, ValueRange{catch1});

        // catches:exit
        rewriter.setInsertionPointToEnd(catchesRegionLast);

        // join blocks
        auto yieldOpCatches = cast<mlir_ts::ResultOp>(catchesRegionLast->getTerminator());

        // we need it to mark end of exception
        auto endCatchFuncName = "__cxa_end_catch";
        auto endCatchFunc = ch.getOrInsertFunction(endCatchFuncName, th.getFunctionType(th.getVoidType(), ArrayRef<Type>{}));

        rewriter.create<LLVM::CallOp>(loc, endCatchFunc, ValueRange{});

        rewriter.replaceOpWithNewOp<BranchOp>(yieldOpCatches, finallyBlockRegion, ValueRange{});
        // rewriter.replaceOpWithNewOp<LLVM::ResumeOp>(yieldOpCatches, landingPadOp);

        // finally:exit
        rewriter.setInsertionPointToEnd(finallyBlockRegionLast);

        auto yieldOpFinallyBlock = cast<mlir_ts::ResultOp>(finallyBlockRegionLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOpFinallyBlock, continuation, yieldOpFinallyBlock.results());

        rewriter.replaceOp(tryOp, continuation->getArguments());

        return success();
    }
};

#else

struct LandingPadOpLowering : public TsLlvmPattern<mlir_ts::LandingPadOp>
{
    using TsLlvmPattern<mlir_ts::LandingPadOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LandingPadOp landingPadOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = landingPadOp.getLoc();

        TypeHelper th(rewriter);

        auto catch1 = landingPadOp.catches().front();

        mlir::Type llvmLandingPadTy = getTypeConverter()->convertType(landingPadOp.getType());
        rewriter.replaceOpWithNewOp<LLVM::LandingpadOp>(landingPadOp, llvmLandingPadTy, false, ValueRange{catch1});

        return success();
    }
};

struct CompareCatchTypeOpLowering : public TsLlvmPattern<mlir_ts::CompareCatchTypeOp>
{
    using TsLlvmPattern<mlir_ts::CompareCatchTypeOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CompareCatchTypeOp compareCatchTypeOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = compareCatchTypeOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(compareCatchTypeOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(compareCatchTypeOp, rewriter);

        auto i8PtrTy = th.getI8PtrType();

        auto loadedI32Value =
            rewriter.create<LLVM::ExtractValueOp>(loc, th.getI32Type(), compareCatchTypeOp.landingPad(), clh.getStructIndexAttr(1));

        auto typeIdFuncName = "llvm.eh.typeid.for";
        auto typeIdFunc = ch.getOrInsertFunction(typeIdFuncName, th.getFunctionType(th.getI32Type(), {i8PtrTy}));

        auto callInfo = rewriter.create<LLVM::CallOp>(loc, typeIdFunc, ValueRange{clh.castToI8Ptr(compareCatchTypeOp.throwTypeInfo())});
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

    LogicalResult matchAndRewrite(mlir_ts::BeginCatchOp beginCatchOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = beginCatchOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(beginCatchOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(beginCatchOp, rewriter);

        auto i8PtrTy = th.getI8PtrType();

        // catches:extract
        auto loadedI8PtrValue =
            rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), beginCatchOp.landingPad(), clh.getStructIndexAttr(0));

        auto beginCatchFuncName = "__cxa_begin_catch";
        auto beginCatchFunc = ch.getOrInsertFunction(beginCatchFuncName, th.getFunctionType(i8PtrTy, {i8PtrTy}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(beginCatchOp, beginCatchFunc, ValueRange{loadedI8PtrValue});

        return success();
    }
};

struct SaveCatchVarOpLowering : public TsLlvmPattern<mlir_ts::SaveCatchVarOp>
{
    using TsLlvmPattern<mlir_ts::SaveCatchVarOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SaveCatchVarOp saveCatchVarOp, ArrayRef<Value> operands,
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
            auto ptrVal = rewriter.create<LLVM::BitcastOp>(loc, th.getPointerType(llvmCatchType), saveCatchVarOp.exceptionInfo());
            catchVal = rewriter.create<LLVM::LoadOp>(loc, llvmCatchType, ptrVal);
        }
        else
        {
            catchVal = rewriter.create<LLVM::BitcastOp>(loc, llvmCatchType, saveCatchVarOp.exceptionInfo());
        }

        rewriter.replaceOpWithNewOp<mlir_ts::StoreOp>(saveCatchVarOp, catchVal, saveCatchVarOp.varStore());

        return success();
    }
};

struct EndCatchOpLowering : public TsLlvmPattern<mlir_ts::EndCatchOp>
{
    using TsLlvmPattern<mlir_ts::EndCatchOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::EndCatchOp endCatchOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        Location loc = endCatchOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(endCatchOp, rewriter, getTypeConverter());

        auto endCatchFuncName = "__cxa_end_catch";
        auto endCatchFunc = ch.getOrInsertFunction(endCatchFuncName, th.getFunctionType(th.getVoidType(), ArrayRef<Type>{}));

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(endCatchOp, endCatchFunc, ValueRange{});

        return success();
    }
};

struct TryOpLowering : public TsLlvmPattern<mlir_ts::TryOp>
{
    using TsLlvmPattern<mlir_ts::TryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TryOp tryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        Location loc = tryOp.getLoc();

        TypeHelper th(rewriter);
        LLVMCodeHelper ch(tryOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(tryOp, rewriter);

        LLVMRTTIHelperVCLinux rttih(tryOp, rewriter, *getTypeConverter());

        auto i8PtrTy = th.getI8PtrType();

        Operation *catchOpPtr = nullptr;
        auto visitorCatchContinue = [&](Operation *op) {
            if (auto catchOp = dyn_cast_or_null<mlir_ts::CatchOp>(op))
            {
                // rttih.setRTTIForType(loc, catchOp.catchArg().getType().cast<mlir_ts::RefType>().getElementType());
                rttih.setType(catchOp.catchArg().getType().cast<mlir_ts::RefType>().getElementType());
                assert(!catchOpPtr);
                catchOpPtr = op;
            }
        };
        tryOp.catches().walk(visitorCatchContinue);

        // add to variables to store exception info: i8*, i32
        auto ptrValueRef =
            rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(th.getI8PtrType()), mlir::Value(), rewriter.getBoolAttr(false));
        auto i32ValueRef =
            rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(th.getI32Type()), mlir::Value(), rewriter.getBoolAttr(false));

        OpBuilder::InsertionGuard guard(rewriter);
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *bodyRegion = &tryOp.body().front();
        auto *bodyRegionLast = &tryOp.body().back();
        auto *catchesRegion = &tryOp.catches().front();
        auto *catchesRegionLast = &tryOp.catches().back();
        auto *finallyBlockRegion = &tryOp.finallyBlock().front();
        auto *finallyBlockRegionLast = &tryOp.finallyBlock().back();

        // logic to set Invoke attribute CallOp
        // auto visitorCallOpContinue = [&](Operation *op) {
        //     if (auto callOp = dyn_cast_or_null<mlir_ts::CallOp>(op))
        //     {
        //         tsLlvmContext->unwind[op] = catchesRegion;
        //     }
        //     else if (auto callIndirectOp = dyn_cast_or_null<mlir_ts::CallIndirectOp>(op))
        //     {
        //         tsLlvmContext->unwind[op] = catchesRegion;
        //     }
        //     else if (auto throwOp = dyn_cast_or_null<mlir_ts::ThrowOp>(op))
        //     {
        //         tsLlvmContext->unwind[op] = catchesRegion;
        //     }
        // };
        // tryOp.body().walk(visitorCallOpContinue);

        // Branch to the "body" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(loc, bodyRegion, ValueRange{});

        rewriter.inlineRegionBefore(tryOp.body(), continuation);

        rewriter.inlineRegionBefore(tryOp.catches(), continuation);

        rewriter.inlineRegionBefore(tryOp.finallyBlock(), continuation);

        // Body:catch vars
        rewriter.setInsertionPointToStart(bodyRegion);
        auto catch1 = rttih.hasType() ? (mlir::Value)rttih.typeInfoPtrValue(loc)
                                      : /*catch all*/ (mlir::Value)rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());

        // filter means - allow all in = catch (...)
        // auto filter = rewriter.create<LLVM::UndefOp>(loc, th.getI8PtrPtrType());

        // Body:exit -> replace ResultOp with br
        rewriter.setInsertionPointToEnd(bodyRegionLast);

        auto resultOp = cast<mlir_ts::ResultOp>(bodyRegionLast->getTerminator());
        // rewriter.replaceOpWithNewOp<BranchOp>(resultOp, continuation, ValueRange{});
        rewriter.replaceOpWithNewOp<BranchOp>(resultOp, finallyBlockRegion, ValueRange{});

        // catches:landingpad
        rewriter.setInsertionPointToStart(catchesRegion);

        auto landingPadTypeLinux = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI8PtrType(), th.getI32Type()}, false);
        auto landingPadOp = rewriter.create<LLVM::LandingpadOp>(loc, landingPadTypeLinux, false, ValueRange{catch1});

        // catches:extract
        auto ptrFromExcept = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), landingPadOp, clh.getStructIndexAttr(0));
        rewriter.create<mlir_ts::StoreOp>(loc, ptrFromExcept, ptrValueRef);
        auto i32FromExcept = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI32Type(), landingPadOp, clh.getStructIndexAttr(1));
        auto storeOpBrPlace = rewriter.create<mlir_ts::StoreOp>(loc, i32FromExcept, i32ValueRef);

        mlir::Value cmpValue;
        if (rttih.hasType())
        {
            // br<->extract, will be inserted later
            // catch: compare
            auto loadedI32Value = rewriter.create<LLVM::LoadOp>(loc, th.getI32Type(), i32ValueRef);

            auto typeIdFuncName = "llvm.eh.typeid.for";
            auto typeIdFunc = ch.getOrInsertFunction(typeIdFuncName, th.getFunctionType(th.getI32Type(), {i8PtrTy}));

            auto callInfo = rewriter.create<LLVM::CallOp>(loc, typeIdFunc, ValueRange{clh.castToI8Ptr(rttih.throwInfoPtrValue(loc))});
            auto typeIdValue = callInfo.getResult(0);

            // icmp
            cmpValue = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, loadedI32Value, typeIdValue);
            // condbr, will be inserted later
        }

        // catch: begin catch
        auto loadedI8PtrValue = rewriter.create<LLVM::LoadOp>(loc, i8PtrTy, ptrValueRef);

        auto beginCatchFuncName = "__cxa_begin_catch";
        auto beginCatchFunc = ch.getOrInsertFunction(beginCatchFuncName, th.getFunctionType(i8PtrTy, {i8PtrTy}));

        auto beginCatchCallInfo = rewriter.create<LLVM::CallOp>(loc, beginCatchFunc, ValueRange{loadedI8PtrValue});

        if (catchOpPtr)
        {
            tsLlvmContext->catchOpData[catchOpPtr] = beginCatchCallInfo->getResult(0);
        }

        // catch: load value
        // TODO:

        // catches: end catch
        rewriter.setInsertionPoint(catchesRegionLast->getTerminator());

        auto endCatchFuncName = "__cxa_end_catch";
        auto endCatchFunc = ch.getOrInsertFunction(endCatchFuncName, th.getFunctionType(th.getVoidType(), ArrayRef<Type>{}));

        rewriter.create<LLVM::CallOp>(loc, endCatchFunc, ValueRange{});

        // exit br
        rewriter.setInsertionPointToEnd(catchesRegionLast);

        auto yieldOpCatches = cast<mlir_ts::ResultOp>(catchesRegionLast->getTerminator());
        // rewriter.replaceOpWithNewOp<BranchOp>(yieldOpCatches, continuation, ValueRange{});
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOpCatches, finallyBlockRegion, ValueRange{});

        // br: insert br after extract values
        rewriter.setInsertionPointAfter(storeOpBrPlace);

        Block *currentBlockBr = rewriter.getInsertionBlock();
        Block *continuationBr = rewriter.splitBlock(currentBlockBr, rewriter.getInsertionPoint());

        rewriter.setInsertionPointToEnd(currentBlockBr);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBr);
        // end of br

        if (cmpValue)
        {
            // condbr
            rewriter.setInsertionPointAfterValue(cmpValue);

            Block *currentBlockBrCmp = rewriter.getInsertionBlock();
            Block *continuationBrCmp = rewriter.splitBlock(currentBlockBrCmp, rewriter.getInsertionPoint());

            rewriter.setInsertionPointAfterValue(cmpValue);
            // TODO: when catch not matching - should go into result (rethrow)
            rewriter.create<CondBranchOp>(loc, cmpValue, continuationBrCmp, continuation);
            // end of condbr
        }

        // end of jumps

        // finally:exit
        rewriter.setInsertionPointToEnd(finallyBlockRegionLast);

        auto yieldOpFinallyBlock = cast<mlir_ts::ResultOp>(finallyBlockRegionLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOpFinallyBlock, continuation, yieldOpFinallyBlock.results());

        rewriter.replaceOp(tryOp, continuation->getArguments());

        return success();
    }
};

#endif

struct CatchOpLowering : public TsLlvmPattern<mlir_ts::CatchOp>
{
    using TsLlvmPattern<mlir_ts::CatchOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CatchOp catchOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        // this is hook to process it later
        auto catchType = catchOp.catchArg().getType().cast<mlir_ts::RefType>().getElementType();
        auto llvmCatchType = getTypeConverter()->convertType(catchType);

        Location loc = catchOp.getLoc();

        auto catchDataValue = tsLlvmContext->catchOpData[catchOp];
        if (catchDataValue)
        {
            // linux version
            mlir::Value val;
            if (!llvmCatchType.isa<LLVM::LLVMPointerType>())
            {
                auto ptrVal = rewriter.create<LLVM::BitcastOp>(loc, th.getPointerType(llvmCatchType), catchDataValue);
                val = rewriter.create<LLVM::LoadOp>(loc, llvmCatchType, ptrVal);
            }
            else
            {
                val = rewriter.create<LLVM::BitcastOp>(loc, llvmCatchType, catchDataValue);
            }

            rewriter.create<LLVM::StoreOp>(loc, val, catchOp.catchArg());
        }
        else
        {
            // windows version
            auto undefVal = rewriter.create<LLVM::UndefOp>(loc, llvmCatchType);
            rewriter.create<LLVM::StoreOp>(loc, undefVal, catchOp.catchArg());
        }

        rewriter.eraseOp(catchOp);

        return success();
    }
};

struct TrampolineOpLowering : public TsLlvmPattern<mlir_ts::TrampolineOp>
{
    using TsLlvmPattern<mlir_ts::TrampolineOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TrampolineOp trampolineOp, ArrayRef<Value> operands,
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

        auto initTrampolineFuncOp =
            ch.getOrInsertFunction("llvm.init.trampoline", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, i8PtrTy}));
        auto adjustTrampolineFuncOp = ch.getOrInsertFunction("llvm.adjust.trampoline", th.getFunctionType(i8PtrTy, {i8PtrTy}));
        // Win32 specifics
        auto enableExecuteStackFuncOp = ch.getOrInsertFunction("__enable_execute_stack", th.getFunctionType(th.getVoidType(), {i8PtrTy}));

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
            auto trampoline = rewriter.create<LLVM::AllocaOp>(location, bufferType, clh.createI32ConstantOf(1));
            auto const0 = clh.createI32ConstantOf(0);
            trampolinePtr = rewriter.create<LLVM::GEPOp>(location, i8PtrTy, ValueRange{trampoline, const0, const0});
        }

        // init trampoline
        rewriter.create<LLVM::CallOp>(
            location, initTrampolineFuncOp,
            ValueRange{trampolinePtr, clh.castToI8Ptr(trampolineOp.callee()), clh.castToI8Ptr(trampolineOp.data_reference())});

        auto callAdjustedTrampoline = rewriter.create<LLVM::CallOp>(location, adjustTrampolineFuncOp, ValueRange{trampolinePtr});
        auto adjustedTrampolinePtr = callAdjustedTrampoline.getResult(0);

        rewriter.create<LLVM::CallOp>(location, enableExecuteStackFuncOp, ValueRange{adjustedTrampolinePtr});

        // mlir::Value castFunc = rewriter.create<mlir_ts::CastOp>(location, trampolineOp.getType(), adjustedTrampolinePtr);
        // replacement
        auto castFunc = castLogic.cast(adjustedTrampolinePtr, trampolineOp.getType());

        rewriter.replaceOp(trampolineOp, castFunc);

        return success();
    }
};

struct CaptureOpLowering : public TsLlvmPattern<mlir_ts::CaptureOp>
{
    using TsLlvmPattern<mlir_ts::CaptureOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CaptureOp captureOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto location = captureOp->getLoc();

        TypeHelper th(rewriter);

        auto captureRefType = captureOp.getType();

        LLVM_DEBUG(llvm::dbgs() << "\n ...capture result type: " << captureRefType << "\n\n";);

        assert(captureRefType.isa<mlir_ts::RefType>());
        auto captureStoreType = captureRefType.cast<mlir_ts::RefType>().getElementType().cast<mlir_ts::TupleType>();

        LLVM_DEBUG(llvm::dbgs() << "\n ...capture store type: " << captureStoreType << "\n\n";);

        // true => we need to allocate capture in heap memory
#ifdef ALLOC_CAPTURE_IN_HEAP
        auto inHeapMemory = true;
#else
        auto inHeapMemory = false;
#endif
        mlir::Value allocTempStorage =
            rewriter.create<mlir_ts::VariableOp>(location, captureRefType, mlir::Value(), rewriter.getBoolAttr(inHeapMemory));

        auto index = 0;
        for (auto val : captureOp.captured())
        {
            // TODO: Hack to detect which sent by ref

            auto thisStoreFieldType = captureStoreType.getType(index);
            auto thisStoreFieldTypeRef = mlir_ts::RefType::get(thisStoreFieldType);
            auto fieldRef = rewriter.create<mlir_ts::PropertyRefOp>(location, thisStoreFieldTypeRef, allocTempStorage,
                                                                    th.getStructIndexAttrValue(index));

            LLVM_DEBUG(llvm::dbgs() << "\n ...storing val: [" << val << "] in (" << index << ") ref: " << fieldRef << "\n\n";);

            // dereference value in case of sending value by ref but stored as value
            // TODO: review capture logic
            if (auto valRefType = val.getType().dyn_cast_or_null<mlir_ts::RefType>())
            {
                if (!thisStoreFieldType.isa<mlir_ts::RefType>() && thisStoreFieldType == valRefType.getElementType())
                {
                    // load value to dereference
                    val = rewriter.create<mlir_ts::LoadOp>(location, valRefType.getElementType(), val);
                }
            }

            rewriter.create<mlir_ts::StoreOp>(location, val, fieldRef);

            index++;
        }

        // mlir::Value newFunc = rewriter.create<mlir_ts::TrampolineOp>(location, captureOp.getType(), captureOp.callee(),
        // allocTempStorage); rewriter.replaceOp(captureOp, newFunc);

        rewriter.replaceOp(captureOp, allocTempStorage);

        return success();
    }
};

struct VTableOffsetRefOpLowering : public TsLlvmPattern<mlir_ts::VTableOffsetRefOp>
{
    using TsLlvmPattern<mlir_ts::VTableOffsetRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::VTableOffsetRefOp vtableOffsetRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = vtableOffsetRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(vtableOffsetRefOp, rewriter);

        auto ptrToArrOfPtrs = rewriter.create<mlir_ts::CastOp>(loc, th.getI8PtrPtrType(), vtableOffsetRefOp.vtable());

        auto index = clh.createI32ConstantOf(vtableOffsetRefOp.index());
        auto methodOrInterfacePtrPtr = rewriter.create<LLVM::GEPOp>(loc, th.getI8PtrPtrType(), ptrToArrOfPtrs, ValueRange{index});
        auto methodOrInterfacePtr = rewriter.create<LLVM::LoadOp>(loc, methodOrInterfacePtrPtr);

        rewriter.replaceOp(vtableOffsetRefOp, ValueRange{methodOrInterfacePtr});

        return success();
    }
};

struct ThisVirtualSymbolRefOpLowering : public TsLlvmPattern<mlir_ts::ThisVirtualSymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::ThisVirtualSymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThisVirtualSymbolRefOp thisVirtualSymbolRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = thisVirtualSymbolRefOp.getLoc();

        TypeHelper th(rewriter);

        auto methodPtr = rewriter.create<mlir_ts::VTableOffsetRefOp>(loc, th.getI8PtrType(), thisVirtualSymbolRefOp.vtable(),
                                                                     thisVirtualSymbolRefOp.index());
        auto methodTyped = rewriter.create<mlir_ts::CastOp>(loc, thisVirtualSymbolRefOp.getType(), methodPtr);

        rewriter.replaceOp(thisVirtualSymbolRefOp, ValueRange{methodTyped});

        return success();
    }
};

struct InterfaceSymbolRefOpLowering : public TsLlvmPattern<mlir_ts::InterfaceSymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::InterfaceSymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InterfaceSymbolRefOp interfaceSymbolRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = interfaceSymbolRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(interfaceSymbolRefOp, rewriter);

        auto vtable = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), interfaceSymbolRefOp.interfaceVal(),
                                                            clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), interfaceSymbolRefOp.interfaceVal(),
                                                             clh.getStructIndexAttr(THIS_VALUE_INDEX));

        auto methodPtr = rewriter.create<mlir_ts::VTableOffsetRefOp>(loc, th.getI8PtrType(), vtable, interfaceSymbolRefOp.index());
        auto methodTyped = rewriter.create<mlir_ts::CastOp>(loc, interfaceSymbolRefOp.getResult(0).getType(), methodPtr);

        rewriter.replaceOp(interfaceSymbolRefOp, ValueRange{methodTyped, thisVal});

        return success();
    }
};

struct NewInterfaceOpLowering : public TsLlvmPattern<mlir_ts::NewInterfaceOp>
{
    using TsLlvmPattern<mlir_ts::NewInterfaceOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::NewInterfaceOp newInterfaceOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = newInterfaceOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(newInterfaceOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto llvmInterfaceType = tch.convertType(newInterfaceOp.getType());

        auto structVal = rewriter.create<mlir_ts::UndefOp>(loc, llvmInterfaceType);
        auto structVal2 = rewriter.create<LLVM::InsertValueOp>(loc, structVal, clh.castToI8Ptr(newInterfaceOp.interfaceVTable()),
                                                               clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, clh.castToI8Ptr(newInterfaceOp.thisVal()),
                                                               clh.getStructIndexAttr(THIS_VALUE_INDEX));

        rewriter.replaceOp(newInterfaceOp, ValueRange{structVal3});

        return success();
    }
};

struct ThisPropertyRefOpLowering : public TsLlvmPattern<mlir_ts::ThisPropertyRefOp>
{
    using TsLlvmPattern<mlir_ts::ThisPropertyRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThisPropertyRefOp thisPropertyRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = thisPropertyRefOp.getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto p1 = rewriter.create<LLVM::PtrToIntOp>(loc, th.getIndexType(), thisPropertyRefOp.objectRef());
        auto p2 = rewriter.create<LLVM::PtrToIntOp>(loc, th.getIndexType(), thisPropertyRefOp.offset());
        auto padded = rewriter.create<LLVM::AddOp>(loc, th.getIndexType(), p1, p2);
        auto typedPtr = rewriter.create<LLVM::IntToPtrOp>(loc, tch.convertType(thisPropertyRefOp.getType()), padded);

        rewriter.replaceOp(thisPropertyRefOp, ValueRange{typedPtr});

        return success();
    }
};

struct LoadBoundRefOpLowering : public TsLlvmPattern<mlir_ts::LoadBoundRefOp>
{
    using TsLlvmPattern<mlir_ts::LoadBoundRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::LoadBoundRefOp loadBoundRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = loadBoundRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(loadBoundRefOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto boundRefType = loadBoundRefOp.reference().getType().cast<mlir_ts::BoundRefType>();

        auto llvmType = tch.convertType(boundRefType.getElementType());
        auto llvmRefType = LLVM::LLVMPointerType::get(llvmType);

        auto thisVal = rewriter.create<LLVM::ExtractValueOp>(loc, th.getI8PtrType(), loadBoundRefOp.reference(),
                                                             clh.getStructIndexAttr(THIS_VALUE_INDEX));
        auto valueRefVal =
            rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, loadBoundRefOp.reference(), clh.getStructIndexAttr(DATA_VALUE_INDEX));

        mlir::Value loadedValue = rewriter.create<LLVM::LoadOp>(loc, valueRefVal);

        if (auto funcType = boundRefType.getElementType().dyn_cast_or_null<mlir::FunctionType>())
        {
            mlir::Value boundMethodValue =
                rewriter.create<mlir_ts::CreateBoundFunctionOp>(loc, loadBoundRefOp.getType(), thisVal, loadedValue);

            LLVM_DEBUG(llvm::dbgs() << "LoadOp Bound Ref: LLVM Type :" << tch.convertType(loadBoundRefOp.getType()) << "\n";);

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

    LogicalResult matchAndRewrite(mlir_ts::StoreBoundRefOp storeBoundRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        auto loc = storeBoundRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(storeBoundRefOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto boundRefType = storeBoundRefOp.reference().getType().cast<mlir_ts::BoundRefType>();

        auto llvmType = tch.convertType(boundRefType.getElementType());
        auto llvmRefType = LLVM::LLVMPointerType::get(llvmType);

        auto valueRefVal =
            rewriter.create<LLVM::ExtractValueOp>(loc, llvmRefType, storeBoundRefOp.reference(), clh.getStructIndexAttr(DATA_VALUE_INDEX));

        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeBoundRefOp, storeBoundRefOp.value(), valueRefVal);
        return success();
    }
};

struct CreateBoundRefOpLowering : public TsLlvmPattern<mlir_ts::CreateBoundRefOp>
{
    using TsLlvmPattern<mlir_ts::CreateBoundRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateBoundRefOp createBoundRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = createBoundRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(createBoundRefOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto llvmBoundRefType = tch.convertType(createBoundRefOp.getType());

        auto structVal = rewriter.create<mlir_ts::UndefOp>(loc, llvmBoundRefType);
        auto structVal2 =
            rewriter.create<LLVM::InsertValueOp>(loc, structVal, createBoundRefOp.valueRef(), clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, clh.castToI8Ptr(createBoundRefOp.thisVal()),
                                                               clh.getStructIndexAttr(THIS_VALUE_INDEX));

        rewriter.replaceOp(createBoundRefOp, ValueRange{structVal3});

        return success();
    }
};

struct CreateBoundFunctionOpLowering : public TsLlvmPattern<mlir_ts::CreateBoundFunctionOp>
{
    using TsLlvmPattern<mlir_ts::CreateBoundFunctionOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CreateBoundFunctionOp createBoundFunctionOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = createBoundFunctionOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(createBoundFunctionOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        assert(createBoundFunctionOp.getType());
        assert(createBoundFunctionOp.getType().isa<mlir_ts::BoundFunctionType>());

        auto llvmBoundFunctionType = tch.convertType(createBoundFunctionOp.getType());

        LLVM_DEBUG(llvm::dbgs() << "CreateBoundFunction: LLVM Type :" << llvmBoundFunctionType << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "CreateBoundFunction: Func Type :" << tch.convertType(createBoundFunctionOp.func().getType()) << "\n";);
        LLVM_DEBUG(llvm::dbgs() << "CreateBoundFunction: This Type :" << createBoundFunctionOp.thisVal().getType() << "\n";);

        auto structVal = rewriter.create<mlir_ts::UndefOp>(loc, llvmBoundFunctionType);
        auto structVal2 =
            rewriter.create<LLVM::InsertValueOp>(loc, structVal, createBoundFunctionOp.func(), clh.getStructIndexAttr(DATA_VALUE_INDEX));
        auto structVal3 = rewriter.create<LLVM::InsertValueOp>(loc, structVal2, createBoundFunctionOp.thisVal(),
                                                               clh.getStructIndexAttr(THIS_VALUE_INDEX));

        rewriter.replaceOp(createBoundFunctionOp, ValueRange{structVal3});

        return success();
    }
};

struct GetThisOpLowering : public TsLlvmPattern<mlir_ts::GetThisOp>
{
    using TsLlvmPattern<mlir_ts::GetThisOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GetThisOp getThisOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        Location loc = getThisOp.getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(getThisOp, rewriter);

        auto llvmThisType = tch.convertType(getThisOp.getType());

        mlir::Value thisVal =
            rewriter.create<LLVM::ExtractValueOp>(loc, llvmThisType, getThisOp.boundFunc(), clh.getStructIndexAttr(THIS_VALUE_INDEX));

        auto thisValCasted = rewriter.create<LLVM::BitcastOp>(loc, tch.convertType(getThisOp.getType()), thisVal);

        rewriter.replaceOp(getThisOp, {thisValCasted});

        return success();
    }
};

struct GetMethodOpLowering : public TsLlvmPattern<mlir_ts::GetMethodOp>
{
    using TsLlvmPattern<mlir_ts::GetMethodOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::GetMethodOp getMethodOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = getMethodOp.getLoc();

        TypeHelper th(rewriter);
        TypeConverterHelper tch(getTypeConverter());
        CodeLogicHelper clh(getMethodOp, rewriter);
        CastLogicHelper castLogic(getMethodOp, rewriter, tch);

        auto boundType = getMethodOp.boundFunc().getType().cast<mlir_ts::BoundFunctionType>();
        auto funcType = rewriter.getFunctionType(boundType.getInputs(), boundType.getResults());
        auto llvmMethodType = tch.convertType(funcType);

        mlir::Value methodVal =
            rewriter.create<LLVM::ExtractValueOp>(loc, llvmMethodType, getMethodOp.boundFunc(), clh.getStructIndexAttr(DATA_VALUE_INDEX));

        if (methodVal.getType() != getMethodOp.getType())
        {
            methodVal = castLogic.cast(methodVal, getMethodOp.getType());
        }

        rewriter.replaceOp(getMethodOp, ValueRange{methodVal});
        return success();
    }
};

struct TypeOfOpLowering : public TsLlvmPattern<mlir_ts::TypeOfOp>
{
    using TsLlvmPattern<mlir_ts::TypeOfOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TypeOfOp typeOfOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        Location loc = typeOfOp.getLoc();

        TypeConverterHelper tch(getTypeConverter());

        TypeOfOpHelper toh(typeOfOp, rewriter, tch);
        auto typeOfValue = toh.typeOfLogic(loc, typeOfOp.value());

        rewriter.replaceOp(typeOfOp, ValueRange{typeOfValue});
        return success();
    }
};

class DebuggerOpLowering : public TsLlvmPattern<mlir_ts::DebuggerOp>
{
  public:
    using TsLlvmPattern<mlir_ts::DebuggerOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::DebuggerOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

struct StateLabelOpLowering : public TsLlvmPattern<mlir_ts::StateLabelOp>
{
    using TsLlvmPattern<mlir_ts::StateLabelOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StateLabelOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.create<mlir::BranchOp>(op.getLoc(), continuationBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

class SwitchStateOpLowering : public TsLlvmPattern<mlir_ts::SwitchStateOp>
{
  public:
    using TsLlvmPattern<mlir_ts::SwitchStateOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::SwitchStateOp switchStateOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(switchStateOp, rewriter);

        auto loc = switchStateOp->getLoc();

        auto defaultBlock = clh.FindReturnBlock();

        SmallVector<int32_t> caseValues;
        SmallVector<Block *> caseDestinations;

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

                auto *opBlock = rewriter.getInsertionBlock();
                auto opPosition = rewriter.getInsertionPoint();
                auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

                rewriter.setInsertionPointToEnd(opBlock);

                rewriter.create<mlir::BranchOp>(stateLabelOp.getLoc(), continuationBlock);

                rewriter.setInsertionPointToStart(continuationBlock);

                rewriter.eraseOp(stateLabelOp);

                // add switch
                caseValues.push_back(index++);
                caseDestinations.push_back(continuationBlock);
            }
        }

        // make switch to be terminator
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // insert 0 state label
        caseValues.insert(caseValues.begin(), 0);
        caseDestinations.insert(caseDestinations.begin(), continuationBlock);

        // switch
        rewriter.setInsertionPointToEnd(opBlock);

        rewriter.create<LLVM::SwitchOp>(loc, switchStateOp.state(), defaultBlock ? defaultBlock : continuationBlock, ValueRange{},
                                        caseValues, caseDestinations);

        rewriter.eraseOp(switchStateOp);

        rewriter.setInsertionPointToStart(continuationBlock);

        LLVM_DEBUG(llvm::dbgs() << *switchStateOp->getParentOp() << "\n";);

        return success();
    }
};

static void populateTypeScriptConversionPatterns(LLVMTypeConverter &converter, mlir::ModuleOp &m)
{
    converter.addConversion([&](mlir_ts::AnyType type) { return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)); });

    converter.addConversion([&](mlir_ts::NullType type) { return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)); });

    converter.addConversion([&](mlir_ts::OpaqueType type) { return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)); });

    converter.addConversion([&](mlir_ts::VoidType type) { return LLVM::LLVMVoidType::get(m.getContext()); });

    converter.addConversion([&](mlir_ts::BooleanType type) {
        TypeHelper th(m.getContext());
        return th.getLLVMBoolType();
    });

    converter.addConversion(
        [&](mlir_ts::CharType type) { return IntegerType::get(m.getContext(), 8 /*, mlir::IntegerType::SignednessSemantics::Unsigned*/); });

    converter.addConversion(
        [&](mlir_ts::ByteType type) { return IntegerType::get(m.getContext(), 8 /*, mlir::IntegerType::SignednessSemantics::Unsigned*/); });

    converter.addConversion([&](mlir_ts::NumberType type) { return Float32Type::get(m.getContext()); });

    converter.addConversion([&](mlir_ts::BigIntType type) {
        return IntegerType::get(m.getContext(), 64 /*, mlir::IntegerType::SignednessSemantics::Signed*/);
    });

    converter.addConversion([&](mlir_ts::StringType type) { return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)); });

    converter.addConversion([&](mlir_ts::EnumType type) { return converter.convertType(type.getElementType()); });

    converter.addConversion(
        [&](mlir_ts::ConstArrayType type) { return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType())); });

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

    converter.addConversion(
        [&](mlir_ts::RefType type) { return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType())); });

    converter.addConversion(
        [&](mlir_ts::ValueRefType type) { return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType())); });

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
        llvmStructType.push_back(LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
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
        llvmStructType.push_back(LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)));
        return LLVM::LLVMStructType::getLiteral(type.getContext(), llvmStructType, false);
    });

    converter.addConversion([&](mlir_ts::ObjectType type) {
        if (type.getStorageType() == mlir_ts::AnyType::get(type.getContext()))
        {
            return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8));
        }

        return LLVM::LLVMPointerType::get(converter.convertType(type.getStorageType()));
    });

    converter.addConversion([&](mlir_ts::UnknownType type) { return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)); });

    converter.addConversion([&](mlir_ts::SymbolType type) { return IntegerType::get(m.getContext(), 32); });

    converter.addConversion([&](mlir_ts::UndefinedType type) { return IntegerType::get(m.getContext(), 1); });

    converter.addConversion([&](mlir_ts::ClassStorageType type) {
        SmallVector<mlir::Type> convertedTypes;
        for (auto subType : type.getFields())
        {
            convertedTypes.push_back(converter.convertType(subType.type));
        }

        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, false);
    });

    converter.addConversion(
        [&](mlir_ts::ClassType type) { return LLVM::LLVMPointerType::get(converter.convertType(type.getStorageType())); });

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
        return IntegerType::get(m.getContext(), 8 /*, mlir::IntegerType::SignednessSemantics::Unsigned*/);
    });

    converter.addConversion([&](mlir_ts::UnionType type) {
        LLVMTypeConverterHelper ltch(converter);

        auto currentSize = 0;
        mlir::Type selectedType;
        for (auto subType : type)
        {
            auto converted = converter.convertType(subType);
            auto typeSize = ltch.getTypeSize(converted);
            if (typeSize > currentSize)
            {
                selectedType = converted;
            }
        }

        SmallVector<mlir::Type> convertedTypes;
        convertedTypes.push_back(selectedType);
        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, false);
    });

    converter.addConversion([&](mlir_ts::IntersectionType type) {
        SmallVector<mlir::Type> convertedTypes;
        for (auto subType : type)
        {
            convertedTypes.push_back(converter.convertType(subType));
        }

        return LLVM::LLVMStructType::getLiteral(type.getContext(), convertedTypes, false);
    });
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TypeScriptToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace
{

struct TypeScriptToLLVMLoweringPass : public PassWrapper<TypeScriptToLLVMLoweringPass, OperationPass<ModuleOp>>
{
    void getDependentDialects(DialectRegistry &registry) const override
    {
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() final;
};

} // end anonymous namespace

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
    LLVMTypeConverter typeConverter(&getContext());

    // Now that the conversion target has been defined, we need to provide the
    // patterns used for lowering. At this point of the compilation process, we
    // have a combination of `typescript`, `affine`, and `std` operations. Luckily, there
    // are already exists a set of patterns to transform `affine` and `std`
    // dialects. These patterns lowering in multiple stages, relying on transitive
    // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
    // patterns must be applied to fully transform an illegal operation into a
    // set of legal ones.
    OwningRewritePatternList patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    populateLoopToStdConversionPatterns(patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
#ifdef ENABLE_ASYNC
    populateAsyncStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
#endif

    // The only remaining operation to lower from the `typescript` dialect, is the PrintOp.
    TsLlvmContext tsLlvmContext;
    patterns.insert<CaptureOpLowering, ReturnOpLowering, ReturnValOpLowering, AddressOfOpLowering, AddressOfConstStringOpLowering,
                    ArithmeticUnaryOpLowering, ArithmeticBinaryOpLowering, AssertOpLowering, CastOpLowering, ConstantOpLowering,
                    CreateOptionalOpLowering, UndefOptionalOpLowering, HasValueOpLowering, ValueOpLowering, SymbolRefOpLowering,
                    GlobalOpLowering, GlobalResultOpLowering, FuncOpLowering, LoadOpLowering, ElementRefOpLowering, PropertyRefOpLowering,
                    ExtractPropertyOpLowering, LogicalBinaryOpLowering, NullOpLowering, NewOpLowering, CreateTupleOpLowering,
                    DeconstructTupleOpLowering, CreateArrayOpLowering, NewEmptyArrayOpLowering, NewArrayOpLowering, PushOpLowering,
                    PopOpLowering, DeleteOpLowering, ParseFloatOpLowering, ParseIntOpLowering, PrintOpLowering, StoreOpLowering,
                    SizeOfOpLowering, InsertPropertyOpLowering, LengthOfOpLowering, StringLengthOpLowering, StringConcatOpLowering,
                    StringCompareOpLowering, CharToStringOpLowering, UndefOpLowering, MemoryCopyOpLowering, LoadSaveValueLowering,
                    ThrowUnwindOpLowering, ThrowCallOpLowering, TrampolineOpLowering, TryOpLowering, CatchOpLowering, VariableOpLowering,
                    InvokeOpLowering, ThisVirtualSymbolRefOpLowering, InterfaceSymbolRefOpLowering, NewInterfaceOpLowering,
                    VTableOffsetRefOpLowering, ThisPropertyRefOpLowering, LoadBoundRefOpLowering, StoreBoundRefOpLowering,
                    CreateBoundRefOpLowering, CreateBoundFunctionOpLowering, GetThisOpLowering, GetMethodOpLowering, TypeOfOpLowering,
                    DebuggerOpLowering, StateLabelOpLowering, SwitchStateOpLowering, UnreachableOpLowering, LandingPadOpLowering,
                    CompareCatchTypeOpLowering, BeginCatchOpLowering, SaveCatchVarOpLowering, EndCatchOpLowering, CallInternalOpLowering>(
        typeConverter, &getContext(), &tsLlvmContext);

    populateTypeScriptConversionPatterns(typeConverter, m);

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "\nDUMP: \n" << module << "\n";);
}

/// Create a pass for lowering operations the remaining `TypeScript` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::typescript::createLowerToLLVMPass()
{
    return std::make_unique<TypeScriptToLLVMLoweringPass>();
}
