#define DEBUG_TYPE "llvm"

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

#define ATTR(attr) StringAttr::get(rewriter.getContext(), attr)
#define IDENT(name) Identifier::get(rewriter.getContext(), name)
#define NAMED_ATTR(name, attr) ArrayAttr::get(rewriter.getContext(), {ATTR(name), ATTR(attr)})

namespace
{
struct TsLlvmContext
{
    // invoke normal, unwind
    DenseMap<Operation *, mlir::Block *> unwind;
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
        for (auto item : op->getOperands())
        {
            auto type = item.getType();
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
                values.push_back(rewriter.create<mlir_ts::ValueOp>(item.getLoc(), o.getElementType(), item));
            }
            else
            {
                values.push_back(item);
            }
        }

        // print new line
        rewriter.create<LLVM::CallOp>(loc, printfFuncOp, values);

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(op);

        return success();
    }
};

class AssertOpLowering : public TsLlvmPattern<mlir_ts::AssertOp>
{
  public:
    using TsLlvmPattern<mlir_ts::AssertOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::AssertOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);
        LLVMCodeHelper ch(op, rewriter, getTypeConverter());

        auto loc = op->getLoc();

        auto line = 0;
        auto column = 0;
        auto fileName = StringRef("");
        TypeSwitch<LocationAttr>(loc).Case<FileLineColLoc>([&](FileLineColLoc loc) {
            fileName = loc.getFilename();
            line = loc.getLine() + 1;
            column = loc.getColumn();
        });

        // Insert the `_assert` declaration if necessary.
        auto i8PtrTy = th.getI8PtrType();
        auto assertFuncOp =
            ch.getOrInsertFunction("_assert", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, rewriter.getI32Type()}));

        // Split block at `assert` operation.
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // Generate IR to call `assert`.
        auto *failureBlock = rewriter.createBlock(opBlock->getParent());

        std::stringstream msgWithNUL;
        msgWithNUL << op.msg().str();

        auto opHash = std::hash<std::string>{}(msgWithNUL.str());

        std::stringstream msgVarName;
        msgVarName << "m_" << opHash;

        std::stringstream fileVarName;
        fileVarName << "f_" << hash_value(fileName);

        std::stringstream fileWithNUL;
        fileWithNUL << fileName.str();

        auto msgCst = ch.getOrCreateGlobalString(msgVarName.str(), msgWithNUL.str());

        auto fileCst = ch.getOrCreateGlobalString(fileVarName.str(), fileName.str());

        // auto nullCst = rewriter.create<LLVM::NullOp>(loc, getI8PtrType(context));

        Value lineNumberRes = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(line));

        rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{msgCst, fileCst, lineNumberRes});
        rewriter.create<LLVM::UnreachableOp>(loc);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(op, op.arg(), continuationBlock, failureBlock);

        return success();
    }
};

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

        // mlir::Value newStringValue = rewriter.create<LLVM::AllocaOp>(op->getLoc(), i8PtrTy, size, true);
        mlir::Value newStringValue = ch.MemoryAllocBitcast(i8PtrTy, size);

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

struct EntryOpLowering : public TsLlvmPattern<mlir_ts::EntryOp>
{
    using TsLlvmPattern<mlir_ts::EntryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::EntryOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        CodeLogicHelper clh(op, rewriter);
        TypeConverterHelper tch(getTypeConverter());

        auto location = op.getLoc();

        mlir::Value allocValue;
        auto anyResult = op.getNumResults() > 0;
        if (anyResult)
        {
            auto result = op.getResult(0);
            allocValue = rewriter.create<LLVM::AllocaOp>(location, tch.convertType(result.getType()), clh.createI32ConstantOf(1));
        }

        // create return block
        auto *opBlock = rewriter.getInsertionBlock();
        auto *region = opBlock->getParent();

        rewriter.createBlock(region);

        if (anyResult)
        {
            auto loadedValue = rewriter.create<LLVM::LoadOp>(op.getLoc(), allocValue);
            rewriter.create<LLVM::ReturnOp>(op.getLoc(), mlir::ValueRange{loadedValue});
            rewriter.replaceOp(op, allocValue);
        }
        else
        {
            rewriter.create<LLVM::ReturnOp>(op.getLoc(), mlir::ValueRange{});
            rewriter.eraseOp(op);
        }

        return success();
    }
};

static mlir::Block *FindReturnBlock(PatternRewriter &rewriter)
{
    auto *region = rewriter.getInsertionBlock()->getParent();
    if (!region)
    {
        return nullptr;
    }

    auto result = std::find_if(region->begin(), region->end(), [&](auto &item) {
        if (item.empty())
        {
            return false;
        }

        auto *op = &item.back();
        // auto name = op->getName().getStringRef();
        auto isReturn = dyn_cast<LLVM::ReturnOp>(op) != nullptr;
        return isReturn;
    });

    if (result == region->end())
    {
        llvm_unreachable("return op. can't be found");
        return nullptr;
    }

    return &*result;
}

struct ReturnOpLowering : public TsLlvmPattern<mlir_ts::ReturnOp>
{
    using TsLlvmPattern<mlir_ts::ReturnOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ReturnOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto retBlock = FindReturnBlock(rewriter);

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
        auto retBlock = FindReturnBlock(rewriter);

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

struct ExitOpLowering : public TsLlvmPattern<mlir_ts::ExitOp>
{
    using TsLlvmPattern<mlir_ts::ExitOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ExitOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        auto retBlock = FindReturnBlock(rewriter);

        rewriter.create<mlir::BranchOp>(op.getLoc(), retBlock);

        rewriter.eraseOp(op);
        return success();
    }
};

struct FuncOpLowering : public TsLlvmPattern<mlir_ts::FuncOp>
{
    using TsLlvmPattern<mlir_ts::FuncOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::FuncOp funcOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
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

        auto newFuncOp = rewriter.create<mlir::FuncOp>(
            funcOp.getLoc(), funcOp.getName(),
            rewriter.getFunctionType(signatureInputsConverter.getConvertedTypes(), signatureResultsConverter.getConvertedTypes()),
            ArrayRef<NamedAttribute>{}, argDictAttrs);
        for (const auto &namedAttr : funcOp->getAttrs())
        {
            if (namedAttr.first == function_like_impl::getTypeAttrName() || namedAttr.first == SymbolTable::getSymbolAttrName())
            {
                continue;
            }

            newFuncOp->setAttr(namedAttr.first, namedAttr.second);
        }

        if (funcOp.personality().hasValue() && funcOp.personality().getValue())
        {
            LLVMRTTIHelperVCWin32 rttih(funcOp, rewriter, typeConverter);
            rttih.setPersonality(newFuncOp);
        }

#ifdef DISABLE_OPT
        // add LLVM attributes to fix issue with shift >> 32
        newFuncOp->setAttr("passthrough", ArrayAttr::get(
                                              {
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
                                              },
                                              rewriter.getContext()));
#endif
        rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
        if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), typeConverter, &signatureInputsConverter)))
        {
            return failure();
        }

        rewriter.eraseOp(funcOp);

        return success();
    }
};

struct CallOpLowering : public TsLlvmPattern<mlir_ts::CallOp>
{
    using TsLlvmPattern<mlir_ts::CallOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        if (auto unwind = tsLlvmContext->unwind[op])
        {
            {
                OpBuilder::InsertionGuard guard(rewriter);

                auto *opBlock = rewriter.getInsertionBlock();
                auto opPosition = rewriter.getInsertionPoint();
                auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

                rewriter.setInsertionPointToEnd(opBlock);

                rewriter.create<LLVM::InvokeOp>(op->getLoc(), op.getResultTypes(), op.calleeAttr(), op.getArgOperands(), continuationBlock,
                                                ValueRange{}, unwind, ValueRange{});
            }

            rewriter.eraseOp(op);

            return success();
        }

        // just replace
        rewriter.replaceOpWithNewOp<mlir::CallOp>(op, op.getCallee(), op.getResultTypes(), op.getArgOperands());
        return success();
    }
};

struct CallIndirectOpLowering : public TsLlvmPattern<mlir_ts::CallIndirectOp>
{
    using TsLlvmPattern<mlir_ts::CallIndirectOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::CallIndirectOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        // just replace
        rewriter.replaceOpWithNewOp<mlir::CallIndirectOp>(op, op.getResultTypes(), op.getCallee(), op.getArgOperands());
        return success();
    }
};

struct InvokeOpLowering : public TsLlvmPattern<mlir_ts::InvokeOp>
{
    using TsLlvmPattern<mlir_ts::InvokeOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::InvokeOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        // just replace
        rewriter.replaceOpWithNewOp<LLVM::InvokeOp>(op, op.getResultTypes(), op.getOperands(), op.normalDest(), op.normalDestOperands(),
                                                    op.unwindDest(), op.unwindDestOperands());
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
        auto res = op.res();
        auto resType = res.getType();

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
        auto isCaptured = varOp.captured().hasValue() && varOp.captured().getValue();

        LLVM_DEBUG(llvm::dbgs() << ">>> variable allocation: " << storageType << "\n";);

        auto allocated = isCaptured ? ch.MemoryAllocBitcast(llvmReferenceType, storageType)
                                    : rewriter.create<LLVM::AllocaOp>(location, llvmReferenceType, clh.createI32ConstantOf(1));

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
        CodeLogicHelper clh(deleteOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);

        auto loc = deleteOp.getLoc();

        auto i8PtrTy = th.getI8PtrType();
        auto freeFuncOp = ch.getOrInsertFunction("free", th.getFunctionType(th.getVoidType(), {i8PtrTy}));

        auto casted = clh.castToI8Ptr(deleteOp.reference());

        rewriter.replaceOpWithNewOp<LLVM::CallOp>(deleteOp, freeFuncOp, ValueRange{casted});
        return success();
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
            value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::eq, CmpFOp, CmpFPredicate, CmpFPredicate::OEQ>(
                logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
            break;
        case SyntaxKind::ExclamationEqualsToken:
        case SyntaxKind::ExclamationEqualsEqualsToken:
            value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::ne, CmpFOp, CmpFPredicate, CmpFPredicate::ONE>(
                logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
            break;
        case SyntaxKind::GreaterThanToken:
            value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::sgt, CmpFOp, CmpFPredicate, CmpFPredicate::OGT>(
                logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
            break;
        case SyntaxKind::GreaterThanEqualsToken:
            value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::sge, CmpFOp, CmpFPredicate, CmpFPredicate::OGE>(
                logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
            break;
        case SyntaxKind::LessThanToken:
            value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::slt, CmpFOp, CmpFPredicate, CmpFPredicate::OLT>(
                logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
            break;
        case SyntaxKind::LessThanEqualsToken:
            value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::sle, CmpFOp, CmpFPredicate, CmpFPredicate::OLE>(
                logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
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

        auto elementType = loadOp.reference().getType().cast<mlir_ts::RefType>().getElementType();
        auto elementTypeConverted = tch.convertType(elementType);

        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(loadOp, elementTypeConverted, loadOp.reference());
        return success();
    }
};

struct StoreOpLowering : public TsLlvmPattern<mlir_ts::StoreOp>
{
    using TsLlvmPattern<mlir_ts::StoreOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::StoreOp storeOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
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

        auto addr = ch.GetAddressOfStructElement(propertyRefOp.getResult().getType(), propertyRefOp.objectRef(), propertyRefOp.position());
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
        // TODO: include initialize block
        lch.createGlobalVarIfNew(globalOp.sym_name(), getTypeConverter()->convertType(globalOp.type()), globalOp.valueAttr(),
                                 globalOp.constant(), globalOp.getInitializerRegion());
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
        auto value = lch.getAddressOfGlobalVar(addressOfOp.global_name());
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
        auto parentModule = addressOfConstStringOp->getParentOfType<ModuleOp>();

        if (auto global = parentModule.lookupSymbol<LLVM::GlobalOp>(addressOfConstStringOp.global_name()))
        {
            auto loc = addressOfConstStringOp->getLoc();
            auto globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
            auto cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getI64Type(), th.getIndexAttrValue(0));
            rewriter.replaceOpWithNewOp<LLVM::GEPOp>(addressOfConstStringOp, th.getI8PtrType(), globalPtr, ArrayRef<Value>({cst0, cst0}));

            return success();
        }

        return failure();
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

        auto valueType = memoryCopyOp.src().getType().cast<LLVM::LLVMPointerType>().getElementType();

        auto size = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), valueType);
        values.push_back(size);

        auto immarg = clh.createI1ConstantOf(false);
        values.push_back(immarg);

        // print new line
        rewriter.create<LLVM::CallOp>(loc, copyMemFuncOp, values);

        // Notify the rewriter that this operation has been removed.
        rewriter.eraseOp(memoryCopyOp);

        return success();
    }
};

struct ThrowOpLoweringVCWin32 : public TsLlvmPattern<mlir_ts::ThrowOp>
{
    // add LLVM attributes to fix issue with shift >> 32
    using TsLlvmPattern<mlir_ts::ThrowOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThrowOp throwOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        LLVMCodeHelper ch(throwOp, rewriter, getTypeConverter());
        CodeLogicHelper clh(throwOp, rewriter);
        TypeConverterHelper tch(getTypeConverter());
        TypeHelper th(rewriter);
        LLVMRTTIHelperVCWin32 rttih(throwOp, rewriter, *getTypeConverter());

        auto loc = throwOp.getLoc();

        auto throwInfoTy = rttih.getThrowInfoTy();
        auto throwInfoPtrTy = rttih.getThrowInfoPtrTy();

        auto i8PtrTy = th.getI8PtrType();

        auto cxxThrowException =
            ch.getOrInsertFunction("_CxxThrowException", th.getFunctionType(th.getVoidType(), {i8PtrTy, throwInfoPtrTy}));

        // prepare RTTI info for throw
        {
            auto parentModule = throwOp->getParentOfType<ModuleOp>();

            OpBuilder::InsertionGuard guard(rewriter);

            rewriter.setInsertionPointToStart(parentModule.getBody());
            ch.seekLast(parentModule.getBody());

            // ??_7type_info@@6B@
            rttih.typeInfo(loc);

            // ??_R0N@8
            rttih.typeDescriptor2(loc);

            // __ImageBase
            rttih.imageBase(loc);

            // _CT??_R0N@88
            rttih.catchableType(loc);

            // _CTA1N
            rttih.catchableArrayType(loc);

            // _TI1N
            rttih.throwInfo(loc);
        }

        // prepare first param
        // we need temp var
        auto value = rewriter.create<mlir_ts::VariableOp>(loc, mlir_ts::RefType::get(throwOp.exception().getType()), throwOp.exception(),
                                                          rewriter.getBoolAttr(false));

        auto throwInfoPtr = rewriter.create<mlir::ConstantOp>(loc, throwInfoPtrTy, FlatSymbolRefAttr::get(rewriter.getContext(), "_TI1N"));

        // throw exception
        rewriter.create<LLVM::CallOp>(loc, cxxThrowException, ValueRange{clh.castToI8Ptr(value), throwInfoPtr});

        rewriter.eraseOp(throwOp);
        return success();
    }
};

struct TryOpLowering : public TsLlvmPattern<mlir_ts::TryOp>
{
    using TsLlvmPattern<mlir_ts::TryOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::TryOp tryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
    {
        TypeHelper th(rewriter);

        Location loc = tryOp.getLoc();

        OpBuilder::InsertionGuard guard(rewriter);
        Block *currentBlock = rewriter.getInsertionBlock();
        Block *continuation = rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

        auto *bodyRegion = &tryOp.body().front();
        auto *bodyRegionLast = &tryOp.body().back();
        auto *catchesRegion = &tryOp.catches().front();
        auto *catchesRegionLast = &tryOp.catches().back();
        // auto *finallyBlockRegion = &tryOp.finallyBlock().front();
        auto *finallyBlockRegionLast = &tryOp.finallyBlock().back();

        // logic to set Invoke attribute CallOp
        auto visitorCallOpContinue = [&](Operation *op) {
            if (auto callOp = dyn_cast_or_null<mlir_ts::CallOp>(op))
            {
                tsLlvmContext->unwind[op] = catchesRegion;
            }
            else if (auto callIndirectOp = dyn_cast_or_null<mlir_ts::CallIndirectOp>(op))
            {
                tsLlvmContext->unwind[op] = catchesRegion;
            }
        };

        // Branch to the "body" region.
        rewriter.setInsertionPointToEnd(currentBlock);
        rewriter.create<BranchOp>(loc, bodyRegion, ValueRange{});

        tryOp.body().walk(visitorCallOpContinue);

        rewriter.inlineRegionBefore(tryOp.body(), continuation);

        rewriter.inlineRegionBefore(tryOp.catches(), continuation);

        rewriter.inlineRegionBefore(tryOp.finallyBlock(), continuation);

        // Body:catch vars
        rewriter.setInsertionPointToStart(bodyRegion);

        auto catch1 = rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());

        // Body:exit -> replace ResultOp with br
        rewriter.setInsertionPointToEnd(bodyRegionLast);

        auto resultOp = cast<mlir_ts::ResultOp>(bodyRegionLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(resultOp, continuation, ValueRange{});

        // catches;landingpad
        rewriter.setInsertionPointToStart(catchesRegion);

        auto landingPadTypeWin32 =
            LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI8PtrType(), th.getI32Type(), th.getI8PtrType()}, false);
        auto landingPadOp = rewriter.create<LLVM::LandingpadOp>(loc, landingPadTypeWin32, false, ValueRange{catch1});
        // rewriter.create<LLVM::ResumeOp>(loc, landingPadOp);

        // catches:exit
        rewriter.setInsertionPointToEnd(catchesRegionLast);

        auto yieldOpCatches = cast<mlir_ts::ResultOp>(catchesRegionLast->getTerminator());
        // rewriter.replaceOpWithNewOp<BranchOp>(yieldOpCatches, continuation, yieldOpCatches.results());
        rewriter.replaceOpWithNewOp<LLVM::ResumeOp>(yieldOpCatches, landingPadOp);

        // finally:exit
        rewriter.setInsertionPointToEnd(finallyBlockRegionLast);

        auto yieldOpFinallyBlock = cast<mlir_ts::ResultOp>(finallyBlockRegionLast->getTerminator());
        rewriter.replaceOpWithNewOp<BranchOp>(yieldOpFinallyBlock, continuation, yieldOpFinallyBlock.results());

        rewriter.replaceOp(tryOp, continuation->getArguments());

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
        CodeLogicHelper clh(trampolineOp, rewriter);
        LLVMCodeHelper ch(trampolineOp, rewriter, getTypeConverter());

        auto i8PtrTy = th.getI8PtrType();

        auto initTrampolineFuncOp =
            ch.getOrInsertFunction("llvm.init.trampoline", th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, i8PtrTy}));
        auto adjustTrampolineFuncOp = ch.getOrInsertFunction("llvm.adjust.trampoline", th.getFunctionType(i8PtrTy, {i8PtrTy}));
        // Win32 specifics
        auto enableExecuteStackFuncOp = ch.getOrInsertFunction("__enable_execute_stack", th.getFunctionType(th.getVoidType(), {i8PtrTy}));

        // allocate temp trampoline
        auto bufferType = th.getPointerType(th.getI8Array(TRAMPOLINE_SIZE));

        auto const0 = clh.createI32ConstantOf(0);

        // auto trampoline = rewriter.create<LLVM::AllocaOp>(location, bufferType, clh.createI32ConstantOf(1));
        // auto trampolinePtr = rewriter.create<LLVM::GEPOp>(location, i8PtrTy, ValueRange{trampoline, const0, const0});

        auto trampoline = ch.MemoryAlloc(bufferType);
        auto trampolinePtr = rewriter.create<LLVM::GEPOp>(location, i8PtrTy, ValueRange{trampoline, const0});

        // init trampoline
        rewriter.create<LLVM::CallOp>(
            location, initTrampolineFuncOp,
            ValueRange{trampolinePtr, clh.castToI8Ptr(trampolineOp.callee()), clh.castToI8Ptr(trampolineOp.data_reference())});

        auto callAdjustedTrampoline = rewriter.create<LLVM::CallOp>(location, adjustTrampolineFuncOp, ValueRange{trampolinePtr});
        auto adjustedTrampolinePtr = callAdjustedTrampoline.getResult(0);

        rewriter.create<LLVM::CallOp>(location, enableExecuteStackFuncOp, ValueRange{adjustedTrampolinePtr});

        mlir::Value castFunc = rewriter.create<mlir_ts::CastOp>(location, trampolineOp.getType(), adjustedTrampolinePtr);
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
        auto captureStoreType = captureRefType.cast<mlir_ts::RefType>().getElementType().cast<mlir_ts::TupleType>();

        LLVM_DEBUG(llvm::dbgs() << "\n ...capture store type: " << captureStoreType << "\n\n";);

        mlir::Value allocTempStorage =
            rewriter.create<mlir_ts::VariableOp>(location, captureRefType, mlir::Value(), rewriter.getBoolAttr(false));

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

struct ThisVirtualSymbolRefLowering : public TsLlvmPattern<mlir_ts::ThisVirtualSymbolRefOp>
{
    using TsLlvmPattern<mlir_ts::ThisVirtualSymbolRefOp>::TsLlvmPattern;

    LogicalResult matchAndRewrite(mlir_ts::ThisVirtualSymbolRefOp thisVirtualSymbolRefOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final
    {
        Location loc = thisVirtualSymbolRefOp.getLoc();

        TypeHelper th(rewriter);
        CodeLogicHelper clh(thisVirtualSymbolRefOp, rewriter);

        auto ptrToArrOfPtrs = rewriter.create<mlir_ts::CastOp>(loc, th.getI8PtrPtrPtrType(), thisVirtualSymbolRefOp.vtable());

        auto index = clh.createI32ConstantOf(thisVirtualSymbolRefOp.index());
        auto methodPtrPtr = rewriter.create<LLVM::GEPOp>(loc, th.getI8PtrPtrType(), ptrToArrOfPtrs, ValueRange{index});
        auto methodPtr = rewriter.create<LLVM::LoadOp>(loc, methodPtrPtr);
        auto methodTyped = rewriter.create<mlir_ts::CastOp>(loc, thisVirtualSymbolRefOp.getType(), methodPtr);

        rewriter.replaceOp(thisVirtualSymbolRefOp, ValueRange{methodTyped});

        return success();
    }
};

static void populateTypeScriptConversionPatterns(LLVMTypeConverter &converter, mlir::ModuleOp &m)
{
    converter.addConversion([&](mlir_ts::AnyType type) { return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8)); });

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

    // The only remaining operation to lower from the `typescript` dialect, is the PrintOp.
    TsLlvmContext tsLlvmContext;
    patterns.insert<CallOpLowering, CallIndirectOpLowering, CaptureOpLowering, ExitOpLowering, ReturnOpLowering, ReturnValOpLowering,
                    AddressOfOpLowering, AddressOfConstStringOpLowering, ArithmeticUnaryOpLowering, ArithmeticBinaryOpLowering,
                    AssertOpLowering, CastOpLowering, ConstantOpLowering, CreateOptionalOpLowering, UndefOptionalOpLowering,
                    HasValueOpLowering, ValueOpLowering, SymbolRefOpLowering, GlobalOpLowering, GlobalResultOpLowering, EntryOpLowering,
                    FuncOpLowering, LoadOpLowering, ElementRefOpLowering, PropertyRefOpLowering, ExtractPropertyOpLowering,
                    LogicalBinaryOpLowering, NullOpLowering, NewOpLowering, CreateArrayOpLowering, NewEmptyArrayOpLowering,
                    NewArrayOpLowering, PushOpLowering, PopOpLowering, DeleteOpLowering, ParseFloatOpLowering, ParseIntOpLowering,
                    PrintOpLowering, StoreOpLowering, SizeOfOpLowering, InsertPropertyOpLowering, LengthOfOpLowering,
                    StringLengthOpLowering, StringConcatOpLowering, StringCompareOpLowering, CharToStringOpLowering, UndefOpLowering,
                    MemoryCopyOpLowering, LoadSaveValueLowering, ThrowOpLoweringVCWin32, TrampolineOpLowering, TryOpLowering,
                    VariableOpLowering, InvokeOpLowering, ThisVirtualSymbolRefLowering>(typeConverter, &getContext(), &tsLlvmContext);

    populateTypeScriptConversionPatterns(typeConverter, m);

    // We want to completely lower to LLVM, so we use a `FullConversion`. This
    // ensures that only legal operations will remain after the conversion.
    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
    {
        signalPassFailure();
    }
}

/// Create a pass for lowering operations the remaining `TypeScript` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::typescript::createLowerToLLVMPass()
{
    return std::make_unique<TypeScriptToLLVMLoweringPass>();
}
