#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

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
    class PrintOpLowering : public OpConversionPattern<mlir_ts::PrintOp>
    {
    public:
        using OpConversionPattern<mlir_ts::PrintOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::PrintOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            LLVMCodeHelper ch(op, rewriter);
            TypeConverterHelper tch(getTypeConverter());

            auto loc = op->getLoc();

            // Get a symbol reference to the printf function, inserting it if necessary.
            auto printfFuncOp =
                ch.getOrInsertFunction(
                    "printf",
                    th.getFunctionType(rewriter.getI32Type(), th.getI8PtrType(), true));

            std::stringstream format;
            auto count = 0;

            std::function<void(mlir::Type)> processFormatForType = [&](mlir::Type type)
            {
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
                    values.push_back(rewriter.create<LLVM::SelectOp>(
                        item.getLoc(),
                        item,
                        ch.getOrCreateGlobalString("__true__", std::string("true")),
                        ch.getOrCreateGlobalString("__false__", std::string("false"))));
                }
                else if (auto o = type.dyn_cast_or_null<mlir_ts::OptionalType>())
                {
                    auto boolPart = rewriter.create<mlir_ts::HasValueOp>(item.getLoc(), th.getBooleanType(), item);
                    values.push_back(rewriter.create<LLVM::SelectOp>(
                        item.getLoc(),
                        boolPart,
                        ch.getOrCreateGlobalString("__true__", std::string("true")),
                        ch.getOrCreateGlobalString("__false__", std::string("false"))));
                    values.push_back(rewriter.create<mlir_ts::ValueOp>(
                        item.getLoc(),
                        o.getElementType(),
                        item));
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

    class AssertOpLowering : public OpConversionPattern<mlir_ts::AssertOp>
    {
    public:
        using OpConversionPattern<mlir_ts::AssertOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::AssertOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            LLVMCodeHelper ch(op, rewriter);

            auto loc = op->getLoc();

            auto line = 0;
            auto column = 0;
            auto fileName = StringRef("");
            TypeSwitch<LocationAttr>(loc)
                .Case<FileLineColLoc>([&](FileLineColLoc loc) {
                    fileName = loc.getFilename();
                    line = loc.getLine();
                    column = loc.getColumn();
                });

            // Insert the `_assert` declaration if necessary.
            auto i8PtrTy = th.getI8PtrType();
            auto assertFuncOp =
                ch.getOrInsertFunction(
                    "_assert",
                    th.getFunctionType(th.getVoidType(), {i8PtrTy, i8PtrTy, rewriter.getI32Type()}));

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

            //auto nullCst = rewriter.create<LLVM::NullOp>(loc, getI8PtrType(context));

            Value lineNumberRes =
                rewriter.create<LLVM::ConstantOp>(
                    loc,
                    rewriter.getI32Type(),
                    rewriter.getI32IntegerAttr(line));

            rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{msgCst, fileCst, lineNumberRes});
            rewriter.create<LLVM::UnreachableOp>(loc);

            // Generate assertion test.
            rewriter.setInsertionPointToEnd(opBlock);
            rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
                op,
                op.arg(),
                continuationBlock,
                failureBlock);

            return success();
        }
    };

    class ParseIntOpLowering : public OpConversionPattern<mlir_ts::ParseIntOp>
    {
    public:
        using OpConversionPattern<mlir_ts::ParseIntOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ParseIntOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            LLVMCodeHelper ch(op, rewriter);

            // Insert the `atoi` declaration if necessary.
            auto i8PtrTy = th.getI8PtrType();
            auto parseIntFuncOp =
                ch.getOrInsertFunction(
                    "atoi",
                    th.getFunctionType(rewriter.getI32Type(), {i8PtrTy}));

            rewriter.replaceOpWithNewOp<LLVM::CallOp>(
                op,
                parseIntFuncOp,
                op->getOperands());

            return success();
        }
    };

    class ParseFloatOpLowering : public OpConversionPattern<mlir_ts::ParseFloatOp>
    {
    public:
        using OpConversionPattern<mlir_ts::ParseFloatOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ParseFloatOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            LLVMCodeHelper ch(op, rewriter);

            // Insert the `atof` declaration if necessary.
            auto i8PtrTy = th.getI8PtrType();
            auto parseFloatFuncOp =
                ch.getOrInsertFunction(
                    "atof",
                    th.getFunctionType(rewriter.getF32Type(), {i8PtrTy}));

            rewriter.replaceOpWithNewOp<LLVM::CallOp>(
                op,
                parseFloatFuncOp,
                op->getOperands());

            return success();
        }
    };

    class StringLengthOpLowering : public OpConversionPattern<mlir_ts::StringLengthOp>
    {
    public:
        using OpConversionPattern<mlir_ts::StringLengthOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::StringLengthOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            LLVMCodeHelper ch(op, rewriter);

            auto loc = op->getLoc();
            auto i8PtrTy = th.getI8PtrType();

            auto strlenFuncOp =
                ch.getOrInsertFunction(
                    "strlen",
                    th.getFunctionType(th.getI64Type(), {i8PtrTy}));                    

            // calc size
            auto size = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, ValueRange{op.op()});
            rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, th.getI32Type(), size.getResult(0));

            return success();
        }
    }; 

    class StringConcatOpLowering : public OpConversionPattern<mlir_ts::StringConcatOp>
    {
    public:
        using OpConversionPattern<mlir_ts::StringConcatOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::StringConcatOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            CodeLogicHelper clh(op, rewriter);
            LLVMCodeHelper ch(op, rewriter);

            auto loc = op->getLoc();

            // TODO implement str concat
            auto i8PtrTy = th.getI8PtrType();
            auto i8PtrPtrTy = th.getI8PtrPtrType();

            auto strlenFuncOp =
                ch.getOrInsertFunction(
                    "strlen",
                    th.getFunctionType(rewriter.getI64Type(), {i8PtrTy}));                    
            auto strcpyFuncOp =
                ch.getOrInsertFunction(
                    "strcpy",
                    th.getFunctionType(i8PtrTy, {i8PtrTy, i8PtrTy}));                    
            auto strcatFuncOp =
                ch.getOrInsertFunction(
                    "strcat",
                    th.getFunctionType(i8PtrTy, {i8PtrTy, i8PtrTy}));   

            mlir::Value size = clh.createI64ConstantOf(1);
            // calc size
            for (auto op : op.ops())
            {
                auto size1 = rewriter.create<LLVM::CallOp>(loc, strlenFuncOp, op);
                size = rewriter.create<LLVM::AddOp>(loc, rewriter.getI64Type(), ValueRange{size, size1.getResult(0)});
            }

            mlir::Value newStringValue = rewriter.create<LLVM::AllocaOp>(op->getLoc(), i8PtrTy, size, true);            

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

    class StringCompareOpLowering : public OpConversionPattern<mlir_ts::StringCompareOp>
    {
    public:
        using OpConversionPattern<mlir_ts::StringCompareOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::StringCompareOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            CodeLogicHelper clh(op, rewriter);
            LLVMCodeHelper ch(op, rewriter);
            LLVMTypeConverterHelper llvmtch(*(LLVMTypeConverter *)getTypeConverter());

            auto loc = op->getLoc();

            auto i8PtrTy = th.getI8PtrType();

            // compare bodies
            auto strcmpFuncOp =
                ch.getOrInsertFunction(
                    "strcmp",
                    th.getFunctionType(th.getI32Type(), {i8PtrTy, i8PtrTy}));        

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

            auto result = clh.conditionalExpressionLowering(th.getBooleanType(), ptrCmpResult, 
                [&](OpBuilder & builder, Location loc) 
                {
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
                [&](OpBuilder & builder, Location loc) 
                {
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

    class CharToStringOpLowering : public OpConversionPattern<mlir_ts::CharToStringOp>
    {
    public:
        using OpConversionPattern<mlir_ts::CharToStringOp>::OpConversionPattern;

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

    struct ConstantOpLowering : public OpConversionPattern<mlir_ts::ConstantOp>
    {
        using OpConversionPattern<mlir_ts::ConstantOp>::OpConversionPattern;

        std::string calc_hash_value(ArrayAttr &arrayAttr, const char* prefix) const
        {
            auto opHash = 0ULL;
            for (auto item : arrayAttr)
            {
                opHash ^= hash_value(item) + 0x9e3779b9 + (opHash<<6) + (opHash>>2);
            }

            // calculate name;
            std::stringstream vecVarName;
            vecVarName << prefix << opHash;     

            return vecVarName.str();
        }

        template <typename T, typename TOp>
        void getOrCreateGlobalArray(TOp constantOp, T type, ConversionPatternRewriter &rewriter) const
        {
            LLVMCodeHelper ch(constantOp, rewriter);
            TypeConverterHelper tch(getTypeConverter());

            auto elementType = type.cast<T>().getElementType();
            auto llvmElementType = tch.convertType(elementType);
            auto arrayAttr = constantOp.value().dyn_cast_or_null<ArrayAttr>();

            auto vecVarName = calc_hash_value(arrayAttr, "a_");      

            auto arrayFirstElementAddrCst = ch.getOrCreateGlobalArray(
                elementType,
                vecVarName, 
                llvmElementType,
                arrayAttr.size(),
                arrayAttr);

            rewriter.replaceOp(constantOp, arrayFirstElementAddrCst);            
        }

        template <typename T, typename TOp>
        void getOrCreateGlobalTuple(TOp constantOp, T type, ConversionPatternRewriter &rewriter) const
        {
            LLVMCodeHelper ch(constantOp, rewriter);
            TypeConverterHelper tch(getTypeConverter());

            auto arrayAttr = constantOp.value().dyn_cast_or_null<ArrayAttr>();

            auto varName = calc_hash_value(arrayAttr, "tp_");      

            auto convertedTupleType = tch.convertType(type);
            auto tupleConstPtr = ch.getOrCreateGlobalTuple(convertedTupleType, varName, arrayAttr);

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
                LLVMCodeHelper ch(constantOp, rewriter);

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

            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(constantOp, tch.convertType(type), constantOp.getValue());
            return success();
        }
    };

    struct SymbolRefOpLowering : public OpConversionPattern<mlir_ts::SymbolRefOp>
    {
        using OpConversionPattern<mlir_ts::SymbolRefOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::SymbolRefOp symbolRefOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(getTypeConverter());
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(symbolRefOp, tch.convertType(symbolRefOp.getType()), symbolRefOp.getValue());
            return success();            
        }
    };

    struct NullOpLowering : public OpConversionPattern<mlir_ts::NullOp>
    {
        using OpConversionPattern<mlir_ts::NullOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::NullOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(getTypeConverter());
            rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, tch.convertType(op.getType()));
            return success();
        }
    };

    class UndefOpLowering : public OpConversionPattern<mlir_ts::UndefOp>
    {
    public:
        using OpConversionPattern<mlir_ts::UndefOp>::OpConversionPattern;

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

    struct EntryOpLowering : public OpConversionPattern<mlir_ts::EntryOp>
    {
        using OpConversionPattern<mlir_ts::EntryOp>::OpConversionPattern;

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
                allocValue =
                    rewriter.create<LLVM::AllocaOp>(
                        location,
                        tch.convertType(result.getType()),
                        clh.createI32ConstantOf(1));
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
            //auto name = op->getName().getStringRef();
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

    struct ReturnOpLowering : public OpRewritePattern<mlir_ts::ReturnOp>
    {
        using OpRewritePattern<mlir_ts::ReturnOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::ReturnOp op, PatternRewriter &rewriter) const final
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

    struct ReturnValOpLowering : public OpRewritePattern<mlir_ts::ReturnValOp>
    {
        using OpRewritePattern<mlir_ts::ReturnValOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::ReturnValOp op, PatternRewriter &rewriter) const final
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

    struct ExitOpLowering : public OpConversionPattern<mlir_ts::ExitOp>
    {
        using OpConversionPattern<mlir_ts::ExitOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ExitOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto retBlock = FindReturnBlock(rewriter);

            rewriter.create<mlir::BranchOp>(op.getLoc(), retBlock);

            rewriter.eraseOp(op);
            return success();
        }
    };

    struct FuncOpLowering : public OpConversionPattern<mlir_ts::FuncOp>
    {
        using OpConversionPattern<mlir_ts::FuncOp>::OpConversionPattern;

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

            auto newFuncOp = rewriter.create<mlir::FuncOp>(
                funcOp.getLoc(),
                funcOp.getName(),
                rewriter.getFunctionType(signatureInputsConverter.getConvertedTypes(), signatureResultsConverter.getConvertedTypes()));
            for (const auto &namedAttr : funcOp.getAttrs())
            {
                if (namedAttr.first == impl::getTypeAttrName() ||
                    namedAttr.first == SymbolTable::getSymbolAttrName())
                {
                    continue;
                }

                newFuncOp->setAttr(namedAttr.first, namedAttr.second);
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

    struct CallOpLowering : public OpRewritePattern<mlir_ts::CallOp>
    {
        using OpRewritePattern<mlir_ts::CallOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::CallOp op, PatternRewriter &rewriter) const final
        {
            // just replace
            rewriter.replaceOpWithNewOp<mlir::CallOp>(
                op,
                op.getCallee(),
                op.getResultTypes(),
                op.getArgOperands());
            return success();
        }
    };

    struct CallIndirectOpLowering : public OpRewritePattern<mlir_ts::CallIndirectOp>
    {
        using OpRewritePattern<mlir_ts::CallIndirectOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::CallIndirectOp op, PatternRewriter &rewriter) const final
        {
            // just replace
            rewriter.replaceOpWithNewOp<mlir::CallIndirectOp>(
                op,
                op.getResultTypes(),
                op.getCallee(),
                op.getArgOperands());
            return success();
        }
    };    

    struct CastOpLowering : public OpConversionPattern<mlir_ts::CastOp>
    {
        using OpConversionPattern<mlir_ts::CastOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::CastOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto loc = op->getLoc();

            TypeConverterHelper tch(getTypeConverter());

            auto in = op.in();
            auto res = op.res();
            auto inType = in.getType();
            auto resType = res.getType();
            auto inLLVMType = tch.convertType(inType);
            auto resLLVMType = tch.convertType(resType);

            CastLogicHelper castLogic(op, rewriter);
            auto result = castLogic.cast(in, inLLVMType, resType, resLLVMType);
            if (!result)
            {
                return failure();
            }

            rewriter.replaceOp(op, result);

            return success();
        }
    };

    struct VariableOpLowering : public OpConversionPattern<mlir_ts::VariableOp>
    {
        using OpConversionPattern<mlir_ts::VariableOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::VariableOp varOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            CodeLogicHelper clh(varOp, rewriter);
            TypeConverterHelper tch(getTypeConverter());

            auto location = varOp.getLoc();

            auto allocated =
                rewriter.create<LLVM::AllocaOp>(
                    location,
                    tch.convertType(varOp.reference().getType()),
                    clh.createI32ConstantOf(1));
            auto value = varOp.initializer();
            if (value)
            {
                // allocate copy
                if (varOp.copy())
                {                    
                    // ...
                    //emitError(location) << "type: " << varOp.initializer().getType() << " llvm:" << tch.convertType(varOp.initializer().getType()) << " llvm as value:" << tch.convertTypeAsValue(varOp.initializer().getType());
                    auto copyAllocated =
                        rewriter.create<LLVM::AllocaOp>(
                            location,
                            tch.convertTypeAsValue(varOp.initializer().getType()),
                            clh.createI32ConstantOf(1));

                    rewriter.create<mlir_ts::MemoryCopyOp>(location, copyAllocated, value);

                    value = copyAllocated;
                }

                rewriter.create<LLVM::StoreOp>(location, value, allocated);
            }

            rewriter.replaceOp(varOp, ValueRange{allocated});
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
                //lhs = clh.createI32ConstantOf(-1);
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

    struct ArithmeticUnaryOpLowering : public OpConversionPattern<mlir_ts::ArithmeticUnaryOp>
    {
        using OpConversionPattern<mlir_ts::ArithmeticUnaryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ArithmeticUnaryOp arithmeticUnaryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

    struct ArithmeticBinaryOpLowering : public OpConversionPattern<mlir_ts::ArithmeticBinaryOp>
    {
        using OpConversionPattern<mlir_ts::ArithmeticBinaryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ArithmeticBinaryOp arithmeticBinaryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto opCode = (SyntaxKind)arithmeticBinaryOp.opCode();
            switch (opCode)
            {
            case SyntaxKind::PlusToken:
                if (arithmeticBinaryOp->getOperand(0).getType().isa<mlir_ts::StringType>())
                {
                    rewriter.replaceOpWithNewOp<mlir_ts::StringConcatOp>(
                        arithmeticBinaryOp, 
                        mlir_ts::StringType::get(rewriter.getContext()), 
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
                BinOp<mlir_ts::ArithmeticBinaryOp, PowFOp, PowFOp>(arithmeticBinaryOp, rewriter);
                return success();                    

            default:
                llvm_unreachable("not implemented");
            }
        }
    };

    struct LogicalBinaryOpLowering : public OpConversionPattern<mlir_ts::LogicalBinaryOp>
    {
        using OpConversionPattern<mlir_ts::LogicalBinaryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::LogicalBinaryOp logicalBinaryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto op = (SyntaxKind)logicalBinaryOp.opCode();

            // int and float
            mlir::Value value;
            switch (op)
            {
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
                value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::eq,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OEQ>(logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                break;
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
                value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::ne,
                        CmpFOp, CmpFPredicate, CmpFPredicate::ONE>(logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                break;
            case SyntaxKind::GreaterThanToken:
                value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::sgt,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OGT>(logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                break;
            case SyntaxKind::GreaterThanEqualsToken:
                value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::sge,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OGE>(logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                break;
            case SyntaxKind::LessThanToken:
                value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::slt,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OLT>(logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                break;
            case SyntaxKind::LessThanEqualsToken:
                value = LogicOp<CmpIOp, CmpIPredicate, CmpIPredicate::sle,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OLE>(logicalBinaryOp, op, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                break;
            default:
                llvm_unreachable("not implemented");
            }

            rewriter.replaceOp(logicalBinaryOp, value);
            return success();
        }
    };

    struct LoadOpLowering : public OpConversionPattern<mlir_ts::LoadOp>
    {
        using OpConversionPattern<mlir_ts::LoadOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::LoadOp loadOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            TypeConverterHelper tch(getTypeConverter());
            CodeLogicHelper clh(loadOp, rewriter);

            auto elementType = loadOp.reference().getType().cast<mlir_ts::RefType>().getElementType();
            auto elementTypeConverted = tch.convertType(elementType);

            rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
                loadOp,
                elementTypeConverted,
                loadOp.reference());
            return success();
        }
    };

    struct StoreOpLowering : public OpConversionPattern<mlir_ts::StoreOp>
    {
        using OpConversionPattern<mlir_ts::StoreOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::StoreOp storeOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, storeOp.value(), storeOp.reference());
            return success();
        }
    };
    
    struct ElementRefOpLowering : public OpConversionPattern<mlir_ts::ElementRefOp>
    {
        using OpConversionPattern<mlir_ts::ElementRefOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ElementRefOp elementOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            LLVMCodeHelper ch(elementOp, rewriter, getTypeConverter());

            auto addr = ch.GetAddressOfArrayElement(elementOp.getResult().getType(), elementOp.array(), elementOp.index());
            rewriter.replaceOp(elementOp, addr);
            return success();
        }
    };

    struct ExtractPropertyOpLowering : public OpConversionPattern<mlir_ts::ExtractPropertyOp>
    {
        using OpConversionPattern<mlir_ts::ExtractPropertyOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ExtractPropertyOp extractPropertyOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(getTypeConverter());

            rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
                extractPropertyOp, 
                tch.convertType(extractPropertyOp.getType()), 
                extractPropertyOp.object(), 
                extractPropertyOp.position());

            return success();
        }
    };

    struct InsertPropertyOpLowering : public OpConversionPattern<mlir_ts::InsertPropertyOp>
    {
        using OpConversionPattern<mlir_ts::InsertPropertyOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::InsertPropertyOp insertPropertyOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(getTypeConverter());
            auto loc = insertPropertyOp->getLoc();

            rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
                insertPropertyOp, 
                tch.convertType(insertPropertyOp.object().getType()),
                insertPropertyOp.object(), 
                insertPropertyOp.value(),
                insertPropertyOp.position());

            return success();
        }
    };     

    struct PropertyRefOpLowering : public OpConversionPattern<mlir_ts::PropertyRefOp>
    {
        using OpConversionPattern<mlir_ts::PropertyRefOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::PropertyRefOp propertyRefOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            LLVMCodeHelper ch(propertyRefOp, rewriter, getTypeConverter());

            auto addr = ch.GetAddressOfStructElement(propertyRefOp.getResult().getType(), propertyRefOp.objectRef(), propertyRefOp.position());
            rewriter.replaceOp(propertyRefOp, addr);

            return success();
        }
    };    

    struct GlobalOpLowering : public OpConversionPattern<mlir_ts::GlobalOp>
    {
        using OpConversionPattern<mlir_ts::GlobalOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::GlobalOp globalOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            TypeConverterHelper tch(getTypeConverter());

            Type type;
            auto hasValue = globalOp.value().hasValue();
            auto value = hasValue ? globalOp.value().getValue() : Attribute();
            Type argType = globalOp.getType();
            if (hasValue && argType.dyn_cast_or_null<mlir_ts::StringType>())
            {
                type = th.getArrayType(th.getI8Type(), value.cast<StringAttr>().getValue().size());
            }
            else
            {
                type = tch.convertType(globalOp.getType());
            }

            rewriter.replaceOpWithNewOp<LLVM::GlobalOp>(
                globalOp,
                type,
                globalOp.constant(),
                LLVM::Linkage::Internal,
                globalOp.sym_name(),
                value);
            return success();
        }
    };

    struct AddressOfOpLowering : public OpConversionPattern<mlir_ts::AddressOfOp>
    {
        using OpConversionPattern<mlir_ts::AddressOfOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::AddressOfOp addressOfOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);            
            TypeConverterHelper tch(getTypeConverter());
            auto parentModule = addressOfOp->getParentOfType<ModuleOp>();

            if (auto global = parentModule.lookupSymbol<LLVM::GlobalOp>(addressOfOp.global_name()))
            {
                rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(addressOfOp, global);
                return success();
            }

            return failure();
        }
    };

    struct AddressOfConstStringOpLowering : public OpConversionPattern<mlir_ts::AddressOfConstStringOp>
    {
        using OpConversionPattern<mlir_ts::AddressOfConstStringOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::AddressOfConstStringOp addressOfConstStringOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);            
            TypeConverterHelper tch(getTypeConverter());
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

    struct CreateOptionalOpLowering : public OpConversionPattern<mlir_ts::CreateOptionalOp>
    {
        using OpConversionPattern<mlir_ts::CreateOptionalOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::CreateOptionalOp createOptionalOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
                CastLogicHelper castLogic(createOptionalOp, rewriter);
                value = castLogic.cast(value, valueLLVMType, boxedType, llvmBoxedType);
                if (!value)
                {
                    return failure();
                }
            }

            auto structValue2 = rewriter.create<LLVM::InsertValueOp>(
                loc, 
                llvmOptType,
                structValue, 
                value,
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));           

            auto trueValue = clh.createI1ConstantOf(true); 
            rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
                createOptionalOp, 
                llvmOptType, 
                structValue2,
                trueValue, 
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

            return success();
        }
    };    

    struct UndefOptionalOpLowering : public OpConversionPattern<mlir_ts::UndefOptionalOp>
    {
        using OpConversionPattern<mlir_ts::UndefOptionalOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::UndefOptionalOp undefOptionalOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
                structValue2 = rewriter.create<LLVM::InsertValueOp>(
                    loc, 
                    llvmOptType,
                    structValue, 
                    defaultValue,
                    rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));                  
            }

            auto falseValue = clh.createI1ConstantOf(false); 
            rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
                undefOptionalOp, 
                llvmOptType, 
                structValue2,
                falseValue, 
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

            return success();
        }
    };   

    struct HasValueOpLowering : public OpConversionPattern<mlir_ts::HasValueOp>
    {
        using OpConversionPattern<mlir_ts::HasValueOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::HasValueOp hasValueOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto loc = hasValueOp->getLoc();

            TypeHelper th(rewriter);            
            TypeConverterHelper tch(getTypeConverter());

            rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
                hasValueOp, 
                th.getLLVMBoolType(), 
                hasValueOp.in(), 
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

            return success();
        }
    };    

    struct ValueOpLowering : public OpConversionPattern<mlir_ts::ValueOp>
    {
        using OpConversionPattern<mlir_ts::ValueOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ValueOp valueOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto loc = valueOp->getLoc();

            TypeConverterHelper tch(getTypeConverter());

            auto valueType = valueOp.res().getType();
            auto llvmValueType = tch.convertType(valueType);

            rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
                valueOp, 
                llvmValueType, 
                valueOp.in(), 
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));

            return success();
        }
    };    

    static void populateTypeScriptConversionPatterns(LLVMTypeConverter &converter, mlir::ModuleOp &m)
    {
        converter.addConversion([&](mlir_ts::AnyType type) {
            return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8));
        });

        converter.addConversion([&](mlir_ts::VoidType type) {
            return LLVM::LLVMVoidType::get(m.getContext());
        });

        converter.addConversion([&](mlir_ts::BooleanType type) {
            TypeHelper th(m.getContext());
            return th.getLLVMBoolType();
        });

        converter.addConversion([&](mlir_ts::CharType type) {
            return IntegerType::get(m.getContext(), 8/*, mlir::IntegerType::SignednessSemantics::Unsigned*/);
        });

        converter.addConversion([&](mlir_ts::ByteType type) {
            return IntegerType::get(m.getContext(), 8/*, mlir::IntegerType::SignednessSemantics::Unsigned*/);
        });

        converter.addConversion([&](mlir_ts::NumberType type) {
            return Float32Type::get(m.getContext());
        });

        converter.addConversion([&](mlir_ts::BigIntType type) {
            return IntegerType::get(m.getContext(), 64/*, mlir::IntegerType::SignednessSemantics::Signed*/);
        });

        converter.addConversion([&](mlir_ts::StringType type) {
            return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8));
        });

        converter.addConversion([&](mlir_ts::EnumType type) {
            return converter.convertType(type.getElementType());
        });        

        converter.addConversion([&](mlir_ts::ConstArrayType type) {
            return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType()));
        });  

        converter.addConversion([&](mlir_ts::ArrayType type) {
            return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType()));
        });        

        converter.addConversion([&](mlir_ts::RefType type) {
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
            return IntegerType::get(m.getContext(), 8/*, mlir::IntegerType::SignednessSemantics::Unsigned*/);
        });                      
    };

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// TypeScriptToLLVMLoweringPass
//===----------------------------------------------------------------------===//

namespace
{
    struct TypeScriptToLLVMLoweringPass
        : public PassWrapper<TypeScriptToLLVMLoweringPass, OperationPass<ModuleOp>>
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
    target.addLegalOp<ModuleOp, ModuleTerminatorOp>();

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
    OwningRewritePatternList patterns;
    populateAffineToStdConversionPatterns(patterns, &getContext());
    populateLoopToStdConversionPatterns(patterns, &getContext());
    populateStdToLLVMConversionPatterns(typeConverter, patterns);

    // The only remaining operation to lower from the `typescript` dialect, is the PrintOp.
    patterns.insert<
        CallOpLowering,
        CallIndirectOpLowering,
        ExitOpLowering,
        ReturnOpLowering,
        ReturnValOpLowering>(&getContext());

    patterns.insert<
        AddressOfOpLowering,
        AddressOfConstStringOpLowering,
        ArithmeticUnaryOpLowering,
        ArithmeticBinaryOpLowering,
        AssertOpLowering,
        CastOpLowering,
        ConstantOpLowering,
        CreateOptionalOpLowering,
        UndefOptionalOpLowering,
        HasValueOpLowering,
        ValueOpLowering,
        SymbolRefOpLowering,
        GlobalOpLowering,
        EntryOpLowering,
        FuncOpLowering,
        LoadOpLowering,
        ElementRefOpLowering,
        PropertyRefOpLowering,
        ExtractPropertyOpLowering,
        LogicalBinaryOpLowering,
        NullOpLowering,
        ParseFloatOpLowering,
        ParseIntOpLowering,
        PrintOpLowering,
        StoreOpLowering,
        InsertPropertyOpLowering,
        StringLengthOpLowering,
        StringConcatOpLowering,
        StringCompareOpLowering,
        CharToStringOpLowering,
        UndefOpLowering,
        VariableOpLowering>(typeConverter, &getContext());

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
