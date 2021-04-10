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

            auto loc = op->getLoc();

            // Get a symbol reference to the printf function, inserting it if necessary.
            auto printfFuncOp =
                ch.getOrInsertFunction(
                    "printf",
                    th.getFunctionType(rewriter.getI32Type(), th.getI8PtrType(), true));

            std::stringstream format;
            auto count = 0;
            for (auto item : op->getOperands())
            {
                auto type = item.getType();

                if (count++ > 0)
                {
                    format << " ";
                }

                if (type.isIntOrIndexOrFloat() && !type.isIntOrIndex())
                {
                    format << "%f";
                }
                else if (type.isIntOrIndex())
                {
                    if (type.isInteger(1))
                    {
                        format << "%s";
                    }
                    else
                    {
                        format << "%d";
                    }
                }
                else if (auto s = type.dyn_cast_or_null<mlir_ts::StringType>())
                {
                    format << "%s";
                }
                else
                {
                    format << "%d";
                }
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
                if (type.isIntOrIndexOrFloat() && !type.isIntOrIndex())
                {
                    values.push_back(rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), item));
                }
                else if (type.isInteger(1))
                {
                    values.push_back(rewriter.create<LLVM::SelectOp>(
                        item.getLoc(),
                        item,
                        ch.getOrCreateGlobalString("__true__", std::string("true")),
                        ch.getOrCreateGlobalString("__false__", std::string("false"))));

                    /*
                    auto valuesCond = ch.conditionalExpressionLowering(
                        i8PtrTy,
                        item,
                        [&](auto &builder, auto loc)
                        {
                            return ch.getOrCreateGlobalString("__true__", std::string("true"));
                        }, 
                        [&](auto &builder, auto loc) 
                        {
                            return ch.getOrCreateGlobalString("__false__", std::string("false"));
                        });

                    values.push_back(valuesCond);
                    */
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

    struct ConstantOpLowering : public OpConversionPattern<mlir_ts::ConstantOp>
    {
        using OpConversionPattern<mlir_ts::ConstantOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::ConstantOp constantOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(*getTypeConverter());
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(constantOp, tch.convertType(constantOp.getType()), constantOp.getValue());
            return success();
        }
    };

    struct NullOpLowering : public OpConversionPattern<mlir_ts::NullOp>
    {
        using OpConversionPattern<mlir_ts::NullOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::NullOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(*getTypeConverter());
            rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, tch.convertType(op.getType()));
            return success();
        }
    };

    class StringOpLowering : public OpConversionPattern<mlir_ts::StringOp>
    {
    public:
        using OpConversionPattern<mlir_ts::StringOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::StringOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            LLVMCodeHelper ch(op, rewriter);

            std::stringstream strWithNUL;
            strWithNUL << op.txt().str();

            auto opHash = std::hash<std::string>{}(strWithNUL.str());

            std::stringstream strVarName;
            strVarName << "s_" << opHash;

            auto txtCst = ch.getOrCreateGlobalString(strVarName.str(), strWithNUL.str());

            rewriter.replaceOp(op, txtCst);

            return success();
        }
    };

    class UndefOpLowering : public OpConversionPattern<mlir_ts::UndefOp>
    {
    public:
        using OpConversionPattern<mlir_ts::UndefOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::UndefOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(*getTypeConverter());
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
            TypeConverterHelper tch(*getTypeConverter());

            auto opTyped = mlir_ts::EntryOpAdaptor(op);
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

    struct CastOpLowering : public OpRewritePattern<mlir_ts::CastOp>
    {
        using OpRewritePattern<mlir_ts::CastOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(mlir_ts::CastOp op, PatternRewriter &rewriter) const final
        {
            auto in = op.in();
            auto res = op.res();
            auto op1 = in.getType();
            auto op2 = res.getType();

            if (op1 == op2)
            {
                // types are equals
                rewriter.replaceOp(op, in);
                return success();
            }

            if (op1.isInteger(32) && op2.isF32())
            {
                rewriter.replaceOpWithNewOp<SIToFPOp>(op, op2, in);
                return success();
            }

            if (op1.isF32() && op2.isInteger(32))
            {
                rewriter.replaceOpWithNewOp<FPToSIOp>(op, op2, in);
                return success();
            }

            if ((op1.isInteger(32) || op1.isInteger(8)) && op2.isInteger(1))
            {
                rewriter.replaceOpWithNewOp<TruncateIOp>(op, op2, in);
                return success();
            }

            auto op1Any = op1.dyn_cast_or_null<mlir_ts::AnyType>();
            auto op2String = op2.dyn_cast_or_null<mlir_ts::StringType>();
            if (op1Any && op2String)
            {
                rewriter.replaceOp(op, op.in());
                return success();
            }

            emitError(op->getLoc(), "invalid cast operator type 1: '") << op1 << "', type 2: '" << op2 << "'";
            llvm_unreachable("not implemented");
        }
    };

    struct VariableOpLowering : public OpConversionPattern<mlir_ts::VariableOp>
    {
        using OpConversionPattern<mlir_ts::VariableOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::VariableOp varOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            CodeLogicHelper clh(varOp, rewriter);
            TypeConverterHelper tch(*getTypeConverter());

            auto location = varOp.getLoc();

            auto allocated =
                rewriter.create<LLVM::AllocaOp>(
                    location,
                    tch.convertType(varOp.reference().getType()),
                    clh.createI32ConstantOf(1));
            auto value = varOp.initializer();
            if (value)
            {
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
            builder.replaceOpWithNewOp<SubIOp>(unaryOp, type, clh.createI32ConstantOf(0), oper);
        }
        else if (!type.isIntOrIndex() && type.isIntOrIndexOrFloat())
        {
            builder.replaceOpWithNewOp<SubFOp>(unaryOp, type, clh.createF32ConstantOf(0.0), oper);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    void NegativeOpBin(mlir_ts::ArithmeticUnaryOp &unaryOp, mlir::PatternRewriter &builder)
    {
        CodeLogicHelper clh(unaryOp, builder);

        auto oper = unaryOp.operand1();
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
                lhs = clh.createI32ConstantOf(0xffff);
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
                NegativeOpBin(arithmeticUnaryOp, rewriter);
                return success();
            case SyntaxKind::PlusToken:
                rewriter.replaceOp(arithmeticUnaryOp, arithmeticUnaryOp.operand1());
                return success();
            case SyntaxKind::MinusToken:
                NegativeOpValue(arithmeticUnaryOp, rewriter);
                return success();
            case SyntaxKind::TildeToken:
                NegativeOpBin(arithmeticUnaryOp, rewriter);
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
                BinOp<mlir_ts::ArithmeticBinaryOp, AddIOp, AddFOp>(arithmeticBinaryOp, rewriter);
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
            switch (op)
            {
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
                LogicOp<mlir_ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::eq,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OEQ>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
                LogicOp<mlir_ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::ne,
                        CmpFOp, CmpFPredicate, CmpFPredicate::ONE>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            case SyntaxKind::GreaterThanToken:
                LogicOp<mlir_ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::sgt,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OGT>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            case SyntaxKind::GreaterThanEqualsToken:
                LogicOp<mlir_ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::sge,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OGE>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            case SyntaxKind::LessThanToken:
                LogicOp<mlir_ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::slt,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OLT>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            case SyntaxKind::LessThanEqualsToken:
                LogicOp<mlir_ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::sle,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OLE>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            default:
                llvm_unreachable("not implemented");
            }
        }
    };

    struct LoadOpLowering : public OpConversionPattern<mlir_ts::LoadOp>
    {
        using OpConversionPattern<mlir_ts::LoadOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::LoadOp loadOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            TypeConverterHelper tch(*getTypeConverter());
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

    struct IfOpLowering : public OpConversionPattern<mlir_ts::IfOp>
    {
        using OpConversionPattern<mlir_ts::IfOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::IfOp ifOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override
        {
            auto loc = ifOp.getLoc();

            auto &typeConverter = *getTypeConverter();

            // Start by splitting the block containing the 'scf.if' into two parts.
            // The part before will contain the condition, the part after will be the
            // continuation point.
            auto *condBlock = rewriter.getInsertionBlock();
            auto opPosition = rewriter.getInsertionPoint();
            auto *remainingOpsBlock = rewriter.splitBlock(condBlock, opPosition);
            Block *continueBlock;
            if (ifOp.getNumResults() == 0)
            {
                continueBlock = remainingOpsBlock;
            }
            else
            {
                continueBlock = rewriter.createBlock(remainingOpsBlock, ifOp.getResultTypes());
                rewriter.create<LLVM::BrOp>(loc, ArrayRef<Value>(), remainingOpsBlock);
            }

            // Move blocks from the "then" region to the region containing 'scf.if',
            // place it before the continuation block, and branch to it.
            auto &thenRegion = ifOp.thenRegion();
            auto *thenBlock = &thenRegion.front();
            Operation *thenTerminator = thenRegion.back().getTerminator();
            ValueRange thenTerminatorOperands = thenTerminator->getOperands();
            rewriter.setInsertionPointToEnd(&thenRegion.back());
            rewriter.create<LLVM::BrOp>(loc, thenTerminatorOperands, continueBlock);
            rewriter.eraseOp(thenTerminator);
            rewriter.inlineRegionBefore(thenRegion, continueBlock);

            if (failed(rewriter.convertRegionTypes(&thenRegion, typeConverter)))
            {
                return failure();
            }

            // Move blocks from the "else" region (if present) to the region containing
            // 'scf.if', place it before the continuation block and branch to it.  It
            // will be placed after the "then" regions.
            auto *elseBlock = continueBlock;
            auto &elseRegion = ifOp.elseRegion();
            if (!elseRegion.empty())
            {
                elseBlock = &elseRegion.front();
                Operation *elseTerminator = elseRegion.back().getTerminator();
                ValueRange elseTerminatorOperands = elseTerminator->getOperands();
                rewriter.setInsertionPointToEnd(&elseRegion.back());
                rewriter.create<LLVM::BrOp>(loc, elseTerminatorOperands, continueBlock);
                rewriter.eraseOp(elseTerminator);
                rewriter.inlineRegionBefore(elseRegion, continueBlock);

                if (failed(rewriter.convertRegionTypes(&elseRegion, typeConverter)))
                {
                    return failure();
                }
            }

            rewriter.setInsertionPointToEnd(condBlock);
            rewriter.create<LLVM::CondBrOp>(
                loc,
                ifOp.condition(),
                thenBlock, /*trueArgs=*/ArrayRef<Value>(),
                elseBlock, /*falseArgs=*/ArrayRef<Value>());

            // Ok, we're done!
            rewriter.replaceOp(ifOp, continueBlock->getArguments());
            return success();
        }
    };

    struct GlobalOpLowering : public OpConversionPattern<mlir_ts::GlobalOp>
    {
        using OpConversionPattern<mlir_ts::GlobalOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(mlir_ts::GlobalOp globalOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeHelper th(rewriter);
            TypeConverterHelper tch(*getTypeConverter());

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
            TypeConverterHelper tch(*getTypeConverter());
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
            TypeConverterHelper tch(*getTypeConverter());
            auto parentModule = addressOfConstStringOp->getParentOfType<ModuleOp>();

            if (auto global = parentModule.lookupSymbol<LLVM::GlobalOp>(addressOfConstStringOp.global_name()))
            {
                auto loc = addressOfConstStringOp->getLoc();
                auto globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
                auto cst0 = rewriter.create<LLVM::ConstantOp>(loc, IntegerType::get(rewriter.getContext(), 64), 
                    rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
                rewriter.replaceOpWithNewOp<LLVM::GEPOp>(addressOfConstStringOp, th.getI8PtrType(), globalPtr, ArrayRef<Value>({cst0, cst0}));

                return success();
            }

            return failure();
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
            return IntegerType::get(m.getContext(), 1);
        });

        converter.addConversion([&](mlir_ts::StringType type) {
            return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8));
        });

        converter.addConversion([&](mlir_ts::RefType type) {
            return LLVM::LLVMPointerType::get(converter.convertType(type.getElementType()));
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
        CastOpLowering,
        ExitOpLowering,
        ReturnOpLowering,
        ReturnValOpLowering>(&getContext());

    patterns.insert<
        AddressOfOpLowering,
        AddressOfConstStringOpLowering,
        ArithmeticBinaryOpLowering,
        ArithmeticUnaryOpLowering,
        AssertOpLowering,
        ConstantOpLowering,
        GlobalOpLowering,
        EntryOpLowering,
        FuncOpLowering,
        IfOpLowering,
        LoadOpLowering,
        LogicalBinaryOpLowering,
        NullOpLowering,
        ParseFloatOpLowering,
        ParseIntOpLowering,
        PrintOpLowering,
        StoreOpLowering,
        StringOpLowering,
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
