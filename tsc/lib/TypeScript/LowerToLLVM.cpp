#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/EnumsAST.h"

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

using namespace mlir;
using namespace ::typescript;
namespace ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace
{
    class PrintOpLowering : public OpConversionPattern<ts::PrintOp>
    {
    public:
        using OpConversionPattern<ts::PrintOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::PrintOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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
                else if (auto s = type.dyn_cast_or_null<ts::StringType>())
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

    class AssertOpLowering : public OpConversionPattern<ts::AssertOp>
    {
    public:
        using OpConversionPattern<ts::AssertOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::AssertOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

    class ParseIntOpLowering : public OpConversionPattern<ts::ParseIntOp>
    {
    public:
        using OpConversionPattern<ts::ParseIntOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::ParseIntOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

    class ParseFloatOpLowering : public OpConversionPattern<ts::ParseFloatOp>
    {
    public:
        using OpConversionPattern<ts::ParseFloatOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::ParseFloatOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

    struct NullOpLowering : public OpConversionPattern<ts::NullOp>
    {
        using OpConversionPattern<ts::NullOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::NullOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(*getTypeConverter());
            rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, tch.typeToConvertedType(op.getType()));
            return success();
        }
    };

    class StringOpLowering : public OpConversionPattern<ts::StringOp>
    {
    public:
        using OpConversionPattern<ts::StringOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::StringOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

    class UndefOpLowering : public OpConversionPattern<ts::UndefOp>
    {
    public:
        using OpConversionPattern<ts::UndefOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::UndefOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(*getTypeConverter());
            rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, tch.typeToConvertedType(op.getType()));
            return success();
        }
    };

    struct EntryOpLowering : public OpConversionPattern<ts::EntryOp>
    {
        using OpConversionPattern<ts::EntryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::EntryOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            CodeLogicHelper clh(op, rewriter);
            TypeConverterHelper tch(*getTypeConverter());

            auto opTyped = ts::EntryOpAdaptor(op);
            auto location = op.getLoc();

            mlir::Value allocValue;
            auto anyResult = op.getNumResults() > 0;
            if (anyResult)
            {
                auto result = op.getResult(0);
                allocValue =
                    rewriter.create<LLVM::AllocaOp>(
                        location,
                        tch.typeToConvertedType(result.getType()),
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

    struct ReturnOpLowering : public OpRewritePattern<ts::ReturnOp>
    {
        using OpRewritePattern<ts::ReturnOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::ReturnOp op, PatternRewriter &rewriter) const final
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

    struct ReturnValOpLowering : public OpRewritePattern<ts::ReturnValOp>
    {
        using OpRewritePattern<ts::ReturnValOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::ReturnValOp op, PatternRewriter &rewriter) const final
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

    struct ExitOpLowering : public OpConversionPattern<ts::ExitOp>
    {
        using OpConversionPattern<ts::ExitOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::ExitOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            auto retBlock = FindReturnBlock(rewriter);

            rewriter.create<mlir::BranchOp>(op.getLoc(), retBlock);

            rewriter.eraseOp(op);
            return success();
        }
    };

    struct FuncOpLowering : public OpConversionPattern<ts::FuncOp>
    {
        using OpConversionPattern<ts::FuncOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::FuncOp funcOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
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

            auto newFuncOp = rewriter.create<FuncOp>(
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

    struct CallOpLowering : public OpRewritePattern<ts::CallOp>
    {
        using OpRewritePattern<ts::CallOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::CallOp op, PatternRewriter &rewriter) const final
        {
            // just replace
            rewriter.replaceOpWithNewOp<CallOp>(
                op,
                op.getCallee(),
                op.getResultTypes(),
                op.getArgOperands());
            return success();
        }
    };

    struct CastOpLowering : public OpRewritePattern<ts::CastOp>
    {
        using OpRewritePattern<ts::CastOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::CastOp op, PatternRewriter &rewriter) const final
        {
            auto in = op.in();
            auto res = op.res();
            auto op1 = in.getType();
            auto op2 = res.getType();

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

            auto op1Any = op1.dyn_cast_or_null<ts::AnyType>();
            auto op2String = op2.dyn_cast_or_null<ts::StringType>();
            if (op1Any && op2String)
            {
                rewriter.replaceOp(op, op.in());
                return success();
            }

            emitError(op->getLoc(), "invalid cast operator type 1: '") << op1 << "', type 2: '" << op2 << "'";
            llvm_unreachable("not implemented");
        }
    };

    struct VariableOpLowering : public OpConversionPattern<ts::VariableOp>
    {
        using OpConversionPattern<ts::VariableOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::VariableOp varOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            CodeLogicHelper clh(varOp, rewriter);
            TypeConverterHelper tch(*getTypeConverter());

            auto location = varOp.getLoc();

            auto allocated =
                rewriter.create<LLVM::AllocaOp>(
                    location,
                    tch.typeToConvertedType(varOp.reference().getType()),
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

    void NegativeOp(ts::ArithmeticUnaryOp &unaryOp, mlir::PatternRewriter &builder)
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

    struct ArithmeticUnaryOpLowering : public OpConversionPattern<ts::ArithmeticUnaryOp>
    {
        using OpConversionPattern<ts::ArithmeticUnaryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::ArithmeticUnaryOp arithmeticUnaryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            switch ((SyntaxKind)arithmeticUnaryOp.opCode())
            {
            case SyntaxKind::ExclamationToken:
                NegativeOp(arithmeticUnaryOp, rewriter);
                return success();

            default:
                llvm_unreachable("not implemented");
            }
        }
    };    

    struct ArithmeticBinaryOpLowering : public OpConversionPattern<ts::ArithmeticBinaryOp>
    {
        using OpConversionPattern<ts::ArithmeticBinaryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::ArithmeticBinaryOp arithmeticBinaryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            switch ((SyntaxKind)arithmeticBinaryOp.opCode())
            {
            case SyntaxKind::PlusToken:
                BinOp<ts::ArithmeticBinaryOp, AddIOp, AddFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::MinusToken:
                BinOp<ts::ArithmeticBinaryOp, SubIOp, SubFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::AsteriskToken:
                BinOp<ts::ArithmeticBinaryOp, MulIOp, MulFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::SlashToken:
                BinOp<ts::ArithmeticBinaryOp, DivFOp, DivFOp>(arithmeticBinaryOp, rewriter);
                return success();

            default:
                llvm_unreachable("not implemented");
            }
        }
    };

    struct LogicalBinaryOpLowering : public OpConversionPattern<ts::LogicalBinaryOp>
    {
        using OpConversionPattern<ts::LogicalBinaryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::LogicalBinaryOp logicalBinaryOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            switch ((SyntaxKind)logicalBinaryOp.opCode())
            {
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
                LogicOp<ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::eq,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OEQ>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
                LogicOp<ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::ne,
                        CmpFOp, CmpFPredicate, CmpFPredicate::ONE>(logicalBinaryOp, rewriter, *(LLVMTypeConverter *)getTypeConverter());
                return success();
            default:
                llvm_unreachable("not implemented");
            }
        }
    };

    struct LoadOpLowering : public OpConversionPattern<ts::LoadOp>
    {
        using OpConversionPattern<ts::LoadOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::LoadOp loadOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            TypeConverterHelper tch(*getTypeConverter());

            auto elementTypeConverted = tch.typeToConvertedType(loadOp.reference().getType().cast<ts::RefType>().getElementType());

            rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
                loadOp,
                elementTypeConverted,
                loadOp.reference());
            return success();
        }
    };

    struct StoreOpLowering : public OpConversionPattern<ts::StoreOp>
    {
        using OpConversionPattern<ts::StoreOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::StoreOp storeOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            rewriter.replaceOpWithNewOp<LLVM::StoreOp>(storeOp, storeOp.value(), storeOp.reference());
            return success();
        }
    };

    struct IfOpLowering : public OpConversionPattern<ts::IfOp>
    {
        using OpConversionPattern<ts::IfOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::IfOp ifOp, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override
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

    static void populateTypeScriptConversionPatterns(LLVMTypeConverter &converter, mlir::ModuleOp &m)
    {
        converter.addConversion([&](ts::AnyType type) {
            return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8));
        });

        converter.addConversion([&](ts::StringType type) {
            return LLVM::LLVMPointerType::get(IntegerType::get(m.getContext(), 8));
        });

        converter.addConversion([&](ts::RefType type) {
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
        ArithmeticBinaryOpLowering,
        ArithmeticUnaryOpLowering,
        AssertOpLowering,
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
