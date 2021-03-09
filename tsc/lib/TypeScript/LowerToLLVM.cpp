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

using namespace mlir;
using namespace ::typescript;
namespace ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScriptToLLVM RewritePatterns
//===----------------------------------------------------------------------===//

namespace
{
    static mlir::Type typeToConvertedType(mlir::Type type, mlir::TypeConverter &typeConverter)
    {
        auto convertedType = typeConverter.convertType(type);
        assert(convertedType);
        return convertedType;
    }

    static Value createI32ConstantOf(Location loc, PatternRewriter &rewriter, unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getI32Type(), value));
    }

    static ValueRange conditionalExpressionLowering(
        Location loc, TypeRange types, Value condition,
        function_ref<void(OpBuilder &, Location)> thenBuilder, 
        function_ref<void(OpBuilder &, Location)> elseBuilder,
        PatternRewriter &rewriter)
    {
        // TODO: or maybe only result should have types arguments as BR to Result has values from branch?

        // Split block
        auto *opBlock = rewriter.getInsertionBlock();
        auto opPosition = rewriter.getInsertionPoint();
        auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

        // then block
        auto *thenBlock = rewriter.createBlock(continuationBlock, types);
        thenBuilder(rewriter, loc);

        // else block
        auto *elseBlock = rewriter.createBlock(continuationBlock, types);
        elseBuilder(rewriter, loc);

        // result block
        auto *resultBlock = rewriter.createBlock(continuationBlock, types);
        rewriter.create<LLVM::BrOp>(
            loc,
            resultBlock->getArguments(),
            continuationBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        rewriter.create<LLVM::BrOp>(
            loc,
            thenBlock->getArguments(),
            resultBlock);

        rewriter.setInsertionPointToEnd(elseBlock);
        rewriter.create<LLVM::BrOp>(
            loc,
            elseBlock->getArguments(),
            resultBlock);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::CondBrOp>(
            loc,
            condition,
            thenBlock,
            elseBlock);

        rewriter.setInsertionPointToStart(continuationBlock);            

        // should I use resultBlock->getArguments();?
        // should I add arguments to continuationBlock?
        //continuationBlock->addArguments(types);
        return continuationBlock->getArguments();
    }

    template <typename T>
    struct OpLowering : public OpConversionPattern<typename T::OpTy>
    {
        using OpConversionPattern::OpConversionPattern;

        LogicalResult matchAndRewrite(typename T::OpTy op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const override
        {
            T logic(getTypeConverter(), op.getOperation(), operands, rewriter);
            return logic.matchAndRewrite();
        }
    };

    class LoweringLogicBase
    {
    public:
        explicit LoweringLogicBase(TypeConverter *typeConverter_, Operation *op_, ArrayRef<Value> &operands_, ConversionPatternRewriter &rewriter_)
            : typeConverter(*typeConverter_),
              op(op_),
              operands(operands_),
              rewriter(rewriter_),
              loc(op->getLoc()),
              parentModule(op->getParentOfType<ModuleOp>()),
              context(parentModule.getContext())
        {
        }

    protected:
        Value getOrCreateGlobalString(StringRef name, std::string value)
        {
            return getOrCreateGlobalString(name, StringRef(value.data(), value.length() + 1));
        }

        /// Return a value representing an access into a global string with the given
        /// name, creating the string if necessary.
        Value getOrCreateGlobalString(StringRef name, StringRef value)
        {
            // Create the global at the entry of the module.
            LLVM::GlobalOp global;
            if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPointToStart(parentModule.getBody());
                auto type = LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), value.size());
                global = rewriter.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, rewriter.getStringAttr(value));
            }

            // Get the pointer to the first character in the global string.
            Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = rewriter.create<LLVM::ConstantOp>(
                loc, IntegerType::get(rewriter.getContext(), 64),
                rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
            return rewriter.create<LLVM::GEPOp>(
                loc,
                getI8PtrType(),
                globalPtr, ArrayRef<Value>({cst0, cst0}));
        }

        LLVM::LLVMFuncOp getOrInsertFunction(const StringRef &name, const LLVM::LLVMFunctionType &llvmFnType)
        {
            if (auto funcOp = parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
            {
                return funcOp;
            }

            // Insert the printf function into the body of the parent module.
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            return rewriter.create<LLVM::LLVMFuncOp>(loc, name, llvmFnType);
        }

        LLVM::LLVMVoidType getVoidType()
        {
            return LLVM::LLVMVoidType::get(context);
        }

        IntegerType getI8Type()
        {
            return IntegerType::get(context, 8);
        }

        IntegerType getI32Type()
        {
            return IntegerType::get(context, 32);
        }

        IntegerType getI64Type()
        {
            return IntegerType::get(context, 64);
        }

        LLVM::LLVMPointerType getPointerType(mlir::Type type)
        {
            return LLVM::LLVMPointerType::get(type);
        }

        LLVM::LLVMPointerType getI8PtrType()
        {
            return getPointerType(getI8Type());
        }

        LLVM::LLVMFunctionType getFunctionType(mlir::Type result, ArrayRef<mlir::Type> arguments, bool isVarArg = false)
        {
            return LLVM::LLVMFunctionType::get(result, arguments, isVarArg);
        }

        LLVM::LLVMFunctionType getFunctionType(ArrayRef<mlir::Type> arguments, bool isVarArg = false)
        {
            return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), arguments, isVarArg);
        }

        Operation *op;
        ArrayRef<Value> &operands;
        ConversionPatternRewriter &rewriter;

        Location loc;
        ModuleOp parentModule;
        MLIRContext *context;
        TypeConverter &typeConverter;
    };

    template <typename T>
    class LoweringLogic : public LoweringLogicBase
    {
    public:
        using OpTy = T;

        explicit LoweringLogic(TypeConverter *typeConverter_, Operation *op_, ArrayRef<Value> &operands_, ConversionPatternRewriter &rewriter_)
            : LoweringLogicBase(typeConverter_, op_, operands_, rewriter_),
              opTyped(cast<OpTy>(op)),
              transformed(operands)
        {
        }

    protected:
        OpTy opTyped;
        typename OpTy::Adaptor transformed;
    };

    class PrintOpLoweringLogic : public LoweringLogic<ts::PrintOp>
    {
    public:
        using LoweringLogic<ts::PrintOp>::LoweringLogic;

        LogicalResult matchAndRewrite()
        {
            // Get a symbol reference to the printf function, inserting it if necessary.
            auto printfFuncOp =
                getOrInsertFunction(
                    "printf",
                    getFunctionType(getI32Type(), getI8PtrType(), true));

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

            auto opHash = OperationEquivalence::computeHash(op, OperationEquivalence::Flags::IgnoreOperands);

            std::stringstream formatVarName;
            formatVarName << "frmt_" << opHash;

            auto formatSpecifierCst = getOrCreateGlobalString(formatVarName.str(), format.str());

            auto i8PtrTy = getI8PtrType();

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
                        getOrCreateGlobalString("__true__", std::string("true")), 
                        getOrCreateGlobalString("__false__", std::string("false"))));

                    /*
                    auto valuesCond = conditionalExpressionLowering(
                        item.getLoc(), 
                        TypeRange{ i8PtrTy }, 
                        item,
                        [&](auto &builder, auto loc) 
                        {
                            getOrCreateGlobalString("__true__", std::string("true"));
                        }, 
                        [&](auto &builder, auto loc) 
                        {
                            getOrCreateGlobalString("__false__", std::string("false"));
                        }, 
                        rewriter);

                    values.push_back(valuesCond.front());
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

    /// Lowers `typescript.print` to a loop nest calling `printf` on each of the individual
    /// elements of the array.
    struct PrintOpLowering : public OpLowering<PrintOpLoweringLogic>
    {
        using OpLowering<PrintOpLoweringLogic>::OpLowering;
    };

    class AssertOpLoweringLogic : public LoweringLogic<ts::AssertOp>
    {
    public:
        using LoweringLogic<ts::AssertOp>::LoweringLogic;

        LogicalResult matchAndRewrite()
        {
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
            auto i8PtrTy = getI8PtrType();
            auto assertFuncOp =
                getOrInsertFunction(
                    "_assert",
                    getFunctionType(getVoidType(), {i8PtrTy, i8PtrTy, getI32Type()}));

            // Split block at `assert` operation.
            auto *opBlock = rewriter.getInsertionBlock();
            auto opPosition = rewriter.getInsertionPoint();
            auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

            // Generate IR to call `assert`.
            auto *failureBlock = rewriter.createBlock(opBlock->getParent());

            auto opHash = OperationEquivalence::computeHash(op, OperationEquivalence::Flags::IgnoreOperands);

            std::stringstream msgVarName;
            msgVarName << "m_" << opHash;

            std::stringstream msgWithNUL;
            msgWithNUL << opTyped.msg().str();

            std::stringstream fileVarName;
            fileVarName << "f_" << hash_value(fileName);

            std::stringstream fileWithNUL;
            fileWithNUL << fileName.str();

            auto msgCst = getOrCreateGlobalString(msgVarName.str(), msgWithNUL.str());

            auto fileCst = getOrCreateGlobalString(fileVarName.str(), fileName.str());

            //auto nullCst = rewriter.create<LLVM::NullOp>(loc, getI8PtrType(context));

            Value lineNumberRes =
                rewriter.create<LLVM::ConstantOp>(
                    loc,
                    getI32Type(),
                    rewriter.getI32IntegerAttr(line));

            rewriter.create<LLVM::CallOp>(loc, assertFuncOp, ValueRange{msgCst, fileCst, lineNumberRes});
            rewriter.create<LLVM::UnreachableOp>(loc);

            // Generate assertion test.
            rewriter.setInsertionPointToEnd(opBlock);
            rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
                op,
                transformed.arg(),
                continuationBlock,
                failureBlock);

            return success();
        }
    };

    struct AssertOpLowering : public OpLowering<AssertOpLoweringLogic>
    {
        using OpLowering<AssertOpLoweringLogic>::OpLowering;
    };

    class ParseIntOpLoweringLogic : public LoweringLogic<ts::ParseIntOp>
    {
    public:
        using LoweringLogic<ts::ParseIntOp>::LoweringLogic;

        LogicalResult matchAndRewrite()
        {
            // Insert the `_assert` declaration if necessary.
            auto i8PtrTy = getI8PtrType();
            auto parseIntFuncOp =
                getOrInsertFunction(
                    "atoi",
                    getFunctionType(rewriter.getI32Type(), {i8PtrTy}));

            rewriter.replaceOpWithNewOp<LLVM::CallOp>(
                op,
                parseIntFuncOp,
                op->getOperands());

            return success();
        }
    };

    struct ParseIntOpLowering : public OpLowering<ParseIntOpLoweringLogic>
    {
        using OpLowering<ParseIntOpLoweringLogic>::OpLowering;
    };

    class ParseFloatOpLoweringLogic : public LoweringLogic<ts::ParseFloatOp>
    {
    public:
        using LoweringLogic<ts::ParseFloatOp>::LoweringLogic;

        LogicalResult matchAndRewrite()
        {
            // Insert the `_assert` declaration if necessary.
            auto i8PtrTy = getI8PtrType();
            auto parseFloatFuncOp =
                getOrInsertFunction(
                    "atof",
                    getFunctionType(rewriter.getF32Type(), {i8PtrTy}));

            rewriter.replaceOpWithNewOp<LLVM::CallOp>(
                op,
                parseFloatFuncOp,
                op->getOperands());

            return success();
        }
    };

    struct ParseFloatOpLowering : public OpLowering<ParseFloatOpLoweringLogic>
    {
        using OpLowering<ParseFloatOpLoweringLogic>::OpLowering;
    };

    struct NullOpLowering : public OpConversionPattern<ts::NullOp>
    {
        using OpConversionPattern<ts::NullOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::NullOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            rewriter.replaceOpWithNewOp<LLVM::NullOp>(op, typeToConvertedType(op.getType(), *getTypeConverter()));
            return success();
        }
    };

    class StringOpLoweringLogic : public LoweringLogic<ts::StringOp>
    {
    public:
        using LoweringLogic<ts::StringOp>::LoweringLogic;

        LogicalResult matchAndRewrite()
        {
            auto opHash = OperationEquivalence::computeHash(op, OperationEquivalence::Flags::IgnoreOperands);

            std::stringstream strVarName;
            strVarName << "s_" << opHash;

            std::stringstream strWithNUL;
            strWithNUL << opTyped.txt().str();

            auto txtCst = getOrCreateGlobalString(strVarName.str(), strWithNUL.str());

            rewriter.replaceOp(op, txtCst);

            return success();
        }
    };

    struct StringOpLowering : public OpLowering<StringOpLoweringLogic>
    {
        using OpLowering<StringOpLoweringLogic>::OpLowering;
    };

    class UndefOpLowering : public OpConversionPattern<ts::UndefOp>
    {
    public:
        using OpConversionPattern<ts::UndefOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::UndefOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
            rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, typeToConvertedType(op.getType(), *getTypeConverter()));
            return success();
        }
    };

    struct EntryOpLowering : public OpConversionPattern<ts::EntryOp>
    {
        using OpConversionPattern<ts::EntryOp>::OpConversionPattern;

        LogicalResult matchAndRewrite(ts::EntryOp op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter) const final
        {
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
                        typeToConvertedType(result.getType(), *getTypeConverter()),
                        createI32ConstantOf(location, rewriter, 1));
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
            auto location = varOp.getLoc();

            auto allocated =
                rewriter.create<LLVM::AllocaOp>(
                    location,
                    typeToConvertedType(varOp.reference().getType(), *getTypeConverter()),
                    createI32ConstantOf(location, rewriter, 1));
            auto value = varOp.initializer();
            if (value)
            {
                rewriter.create<LLVM::StoreOp>(location, value, allocated);
            }

            rewriter.replaceOp(varOp, ValueRange{allocated});
            return success();
        }
    };

    template <typename BinOpTy, typename StdIOpTy, typename StdFOpTy>
    void BinOp(BinOpTy &binOp, mlir::PatternRewriter &builder)
    {
        auto leftType = binOp.getOperand(0).getType();
        if (leftType.isIntOrIndex())
        {
            builder.replaceOpWithNewOp<StdIOpTy>(binOp, binOp.getOperand(0), binOp.getOperand(1));
        }
        else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
        {
            builder.replaceOpWithNewOp<StdFOpTy>(binOp, binOp.getOperand(0), binOp.getOperand(1));
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    template <typename BinOpTy, typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    void LogicOp(BinOpTy &binOp, mlir::PatternRewriter &builder)
    {
        auto leftType = binOp.getOperand(0).getType();
        if (leftType.isIntOrIndex())
        {
            builder.replaceOpWithNewOp<StdIOpTy>(binOp, v1, binOp.getOperand(0), binOp.getOperand(1));
        }
        else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
        {
            builder.replaceOpWithNewOp<StdFOpTy>(binOp, v2, binOp.getOperand(0), binOp.getOperand(1));
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }

    struct ArithmeticBinaryOpLowering : public OpRewritePattern<ts::ArithmeticBinaryOp>
    {
        using OpRewritePattern<ts::ArithmeticBinaryOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::ArithmeticBinaryOp arithmeticBinaryOp, PatternRewriter &rewriter) const final
        {
            switch ((SyntaxKind)arithmeticBinaryOp.opCode())
            {
            case SyntaxKind::PlusToken:
                BinOp<ts::ArithmeticBinaryOp, AddIOp, AddFOp>(arithmeticBinaryOp, rewriter);
                return success();

            case SyntaxKind::MinusToken:
                BinOp<ts::ArithmeticBinaryOp, SubIOp, SubFOp>(arithmeticBinaryOp, rewriter);
                return success();
            default:
                llvm_unreachable("not implemented");
            }
        }
    };

    struct LogicalBinaryOpLowering : public OpRewritePattern<ts::LogicalBinaryOp>
    {
        using OpRewritePattern<ts::LogicalBinaryOp>::OpRewritePattern;

        LogicalResult matchAndRewrite(ts::LogicalBinaryOp logicalBinaryOp, PatternRewriter &rewriter) const final
        {
            switch ((SyntaxKind)logicalBinaryOp.opCode())
            {
            case SyntaxKind::EqualsEqualsToken:
            case SyntaxKind::EqualsEqualsEqualsToken:
                LogicOp<ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::eq,
                        CmpFOp, CmpFPredicate, CmpFPredicate::OEQ>(logicalBinaryOp, rewriter);
                return success();
            case SyntaxKind::ExclamationEqualsToken:
            case SyntaxKind::ExclamationEqualsEqualsToken:
                LogicOp<ts::LogicalBinaryOp,
                        CmpIOp, CmpIPredicate, CmpIPredicate::ne,
                        CmpFOp, CmpFPredicate, CmpFPredicate::ONE>(logicalBinaryOp, rewriter);
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
            auto elementTypeConverted = typeToConvertedType(loadOp.reference().getType().cast<ts::RefType>().getElementType(), *getTypeConverter());

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
            registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
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
        ArithmeticBinaryOpLowering,
        CallOpLowering,
        CastOpLowering,
        ExitOpLowering,
        LogicalBinaryOpLowering,
        ReturnOpLowering,
        ReturnValOpLowering>(&getContext());

    patterns.insert<
        AssertOpLowering,
        EntryOpLowering,
        FuncOpLowering,
        LoadOpLowering,
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
