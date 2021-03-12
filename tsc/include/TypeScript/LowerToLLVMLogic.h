#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_

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

namespace typescript
{
    mlir::Type typeToConvertedType(mlir::Type type, mlir::TypeConverter &typeConverter);
    
    Value createI32ConstantOf(Location loc, PatternRewriter &rewriter, unsigned value);
    
    Value createI1ConstantOf(Location loc, PatternRewriter &rewriter, bool value);
    
    Type getIntPtrType(unsigned addressSpace, LLVMTypeConverter &typeConverter);
    
    Value conditionalExpressionLowering(Location loc, Type type, Value condition, 
        function_ref<Value(OpBuilder &, Location)> thenBuilder, 
        function_ref<Value(OpBuilder &, Location)> elseBuilder,
        PatternRewriter &rewriter);

    void NegativeOp(ts::ArithmeticUnaryOp &unaryOp, mlir::PatternRewriter &builder);

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

    template <typename UnaryOpTy, typename StdIOpTy, typename StdFOpTy>
    void UnaryOp(UnaryOpTy &unaryOp, mlir::PatternRewriter &builder)
    {
        auto oper = unaryOp.operand1();
        auto type = oper.getType();
        if (type.isIntOrIndex())
        {
            builder.replaceOpWithNewOp<StdIOpTy>(unaryOp, type, oper);
        }
        else if (!type.isIntOrIndex() && type.isIntOrIndexOrFloat())
        {
            builder.replaceOpWithNewOp<StdFOpTy>(unaryOp, type, oper);
        }
        else
        {
            llvm_unreachable("not implemented");
        }
    }  

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
    void LogicOp(BinOpTy &binOp, ConversionPatternRewriter &builder, LLVMTypeConverter &typeConverter)
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
        else if (leftType.dyn_cast_or_null<ts::StringType>() || leftType.dyn_cast_or_null<ts::AnyType>())
        {
            auto left = binOp.getOperand(0);
            auto right = binOp.getOperand(1);

            auto intPtrType = getIntPtrType(0, typeConverter);

            Value leftPtrValue = builder.create<LLVM::PtrToIntOp>(binOp.getLoc(), intPtrType, left);
            Value rightPtrValue = builder.create<LLVM::PtrToIntOp>(binOp.getLoc(), intPtrType, right);

            builder.replaceOpWithNewOp<StdIOpTy>(binOp, v1, leftPtrValue, rightPtrValue);
        }
        else
        {
            emitError(binOp.getLoc(), "Not implemented operator for type 1: '") << leftType << "'";
            llvm_unreachable("not implemented");
        }
    }
}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
