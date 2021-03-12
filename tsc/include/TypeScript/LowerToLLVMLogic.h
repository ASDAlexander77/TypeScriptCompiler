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
    class TypeHelper
    {
        PatternRewriter &rewriter;
    public:        
        TypeHelper(PatternRewriter &rewriter) : rewriter(rewriter) {}

        LLVM::LLVMVoidType getVoidType()
        {
            return LLVM::LLVMVoidType::get(rewriter.getContext());
        }

        Type getI8Type()
        {
            return rewriter.getIntegerType(8);
        }

        LLVM::LLVMPointerType getI8PtrType()
        {
            return LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));
        }

        LLVM::LLVMArrayType getArrayType(Type elementType, size_t size)
        {
            return LLVM::LLVMArrayType::get(elementType, size);
        }

        LLVM::LLVMFunctionType getFunctionType(mlir::Type result, ArrayRef<mlir::Type> arguments, bool isVarArg = false)
        {
            return LLVM::LLVMFunctionType::get(result, arguments, isVarArg);
        }

        LLVM::LLVMFunctionType getFunctionType(ArrayRef<mlir::Type> arguments, bool isVarArg = false)
        {
            return LLVM::LLVMFunctionType::get(getVoidType(), arguments, isVarArg);
        }
    };

    class TypeConverterHelper
    {
        TypeConverter &typeConverter;
    public:
        TypeConverterHelper(TypeConverter &typeConverter) : typeConverter(typeConverter) {}

        mlir::Type typeToConvertedType(mlir::Type type)
        {
            auto convertedType = typeConverter.convertType(type);
            assert(convertedType);
            return convertedType;
        }
    };

    class LLVMTypeConverterHelper
    {
        LLVMTypeConverter &typeConverter;
    public:
        LLVMTypeConverterHelper(LLVMTypeConverter &typeConverter) : typeConverter(typeConverter) {}

        Type getIntPtrType(unsigned addressSpace)
        {
            return IntegerType::get(&typeConverter.getContext(), typeConverter.getPointerBitwidth(addressSpace));
        }
    };    

    class LLVMCodeHelper
    {
        Operation *op;
        PatternRewriter &rewriter;
    public:        
        LLVMCodeHelper(Operation *op, PatternRewriter &rewriter) : op(op), rewriter(rewriter) {}

        /// Return a value representing an access into a global string with the given
        /// name, creating the string if necessary.
        Value getOrCreateGlobalString(StringRef name, StringRef value)
        {
            auto loc = op->getLoc();
            auto parentModule = op->getParentOfType<ModuleOp>();

            TypeHelper th(rewriter);

            // Create the global at the entry of the module.
            LLVM::GlobalOp global;
            if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPointToStart(parentModule.getBody());
                auto type = th.getArrayType(th.getI8Type(), value.size());
                global = rewriter.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, rewriter.getStringAttr(value));
            }

            // Get the pointer to the first character in the global string.
            Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = rewriter.create<LLVM::ConstantOp>(
                loc, IntegerType::get(rewriter.getContext(), 64),
                rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
            return rewriter.create<LLVM::GEPOp>(
                loc,
                th.getI8PtrType(),
                globalPtr, ArrayRef<Value>({cst0, cst0}));
        }

        Value getOrCreateGlobalString(StringRef name, std::string value, mlir::PatternRewriter &rewriter)
        {
            return getOrCreateGlobalString(name, StringRef(value.data(), value.length() + 1));
        }

        LLVM::LLVMFuncOp getOrInsertFunction(const StringRef &name, const LLVM::LLVMFunctionType &llvmFnType)
        {
            auto parentModule = op->getParentOfType<ModuleOp>();

            if (auto funcOp = parentModule.lookupSymbol<LLVM::LLVMFuncOp>(name))
            {
                return funcOp;
            }

            auto loc = op->getLoc();

            // Insert the printf function into the body of the parent module.
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());
            return rewriter.create<LLVM::LLVMFuncOp>(loc, name, llvmFnType);
        }        
    };

    class CodeLogicHelper
    {
        Operation *op;
        PatternRewriter &rewriter;
    public:        
        CodeLogicHelper(Operation *op, PatternRewriter &rewriter) : op(op), rewriter(rewriter) {}

        Value createI32ConstantOf(unsigned value)
        {
            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getI32Type(), value));
        }

        Value createI1ConstantOf(bool value)
        {
            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(1), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
        }    

        Value conditionalExpressionLowering(
            Type type, Value condition,
            function_ref<Value(OpBuilder &, Location)> thenBuilder,
            function_ref<Value(OpBuilder &, Location)> elseBuilder)
        {
            auto loc = op->getLoc();

            // Split block
            auto *opBlock = rewriter.getInsertionBlock();
            auto opPosition = rewriter.getInsertionPoint();
            auto *continuationBlock = rewriter.splitBlock(opBlock, opPosition);

            // then block
            auto *thenBlock = rewriter.createBlock(continuationBlock);
            auto thenValue = thenBuilder(rewriter, loc);

            // else block
            auto *elseBlock = rewriter.createBlock(continuationBlock);
            auto elseValue = elseBuilder(rewriter, loc);

            // result block
            auto *resultBlock = rewriter.createBlock(continuationBlock, TypeRange{type});
            rewriter.create<LLVM::BrOp>(
                loc,
                ValueRange{},
                continuationBlock);

            rewriter.setInsertionPointToEnd(thenBlock);
            rewriter.create<LLVM::BrOp>(
                loc,
                ValueRange{thenValue},
                resultBlock);

            rewriter.setInsertionPointToEnd(elseBlock);
            rewriter.create<LLVM::BrOp>(
                loc,
                ValueRange{elseValue},
                resultBlock);

            // Generate assertion test.
            rewriter.setInsertionPointToEnd(opBlock);
            rewriter.create<LLVM::CondBrOp>(
                loc,
                condition,
                thenBlock,
                elseBlock);

            rewriter.setInsertionPointToStart(continuationBlock);

            return resultBlock->getArguments().front();
        }
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
        LLVMTypeConverterHelper llvmtch(typeConverter);

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

            auto intPtrType = llvmtch.getIntPtrType(0);

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
