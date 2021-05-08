#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_

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

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

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

        Type getBooleanType()
        {
            return mlir_ts::BooleanType::get(rewriter.getContext());
        }

        Type getI8Type()
        {
            return rewriter.getIntegerType(8);
        }

        Type getI32Type()
        {
            return rewriter.getIntegerType(32);
        }

        Type getI64Type()
        {
            return rewriter.getIntegerType(64);
        }

        Type getF32Type()
        {
            return rewriter.getF32Type();
        }

        IntegerAttr getIndexAttrValue(int64_t value)
        {
            return rewriter.getIntegerAttr(rewriter.getIndexType(), value);
        }

        LLVM::LLVMPointerType getI8PtrType()
        {
            return LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));
        }

        LLVM::LLVMPointerType getI8PtrPtrType()
        {
            return LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
        }


        LLVM::LLVMPointerType getPointerType(Type elementType)
        {
            return LLVM::LLVMPointerType::get(elementType);
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

        mlir::Type convertType(mlir::Type type)
        {
            if (type)
            {
                if (auto convertedType = typeConverter.convertType(type))
                {
                    return convertedType;
                }
            }

            return type;
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
        TypeConverter *typeConverter;
    public:        
        LLVMCodeHelper(Operation *op, PatternRewriter &rewriter) : op(op), rewriter(rewriter) {}
        LLVMCodeHelper(Operation *op, PatternRewriter &rewriter, TypeConverter *typeConverter) : op(op), rewriter(rewriter), typeConverter(typeConverter) {}

    private:
        /// Return a value representing an access into a global string with the given
        /// name, creating the string if necessary.
        Value getOrCreateGlobalString_(StringRef name, StringRef value)
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
                globalPtr, 
                ArrayRef<Value>({cst0, cst0}));
        }

    public:

        std::string getStorageStringName(std::string value)
        {
            auto opHash = std::hash<std::string>{}(value);

            std::stringstream strVarName;
            strVarName << "s_" << opHash;

            return strVarName.str();
        }

        Value getOrCreateGlobalString(std::string value)
        {
            return getOrCreateGlobalString(getStorageStringName(value), value);
        }        

        Value getOrCreateGlobalString(StringRef name, std::string value)
        {
            return getOrCreateGlobalString_(name, StringRef(value.data(), value.length() + 1));
        }

        Value getOrCreateGlobalArray(mlir::Type originalElementType, StringRef name, mlir::Type llvmElementType, unsigned size, ArrayAttr arrayAttr)
        {
            auto loc = op->getLoc();
            auto parentModule = op->getParentOfType<ModuleOp>();

            TypeHelper th(rewriter);

            auto pointerType = LLVM::LLVMPointerType::get(llvmElementType);
            auto arrayType = th.getArrayType(llvmElementType, size);

            // Create the global at the entry of the module.
            LLVM::GlobalOp global;
            if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPointToStart(parentModule.getBody());

                // dense value
                auto value = arrayAttr.getValue();
                if (llvmElementType.isIntOrFloat())
                {
                    // end
                    auto dataType = mlir::VectorType::get({static_cast<int64_t>(value.size())}, llvmElementType);
                    auto attr = mlir::DenseElementsAttr::get(dataType, value);                
                    global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, attr);
                }
                else if (originalElementType.dyn_cast_or_null<mlir_ts::StringType>())
                {
                    mlir::OpBuilder::InsertionGuard guard(rewriter);
                    
                    global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, Attribute{});

                    global.getInitializerRegion().push_back(new Block());
                    rewriter.setInsertionPointToStart(global.getInitializerBlock());

                    mlir::Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

                    auto position = 0;
                    for (auto item : arrayAttr.getValue())
                    {
                        auto strValue = item.cast<StringAttr>().getValue().str();
                        auto itemVal = getOrCreateGlobalString(strValue);                        

                        arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, itemVal, rewriter.getI64ArrayAttr(position++));
                    }

                    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{arrayVal});
                }
                else
                {
                    llvm_unreachable("array literal is not implemented(1)");
                }
            }

            // Get the pointer to the first character in the global string.
            Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = rewriter.create<LLVM::ConstantOp>(
                loc, IntegerType::get(rewriter.getContext(), 64),
                rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
            return rewriter.create<LLVM::GEPOp>(
                loc,
                pointerType,
                globalPtr, 
                ArrayRef<Value>({cst0, cst0}));            
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

        mlir::Value GetAddressOfElement(mlir::Value array, mlir::Value index)
        {
            TypeHelper th(rewriter);
            TypeConverterHelper tch(*typeConverter);

            auto loc = op->getLoc();
            auto globalPtr = array;
            auto addr = rewriter.create<LLVM::GEPOp>(
                loc,
                th.getPointerType(tch.convertType(array.getType().cast<mlir_ts::ArrayType>().getElementType())), 
                globalPtr,
                ValueRange{index});

            return addr;            
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

        Value createI64ConstantOf(unsigned value)
        {
            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(64), rewriter.getIntegerAttr(rewriter.getI64Type(), value));
        }

        Value createI1ConstantOf(bool value)
        {
            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(1), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
        }    

        Value createF32ConstantOf(float value)
        {
            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getF32Type(), rewriter.getIntegerAttr(rewriter.getF32Type(), value));
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

        template <typename OpTy>
        void saveResult(OpTy& op, mlir::Value result)
        {
            auto defOp = op.operand1().getDefiningOp();
            // TODO: finish it for field access
            if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(defOp))
            {
                rewriter.create<mlir_ts::StoreOp>(op->getLoc(), result, loadOp.reference());
            }
            else if (auto loadElementOp = dyn_cast<mlir_ts::LoadElementOp>(defOp))
            {
                rewriter.create<mlir_ts::StoreElementOp>(op->getLoc(), result, loadElementOp.array(), loadElementOp.index());
            }
            else
            {
                llvm_unreachable("not implemented");
            }
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
            emitError(binOp.getLoc(), "Not implemented operator for type 1: '") << type << "'";
            llvm_unreachable("not implemented");
        }
    }  

    template <typename BinOpTy>
    bool IsStringArg(BinOpTy &binOp)
    {
        auto leftType = binOp.getOperand(0).getType();
        if (leftType.dyn_cast_or_null<mlir_ts::StringType>())
        {
            return true;
        }

        return false;
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
        else if (leftType.dyn_cast_or_null<mlir_ts::NumberType>())
        {
            auto castLeft = builder.create<mlir_ts::CastOp>(binOp->getLoc(), builder.getF32Type(), binOp.getOperand(0));
            auto castRight = builder.create<mlir_ts::CastOp>(binOp->getLoc(), builder.getF32Type(), binOp.getOperand(1));
            builder.replaceOpWithNewOp<StdFOpTy>(binOp, castLeft, castRight);
        }
        else
        {
            emitError(binOp.getLoc(), "Not implemented operator for type 1: '") << leftType << "'";
            llvm_unreachable("not implemented");
        }
    }

    template <typename BinOpTy, typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    void LogicOp(BinOpTy &binOp, ConversionPatternRewriter &builder, LLVMTypeConverter &typeConverter)
    {
        LLVMTypeConverterHelper llvmtch(typeConverter);

        auto leftType = binOp.getOperand(0).getType();
        if (leftType.isIntOrIndex() || leftType.dyn_cast_or_null<mlir_ts::BooleanType>())
        {
            builder.replaceOpWithNewOp<StdIOpTy>(binOp, v1, binOp.getOperand(0), binOp.getOperand(1));
        }
        else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
        {
            builder.replaceOpWithNewOp<StdFOpTy>(binOp, v2, binOp.getOperand(0), binOp.getOperand(1));
        }
        else if (/*leftType.dyn_cast_or_null<mlir_ts::StringType>() || */leftType.dyn_cast_or_null<mlir_ts::AnyType>())
        {
            // excluded string
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
