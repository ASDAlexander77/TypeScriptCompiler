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
#include "llvm/IR/DataLayout.h"

#include "TypeScript/CommonGenLogic.h"

#include "scanner_enums.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace typescript
{
    class TypeHelper
    {
        MLIRContext *context;
    public:        
        TypeHelper(PatternRewriter &rewriter) : context(rewriter.getContext()) {}
        TypeHelper(MLIRContext *context) : context(context) {}

        Type getBooleanType()
        {
            return mlir_ts::BooleanType::get(context);
        }

        Type getI8Type()
        {
            return IntegerType::get(context, 8);
        }

        Type getI32Type()
        {
            return IntegerType::get(context, 32);
        }

        Type getI64Type()
        {
            return IntegerType::get(context, 64);
        }

        Type getF32Type()
        {
            return FloatType::getF32(context);
        }

        IntegerAttr getStructIndexAttrValue(int32_t value)
        {
            return IntegerAttr::get(getI32Type(), APInt(32, value));
        }

        Type getIndexType() 
        { 
            return getI64Type(); 
        }

        IntegerAttr getIndexAttrValue(int64_t value)
        {
            return IntegerAttr::get(getIndexType(), APInt(64, value));
        }

        Type getLLVMBoolType()
        {
            return IntegerType::get(context, 1/*, IntegerType::SignednessSemantics::Unsigned*/);
        }

        LLVM::LLVMVoidType getVoidType()
        {
            return LLVM::LLVMVoidType::get(context);
        }

        LLVM::LLVMPointerType getI8PtrType()
        {
            return LLVM::LLVMPointerType::get(getI8Type());
        }

        LLVM::LLVMPointerType getI8PtrPtrType()
        {
            return LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(getI8Type()));
        }

        LLVM::LLVMPointerType getPointerType(Type elementType)
        {
            return LLVM::LLVMPointerType::get(elementType);
        }

        LLVM::LLVMArrayType getArrayType(Type elementType, size_t size)
        {
            return LLVM::LLVMArrayType::get(elementType, size);
        }

        LLVM::LLVMFunctionType getFunctionType(Type result, ArrayRef<Type> arguments, bool isVarArg = false)
        {
            return LLVM::LLVMFunctionType::get(result, arguments, isVarArg);
        }

        LLVM::LLVMFunctionType getFunctionType(ArrayRef<Type> arguments, bool isVarArg = false)
        {
            return LLVM::LLVMFunctionType::get(getVoidType(), arguments, isVarArg);
        }

        llvm::TypeSize getTypeSize(mlir::Type valueType)
        {
            auto size = LLVM::getPrimitiveTypeSizeInBits(valueType);
            if (size > 0)
            {
                return size;
            }

            // TODO: review DataLayout.getTypeAllocSizeInBits, https://llvm.org/docs/LangRef.html#langref-datalayout as class init

            auto calcSize = llvm::TypeSwitch<Type, llvm::TypeSize>(valueType)
                .Case<LLVM::LLVMArrayType>([&](LLVM::LLVMArrayType aty) 
                { 
                    auto sizeElement = getTypeSize(aty.getElementType());
                    auto count = aty.getNumElements();
                    return llvm::TypeSize::Fixed(sizeElement * count); 
                })
                .Case<LLVM::LLVMPointerType>([&](LLVM::LLVMPointerType pty) 
                { 
                    return llvm::TypeSize::Fixed(64); 
                })                
                .Default([](Type ty) 
                {
                    assert(false);
                    return llvm::TypeSize::Fixed(0);
                });

            return calcSize;
        }
    };

    class TypeConverterHelper
    {
        TypeConverter &typeConverter;
    public:
        TypeConverterHelper(TypeConverter *typeConverter) : typeConverter(*typeConverter) {
            assert(typeConverter);
        }

        Type convertType(Type type)
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

        Type makePtrToValue(Type type)
        {
            if (auto constArray = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
            {
                return LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(convertType(constArray.getElementType()), constArray.getSize()));
            }

            llvm_unreachable("not implemented");
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

        int32_t getPointerBitwidth(unsigned addressSpace)
        {
            return typeConverter.getPointerBitwidth(addressSpace);
        }
    };    

    class CodeLogicHelper
    {
        Operation *op;
        PatternRewriter &rewriter;
    public:        
        CodeLogicHelper(Operation *op, PatternRewriter &rewriter) : op(op), rewriter(rewriter) {}

        Value createIConstantOf(unsigned width, unsigned value)
        {
            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(width), rewriter.getIntegerAttr(rewriter.getIntegerType(width), value));
        }

        Value createFConstantOf(unsigned width, double value)
        {
            auto ftype = rewriter.getF32Type();
            if (width == 16)
            {
                ftype = rewriter.getF16Type();
            }
            else if (width == 64)
            {
                ftype = rewriter.getF64Type();
            }
            else if (width == 128)
            {
                ftype = rewriter.getF128Type();
            }

            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), ftype, rewriter.getFloatAttr(ftype, value));
        }

        Value createI8ConstantOf(unsigned value)
        {
            return rewriter.create<LLVM::ConstantOp>(op->getLoc(), rewriter.getIntegerType(8), rewriter.getIntegerAttr(rewriter.getIntegerType(8), value));
        }

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

        Value castToI8Ptr(mlir::Value value)
        {
            TypeHelper th(rewriter);
            return rewriter.create<LLVM::BitcastOp>(op->getLoc(), th.getI8PtrType(), value);
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
        void saveResult(OpTy& op, Value result)
        {
            auto defOp = op.operand1().getDefiningOp();
            // TODO: finish it for field access
            if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(defOp))
            {
                rewriter.create<mlir_ts::StoreOp>(op->getLoc(), result, loadOp.reference());
            }
            else
            {
                llvm_unreachable("not implemented");
            }
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

        template <typename T>
        void seekLast(Block *block)
        {
            // find last string
            auto lastUse = [&](Operation* op) {
                if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op)) {
                    if (globalOp.valueAttr() && globalOp.valueAttr().isa<T>()) {
                        rewriter.setInsertionPointAfter(globalOp);
                    }
                }
            };

            block->walk(lastUse);      
        }

        void seekLast(Block *block)
        {
            // find last string
            auto lastUse = [&](Operation* op) {
                if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op)) {
                    rewriter.setInsertionPointAfter(globalOp);
                }
            };

            block->walk(lastUse);      
        }

        void seekLastWithBody(Block *block)
        {
            // find last string
            auto lastUse = [&](Operation* op) {
                if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op)) {
                    if (globalOp.getInitializerBlock()) {
                        rewriter.setInsertionPointAfter(globalOp);
                    }
                }
            };

            block->walk(lastUse);      
        }

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

                seekLast<StringAttr>(parentModule.getBody());

                auto type = th.getArrayType(th.getI8Type(), value.size());
                global = rewriter.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, rewriter.getStringAttr(value));
            }

            // Get the pointer to the first character in the global string.
            Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = rewriter.create<LLVM::ConstantOp>(
                loc, 
                th.getIndexType(),
                th.getIndexAttrValue(0));
            return rewriter.create<LLVM::GEPOp>(
                loc,
                th.getI8PtrType(),
                globalPtr, 
                ArrayRef<Value>({cst0, cst0}));
        }

    public:

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

        std::string getStorageStringName(std::string value)
        {
            auto opHash = std::hash<std::string>{}(value);

            std::stringstream strVarName;
            strVarName << "s_" << opHash;

            return strVarName.str();
        }

        std::string getStorageTupleName(std::string value)
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

        Value getOrCreateGlobalArray(Type originalElementType, Type llvmElementType, unsigned size, ArrayAttr arrayAttr)
        {
            auto vecVarName = calc_hash_value(arrayAttr, "a_"); 
            return getOrCreateGlobalArray(originalElementType, vecVarName, llvmElementType, size, arrayAttr);
        }

        Value getReadOnlyRTArray(mlir::Location loc, mlir_ts::ArrayType originalArrayType, LLVM::LLVMStructType llvmArrayType, ArrayAttr arrayValue)
        {
            auto llvmSubElementType = llvmArrayType.getBody()[0].cast<LLVM::LLVMPointerType>().getElementType();

            auto size = arrayValue.size();
            auto itemValArrayPtr = getOrCreateGlobalArray(originalArrayType.getElementType(), llvmSubElementType, size, arrayValue);        

            // create ReadOnlyRuntimeArrayType
            auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmArrayType);
            //auto arrayPtrType = LLVM::LLVMPointerType::get(llvmSubElementType);
            //auto arrayValueSize = LLVM::LLVMArrayType::get(llvmSubElementType, size);
            //auto ptrToArray = LLVM::LLVMPointerType::get(arrayValueSize);

            auto sizeValue = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getI32Type(), arrayValue.size()));

            auto structValue2 = rewriter.create<LLVM::InsertValueOp>(
                loc, 
                llvmArrayType,
                structValue, 
                itemValArrayPtr,
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));           

            auto structValue3 = rewriter.create<LLVM::InsertValueOp>(
                loc, 
                llvmArrayType, 
                structValue2,
                sizeValue, 
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

            return structValue3;
        }

        Value getOrCreateGlobalArray(Type originalElementType, StringRef name, Type llvmElementType, unsigned size, ArrayAttr arrayAttr)
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
                    seekLast<DenseElementsAttr>(parentModule.getBody());

                    // end
                    auto dataType = VectorType::get({static_cast<int64_t>(value.size())}, llvmElementType);
                    auto attr = DenseElementsAttr::get(dataType, value);                
                    global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, attr);
                }
                else if (originalElementType.dyn_cast_or_null<mlir_ts::StringType>())
                {
                    seekLast(parentModule.getBody());

                    OpBuilder::InsertionGuard guard(rewriter);
                    
                    global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, Attribute{});

                    Region &region = global.getInitializerRegion();
                    Block *block = rewriter.createBlock(&region);

                    // Initialize the tuple
                    rewriter.setInsertionPoint(block, block->begin());

                    Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

                    auto position = 0;
                    for (auto item : arrayAttr.getValue())
                    {
                        auto strValue = item.cast<StringAttr>().getValue().str();
                        auto itemVal = getOrCreateGlobalString(strValue);                        

                        arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, itemVal, rewriter.getI64ArrayAttr(position++));
                    }

                    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{arrayVal});
                }
                else if (auto originalArrayType = originalElementType.dyn_cast_or_null<mlir_ts::ArrayType>())
                { 
                    seekLast(parentModule.getBody());

                    OpBuilder::InsertionGuard guard(rewriter);
                    
                    global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, Attribute{});

                    Region &region = global.getInitializerRegion();
                    Block *block = rewriter.createBlock(&region);

                    // Initialize the tuple
                    rewriter.setInsertionPoint(block, block->begin());

                    Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

                    // TODO: implement ReadOnlyRTArray; as RTArray may contains ConstArray data (so using not editable memory)

                    auto position = 0;
                    for (auto item : arrayAttr.getValue())
                    {
                        auto arrayValue = item.cast<ArrayAttr>();
                        auto itemVal = getReadOnlyRTArray(loc, originalArrayType, llvmElementType.cast<LLVM::LLVMStructType>(), arrayValue);

                        arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, itemVal, rewriter.getI64ArrayAttr(position++));
                    }

                    rewriter.create<LLVM::ReturnOp>(loc, ValueRange{arrayVal});                    
                }
                else if (originalElementType.dyn_cast_or_null<mlir_ts::ConstArrayType>())
                {                                        
                    //
                    llvm_unreachable("ConstArrayType must not be used in array, use normal ArrayType (the same way as StringType)");
                }
                else if (auto constTupleType = originalElementType.dyn_cast_or_null<mlir_ts::ConstTupleType>())
                {                                        
                    seekLast(parentModule.getBody());

                    OpBuilder::InsertionGuard guard(rewriter);
                    
                    global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, Attribute{});

                    Region &region = global.getInitializerRegion();
                    Block *block = rewriter.createBlock(&region);

                    // Initialize the tuple
                    rewriter.setInsertionPoint(block, block->begin());

                    Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

                    auto position = 0;
                    for (auto item : arrayAttr.getValue())
                    {
                        auto tupleVal = getTupleFromArrayAttr(loc, constTupleType, llvmElementType.cast<LLVM::LLVMStructType>(), item.dyn_cast_or_null<ArrayAttr>());
                        arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, tupleVal, rewriter.getI64ArrayAttr(position++));
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
                loc,
                th.getIndexType(),
                th.getIndexAttrValue(0));
            return rewriter.create<LLVM::GEPOp>(
                loc,
                pointerType,
                globalPtr, 
                ArrayRef<Value>({cst0, cst0}));            
        }        

        Value getTupleFromArrayAttr(Location loc, mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType, ArrayAttr arrayAttr)
        {
            Value tupleVal = rewriter.create<LLVM::UndefOp>(loc, llvmStructType);

            auto typesRange = llvmStructType.getBody();

            auto position = 0;
            for (auto item : arrayAttr.getValue())
            {
                auto type = originalType.getType(position);

                auto llvmType = typesRange[position];
                if (item.isa<StringAttr>())
                {
                    OpBuilder::InsertionGuard guard(rewriter);
                    
                    auto strValue = item.cast<StringAttr>().getValue().str();
                    auto itemVal = getOrCreateGlobalString(strValue);                        

                    tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, rewriter.getI64ArrayAttr(position++));
                }
                else if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
                {
                    llvm_unreachable("not used.");
                }       
                else if (auto arrayType = type.dyn_cast_or_null<mlir_ts::ArrayType>())
                {
                    auto subArrayAttr = item.dyn_cast_or_null<ArrayAttr>();

                    OpBuilder::InsertionGuard guard(rewriter);

                    auto itemVal = getReadOnlyRTArray(loc, arrayType, llvmType.cast<LLVM::LLVMStructType>(), subArrayAttr);
                    tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, rewriter.getI64ArrayAttr(position++));
                }                               
                else if (auto constTupleType = type.dyn_cast_or_null<mlir_ts::ConstTupleType>())
                {
                    auto subArrayAttr = item.dyn_cast_or_null<ArrayAttr>();

                    OpBuilder::InsertionGuard guard(rewriter);

                    auto subTupleVal = getTupleFromArrayAttr(loc, constTupleType, llvmType.cast<LLVM::LLVMStructType>(), subArrayAttr);

                    tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, subTupleVal, rewriter.getI64ArrayAttr(position++));
                }                
                else
                {
                    // DO NOT Replace with LLVM::ConstantOp - to use AddressOf for global symbol names
                    auto itemValue = rewriter.create<mlir::ConstantOp>(loc, llvmType, item);
                    tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemValue, rewriter.getI64ArrayAttr(position++));
                }
            }

            return tupleVal;
        }

        Value getOrCreateGlobalTuple(mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType, ArrayAttr arrayAttr)
        {
            auto varName = calc_hash_value(arrayAttr, "tp_");   
            return getOrCreateGlobalTuple(originalType, llvmStructType, varName, arrayAttr);
        }

        Value getOrCreateGlobalTuple(mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType, StringRef name, ArrayAttr arrayAttr)
        {
            auto loc = op->getLoc();
            auto parentModule = op->getParentOfType<ModuleOp>();

            TypeHelper th(rewriter);
            
            auto pointerType = LLVM::LLVMPointerType::get(llvmStructType);

            // Create the global at the entry of the module.
            LLVM::GlobalOp global;
            if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
            {
                OpBuilder::InsertionGuard insertGuard(rewriter);
                rewriter.setInsertionPointToStart(parentModule.getBody());

                seekLast(parentModule.getBody());                
              
                global = rewriter.create<LLVM::GlobalOp>(loc, llvmStructType, true, LLVM::Linkage::Internal, name, Attribute{});

                Region &region = global.getInitializerRegion();
                Block *block = rewriter.createBlock(&region);

                // Initialize the tuple
                rewriter.setInsertionPoint(block, block->begin());

                auto tupleVal = getTupleFromArrayAttr(loc, originalType, llvmStructType, arrayAttr);
                rewriter.create<LLVM::ReturnOp>(loc, ValueRange{tupleVal});
            }

            // Get the pointer to the first character in the global string.
            Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = rewriter.create<LLVM::ConstantOp>(
                loc, 
                th.getIndexType(),
                th.getIndexAttrValue(0));
            return rewriter.create<LLVM::GEPOp>(
                loc,
                pointerType,
                globalPtr, 
                ArrayRef<Value>({cst0}));  
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

        Value GetAddressOfArrayElement(Type elementRefType, Value arrayOrStringOrTuple, Value index)
        {
            TypeHelper th(rewriter);
            TypeConverterHelper tch(typeConverter);
            CodeLogicHelper clh(op, rewriter);

            auto loc = op->getLoc();

            assert(elementRefType.isa<mlir_ts::RefType>());

            auto ptrType = tch.convertType(elementRefType);

            auto dataPtr = arrayOrStringOrTuple;
            if (auto arrayType = arrayOrStringOrTuple.getType().isa<mlir_ts::ArrayType>())
            {
                // extract pointer from struct
                dataPtr = rewriter.create<LLVM::ExtractValueOp>(
                    loc, 
                    ptrType, 
                    arrayOrStringOrTuple, 
                    rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));
            }

            auto addr = rewriter.create<LLVM::GEPOp>(
                loc,
                ptrType, 
                dataPtr,
                ValueRange{index});

            return addr;            
        }     

        Value GetAddressOfStructElement(Type elementRefType, Value arrayOrStringOrTuple, int32_t index)
        {
            // index of struct MUST BE 32 bit
            TypeHelper th(rewriter);
            TypeConverterHelper tch(typeConverter);

            auto loc = op->getLoc();
            auto globalPtr = arrayOrStringOrTuple;

            assert(elementRefType.isa<mlir_ts::RefType>());

            auto ptrType = tch.convertType(elementRefType);

            SmallVector<Value> indexes;
            // add first index which 64 bit (struct field MUST BE 32 bit index)
            auto firstIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
            indexes.push_back(firstIndex);
            auto fieldIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(index));
            indexes.push_back(fieldIndex);

            auto addr = rewriter.create<LLVM::GEPOp>(
                loc,
                ptrType, 
                globalPtr,
                indexes);

            return addr;            
        }        
    };

    class CastLogicHelper
    {
        Operation *op;
        PatternRewriter &rewriter;
        TypeConverterHelper &tch;
        TypeHelper th;
        LLVMCodeHelper ch;
        CodeLogicHelper clh;
        Location loc;

    public:        
        CastLogicHelper(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch) : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter), clh(op, rewriter), loc(op->getLoc()) {}

        Value cast(mlir::Value in, mlir::Type inLLVMType, mlir::Type resType, mlir::Type resLLVMType)
        {
            if (inLLVMType == resLLVMType)
            {
                return in;
            }

            auto inType = in.getType();

            if (inType.dyn_cast_or_null<mlir_ts::CharType>() && resType.dyn_cast_or_null<mlir_ts::StringType>())
            {
                // types are equals
                return rewriter.create<mlir_ts::CharToStringOp>(loc, mlir_ts::StringType::get(rewriter.getContext()), in);
            }

            if ((inLLVMType.isInteger(32) || inLLVMType.isInteger(64)) && (resLLVMType.isF32() || resLLVMType.isF64()))
            {
                return rewriter.create<SIToFPOp>(loc, resLLVMType, in);
            }

            if ((inLLVMType.isF32() || inLLVMType.isF64()) && (resLLVMType.isInteger(32) || resLLVMType.isInteger(64)))
            {
                return rewriter.create<FPToSIOp>(loc, resLLVMType, in);
            }

            if ((inLLVMType.isInteger(64) || inLLVMType.isInteger(32) || inLLVMType.isInteger(8)) && resLLVMType.isInteger(1))
            {
                return rewriter.create<CmpIOp>(loc, CmpIPredicate::ne, in, clh.createI32ConstantOf(0));
            }

            if (inLLVMType.isInteger(1) && (resLLVMType.isInteger(8) || resLLVMType.isInteger(32) || resLLVMType.isInteger(64)))
            {
                return rewriter.create<ZeroExtendIOp>(loc, in, resLLVMType);
            }            

            if (inLLVMType.isInteger(8) && resLLVMType.isInteger(32))
            {
                return rewriter.create<ZeroExtendIOp>(loc, in, resLLVMType);
            }               

            if ((inLLVMType.isInteger(8) || inLLVMType.isInteger(32)) && resLLVMType.isInteger(64))
            {
                return rewriter.create<ZeroExtendIOp>(loc, in, resLLVMType);
            }               

            if ((inLLVMType.isInteger(64) || inLLVMType.isInteger(32) || inLLVMType.isInteger(16)) && resLLVMType.isInteger(8))
            {
                return rewriter.create<TruncateIOp>(loc, in, resLLVMType);
            }

            if (inLLVMType.isInteger(64) && resLLVMType.isInteger(32))
            {
                return rewriter.create<TruncateIOp>(loc, in, resLLVMType);
            }

            if (inLLVMType.isF32() && (resLLVMType.isF64() || resLLVMType.isF128()))
            {
                return rewriter.create<FPExtOp>(loc, in, resLLVMType);
            }

            if ((inLLVMType.isF64() || inLLVMType.isF128()) && resLLVMType.isF32())
            {
                return rewriter.create<FPTruncOp>(loc, in, resLLVMType);
            }

            auto isResString = resType.dyn_cast_or_null<mlir_ts::StringType>();

            if (inLLVMType.isInteger(1) && isResString)
            {
                return castBoolToString(in);
            }

            if (inLLVMType.isInteger(32) && isResString)
            {
                return castI32ToString(in);
            }

            if (inLLVMType.isInteger(64) && isResString)
            {
                return castI64ToString(in);
            }            

            if ((inLLVMType.isF32() || inLLVMType.isF64()) && isResString)
            {
                return castF32orF64ToString(in);
            }

            if (auto arrType = resType.dyn_cast_or_null<mlir_ts::ArrayType>())
            {
                return castToArrayType(in, resType);
            }

            auto isInString = inType.dyn_cast_or_null<mlir_ts::StringType>();
            if (isInString && (resLLVMType.isInteger(32) || resLLVMType.isInteger(64)))
            {
                return rewriter.create<mlir_ts::ParseIntOp>(loc, resLLVMType, in);
            }

            if (isInString && (resLLVMType.isF32() || resLLVMType.isF64()))
            {
                return rewriter.create<mlir_ts::ParseFloatOp>(loc, resLLVMType, in);
            }

            // cast value value to optional value
            if (auto optType = resType.dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                return rewriter.create<mlir_ts::CreateOptionalOp>(loc, resType, in);
            }

            if (auto optType = inType.dyn_cast_or_null<mlir_ts::OptionalType>())
            {
                auto val = rewriter.create<mlir_ts::ValueOp>(loc, optType.getElementType(), in);
                return cast(val, tch.convertType(val.getType()), resType, resLLVMType);
            }            

            // ptrs cast
            if (inLLVMType.isa<LLVM::LLVMPointerType>() && resLLVMType.isa<LLVM::LLVMPointerType>())
            {
                return rewriter.create<LLVM::BitcastOp>(loc, resLLVMType, in);
            }            

            emitError(loc, "invalid cast operator type 1: '") << inLLVMType << "', type 2: '" << resLLVMType << "'";
            llvm_unreachable("not implemented");

            return mlir::Value();
        }        

        mlir::Value castBoolToString(mlir::Value in)
        {
            return rewriter.create<LLVM::SelectOp>(
                loc,
                in,
                ch.getOrCreateGlobalString("__true__", std::string("true")),
                ch.getOrCreateGlobalString("__false__", std::string("false"))); 
        }
        
        mlir::Value castI32ToString(mlir::Value in)
        {
            auto i8PtrTy = th.getI8PtrType();

            auto _itoaFuncOp =
                ch.getOrInsertFunction(
                    "_itoa",
                    th.getFunctionType(th.getI8PtrType(), ArrayRef<mlir::Type>{rewriter.getI32Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

            auto bufferSizeValue = clh.createI32ConstantOf(50);
            auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
            auto base = clh.createI32ConstantOf(10);

            return rewriter.create<LLVM::CallOp>(loc, _itoaFuncOp, ValueRange{in, newStringValue, base}).getResult(0);
        }
        
        mlir::Value castI64ToString(mlir::Value in)
        {
            auto i8PtrTy = th.getI8PtrType();

            auto _itoaFuncOp =
                ch.getOrInsertFunction(
                    "_i64toa",
                    th.getFunctionType(th.getI8PtrType(), ArrayRef<mlir::Type>{rewriter.getI32Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

            auto bufferSizeValue = clh.createI32ConstantOf(50);
            auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
            auto base = clh.createI32ConstantOf(10);

            return rewriter.create<LLVM::CallOp>(loc, _itoaFuncOp, ValueRange{in, newStringValue, base}).getResult(0);
        }        

        mlir::Value castF32orF64ToString(mlir::Value in)
        {
            auto i8PtrTy = th.getI8PtrType();

            auto _gcvtFuncOp =
                ch.getOrInsertFunction(
                    "_gcvt",
                    th.getFunctionType(th.getI8PtrType(), ArrayRef<mlir::Type>{rewriter.getF64Type(), rewriter.getI32Type(), th.getI8PtrType()}, true));

            auto bufferSizeValue = clh.createI32ConstantOf(50);
            auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
            auto doubleValue = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), in);
            auto precision = clh.createI32ConstantOf(16);

            return rewriter.create<LLVM::CallOp>(loc, _gcvtFuncOp, ValueRange{doubleValue, precision, newStringValue}).getResult(0);
        }

        mlir::Value castToArrayType(mlir::Value in, mlir::Type resType)
        {
            mlir::Type llvmElementType;
            auto type = in.getType();
            auto size = 0;
            if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
            {
                size = constArrayType.getSize();
                llvmElementType = tch.convertType(constArrayType.getElementType());
            }
            else if (auto ptrValue = type.dyn_cast_or_null<LLVM::LLVMPointerType>())
            {
                auto elementType = ptrValue.getElementType();
                if (auto arrayType = elementType.dyn_cast_or_null<LLVM::LLVMArrayType>())
                {
                    size = arrayType.getNumElements();
                    llvmElementType = tch.convertType(arrayType.getElementType());
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            }
            else
            {
                llvm_unreachable("not implemented");
            }

            auto llvmRtArrayStructType = tch.convertType(resType);

            auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
            auto arrayPtrType = LLVM::LLVMPointerType::get(llvmElementType);
            auto arrayValueSize = LLVM::LLVMArrayType::get(llvmElementType, size);
            auto ptrToArray = LLVM::LLVMPointerType::get(arrayValueSize);

            auto sizeValue = clh.createI32ConstantOf(size); 

            mlir::Value arrayPtr;
            bool byValue = true;
            if (byValue)
            {
                auto copyAllocated =
                    rewriter.create<LLVM::AllocaOp>(
                        loc,
                        arrayPtrType,
                        sizeValue);

                auto ptrToArraySrc = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, in);
                auto ptrToArrayDst = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, copyAllocated);
                rewriter.create<mlir_ts::LoadSaveOp>(loc, ptrToArrayDst, ptrToArraySrc);              

                arrayPtr = copyAllocated;           
            }
            else
            {
                // copy ptr only (const ptr -> ptr)
                // TODO: here we need to clone body to make it writable (and remove logic from VariableOp)
                arrayPtr = rewriter.create<LLVM::BitcastOp>(loc, arrayPtrType, in);
            }

            auto structValue2 = rewriter.create<LLVM::InsertValueOp>(
                loc, 
                llvmRtArrayStructType,
                structValue, 
                arrayPtr,
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));           

            auto structValue3 = rewriter.create<LLVM::InsertValueOp>(
                loc, 
                llvmRtArrayStructType, 
                structValue2,
                sizeValue, 
                rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(1)));

            return structValue3;
        }
    };    

    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value LogicOp_(Operation *, mlir::Value, mlir::Value, PatternRewriter &, LLVMTypeConverter &);

    class OptionalLogicHelper
    {
        Operation *op;
        PatternRewriter &rewriter;
        LLVMTypeConverter &typeConverter;
    public:        
        OptionalLogicHelper(Operation *op, PatternRewriter &rewriter, LLVMTypeConverter &typeConverter) : op(op), rewriter(rewriter), typeConverter(typeConverter) {}

        template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
        Value logicalOp(Operation *binOp, SyntaxKind opCmpCode)
        {
            auto loc = binOp->getLoc();

            TypeHelper th(rewriter);
            CodeLogicHelper clh(op, rewriter);

            auto left = binOp->getOperand(0);
            auto right = binOp->getOperand(1);
            auto leftType = left.getType();
            auto rightType = right.getType();
            auto leftOptType = leftType.dyn_cast_or_null<mlir_ts::OptionalType>();
            auto rightOptType = rightType.dyn_cast_or_null<mlir_ts::OptionalType>();

            assert(leftOptType || rightOptType);

            // case 1, when both are optional
            if (leftOptType && rightOptType)
            {
                // both are optional types
                // compare hasvalue first
                auto leftUndefFlagValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), left);
                auto rightUndefFlagValueBool = rewriter.create<mlir_ts::HasValueOp>(loc, th.getBooleanType(), right);

                auto leftUndefFlagValue = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), leftUndefFlagValueBool);
                auto rightUndefFlagValue = rewriter.create<mlir_ts::CastOp>(loc, th.getI32Type(), rightUndefFlagValueBool);

                auto whenBothHasNoValues = [&](OpBuilder & builder, Location loc) 
                {
                    mlir::Value undefFlagCmpResult;
                    switch (opCmpCode)
                    {
                        case SyntaxKind::EqualsEqualsToken:
                        case SyntaxKind::EqualsEqualsEqualsToken:
                            undefFlagCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, leftUndefFlagValue, rightUndefFlagValue);
                            break;
                        case SyntaxKind::ExclamationEqualsToken:
                        case SyntaxKind::ExclamationEqualsEqualsToken:
                            undefFlagCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, leftUndefFlagValue, rightUndefFlagValue);
                            break;
                        case SyntaxKind::GreaterThanToken:
                            undefFlagCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, leftUndefFlagValue, rightUndefFlagValue);
                            break;
                        case SyntaxKind::GreaterThanEqualsToken:
                            undefFlagCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, leftUndefFlagValue, rightUndefFlagValue);
                            break;
                        case SyntaxKind::LessThanToken:
                            undefFlagCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, leftUndefFlagValue, rightUndefFlagValue);
                            break;
                        case SyntaxKind::LessThanEqualsToken:
                            undefFlagCmpResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, leftUndefFlagValue, rightUndefFlagValue);
                            break;
                        default:
                            llvm_unreachable("not implemented");
                    }                    

                    return undefFlagCmpResult;
                };

                if (leftOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>() || rightOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
                {
                    // when we have undef in 1 of values we do not condition to test actual values
                    return whenBothHasNoValues(rewriter, loc);
                }

                auto andOpResult = rewriter.create<mlir::AndOp>(loc, th.getI32Type(), leftUndefFlagValue, rightUndefFlagValue);
                auto const0 = clh.createI32ConstantOf(0);
                auto bothHasResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, andOpResult, const0);

                auto result = clh.conditionalExpressionLowering(th.getBooleanType(), bothHasResult, 
                    [&](OpBuilder & builder, Location loc) 
                    {
                        auto leftSubType = leftOptType.getElementType();
                        auto rightSubType = rightOptType.getElementType();
                        left = rewriter.create<mlir_ts::ValueOp>(loc, leftSubType, left);
                        right = rewriter.create<mlir_ts::ValueOp>(loc, rightSubType, right);
                        return LogicOp_<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode, left, right, rewriter, typeConverter);
                    }, 
                    whenBothHasNoValues);

                return result;
            }
            else
            {
                // case when 1 value is optional
                auto whenOneValueIsUndef = [&](OpBuilder & builder, Location loc) 
                {
                    mlir::Value undefFlagCmpResult;
                    switch (opCmpCode)
                    {
                        case SyntaxKind::EqualsEqualsToken:
                        case SyntaxKind::EqualsEqualsEqualsToken:
                            undefFlagCmpResult = clh.createI1ConstantOf(false);
                            break;
                        case SyntaxKind::ExclamationEqualsToken:
                        case SyntaxKind::ExclamationEqualsEqualsToken:
                            undefFlagCmpResult = clh.createI1ConstantOf(true);
                            break;
                        case SyntaxKind::GreaterThanToken:
                            undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? false : true);
                            break;
                        case SyntaxKind::GreaterThanEqualsToken:
                            undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? false : true);
                            break;
                        case SyntaxKind::LessThanToken:
                            undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? true : false);
                            break;
                        case SyntaxKind::LessThanEqualsToken:
                            undefFlagCmpResult = clh.createI1ConstantOf(leftOptType ? true : false);
                            break;
                        default:
                            llvm_unreachable("not implemented");
                    }                    

                    return undefFlagCmpResult;
                };

                // when 1 of them is optional
                if (leftOptType && leftOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
                {
                    // result is false already
                    return whenOneValueIsUndef(rewriter, loc);
                }

                if (rightOptType && rightOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
                {
                    // result is false already
                    return whenOneValueIsUndef(rewriter, loc);
                }

                if (leftOptType)
                {
                    auto leftSubType = leftOptType.getElementType();
                    left = rewriter.create<mlir_ts::ValueOp>(loc, leftSubType, left);
                }

                if (rightOptType)
                {
                    auto rightSubType = rightOptType.getElementType();
                    right = rewriter.create<mlir_ts::ValueOp>(loc, rightSubType, right);
                }

                return LogicOp_<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, opCmpCode, left, right, rewriter, typeConverter);
            }
        }        
    };        

    template <typename UnaryOpTy, typename StdIOpTy, typename StdFOpTy>
    void UnaryOp(UnaryOpTy &unaryOp, PatternRewriter &builder)
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

    template <typename BinOpTy, typename StdIOpTy, typename StdFOpTy>
    void BinOp(BinOpTy &binOp, PatternRewriter &builder)
    {
        auto loc = binOp->getLoc();

        auto left = binOp->getOperand(0);
        auto right = binOp->getOperand(1);        
        auto leftType = left.getType();
        if (leftType.isIntOrIndex())
        {
            builder.replaceOpWithNewOp<StdIOpTy>(binOp, left, right);
        }
        else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
        {
            builder.replaceOpWithNewOp<StdFOpTy>(binOp, left, right);
        }
        else if (leftType.dyn_cast_or_null<mlir_ts::NumberType>())
        {
            auto castLeft = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), left);
            auto castRight = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), right);
            builder.replaceOpWithNewOp<StdFOpTy>(binOp, castLeft, castRight);
        }
        else
        {
            emitError(binOp.getLoc(), "Not implemented operator for type 1: '") << leftType << "'";
            llvm_unreachable("not implemented");
        }
    }

    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value LogicOp_(Operation *binOp, SyntaxKind op, mlir::Value left, mlir::Value right, PatternRewriter &builder, LLVMTypeConverter &typeConverter)
    {
        auto loc = binOp->getLoc();
        
        LLVMTypeConverterHelper llvmtch(typeConverter);

        auto leftType = left.getType();

        if (leftType.isIntOrIndex() || leftType.dyn_cast_or_null<mlir_ts::BooleanType>())
        {
            auto value = builder.create<StdIOpTy>(loc, v1, left, right);
            return value;
        }
        else if (!leftType.isIntOrIndex() && leftType.isIntOrIndexOrFloat())
        {
            auto value = builder.create<StdFOpTy>(loc, v2, left, right);
            return value;
        }
        else if (leftType.dyn_cast_or_null<mlir_ts::NumberType>())
        {
            auto castLeft = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), left);
            auto castRight = builder.create<mlir_ts::CastOp>(loc, builder.getF32Type(), right);
            auto value = builder.create<StdFOpTy>(loc, v2, castLeft, castRight);
            return value;
        }        
        /*
        else if (auto leftEnumType = leftType.dyn_cast_or_null<mlir_ts::EnumType>())
        {
            auto castLeft = builder.create<mlir_ts::CastOp>(loc, leftEnumType.getElementType(), left);
            auto castRight = builder.create<mlir_ts::CastOp>(loc, leftEnumType.getElementType(), right);
            auto res = builder.create<StdFOpTy>(loc, v2, castLeft, castRight);
            builder.create<mlir_ts::CastOp>(binOp, leftEnumType, res);
            return value;
        } 
        */         
        else if (leftType.dyn_cast_or_null<mlir_ts::OptionalType>())
        {
            OptionalLogicHelper olh(binOp, builder, typeConverter);
            auto value = olh.logicalOp<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, op);
            return value;
        }
        else if (leftType.dyn_cast_or_null<mlir_ts::StringType>())
        {
            if (left.getType() != right.getType()) 
            {
                right = builder.create<mlir_ts::CastOp>(loc, left.getType(), right);
            }

            auto value = builder.create<mlir_ts::StringCompareOp>(
                loc, 
                mlir_ts::BooleanType::get(builder.getContext()), 
                left, 
                right,
                builder.getI32IntegerAttr((int)op));        

            return value;        
        }
        else if (leftType.dyn_cast_or_null<mlir_ts::AnyType>())
        {
            // excluded string
            auto intPtrType = llvmtch.getIntPtrType(0);

            Value leftPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, left);
            Value rightPtrValue = builder.create<LLVM::PtrToIntOp>(loc, intPtrType, right);

            auto value = builder.create<StdIOpTy>(loc, v1, leftPtrValue, rightPtrValue);
            return value;
        }        
        else
        {
            emitError(loc, "Not implemented operator for type 1: '") << leftType << "'";
            llvm_unreachable("not implemented");
        }
    }   

    template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
    mlir::Value LogicOp(Operation *binOp, SyntaxKind op, PatternRewriter &builder, LLVMTypeConverter &typeConverter)
    {
        auto left = binOp->getOperand(0);
        auto right = binOp->getOperand(1);
        return LogicOp_<StdIOpTy, V1, v1, StdFOpTy, V2, v2>(binOp, op, left, right, builder, typeConverter);
    }
}

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
