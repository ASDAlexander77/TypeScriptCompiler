#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_

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
    TypeHelper(PatternRewriter &rewriter) : context(rewriter.getContext())
    {
    }
    TypeHelper(MLIRContext *context) : context(context)
    {
    }

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
        return IntegerType::get(context, 1 /*, IntegerType::SignednessSemantics::Unsigned*/);
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

    LLVM::LLVMPointerType getI8PtrPtrPtrType()
    {
        return LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(LLVM::LLVMPointerType::get(getI8Type())));
    }

    LLVM::LLVMArrayType getI8Array(unsigned size)
    {
        return LLVM::LLVMArrayType::get(getI8Type(), size);
    }

    LLVM::LLVMArrayType getI32Array(unsigned size)
    {
        return LLVM::LLVMArrayType::get(getI32Type(), size);
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
};

class TypeConverterHelper
{
  public:
    TypeConverter &typeConverter;

    TypeConverterHelper(TypeConverter *typeConverter) : typeConverter(*typeConverter)
    {
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
    LLVMTypeConverterHelper(LLVMTypeConverter &typeConverter) : typeConverter(typeConverter)
    {
    }

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
    Location loc;
    PatternRewriter &rewriter;

  public:
    CodeLogicHelper(Operation *op, PatternRewriter &rewriter) : loc(op->getLoc()), rewriter(rewriter)
    {
    }

    CodeLogicHelper(Location loc, PatternRewriter &rewriter) : loc(loc), rewriter(rewriter)
    {
    }

    ArrayAttr getStructIndexAttr(int index)
    {
        return rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(index));
    }

    Value createIConstantOf(unsigned width, unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(width),
                                                 rewriter.getIntegerAttr(rewriter.getIntegerType(width), value));
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

        return rewriter.create<LLVM::ConstantOp>(loc, ftype, rewriter.getFloatAttr(ftype, value));
    }

    Value createI8ConstantOf(unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(8),
                                                 rewriter.getIntegerAttr(rewriter.getIntegerType(8), value));
    }

    Value createI32ConstantOf(unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32), rewriter.getIntegerAttr(rewriter.getI32Type(), value));
    }

    Value createI64ConstantOf(unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getIntegerAttr(rewriter.getI64Type(), value));
    }

    Value createI1ConstantOf(bool value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(1), rewriter.getIntegerAttr(rewriter.getI1Type(), value));
    }

    Value createF32ConstantOf(float value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getF32Type(), rewriter.getIntegerAttr(rewriter.getF32Type(), value));
    }

    Value createIndexConstantOf(unsigned value)
    {
        return rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(64), rewriter.getIntegerAttr(rewriter.getI64Type(), value));
    }

    Value castToI8Ptr(mlir::Value value)
    {
        TypeHelper th(rewriter);
        return rewriter.create<LLVM::BitcastOp>(loc, th.getI8PtrType(), value);
    }

    Value conditionalExpressionLowering(Type type, Value condition, function_ref<Value(OpBuilder &, Location)> thenBuilder,
                                        function_ref<Value(OpBuilder &, Location)> elseBuilder)
    {
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
        rewriter.create<LLVM::BrOp>(loc, ValueRange{}, continuationBlock);

        rewriter.setInsertionPointToEnd(thenBlock);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{thenValue}, resultBlock);

        rewriter.setInsertionPointToEnd(elseBlock);
        rewriter.create<LLVM::BrOp>(loc, ValueRange{elseValue}, resultBlock);

        // Generate assertion test.
        rewriter.setInsertionPointToEnd(opBlock);
        rewriter.create<LLVM::CondBrOp>(loc, condition, thenBlock, elseBlock);

        rewriter.setInsertionPointToStart(continuationBlock);

        return resultBlock->getArguments().front();
    }

    template <typename OpTy> void saveResult(OpTy &op, Value result)
    {
        auto defOp = op.operand1().getDefiningOp();
        // TODO: finish it for field access
        if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(defOp))
        {
            rewriter.create<mlir_ts::StoreOp>(loc, result, loadOp.reference());
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
    LLVMCodeHelper(Operation *op, PatternRewriter &rewriter, TypeConverter *typeConverter)
        : op(op), rewriter(rewriter), typeConverter(typeConverter)
    {
    }

    template <typename T> void seekLast(Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                if (globalOp.valueAttr() && globalOp.valueAttr().isa<T>())
                {
                    rewriter.setInsertionPointAfter(globalOp);
                }
            }
        };

        block->walk(lastUse);
    }

    void seekLast(Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                rewriter.setInsertionPointAfter(globalOp);
            }
        };

        block->walk(lastUse);
    }

    void seekLastWithBody(Block *block)
    {
        // find last string
        auto lastUse = [&](Operation *op) {
            if (auto globalOp = dyn_cast_or_null<LLVM::GlobalOp>(op))
            {
                if (globalOp.getInitializerBlock())
                {
                    rewriter.setInsertionPointAfter(globalOp);
                }
            }
        };

        block->walk(lastUse);
    }

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

            seekLast<StringAttr>(parentModule.getBody());

            auto type = th.getArrayType(th.getI8Type(), value.size());
            global = rewriter.create<LLVM::GlobalOp>(loc, type, true, LLVM::Linkage::Internal, name, rewriter.getStringAttr(value));
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getIndexType(), th.getIndexAttrValue(0));
        return rewriter.create<LLVM::GEPOp>(loc, th.getI8PtrType(), globalPtr, ArrayRef<Value>({cst0, cst0}));
    }

  public:
    std::string calc_hash_value(ArrayAttr &arrayAttr, const char *prefix) const
    {
        auto opHash = 0ULL;
        for (auto item : arrayAttr)
        {
            opHash ^= hash_value(item) + 0x9e3779b9 + (opHash << 6) + (opHash >> 2);
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

    mlir::LogicalResult createGlobalVarIfNew(StringRef name, mlir::Type type, mlir::Attribute value, bool isConst, mlir::Region &initRegion)
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

            seekLast(parentModule.getBody());

            global = rewriter.create<LLVM::GlobalOp>(loc, type, isConst, LLVM::Linkage::Internal, name, value);

            if (!value && !initRegion.empty())
            {
                setStructWritingPoint(global);

                rewriter.inlineRegionBefore(initRegion, &global.initializer().back());
                rewriter.eraseBlock(&global.initializer().back());
            }

            return success();
        }

        return failure();
    }

    Value getAddressOfGlobalVar(StringRef name)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);

        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (global = parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            // Get the pointer to the first character in the global string.
            Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
            Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getIndexType(), th.getIndexAttrValue(0));
            return rewriter.create<LLVM::GEPOp>(loc, globalPtr.getType(), globalPtr, ArrayRef<Value>({cst0}));
        }

        assert(false);
        return mlir::Value();
    }

    Value getOrCreateGlobalString(std::string value)
    {
        return getOrCreateGlobalString(getStorageStringName(value), value);
    }

    StringAttr getStringAttrWith0(std::string value)
    {
        return rewriter.getStringAttr(StringRef(value.data(), value.length() + 1));
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

    Value getReadOnlyRTArray(mlir::Location loc, mlir_ts::ArrayType originalArrayType, LLVM::LLVMStructType llvmArrayType,
                             ArrayAttr arrayValue)
    {
        auto llvmSubElementType = llvmArrayType.getBody()[0].cast<LLVM::LLVMPointerType>().getElementType();

        auto size = arrayValue.size();
        auto itemValArrayPtr = getOrCreateGlobalArray(originalArrayType.getElementType(), llvmSubElementType, size, arrayValue);

        // create ReadOnlyRuntimeArrayType
        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmArrayType);
        // auto arrayPtrType = LLVM::LLVMPointerType::get(llvmSubElementType);
        // auto arrayValueSize = LLVM::LLVMArrayType::get(llvmSubElementType, size);
        // auto ptrToArray = LLVM::LLVMPointerType::get(arrayValueSize);

        auto sizeValue = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getIntegerType(32),
                                                           rewriter.getIntegerAttr(rewriter.getI32Type(), arrayValue.size()));

        auto structValue2 = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, structValue, itemValArrayPtr,
                                                                 rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));

        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, structValue2, sizeValue,
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

                setStructWritingPoint(global);

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
            else if (originalElementType.dyn_cast_or_null<mlir_ts::AnyType>())
            {
                seekLast(parentModule.getBody());

                OpBuilder::InsertionGuard guard(rewriter);

                global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, Attribute{});

                setStructWritingPoint(global);

                Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

                for (auto item : arrayAttr.getValue())
                {
                    // it must be '[]' empty array
                    assert(false);
                }

                rewriter.create<LLVM::ReturnOp>(loc, ValueRange{arrayVal});
            }
            else if (auto originalArrayType = originalElementType.dyn_cast_or_null<mlir_ts::ArrayType>())
            {
                seekLast(parentModule.getBody());

                OpBuilder::InsertionGuard guard(rewriter);

                global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, Attribute{});

                setStructWritingPoint(global);

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

                setStructWritingPoint(global);

                Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

                auto position = 0;
                for (auto item : arrayAttr.getValue())
                {
                    auto tupleVal = getTupleFromArrayAttr(loc, constTupleType, llvmElementType.cast<LLVM::LLVMStructType>(),
                                                          item.dyn_cast_or_null<ArrayAttr>());
                    arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, tupleVal, rewriter.getI64ArrayAttr(position++));
                }

                rewriter.create<LLVM::ReturnOp>(loc, ValueRange{arrayVal});
            }
            else
            {
                LLVM_DEBUG(llvm::dbgs() << "type: "; originalElementType.dump(); llvm::dbgs() << "\n";);

                llvm_unreachable("array literal is not implemented(1)");
            }
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getIndexType(), th.getIndexAttrValue(0));
        return rewriter.create<LLVM::GEPOp>(loc, pointerType, globalPtr, ArrayRef<Value>({cst0, cst0}));
    }

    mlir::LogicalResult setStructWritingPoint(LLVM::GlobalOp globalOp)
    {
        Region &region = globalOp.getInitializerRegion();
        Block *block = rewriter.createBlock(&region);

        rewriter.setInsertionPoint(block, block->begin());

        return mlir::success();
    }

    mlir::LogicalResult setStructValue(mlir::Location loc, mlir::Value &structVal, mlir::Value itemValue, unsigned index)
    {
        structVal = rewriter.create<LLVM::InsertValueOp>(loc, structVal, itemValue, rewriter.getI64ArrayAttr(index));
        return mlir::success();
    }

    Value getStructFromArrayAttr(Location loc, LLVM::LLVMStructType llvmStructType, ArrayAttr arrayAttr)
    {
        Value structVal = rewriter.create<LLVM::UndefOp>(loc, llvmStructType);

        auto typesRange = llvmStructType.getBody();

        auto position = 0;
        for (auto item : arrayAttr.getValue())
        {
            auto llvmType = typesRange[position];

            // DO NOT Replace with LLVM::ConstantOp - to use AddressOf for global symbol names
            auto itemValue = rewriter.create<mlir::ConstantOp>(loc, llvmType, item);
            structVal = rewriter.create<LLVM::InsertValueOp>(loc, structVal, itemValue, rewriter.getI64ArrayAttr(position++));
        }

        return structVal;
    }

    Value getTupleFromArrayAttr(Location loc, mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType,
                                ArrayAttr arrayAttr)
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

    Value getOrCreateGlobalTuple(mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType, StringRef name,
                                 ArrayAttr arrayAttr)
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

            setStructWritingPoint(global);

            auto tupleVal = getTupleFromArrayAttr(loc, originalType, llvmStructType, arrayAttr);
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{tupleVal});
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getIndexType(), th.getIndexAttrValue(0));
        return rewriter.create<LLVM::GEPOp>(loc, pointerType, globalPtr, ArrayRef<Value>({cst0}));
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
            dataPtr = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, arrayOrStringOrTuple,
                                                            rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));
        }

        auto addr = rewriter.create<LLVM::GEPOp>(loc, ptrType, dataPtr, ValueRange{index});

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
        // auto firstIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        auto firstIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        indexes.push_back(firstIndex);
        auto fieldIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(index));
        indexes.push_back(fieldIndex);

        auto addr = rewriter.create<LLVM::GEPOp>(loc, ptrType, globalPtr, indexes);

        return addr;
    }

    template <typename T> Value _MemoryAlloc(mlir::Value sizeOfAlloc);
    Value MemoryAlloc(mlir::Value sizeOfAlloc)
    {
        return _MemoryAlloc<int>(sizeOfAlloc);
    }

    Value MemoryAlloc(mlir::Type storageType)
    {
        TypeHelper th(rewriter);

        auto loc = op->getLoc();

        auto sizeOfTypeValue = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), storageType);
        return MemoryAlloc(sizeOfTypeValue);
    }

    Value MemoryAlloc(mlir::Type res, mlir::Type storageType)
    {
        auto loc = op->getLoc();

        auto alloc = MemoryAlloc(storageType);
        auto val = rewriter.create<LLVM::BitcastOp>(loc, res, alloc);
        return val;
    }

    Value MemoryAlloc(mlir::Type res, mlir::Value sizeOfAlloc)
    {
        auto loc = op->getLoc();

        auto alloc = MemoryAlloc(sizeOfAlloc);
        auto val = rewriter.create<LLVM::BitcastOp>(loc, res, alloc);
        return val;
    }

    template <typename T> Value _MemoryRealloc(mlir::Value ptrValue, mlir::Value sizeOfAlloc);
    Value MemoryRealloc(mlir::Value ptrValue, mlir::Value sizeOfAlloc)
    {
        return _MemoryRealloc<int>(ptrValue, sizeOfAlloc);
    }

    Value MemoryRealloc(mlir::Type res, mlir::Value ptrValue, mlir::Value sizeOfAlloc)
    {
        auto loc = op->getLoc();

        auto alloc = MemoryRealloc(ptrValue, sizeOfAlloc);
        auto val = rewriter.create<LLVM::BitcastOp>(loc, res, alloc);
        return val;
    }
};

class LLVMRTTIHelperVCWin32
{
    Operation *op;
    PatternRewriter &rewriter;
    ModuleOp parentModule;
    TypeHelper th;
    LLVMCodeHelper ch;

  public:
    LLVMRTTIHelperVCWin32(Operation *op, PatternRewriter &rewriter, TypeConverter &typeConverter)
        : op(op), rewriter(rewriter), parentModule(op->getParentOfType<ModuleOp>()), th(rewriter), ch(op, rewriter, &typeConverter)
    {
    }

    LogicalResult setPersonality(mlir::FuncOp newFuncOp)
    {
        auto cxxFrameHandler3 = ch.getOrInsertFunction("__CxxFrameHandler3", th.getFunctionType(th.getI32Type(), {}, true));

        newFuncOp->setAttr(rewriter.getIdentifier("personality"), FlatSymbolRefAttr::get(rewriter.getContext(), "__CxxFrameHandler3"));
        return success();
    }

    LogicalResult typeInfo(mlir::Location loc)
    {
        auto name = "??_7type_info@@6B@";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8PtrType(), true, LLVM::Linkage::External, name, Attribute{});
        return success();
    }

    LogicalResult typeDescriptor2(mlir::Location loc)
    {
        auto name = "??_R0N@8";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        auto rttiTypeDescriptor2Ty = getRttiTypeDescriptor2Ty();
        auto _r0n_Value = rewriter.create<LLVM::GlobalOp>(loc, rttiTypeDescriptor2Ty, false, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_r0n_Value);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, rttiTypeDescriptor2Ty);

            auto itemValue1 = rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrPtrType(),
                                                                FlatSymbolRefAttr::get(rewriter.getContext(), "??_7type_info@@6B@"));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            auto itemValue2 = rewriter.create<LLVM::NullOp>(loc, th.getI8PtrType());
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir::ConstantOp>(loc, th.getI8Array(3), ch.getStringAttrWith0(".N"));
            ch.setStructValue(loc, structVal, itemValue3, 2);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_r0n_Value);
        }

        return success();
    }

    LogicalResult imageBase(mlir::Location loc)
    {
        auto name = "__ImageBase";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        rewriter.create<LLVM::GlobalOp>(loc, th.getI8Type(), true, LLVM::Linkage::External, name, Attribute{});
        return success();
    }

    LogicalResult catchableType(mlir::Location loc)
    {
        auto name = "_CT??_R0N@88";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        // _CT??_R0N@88
        auto ehCatchableTypeTy = getCatchableTypeTy();
        auto _ct_r0n_Value = rewriter.create<LLVM::GlobalOp>(loc, ehCatchableTypeTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_ct_r0n_Value);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableTypeTy);

            auto itemValue1 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(1));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            // value 2
            auto rttiTypeDescriptor2PtrValue = rewriter.create<mlir::ConstantOp>(loc, getRttiTypeDescriptor2PtrTy(),
                                                                                 FlatSymbolRefAttr::get(rewriter.getContext(), "??_R0N@8"));
            auto rttiTypeDescriptor2IntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiTypeDescriptor2PtrValue);

            auto imageBasePtrValue =
                rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), "__ImageBase"));
            auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

            // sub
            auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiTypeDescriptor2IntValue, imageBaseIntValue);

            // trunc
            auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);

            auto itemValue2 = subRes32Value;
            ch.setStructValue(loc, structVal, itemValue2, 1);

            auto itemValue3 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue3, 2);

            auto itemValue4 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(-1));
            ch.setStructValue(loc, structVal, itemValue4, 3);

            auto itemValue5 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue5, 4);

            auto itemValue6 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(8));
            ch.setStructValue(loc, structVal, itemValue6, 5);

            auto itemValue7 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(0));
            ch.setStructValue(loc, structVal, itemValue7, 6);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_ct_r0n_Value);
        }

        return success();
    }

    LogicalResult catchableArrayType(mlir::Location loc)
    {
        auto name = "_CTA1N";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        // _CT??_R0N@88
        auto ehCatchableArrayTypeTy = getCatchableArrayTypeTy();
        auto _cta1nValue =
            rewriter.create<LLVM::GlobalOp>(loc, ehCatchableArrayTypeTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        {
            ch.setStructWritingPoint(_cta1nValue);

            // begin
            Value structVal = rewriter.create<LLVM::UndefOp>(loc, ehCatchableArrayTypeTy);

            auto itemValue1 = rewriter.create<mlir::ConstantOp>(loc, th.getI32Type(), rewriter.getI32IntegerAttr(1));
            ch.setStructValue(loc, structVal, itemValue1, 0);

            // value 2
            auto rttiCatchableTypePtrValue = rewriter.create<mlir::ConstantOp>(
                loc, getCatchableTypePtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), "_CT??_R0N@88"));
            auto rttiCatchableTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableTypePtrValue);

            auto imageBasePtrValue =
                rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), "__ImageBase"));
            auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

            // sub
            auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiCatchableTypeIntValue, imageBaseIntValue);

            // trunc
            auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);

            // make array
            Value array1Val = rewriter.create<LLVM::UndefOp>(loc, th.getArrayType(th.getI32Type(), 1));
            ch.setStructValue(loc, array1Val, subRes32Value, 0);

            auto itemValue2 = array1Val;
            ch.setStructValue(loc, structVal, itemValue2, 1);

            // end
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structVal});

            rewriter.setInsertionPointAfter(_cta1nValue);
        }

        return success();
    }

    LogicalResult throwInfo(mlir::Location loc)
    {
        auto name = "_TI1N";
        if (parentModule.lookupSymbol<LLVM::GlobalOp>(name))
        {
            return failure();
        }

        auto throwInfoTy = getThrowInfoTy();
        auto _TI1NValue = rewriter.create<LLVM::GlobalOp>(loc, throwInfoTy, true, LLVM::Linkage::LinkonceODR, name, Attribute{});

        ch.setStructWritingPoint(_TI1NValue);

        Value structValue = ch.getStructFromArrayAttr(
            loc, throwInfoTy,
            rewriter.getArrayAttr({rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0)}));

        // value 3
        auto rttiCatchableArrayTypePtrValue =
            rewriter.create<mlir::ConstantOp>(loc, getCatchableArrayTypePtrTy(), FlatSymbolRefAttr::get(rewriter.getContext(), "_CTA1N"));
        auto rttiCatchableArrayTypeIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), rttiCatchableArrayTypePtrValue);

        auto imageBasePtrValue =
            rewriter.create<mlir::ConstantOp>(loc, th.getI8PtrType(), FlatSymbolRefAttr::get(rewriter.getContext(), "__ImageBase"));
        auto imageBaseIntValue = rewriter.create<LLVM::PtrToIntOp>(loc, th.getI64Type(), imageBasePtrValue);

        // sub
        auto subResValue = rewriter.create<LLVM::SubOp>(loc, th.getI64Type(), rttiCatchableArrayTypeIntValue, imageBaseIntValue);

        // trunc
        auto subRes32Value = rewriter.create<LLVM::TruncOp>(loc, th.getI32Type(), subResValue);
        ch.setStructValue(loc, structValue, subRes32Value, 3);

        rewriter.create<LLVM::ReturnOp>(loc, ValueRange{structValue});

        rewriter.setInsertionPointAfter(_TI1NValue);

        return success();
    }

    LLVM::LLVMStructType getThrowInfoTy()
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type()},
                                                false);
    }

    LLVM::LLVMPointerType getThrowInfoPtrTy()
    {
        return LLVM::LLVMPointerType::get(getThrowInfoTy());
    }

    LLVM::LLVMStructType getRttiTypeDescriptor2Ty()
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI8PtrPtrType(), th.getI8PtrType(), th.getI8Array(3)}, false);
    }

    LLVM::LLVMPointerType getRttiTypeDescriptor2PtrTy()
    {
        return LLVM::LLVMPointerType::get(getRttiTypeDescriptor2Ty());
    }

    LLVM::LLVMStructType getCatchableTypeTy()
    {
        return LLVM::LLVMStructType::getLiteral(
            rewriter.getContext(),
            {th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type(), th.getI32Type()}, false);
    }

    LLVM::LLVMPointerType getCatchableTypePtrTy()
    {
        return LLVM::LLVMPointerType::get(getCatchableTypeTy());
    }

    LLVM::LLVMStructType getCatchableArrayTypeTy()
    {
        return LLVM::LLVMStructType::getLiteral(rewriter.getContext(), {th.getI32Type(), th.getI32Array(1)}, false);
    }

    LLVM::LLVMPointerType getCatchableArrayTypePtrTy()
    {
        return LLVM::LLVMPointerType::get(getCatchableArrayTypeTy());
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
    CastLogicHelper(Operation *op, PatternRewriter &rewriter, TypeConverterHelper &tch)
        : op(op), rewriter(rewriter), tch(tch), th(rewriter), ch(op, rewriter, &tch.typeConverter), clh(op, rewriter), loc(op->getLoc())
    {
    }

    Value cast(mlir::Value in, mlir::Type resType)
    {
        auto inType = in.getType();
        auto inLLVMType = tch.convertType(inType);
        auto resLLVMType = tch.convertType(resType);
        return cast(in, inLLVMType, resType, resLLVMType);
    }

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

        // array to ref of element
        if (auto arrayType = inType.dyn_cast_or_null<mlir_ts::ArrayType>())
        {
            if (auto refType = resType.dyn_cast_or_null<mlir_ts::RefType>())
            {
                if (arrayType.getElementType() == refType.getElementType())
                {

                    return rewriter.create<LLVM::ExtractValueOp>(loc, resLLVMType, in,
                                                                 rewriter.getI32ArrayAttr(mlir::ArrayRef<int32_t>(0)));
                }
            }
        }

        emitError(loc, "invalid cast operator type 1: '") << inLLVMType << "', type 2: '" << resLLVMType << "'";
        llvm_unreachable("not implemented");

        return mlir::Value();
    }

    mlir::Value castBoolToString(mlir::Value in)
    {
        return rewriter.create<LLVM::SelectOp>(loc, in, ch.getOrCreateGlobalString("__true__", std::string("true")),
                                               ch.getOrCreateGlobalString("__false__", std::string("false")));
    }

    mlir::Value castI32ToString(mlir::Value in)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _itoaFuncOp = ch.getOrInsertFunction(
            "_itoa", th.getFunctionType(th.getI8PtrType(),
                                        ArrayRef<mlir::Type>{rewriter.getI32Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
        auto base = clh.createI32ConstantOf(10);

        return rewriter.create<LLVM::CallOp>(loc, _itoaFuncOp, ValueRange{in, newStringValue, base}).getResult(0);
    }

    mlir::Value castI64ToString(mlir::Value in)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _itoaFuncOp = ch.getOrInsertFunction(
            "_i64toa", th.getFunctionType(th.getI8PtrType(),
                                          ArrayRef<mlir::Type>{rewriter.getI32Type(), th.getI8PtrType(), rewriter.getI32Type()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
        auto base = clh.createI32ConstantOf(10);

        return rewriter.create<LLVM::CallOp>(loc, _itoaFuncOp, ValueRange{in, newStringValue, base}).getResult(0);
    }

    mlir::Value castF32orF64ToString(mlir::Value in)
    {
        auto i8PtrTy = th.getI8PtrType();

        auto _gcvtFuncOp = ch.getOrInsertFunction(
            "_gcvt", th.getFunctionType(th.getI8PtrType(),
                                        ArrayRef<mlir::Type>{rewriter.getF64Type(), rewriter.getI32Type(), th.getI8PtrType()}, true));

        auto bufferSizeValue = clh.createI32ConstantOf(50);
        auto newStringValue = rewriter.create<LLVM::AllocaOp>(loc, i8PtrTy, bufferSizeValue, true);
        auto doubleValue = rewriter.create<LLVM::FPExtOp>(loc, rewriter.getF64Type(), in);
        auto precision = clh.createI32ConstantOf(16);

        return rewriter.create<LLVM::CallOp>(loc, _gcvtFuncOp, ValueRange{doubleValue, precision, newStringValue}).getResult(0);
    }

    mlir::Value castToArrayType(mlir::Value in, mlir::Type arrayType)
    {
        mlir::Type srcElementType;
        mlir::Type llvmSrcElementType;
        auto type = in.getType();
        auto size = 0;
        if (auto constArrayType = type.dyn_cast_or_null<mlir_ts::ConstArrayType>())
        {
            size = constArrayType.getSize();
            srcElementType = constArrayType.getElementType();
            llvmSrcElementType = tch.convertType(srcElementType);

            auto dstElementType = arrayType.cast<mlir_ts::ArrayType>().getElementType();
            auto llvmDstElementType = tch.convertType(dstElementType);
            if (size > 0 && llvmDstElementType != llvmSrcElementType)
            {
                emitError(loc) << "source array and destination array have different types, src: " << srcElementType
                               << " dst: " << dstElementType;
                return mlir::Value();
            }
        }
        else if (auto ptrValue = type.dyn_cast_or_null<LLVM::LLVMPointerType>())
        {
            auto elementType = ptrValue.getElementType();
            if (auto arrayType = elementType.dyn_cast_or_null<LLVM::LLVMArrayType>())
            {
                size = arrayType.getNumElements();
                llvmSrcElementType = tch.convertType(arrayType.getElementType());
            }
            else
            {
                LLVM_DEBUG(llvm::dbgs() << "[castToArrayType(2)] from value: " << in << " type: " << in.getType()
                                        << " to type: " << arrayType << "\n";);
                llvm_unreachable("not implemented");
            }
        }
        else
        {
            LLVM_DEBUG(llvm::dbgs() << "[castToArrayType(1)] from value: " << in << " type: " << in.getType() << " to type: " << arrayType
                                    << "\n";);
            llvm_unreachable("not implemented");
        }

        auto sizeValue = clh.createI32ConstantOf(size);
        auto llvmRtArrayStructType = tch.convertType(arrayType);
        auto destArrayElement = arrayType.cast<mlir_ts::ArrayType>().getElementType();
        auto llvmDestArrayElement = tch.convertType(destArrayElement);
        auto llvmDestArray = LLVM::LLVMPointerType::get(llvmDestArrayElement);

        auto structValue = rewriter.create<LLVM::UndefOp>(loc, llvmRtArrayStructType);
        auto arrayPtrType = LLVM::LLVMPointerType::get(llvmSrcElementType);
        auto arrayValueSize = LLVM::LLVMArrayType::get(llvmSrcElementType, size);
        auto ptrToArray = LLVM::LLVMPointerType::get(arrayValueSize);

        mlir::Value arrayPtr;
        bool byValue = true;
        if (byValue)
        {
            auto bytesSize = rewriter.create<mlir_ts::SizeOfOp>(loc, th.getIndexType(), arrayValueSize);
            // TODO: create MemRef which will store information about memory. stack of heap, to use in array push to realloc
            // auto copyAllocated = rewriter.create<LLVM::AllocaOp>(loc, arrayPtrType, bytesSize);
            auto copyAllocated = ch.MemoryAlloc(arrayPtrType, bytesSize);

            auto ptrToArraySrc = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, in);
            auto ptrToArrayDst = rewriter.create<LLVM::BitcastOp>(loc, ptrToArray, copyAllocated);
            rewriter.create<mlir_ts::LoadSaveOp>(loc, ptrToArrayDst, ptrToArraySrc);
            // rewriter.create<mlir_ts::MemoryCopyOp>(loc, ptrToArrayDst, ptrToArraySrc);

            arrayPtr = rewriter.create<LLVM::BitcastOp>(loc, llvmDestArray, copyAllocated);
        }
        else
        {
            // copy ptr only (const ptr -> ptr)
            // TODO: here we need to clone body to make it writable (and remove logic from VariableOp)
            arrayPtr = rewriter.create<LLVM::BitcastOp>(loc, arrayPtrType, in);
        }

        auto structValue2 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue, arrayPtr, clh.getStructIndexAttr(0));

        auto structValue3 =
            rewriter.create<LLVM::InsertValueOp>(loc, llvmRtArrayStructType, structValue2, sizeValue, clh.getStructIndexAttr(1));

        return structValue3;
    }
};

template <typename T> Value LLVMCodeHelper::_MemoryAlloc(mlir::Value sizeOfAlloc)
{
    TypeHelper th(rewriter);
    TypeConverterHelper tch(typeConverter);

    auto loc = op->getLoc();

    auto i8PtrTy = th.getI8PtrType();
    auto mallocFuncOp = getOrInsertFunction("malloc", th.getFunctionType(i8PtrTy, {th.getIndexType()}));

    auto effectiveSize = sizeOfAlloc;
    if (effectiveSize.getType() != th.getIndexType())
    {
        CastLogicHelper castLogic(op, rewriter, tch);
        effectiveSize = castLogic.cast(effectiveSize, th.getIndexType());
    }

    auto callResults = rewriter.create<LLVM::CallOp>(loc, mallocFuncOp, ValueRange{effectiveSize});
    return callResults.getResult(0);
}

template <typename T> Value LLVMCodeHelper::_MemoryRealloc(mlir::Value ptrValue, mlir::Value sizeOfAlloc)
{
    TypeHelper th(rewriter);
    TypeConverterHelper tch(typeConverter);

    auto loc = op->getLoc();

    auto i8PtrTy = th.getI8PtrType();

    auto effectivePtrValue = ptrValue;
    if (ptrValue.getType() != i8PtrTy)
    {
        effectivePtrValue = rewriter.create<LLVM::BitcastOp>(loc, i8PtrTy, ptrValue);
    }

    auto mallocFuncOp = getOrInsertFunction("realloc", th.getFunctionType(i8PtrTy, {i8PtrTy, th.getIndexType()}));

    auto effectiveSize = sizeOfAlloc;
    if (effectiveSize.getType() != th.getIndexType())
    {
        CastLogicHelper castLogic(op, rewriter, tch);
        effectiveSize = castLogic.cast(effectiveSize, th.getIndexType());
    }

    auto callResults = rewriter.create<LLVM::CallOp>(loc, mallocFuncOp, ValueRange{effectivePtrValue, effectiveSize});
    return callResults.getResult(0);
}

template <typename StdIOpTy, typename V1, V1 v1, typename StdFOpTy, typename V2, V2 v2>
mlir::Value LogicOp_(Operation *, SyntaxKind, mlir::Value, mlir::Value, PatternRewriter &, LLVMTypeConverter &);

class OptionalLogicHelper
{
    Operation *op;
    PatternRewriter &rewriter;
    LLVMTypeConverter &typeConverter;

  public:
    OptionalLogicHelper(Operation *op, PatternRewriter &rewriter, LLVMTypeConverter &typeConverter)
        : op(op), rewriter(rewriter), typeConverter(typeConverter)
    {
    }

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

            auto whenBothHasNoValues = [&](OpBuilder &builder, Location loc) {
                mlir::Value undefFlagCmpResult;
                switch (opCmpCode)
                {
                case SyntaxKind::EqualsEqualsToken:
                case SyntaxKind::EqualsEqualsEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::ExclamationEqualsToken:
                case SyntaxKind::ExclamationEqualsEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::GreaterThanToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sgt, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::GreaterThanEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sge, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::LessThanToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                case SyntaxKind::LessThanEqualsToken:
                    undefFlagCmpResult =
                        rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::sle, leftUndefFlagValue, rightUndefFlagValue);
                    break;
                default:
                    llvm_unreachable("not implemented");
                }

                return undefFlagCmpResult;
            };

            if (leftOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>() ||
                rightOptType.getElementType().isa<mlir_ts::UndefPlaceHolderType>())
            {
                // when we have undef in 1 of values we do not condition to test actual values
                return whenBothHasNoValues(rewriter, loc);
            }

            auto andOpResult = rewriter.create<mlir::AndOp>(loc, th.getI32Type(), leftUndefFlagValue, rightUndefFlagValue);
            auto const0 = clh.createI32ConstantOf(0);
            auto bothHasResult = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne, andOpResult, const0);

            auto result = clh.conditionalExpressionLowering(
                th.getBooleanType(), bothHasResult,
                [&](OpBuilder &builder, Location loc) {
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
            auto whenOneValueIsUndef = [&](OpBuilder &builder, Location loc) {
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

template <typename UnaryOpTy, typename StdIOpTy, typename StdFOpTy> void UnaryOp(UnaryOpTy &unaryOp, PatternRewriter &builder)
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
        emitError(unaryOp.getLoc(), "Not implemented operator for type 1: '") << type << "'";
        llvm_unreachable("not implemented");
    }
}

template <typename BinOpTy, typename StdIOpTy, typename StdFOpTy> void BinOp(BinOpTy &binOp, PatternRewriter &builder)
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
    else if (leftType.template dyn_cast_or_null<mlir_ts::NumberType>())
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
mlir::Value LogicOp_(Operation *binOp, SyntaxKind op, mlir::Value left, mlir::Value right, PatternRewriter &builder,
                     LLVMTypeConverter &typeConverter)
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

        auto value = builder.create<mlir_ts::StringCompareOp>(loc, mlir_ts::BooleanType::get(builder.getContext()), left, right,
                                                              builder.getI32IntegerAttr((int)op));

        return value;
    }
    else if (leftType.dyn_cast_or_null<mlir_ts::AnyType>() || leftType.dyn_cast_or_null<mlir_ts::ClassType>())
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
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
