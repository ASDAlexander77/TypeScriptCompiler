#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/CastLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMCodeHelper : public LLVMCodeHelperBase
{
  public:
    LLVMCodeHelper(Operation *op, PatternRewriter &rewriter, TypeConverter *typeConverter) : LLVMCodeHelperBase(op, rewriter, typeConverter)
    {
    }

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

    Value getAddressOfGlobalVar(StringRef name, mlir::Type type)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);

        // Get the pointer to the first character in the global string.
        Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, type, name);
        Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, th.getIndexType(), th.getIndexAttrValue(0));
        return rewriter.create<LLVM::GEPOp>(loc, globalPtr.getType(), globalPtr, ArrayRef<Value>({cst0}));
    }

    StringAttr getStringAttrWith0(std::string value)
    {
        return rewriter.getStringAttr(StringRef(value.data(), value.length() + 1));
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
            if (auto unitAttr = item.dyn_cast_or_null<UnitAttr>())
            {
                auto itemValue = rewriter.create<mlir_ts::UndefOp>(loc, llvmType);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemValue, rewriter.getI64ArrayAttr(position++));
            }
            else if (auto stringAttr = item.dyn_cast_or_null<StringAttr>())
            {
                OpBuilder::InsertionGuard guard(rewriter);

                auto strValue = stringAttr.getValue().str();
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

        auto isRefType = elementRefType.isa<mlir_ts::RefType>();
        auto isBoundRefType = elementRefType.isa<mlir_ts::BoundRefType>();

        assert(isRefType || isBoundRefType);

        auto elementType = isRefType ? elementRefType.cast<mlir_ts::RefType>().getElementType()
                                     : elementRefType.cast<mlir_ts::BoundRefType>().getElementType();

        auto ptrType = LLVM::LLVMPointerType::get(tch.convertType(elementType));

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
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPER_H_
