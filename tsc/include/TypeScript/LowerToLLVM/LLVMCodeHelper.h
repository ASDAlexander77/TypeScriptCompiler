#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPER_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPER_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/CastLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelperBase.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llvm"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMCodeHelper : public LLVMCodeHelperBase
{
  public:
    LLVMCodeHelper(Operation *op, PatternRewriter &rewriter, TypeConverter *typeConverter, CompileOptions &compileOptions) 
        : LLVMCodeHelperBase(op, rewriter, typeConverter, compileOptions)
    {
    }

    std::string calc_hash_value(ArrayAttr &arrayAttr, mlir::Type llvmType, const char *prefix) const
    {
        auto opHash = 0ULL;
        opHash ^= hash_value(llvmType) + 0x9e3779b9 + (opHash << 6) + (opHash >> 2);
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

    LLVM::Linkage getLinkage(mlir::Operation *op)
    {
        //auto linkage = LLVM::Linkage::Internal;
        auto linkage = LLVM::Linkage::External;
        if (auto linkageAttr = op->getAttrOfType<StringAttr>("Linkage"))
        {
            auto val = linkageAttr.getValue();
            if (val == "External")
            {
                linkage = LLVM::Linkage::External;
            }
            else if (val == "Linkonce")
            {
                linkage = LLVM::Linkage::Linkonce;
            }
            else if (val == "LinkonceODR")
            {
                linkage = LLVM::Linkage::LinkonceODR;
            }
            else if (val == "Appending")
            {
                linkage = LLVM::Linkage::Appending;
            }
        }

        return linkage;
    }

    LLVM::GlobalOp createUndefGlobalVarIfNew(StringRef name, mlir::Type type, mlir::Attribute value, bool isConst,
                                                  LLVM::Linkage linkage = LLVM::Linkage::Internal)
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

            global = rewriter.create<LLVM::GlobalOp>(loc, type, isConst, linkage, name, value);

            {
                setStructWritingPoint(global);

                mlir::Value undefVal = rewriter.create<LLVM::UndefOp>(loc, type);
                rewriter.create<LLVM::ReturnOp>(loc, mlir::ValueRange{undefVal});
            }

            return global;
        }

        return global;
    }

    LLVM::GlobalOp createGlobalVarIfNew(StringRef name, mlir::Type type, mlir::Attribute value, bool isConst, mlir::Region &initRegion,
                                             LLVM::Linkage linkage = LLVM::Linkage::Internal)
    {
        LLVM::GlobalOp global;

        if (!type)
        {
            return global;
        }

        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);

        // Create the global at the entry of the module.
        if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            seekLast(parentModule.getBody());

            global = rewriter.create<LLVM::GlobalOp>(loc, type, isConst, linkage, name, value);

            if (!value && !initRegion.empty())
            {
                setStructWritingPoint(global);

                rewriter.inlineRegionBefore(initRegion, &global.getInitializer().back());
                rewriter.eraseBlock(&global.getInitializer().back());
            }

            return global;
        }

        return global;
    }

    mlir::func::FuncOp createFunctionFromRegion(mlir::Location location, StringRef name, mlir::Region &initRegion, StringRef saveToGlobalName)
    {
        TypeHelper th(rewriter);

        // TODO: finish it
        auto newFuncOp = rewriter.create<mlir::func::FuncOp>(
            location, name, mlir::FunctionType::get(rewriter.getContext(), llvm::ArrayRef<mlir::Type>(), llvm::ArrayRef<mlir::Type>()));
        if (!initRegion.empty())
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);

            mlir::Block *block = rewriter.createBlock(&newFuncOp.getBody());
            rewriter.setInsertionPoint(block, block->begin());

            rewriter.inlineRegionBefore(initRegion, &newFuncOp.getBody().back());

            // last inserted block
            auto lastBlock = newFuncOp.getBody().back().getPrevNode();

            LLVM_DEBUG(llvm::dbgs() << "\n!! new func last block: "; lastBlock->dump(); llvm::dbgs() << "\n";);

            LLVM_DEBUG(llvm::dbgs() << "\n!! new func terminator: " << *lastBlock->getTerminator() << "\n";);

            rewriter.setInsertionPoint(lastBlock->getTerminator());

            if (!saveToGlobalName.empty())
            {
                auto value = lastBlock->getTerminator()->getOperand(0);
                auto resultType = value.getType();

                LLVM_DEBUG(llvm::dbgs() << "\n!! value type: " << resultType << "\n";);

                auto addrToSave = rewriter.create<mlir_ts::AddressOfOp>(
                    location, mlir_ts::RefType::get(resultType), ::mlir::FlatSymbolRefAttr::get(rewriter.getContext(), saveToGlobalName),
                    ::mlir::IntegerAttr());
                rewriter.create<mlir_ts::StoreOp>(location, value, addrToSave);
            }

            rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(lastBlock->getTerminator());

            rewriter.eraseBlock(&newFuncOp.getBody().back());

            LLVM_DEBUG(llvm::dbgs() << "\n!! new func: " << newFuncOp << "\n";);
        }

        return newFuncOp;
    }

    mlir::LogicalResult createGlobalConstructorIfNew(StringRef name, mlir::Type type, LLVM::Linkage linkage,
                                                     std::function<void(LLVMCodeHelper *)> buildFunc)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);

        // Create the global at the entry of the module.
        LLVM::GlobalOp global = parentModule.lookupSymbol<LLVM::GlobalOp>(name);
        if (!global)
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            seekLast(parentModule.getBody());

            global = rewriter.create<LLVM::GlobalOp>(loc, type, true, linkage, name, mlir::Attribute());

            {
                setStructWritingPoint(global);
                buildFunc(this);
            }

            return success();
        }

        return failure();
    }

    mlir::Value getAddressOfGlobalVar(StringRef name, mlir::Type type, int32_t index = 0)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());

        // Get the pointer to the first character in the global string.
        mlir::Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, type, name);
        mlir::Value cstIdx = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, th.getIndexAttrValue(llvmIndexType, index));
        return rewriter.create<LLVM::GEPOp>(loc, globalPtr.getType(), globalPtr, ArrayRef<mlir::Value>({cstIdx}));
    }

    StringAttr getStringAttrWith0(std::string value)
    {
        return rewriter.getStringAttr(StringRef(value.data(), value.length() + 1));
    }

    mlir::Value getOrCreateGlobalArray(mlir::Type originalElementType, mlir::Type llvmElementType, unsigned size, ArrayAttr arrayAttr)
    {
        std::stringstream ss;
        ss << "a_" << size;
        auto vecVarName = calc_hash_value(arrayAttr, llvmElementType, ss.str().c_str());
        return getOrCreateGlobalArray(originalElementType, vecVarName, llvmElementType, size, arrayAttr);
    }

    mlir::Value getReadOnlyRTArray(mlir::Location loc, mlir_ts::ArrayType originalArrayType, LLVM::LLVMStructType llvmArrayType,
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
                                                                 MLIRHelper::getStructIndex(rewriter, 0));

        auto structValue3 = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayType, structValue2, sizeValue,
                                                                 MLIRHelper::getStructIndex(rewriter, 1));

        return structValue3;
    }

    mlir::Value getArrayValue(mlir::Type originalElementType, mlir::Type llvmElementType, unsigned size,
                                       ArrayAttr arrayAttr)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);

        auto arrayType = th.getArrayType(llvmElementType, size);
        mlir::Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

        // dense value
        auto value = arrayAttr.getValue();
        if (value.size() == 0/*|| originalElementType.dyn_cast<mlir_ts::AnyType>()*/)
        {
            for (auto item : arrayAttr.getValue())
            {
                // it must be '[]' empty array
                assert(false);
            }

            return arrayVal;
        }
        else if (llvmElementType.isIntOrFloat())
        {
            llvm_unreachable("it should be process in constant with denseattr value");
        }
        else if (originalElementType.dyn_cast<mlir_ts::StringType>())
        {
            auto position = 0;
            for (auto item : arrayAttr.getValue())
            {
                auto strValue = item.cast<StringAttr>().getValue().str();
                auto itemVal = getOrCreateGlobalString(strValue);

                arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }

            return arrayVal;
        }
        else if (auto originalArrayType = originalElementType.dyn_cast<mlir_ts::ArrayType>())
        {
            // TODO: implement ReadOnlyRTArray; as RTArray may contains ConstArray data (so using not editable memory)
            auto position = 0;
            for (auto item : arrayAttr.getValue())
            {
                auto arrayValue = item.cast<ArrayAttr>();
                auto itemVal = getReadOnlyRTArray(loc, originalArrayType, llvmElementType.cast<LLVM::LLVMStructType>(), arrayValue);

                arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }

            return arrayVal;
        }
        else if (originalElementType.dyn_cast<mlir_ts::ConstArrayType>())
        {
            llvm_unreachable("ConstArrayType must not be used in array, use normal ArrayType (the same way as StringType)");
        }
        else if (auto tupleType = originalElementType.dyn_cast<mlir_ts::TupleType>())
        {
            MLIRTypeHelper mth(rewriter.getContext());
            auto position = 0;
            for (auto item : arrayAttr.getValue())
            {
                auto tupleVal = getTupleFromArrayAttr(loc, mth.convertTupleTypeToConstTupleType(tupleType).cast<mlir_ts::ConstTupleType>(), llvmElementType.cast<LLVM::LLVMStructType>(),
                                                        item.dyn_cast<ArrayAttr>());
                arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, tupleVal, MLIRHelper::getStructIndex(rewriter, position++));
            }

            return arrayVal;
        }
        else if (originalElementType.dyn_cast<mlir_ts::ConstTupleType>())
        {
            llvm_unreachable("ConstTupleType must not be used in array, use normal TupleType (the same way as StringType)");
        }            

        LLVM_DEBUG(llvm::dbgs() << "type: "; originalElementType.dump(); llvm::dbgs() << "\n";);
        llvm_unreachable("array literal is not implemented(1)");
    }

    mlir::Value getOrCreateGlobalArray(mlir::Type originalElementType, StringRef name, mlir::Type llvmElementType, unsigned size,
                                       ArrayAttr arrayAttr)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());

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
            if (value.size() > 0 && llvmElementType.isIntOrIndexOrFloat())
            {
                seekLast<DenseElementsAttr>(parentModule.getBody());

                // end
                auto dataType = mlir::VectorType::get({static_cast<int64_t>(value.size())}, llvmElementType);

                DenseElementsAttr attr;
                if (llvmElementType.isIntOrIndex())
                {
                    SmallVector<APInt> values;
                    std::for_each(std::begin(value), std::end(value), [&] (auto &value_) {
                        values.push_back(value_.template cast<mlir::IntegerAttr>().getValue());
                    });

                    attr = DenseElementsAttr::get(dataType, values);
                }
                else
                {
                    SmallVector<APFloat> values;
                    std::for_each(std::begin(value), std::end(value), [&] (auto &value_) {
                        values.push_back(value_.template cast<mlir::FloatAttr>().getValue());
                    });

                    attr = DenseElementsAttr::get(dataType, values);
                }

                global = rewriter.create<LLVM::GlobalOp>(loc, /*arrayType*/dataType, true, LLVM::Linkage::Internal, name, attr);
            }
            else
            {
                seekLast(parentModule.getBody());

                OpBuilder::InsertionGuard guard(rewriter);

                global = rewriter.create<LLVM::GlobalOp>(loc, arrayType, true, LLVM::Linkage::Internal, name, mlir::Attribute{});

                setStructWritingPoint(global);

                mlir::Value arrayVal = getArrayValue(originalElementType, llvmElementType, size, arrayAttr);

                rewriter.create<LLVM::ReturnOp>(loc, ValueRange{arrayVal});
            }
        }

        // Get the pointer to the first character in the global string.
        mlir::Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        mlir::Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, th.getIndexAttrValue(llvmIndexType, 0));
        return rewriter.create<LLVM::GEPOp>(loc, pointerType, globalPtr, ArrayRef<mlir::Value>({cst0, cst0}));
    }

    mlir::LogicalResult setStructWritingPoint(LLVM::GlobalOp globalOp)
    {
        Region &region = globalOp.getInitializerRegion();
        mlir::Block *block = rewriter.createBlock(&region);

        rewriter.setInsertionPoint(block, block->begin());

        return mlir::success();
    }

    mlir::LogicalResult setStructWritingPointToStart(LLVM::GlobalOp globalOp)
    {
        rewriter.setInsertionPointToStart(&globalOp.getInitializerRegion().front());
        return mlir::success();
    }

    mlir::LogicalResult setStructWritingPointToEnd(LLVM::GlobalOp globalOp)
    {
        rewriter.setInsertionPoint(globalOp.getInitializerRegion().back().getTerminator());
        return mlir::success();
    }

    mlir::LogicalResult setStructValue(mlir::Location loc, mlir::Value &structVal, mlir::Value itemValue, unsigned index)
    {
        structVal = rewriter.create<LLVM::InsertValueOp>(loc, structVal, itemValue, MLIRHelper::getStructIndex(rewriter, index));
        return mlir::success();
    }

    mlir::Value getStructFromArrayAttr(Location loc, LLVM::LLVMStructType llvmStructType, ArrayAttr arrayAttr)
    {
        mlir::Value structVal = rewriter.create<LLVM::UndefOp>(loc, llvmStructType);

        auto typesRange = llvmStructType.getBody();

        auto position = 0;
        for (auto item : arrayAttr.getValue())
        {
            auto llvmType = typesRange[position];

            // DO NOT Replace with LLVM::ConstantOp - to use AddressOf for global symbol names
            auto itemValue = rewriter.create<mlir::arith::ConstantOp>(loc, llvmType, cast<mlir::TypedAttr>(item));
            structVal = rewriter.create<LLVM::InsertValueOp>(loc, structVal, itemValue, MLIRHelper::getStructIndex(rewriter, position++));
        }

        return structVal;
    }

    mlir::Value getTupleFromArrayAttr(Location loc, mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType,
                                      ArrayAttr arrayAttr)
    {
        mlir::Value tupleVal = rewriter.create<LLVM::UndefOp>(loc, llvmStructType);

        auto typesRange = llvmStructType.getBody();

        auto position = 0;
        for (auto item : arrayAttr.getValue())
        {
            auto type = originalType.getType(position);

            auto llvmType = typesRange[position];
            if (auto unitAttr = item.dyn_cast<UnitAttr>())
            {
                LLVM_DEBUG(llvm::dbgs() << "!! Unit Attr is type of '" << llvmType << "'\n");

                auto itemValue = rewriter.create<mlir_ts::UndefOp>(loc, llvmType);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemValue, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto stringAttr = item.dyn_cast<StringAttr>())
            {
                OpBuilder::InsertionGuard guard(rewriter);

                auto strValue = stringAttr.getValue().str();
                auto itemVal = getOrCreateGlobalString(strValue);

                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto constArrayType = type.dyn_cast<mlir_ts::ConstArrayType>())
            {
                llvm_unreachable("not used.");
                /*
                auto subArrayAttr = item.dyn_cast<ArrayAttr>();

                MLIRTypeHelper mth(rewriter.getContext());
                auto arrayType = mth.convertConstArrayTypeToArrayType(constArrayType);

                LLVM_DEBUG(llvm::dbgs() << "\n!! llvmType: " << llvmType << "\n";);

                OpBuilder::InsertionGuard guard(rewriter);

                auto itemVal =
                    getReadOnlyRTArray(loc, arrayType.cast<mlir_ts::ArrayType>(), llvmType.cast<LLVM::LLVMStructType>(), subArrayAttr);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
                */
            }
            else if (auto arrayType = type.dyn_cast<mlir_ts::ArrayType>())
            {
                auto subArrayAttr = item.dyn_cast<ArrayAttr>();

                OpBuilder::InsertionGuard guard(rewriter);

                auto itemVal = getReadOnlyRTArray(loc, arrayType, llvmType.cast<LLVM::LLVMStructType>(), subArrayAttr);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto constTupleType = type.dyn_cast<mlir_ts::ConstTupleType>())
            {
                auto subArrayAttr = item.dyn_cast<ArrayAttr>();

                OpBuilder::InsertionGuard guard(rewriter);

                auto subTupleVal = getTupleFromArrayAttr(loc, constTupleType, llvmType.cast<LLVM::LLVMStructType>(), subArrayAttr);

                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, subTupleVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto constTupleType = type.dyn_cast<mlir_ts::TupleType>())
            {
                auto subArrayAttr = item.dyn_cast<ArrayAttr>();

                OpBuilder::InsertionGuard guard(rewriter);

                ::typescript::MLIRTypeHelper mth(rewriter.getContext());

                auto subTupleVal =
                    getTupleFromArrayAttr(loc, mth.convertTupleTypeToConstTupleType(constTupleType).cast<mlir_ts::ConstTupleType>(),
                                          llvmType.cast<LLVM::LLVMStructType>(), subArrayAttr);

                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, subTupleVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto flatSymbRef = item.dyn_cast<mlir::FlatSymbolRefAttr>())
            {
                auto itemValue = rewriter.create<LLVM::AddressOfOp>(loc, llvmType, flatSymbRef);                
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemValue, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else
            {
                // DO NOT Replace with LLVM::ConstantOp - to use AddressOf for global symbol names
                auto itemValue = rewriter.create<LLVM::ConstantOp>(loc, llvmType, item);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemValue, MLIRHelper::getStructIndex(rewriter, position++));
            }
        }

        return tupleVal;
    }

    mlir::Value getOrCreateGlobalTuple(mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType, ArrayAttr arrayAttr)
    {
        auto varName = calc_hash_value(arrayAttr, llvmStructType, "tp_");
        return getOrCreateGlobalTuple(originalType, llvmStructType, varName, arrayAttr);
    }

    mlir::Value getOrCreateGlobalTuple(mlir_ts::ConstTupleType originalType, LLVM::LLVMStructType llvmStructType, StringRef name,
                                       ArrayAttr arrayAttr)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());

        auto pointerType = LLVM::LLVMPointerType::get(llvmStructType);

        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            seekLast(parentModule.getBody());

            global = rewriter.create<LLVM::GlobalOp>(loc, llvmStructType, true, LLVM::Linkage::Internal, name, mlir::Attribute{});

            setStructWritingPoint(global);

            auto tupleVal = getTupleFromArrayAttr(loc, originalType, llvmStructType, arrayAttr);
            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{tupleVal});
        }

        // Get the pointer to the first character in the global string.
        mlir::Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
        mlir::Value cst0 = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, th.getIndexAttrValue(llvmIndexType, 0));
        return rewriter.create<LLVM::GEPOp>(loc, pointerType, globalPtr, ArrayRef<mlir::Value>({cst0}));
    }

    mlir::Value GetAddressOfArrayElement(mlir::Type elementRefType, mlir::Type arrayOrStringOrTupleMlirTSType, mlir::Value arrayOrStringOrTuple, mlir::Value index)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);
        CodeLogicHelper clh(op, rewriter);

        auto loc = op->getLoc();

        assert(elementRefType.isa<mlir_ts::RefType>());

        auto ptrType = tch.convertType(elementRefType);

        auto dataPtr = arrayOrStringOrTuple;
        if (arrayOrStringOrTupleMlirTSType.isa<mlir_ts::ArrayType>())
        {
            // extract pointer from struct
            dataPtr = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, arrayOrStringOrTuple,
                                                            MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
        }

        auto addr = rewriter.create<LLVM::GEPOp>(loc, ptrType, dataPtr, ValueRange{index});
        return addr;
    }

    mlir::Value GetAddressOfStructElement(mlir::Type elementRefType, mlir::Value arrayOrStringOrTuple, int32_t index)
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
                         : isBoundRefType ? elementRefType.cast<mlir_ts::BoundRefType>().getElementType()
                         : mlir::Type();

        if (!elementType)
        {
            return mlir::Value();
        }

        auto ptrType = LLVM::LLVMPointerType::get(tch.convertType(elementType));

        SmallVector<mlir::Value> indexes;
        // add first index which 64 bit (struct field MUST BE 32 bit index)
        // auto firstIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(0));
        auto firstIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        indexes.push_back(firstIndex);
        auto fieldIndex = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(index));
        indexes.push_back(fieldIndex);

        auto addr = rewriter.create<LLVM::GEPOp>(loc, ptrType, globalPtr, indexes);

        return addr;
    }

    mlir::Value GetAddressOfPointerOffset(mlir::Type elementRefType, mlir::Value refValue, mlir::Value index)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);
        CodeLogicHelper clh(op, rewriter);

        auto loc = op->getLoc();

        auto ptrType = tch.convertType(elementRefType);

        assert(ptrType.isa<LLVM::LLVMPointerType>());

        auto dataPtr = refValue;

        auto addr = rewriter.create<LLVM::GEPOp>(loc, ptrType, dataPtr, ValueRange{index});
        return addr;
    }

};

} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPER_H_
