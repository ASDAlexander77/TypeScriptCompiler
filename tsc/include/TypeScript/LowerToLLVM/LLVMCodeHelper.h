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
    LLVMCodeHelper(Operation *op, PatternRewriter &rewriter, const TypeConverter *typeConverter, CompileOptions &compileOptions) 
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
                                             LLVM::Linkage linkage = LLVM::Linkage::Internal, StringRef section = "")
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

            auto effectiveValue = value;
            if (effectiveValue && linkage == LLVM::Linkage::Appending && !isa<mlir::ArrayAttr>(effectiveValue))
            {
                effectiveValue = rewriter.getArrayAttr({effectiveValue});
            }

            global = rewriter.create<LLVM::GlobalOp>(loc, type, isConst, linkage, name, effectiveValue);
            if (section.size() > 0)
            {
                global.setSection(section);
            }

            if (!effectiveValue && !initRegion.empty())
            {
                setStructWritingPoint(global);

                rewriter.inlineRegionBefore(initRegion, &global.getInitializer().back());
                rewriter.eraseBlock(&global.getInitializer().back());
            }

            return global;
        } else if (linkage == LLVM::Linkage::Appending && global.getLinkageAttr().getLinkage() == linkage) {
            SmallVector<mlir::Attribute> values(cast<mlir::ArrayAttr>(global.getValueAttr()).getValue());
            values.push_back(value);
            
            auto newArrayAttr = rewriter.getArrayAttr(values);
            
            global.setValueAttr(newArrayAttr);

            auto arrayType = cast<LLVM::LLVMArrayType>(global.getGlobalType());
            global.setGlobalType(LLVM::LLVMArrayType::get(arrayType.getElementType(), arrayType.getNumElements() + 1));
        }

        return global;
    }

    LLVM::GlobalOp createGlobalVarRegionWithAppendingSymbolRef(StringRef name, FlatSymbolRefAttr nameValue, StringRef section = "")
    {
        LLVM::GlobalOp global;

        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);
        auto ptrType = th.getPtrType();

        // Create the global at the entry of the module.
        if (!(global = parentModule.lookupSymbol<LLVM::GlobalOp>(name)))
        {
            OpBuilder::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(parentModule.getBody());

            seekLast(parentModule.getBody());

            TypeHelper th(rewriter);
            auto arrayPtrType = LLVM::LLVMArrayType::get(ptrType, 1);

            global = rewriter.create<LLVM::GlobalOp>(loc, arrayPtrType, false, LLVM::Linkage::Appending, name, mlir::Attribute{});
            if (section.size() > 0)
            {
                global.setSection(section);
            }

            setStructWritingPoint(global);

            auto arrayType = th.getArrayType(ptrType, 1);
            mlir::Value arrayVal = rewriter.create<LLVM::UndefOp>(loc, arrayType);

            auto addrValue = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, nameValue);
            auto globalNameAddr = rewriter.create<LLVM::GEPOp>(loc, ptrType, ptrType, addrValue, ArrayRef<LLVM::GEPArg>{0});

            arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, globalNameAddr, MLIRHelper::getStructIndex(rewriter, 0));

            rewriter.create<LLVM::ReturnOp>(loc, ValueRange{arrayVal});

            return global;
        } else if (global.getLinkageAttr().getLinkage() == LLVM::Linkage::Appending) {
            auto arrayType = cast<LLVM::LLVMArrayType>(global.getGlobalType());

            auto newIndex = arrayType.getNumElements();
            auto newCount = newIndex + 1;
            auto newArrayType = LLVM::LLVMArrayType::get(arrayType.getElementType(), newCount);

            global.setGlobalType(newArrayType);

            global.getBodyRegion().walk(
                [&](mlir::Operation *op) {
                    if (auto undefOp = dyn_cast_or_null<LLVM::UndefOp>(op))
                    {
                        undefOp.getResult().setType(newArrayType);
                    }
                    else if (auto insertValueOp = dyn_cast_or_null<LLVM::InsertValueOp>(op))
                    {
                        insertValueOp.getResult().setType(newArrayType);
                    }
                });

            auto returnOp = global.getBodyRegion().back().getTerminator();
            rewriter.setInsertionPoint(returnOp);

            auto returnOpTyped = cast<LLVM::ReturnOp>(returnOp);
            auto arrayVal = returnOpTyped.getArg();

            auto addrValue = rewriter.create<LLVM::AddressOfOp>(loc, ptrType, nameValue);
            auto globalNameAddr = rewriter.create<LLVM::GEPOp>(loc, ptrType, ptrType, addrValue, ArrayRef<LLVM::GEPArg>{0});

            arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, globalNameAddr, MLIRHelper::getStructIndex(rewriter, newIndex));

            returnOpTyped.getArgMutable().assign({arrayVal});
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

        // Get the pointer to the first character in the global string.
        mlir::Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, type, name);
        return rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), type, globalPtr, ArrayRef<LLVM::GEPArg>{index});
    }

    StringAttr getStringAttrWith0(std::string value)
    {
        return rewriter.getStringAttr(StringRef(value.data(), value.length() + 1));
    }

    mlir::Value getOrCreateGlobalArray(mlir::Type originalElementType, unsigned size, ArrayAttr arrayAttr)
    {
        std::stringstream ss;
        ss << "a_" << size;
        auto vecVarName = calc_hash_value(arrayAttr, originalElementType, ss.str().c_str());
        return getOrCreateGlobalArray(originalElementType, vecVarName, size, arrayAttr);
    }

    mlir::Value getReadOnlyRTArray(mlir::Location loc, mlir_ts::ArrayType originalArrayType, LLVM::LLVMStructType llvmArrayType,
                                   ArrayAttr arrayValue)
    {
        auto size = arrayValue.size();
        auto itemValArrayPtr = getOrCreateGlobalArray(originalArrayType.getElementType(), size, arrayValue);

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
        if (value.size() == 0/*|| dyn_cast<mlir_ts::AnyType>(originalElementType)*/)
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
        else if (isa<mlir_ts::StringType>(originalElementType))
        {
            auto position = 0;
            for (auto item : arrayAttr.getValue())
            {
                auto strValue = mlir::cast<StringAttr>(item).getValue().str();
                auto itemVal = getOrCreateGlobalString(strValue);

                arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }

            return arrayVal;
        }
        else if (auto originalArrayType = dyn_cast<mlir_ts::ArrayType>(originalElementType))
        {
            // TODO: implement ReadOnlyRTArray; as RTArray may contains ConstArray data (so using not editable memory)
            auto position = 0;
            for (auto item : arrayAttr.getValue())
            {
                auto arrayValue = mlir::cast<ArrayAttr>(item);
                auto itemVal = getReadOnlyRTArray(loc, originalArrayType, mlir::cast<LLVM::LLVMStructType>(llvmElementType), arrayValue);

                arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }

            return arrayVal;
        }
        else if (isa<mlir_ts::ConstArrayType>(originalElementType))
        {
            llvm_unreachable("ConstArrayType must not be used in array, use normal ArrayType (the same way as StringType)");
        }
        else if (auto tupleType = dyn_cast<mlir_ts::TupleType>(originalElementType))
        {
            MLIRTypeHelper mth(rewriter.getContext());
            auto position = 0;
            for (auto item : arrayAttr.getValue())
            {
                auto tupleVal = getTupleFromArrayAttr(loc, cast<mlir_ts::ConstTupleType>(mth.convertTupleTypeToConstTupleType(tupleType)), mlir::cast<LLVM::LLVMStructType>(llvmElementType),
                                                        dyn_cast<ArrayAttr>(item));
                arrayVal = rewriter.create<LLVM::InsertValueOp>(loc, arrayVal, tupleVal, MLIRHelper::getStructIndex(rewriter, position++));
            }

            return arrayVal;
        }
        else if (isa<mlir_ts::ConstTupleType>(originalElementType))
        {
            llvm_unreachable("ConstTupleType must not be used in array, use normal TupleType (the same way as StringType)");
        }            

        LLVM_DEBUG(llvm::dbgs() << "type: "; originalElementType.dump(); llvm::dbgs() << "\n";);
        llvm_unreachable("array literal is not implemented(1)");
    }

    mlir::Value getOrCreateGlobalArray(mlir::Type originalElementType, StringRef name, unsigned size,
                                       ArrayAttr arrayAttr)
    {
        auto loc = op->getLoc();
        auto parentModule = op->getParentOfType<ModuleOp>();

        TypeHelper th(rewriter);
        auto llvmIndexType = typeConverter->convertType(th.getIndexType());
        auto llvmElementType = typeConverter->convertType(originalElementType);

        auto ptrType = th.getPtrType();
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
                        values.push_back(cast<mlir::IntegerAttr>(value_).getValue());
                    });

                    attr = DenseElementsAttr::get(dataType, values);
                }
                else
                {
                    SmallVector<APFloat> values;
                    std::for_each(std::begin(value), std::end(value), [&] (auto &value_) {
                        values.push_back(cast<mlir::FloatAttr>(value_).getValue());
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
        return rewriter.create<LLVM::GEPOp>(loc, ptrType, global.getType(), globalPtr, ArrayRef<LLVM::GEPArg>{0, 0});
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
            if (auto unitAttr = dyn_cast<UnitAttr>(item))
            {
                LLVM_DEBUG(llvm::dbgs() << "!! Unit Attr is type of '" << llvmType << "'\n");

                auto itemValue = rewriter.create<mlir_ts::UndefOp>(loc, llvmType);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemValue, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto stringAttr = dyn_cast<StringAttr>(item))
            {
                OpBuilder::InsertionGuard guard(rewriter);

                auto strValue = stringAttr.getValue().str();
                auto itemVal = getOrCreateGlobalString(strValue);

                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto constArrayType = dyn_cast<mlir_ts::ConstArrayType>(type))
            {
                llvm_unreachable("not used.");
                /*
                auto subArrayAttr = dyn_cast<ArrayAttr>(item);

                MLIRTypeHelper mth(rewriter.getContext());
                auto arrayType = mth.convertConstArrayTypeToArrayType(constArrayType);

                LLVM_DEBUG(llvm::dbgs() << "\n!! llvmType: " << llvmType << "\n";);

                OpBuilder::InsertionGuard guard(rewriter);

                auto itemVal =
                    getReadOnlyRTArray(loc, mlir::cast<mlir_ts::ArrayType>(arrayType), mlir::cast<LLVM::LLVMStructType>(llvmType), subArrayAttr);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
                */
            }
            else if (auto arrayType = dyn_cast<mlir_ts::ArrayType>(type))
            {
                auto subArrayAttr = dyn_cast<ArrayAttr>(item);

                OpBuilder::InsertionGuard guard(rewriter);

                auto itemVal = getReadOnlyRTArray(loc, arrayType, mlir::cast<LLVM::LLVMStructType>(llvmType), subArrayAttr);
                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, itemVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                auto subArrayAttr = dyn_cast<ArrayAttr>(item);

                OpBuilder::InsertionGuard guard(rewriter);

                auto subTupleVal = getTupleFromArrayAttr(loc, constTupleType, mlir::cast<LLVM::LLVMStructType>(llvmType), subArrayAttr);

                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, subTupleVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto constTupleType = dyn_cast<mlir_ts::TupleType>(type))
            {
                auto subArrayAttr = dyn_cast<ArrayAttr>(item);

                OpBuilder::InsertionGuard guard(rewriter);

                ::typescript::MLIRTypeHelper mth(rewriter.getContext());

                auto subTupleVal =
                    getTupleFromArrayAttr(loc, cast<mlir_ts::ConstTupleType>(mth.convertTupleTypeToConstTupleType(constTupleType)),
                                          mlir::cast<LLVM::LLVMStructType>(llvmType), subArrayAttr);

                tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleVal, subTupleVal, MLIRHelper::getStructIndex(rewriter, position++));
            }
            else if (auto flatSymbRef = dyn_cast<mlir::FlatSymbolRefAttr>(item))
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

        auto pointerType = th.getPtrType();

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
        return rewriter.create<LLVM::GEPOp>(loc, pointerType, global.getType(), globalPtr, ArrayRef<mlir::Value>({cst0}));
    }

    mlir::Value GetAddressOfArrayElement(mlir::Type elementType, mlir::Type arrayOrStringOrTupleMlirTSType, mlir::Value arrayOrStringOrTuple, mlir::Value index)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);
        CodeLogicHelper clh(op, rewriter);

        auto loc = op->getLoc();

        auto ptrType = th.getPtrType();
        auto llvmElementType = tch.convertType(elementType);

        auto dataPtr = arrayOrStringOrTuple;
        if (isa<mlir_ts::ArrayType>(arrayOrStringOrTupleMlirTSType))
        {
            // extract pointer from struct
            dataPtr = rewriter.create<LLVM::ExtractValueOp>(loc, ptrType, arrayOrStringOrTuple,
                                                            MLIRHelper::getStructIndex(rewriter, ARRAY_DATA_INDEX));
        }

        auto addr = rewriter.create<LLVM::GEPOp>(loc, ptrType, llvmElementType, dataPtr, ArrayRef<LLVM::GEPArg>{index});
        return addr;
    }

    mlir::Value GetAddressOfStructElement(mlir::Type objectRefType, mlir::Value arrayOrStringOrTuple, int32_t index)
    {
        // index of struct MUST BE 32 bit
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto loc = op->getLoc();
        auto globalPtr = arrayOrStringOrTuple;

        auto elementType = MLIRHelper::getElementTypeOrSelf(objectRefType);
        if (!elementType)
        {
            return mlir::Value();
        }

        auto llvmElementType = tch.convertType(elementType);

        LLVM_DEBUG(llvm::dbgs() << "\n!! GetAddressOfStructElement: index #" << index << " type - " << elementType << " llvm: " << llvmElementType << "\n";);

        auto addr = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, globalPtr, ArrayRef<LLVM::GEPArg>{0, index});
        return addr;
    }

    mlir::Value GetAddressOfPointerOffset(mlir::Type elementType, mlir::Value refValue, mlir::Value index)
    {
        TypeHelper th(rewriter);
        TypeConverterHelper tch(typeConverter);

        auto llvmElementType = tch.convertType(elementType);

        LLVM_DEBUG(llvm::dbgs() << "\n!! GetAddressOfPointerOffset: index #" << index << " type - " << elementType << " llvm: " << llvmElementType << "\n";);

        auto loc = op->getLoc();
        auto addr = rewriter.create<LLVM::GEPOp>(loc, th.getPtrType(), llvmElementType, refValue, ValueRange{index});
        return addr;
    }
};

} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMCODEHELPER_H_
