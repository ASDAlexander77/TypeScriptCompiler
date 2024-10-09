#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUX_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUX_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCLinuxConst.h"
#include "TypeScript/LowerToLLVM/TypeHelper.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <sstream>

#define DEBUG_TYPE "llvm"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

class LLVMRTTIHelperVCLinux
{
    Operation *op;
    PatternRewriter &rewriter;
    ModuleOp parentModule;
    TypeHelper th;
    LLVMCodeHelper ch;
    CodeLogicHelper clh;

    bool classType;
    bool rethrow;
    SmallVector<std::string> types;

  public:
    LLVMRTTIHelperVCLinux(Operation *op, PatternRewriter &rewriter, TypeConverter &typeConverter, CompileOptions &compileOptions)
        : op(op), rewriter(rewriter), parentModule(op->getParentOfType<ModuleOp>()), th(rewriter), ch(op, rewriter, &typeConverter, compileOptions),
          clh(op, rewriter), classType(false), rethrow(false)
    {
    }

    bool isRethrow()
    {
        return rethrow;
    }

    void setF32AsCatchType()
    {
        types.push_back({linux::F32Type::typeName});
    }

    void setF64AsCatchType()
    {
        types.push_back({linux::F64Type::typeName});
    }

    void setI32AsCatchType()
    {
        types.push_back({linux::I32Type::typeName});
    }

    void setStringTypeAsCatchType()
    {
        types.push_back({linux::StringType::typeName});
    }

    void setI8PtrAsCatchType()
    {
        types.push_back({linux::I8PtrType::typeName});
    }

    void setClassTypeAsCatchType(StringRef name)
    {
        std::stringstream ss;
        ss << "_ZTIP";
        ss << name.str().size();
        ss << name.str();

        types.push_back({ss.str()});
        classType = true;
    }

    LogicalResult setPersonality(mlir::func::FuncOp newFuncOp)
    {
        auto name = "__gxx_personality_v0";
        auto cxxFrameHandler3 = ch.getOrInsertFunction(name, th.getFunctionType(th.getI32Type(), {}, true));

        newFuncOp->setAttr(ATTR("personality"), FlatSymbolRefAttr::get(rewriter.getContext(), name));
        return success();
    }

    void setType(mlir::Type type)
    {
        auto normalizedType = MLIRHelper::stripLiteralType(type);
        if (auto enumType = normalizedType.dyn_cast<mlir_ts::EnumType>())
        {
            normalizedType = enumType.getElementType();
        }

        mlir::TypeSwitch<mlir::Type>(normalizedType)
            .Case<mlir::IntegerType>([&](auto intType) {
                if (intType.getIntOrFloatBitWidth() == 32)
                {
                    setI32AsCatchType();
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Case<mlir::FloatType>([&](auto floatType) {
                auto width = floatType.getIntOrFloatBitWidth();
                if (width == 32)
                {
                    setF32AsCatchType();
                }
                else if (width == 64)
                {
                    setF64AsCatchType();
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Case<mlir_ts::NumberType>([&](auto numberType) {
#ifdef NUMBER_F64
                setF64AsCatchType();
#else
                setF32AsCatchType();
#endif
            })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) { setClassTypeAsCatchType(classType.getName().getValue()); })
            .Case<mlir_ts::AnyType>([&](auto anyType) { setI8PtrAsCatchType(); })
            .Case<mlir_ts::NullType>([&](auto nullType) { rethrow = true; })
            .Default([&](auto type) { llvm_unreachable("not implemented"); });
    }

    bool hasType()
    {
        return types.size() > 0;
    }

    mlir::Value typeInfoPtrValue(mlir::Location loc)
    {
        // TODO:
        return throwInfoPtrValue(loc);
    }

    mlir::Value throwInfoPtrValue(mlir::Location loc)
    {
        auto typeName = types.front();

        assert(typeName.size() > 0);

        LLVM_DEBUG(llvm::dbgs() << "\n Throw info name: " << typeName << "\n");

        mlir::Type tiType;
        if (classType)
        {
            SmallVector<mlir::Type> tiTypes;
            tiTypes.push_back(th.getI8PtrType());
            tiTypes.push_back(th.getI8PtrType());
            tiTypes.push_back(th.getI32Type());
            tiTypes.push_back(th.getI8PtrType());

            tiType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(), tiTypes, false);
        }
        else
        {
            tiType = th.getI8PtrType();
        }

        mlir::Value throwInfoPtr =
            rewriter.create<LLVM::AddressOfOp>(loc, th.getPointerType(tiType), FlatSymbolRefAttr::get(rewriter.getContext(), typeName));
        if (classType)
        {
            throwInfoPtr = clh.castToI8Ptr(throwInfoPtr);
        }

        return throwInfoPtr;
    }
};
} // namespace typescript

#undef DEBUG_TYPE

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUX_H_
