#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCLINUX_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCLINUX_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#include "TypeScript/MLIRLogic/MLIRCodeLogic.h"
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCLinuxConst.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <sstream>
#include <functional>

using namespace ::typescript;
using namespace ts;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

struct TypeNames
{
    std::string typeName;
};

class MLIRRTTIHelperVCLinux
{
    mlir::OpBuilder &rewriter;
    mlir::ModuleOp &parentModule;
    MLIRTypeHelper mth;
    MLIRLogicHelper mlh;

    SmallVector<TypeNames> types;

  public:
    MLIRRTTIHelperVCLinux(mlir::OpBuilder &rewriter, mlir::ModuleOp &parentModule)
        : rewriter(rewriter), parentModule(parentModule), mth(rewriter.getContext()), mlh()
    {
        // setI32AsCatchType();
    }

    void setF32AsCatchType()
    {
        types.push_back({F32Type::typeName});
    }

    void setI32AsCatchType()
    {
        types.push_back({I32Type::typeName});
    }

    void setStringTypeAsCatchType()
    {
        types.push_back({StringType::typeName});
    }

    void setI8PtrAsCatchType()
    {
        types.push_back({I8PtrType::typeName});
    }

    void setClassTypeAsCatchType(ArrayRef<StringRef> names)
    {
        types.push_back({ClassType::typeName});
    }

    void setClassTypeAsCatchType(StringRef name)
    {
        types.push_back({ClassType::typeName});
    }

    void setType(mlir::Type type, std::function<ClassInfo::TypePtr(StringRef fullClassName)> resolveClassInfo)
    {
        llvm::TypeSwitch<mlir::Type>(type)
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
                if (floatType.getIntOrFloatBitWidth() == 32)
                {
                    setF32AsCatchType();
                }
                else
                {
                    llvm_unreachable("not implemented");
                }
            })
            .Case<mlir_ts::NumberType>([&](auto numberType) { setF32AsCatchType(); })
            .Case<mlir_ts::StringType>([&](auto stringType) { setStringTypeAsCatchType(); })
            .Case<mlir_ts::ClassType>([&](auto classType) {
                // we need all bases as well
                auto classInfo = resolveClassInfo(classType.getName().getValue());

                SmallVector<StringRef> classAndBases;
                classInfo->getBasesWithRoot(classAndBases);

                setClassTypeAsCatchType(classAndBases);
            })
            .Case<mlir_ts::AnyType>([&](auto anyType) { setI8PtrAsCatchType(); })
            .Default([&](auto type) { llvm_unreachable("not implemented"); });
    }

    void seekLast(mlir::Block *block)
    {
        // find last string
        auto lastUse = [&](mlir::Operation *op) {
            if (auto globalOp = dyn_cast_or_null<mlir_ts::GlobalOp>(op))
            {
                rewriter.setInsertionPointAfter(globalOp);
            }
        };

        block->walk(lastUse);
    }

    void setRTTIForType(mlir::Location loc, mlir::Type type, std::function<ClassInfo::TypePtr(StringRef fullClassName)> resolveClassInfo)
    {
        setType(type, resolveClassInfo);

        mlir::OpBuilder::InsertionGuard guard(rewriter);

        rewriter.setInsertionPointToStart(parentModule.getBody());
        seekLast(parentModule.getBody());

        // _ZTId
        for (auto type : types)
        {
            typeInfo(loc, type.typeName);
        }
    }

    mlir::LogicalResult typeInfo(mlir::Location loc, StringRef typeName)
    {
        auto name = typeName;
        if (parentModule.lookupSymbol<mlir_ts::GlobalOp>(name))
        {
            return mlir::failure();
        }

        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back({IDENT("Linkage"), ATTR("External")});

        rewriter.create<mlir_ts::GlobalOp>(loc, mth.getOpaqueType(), true, /*LLVM::Linkage::External,*/ name, mlir::Attribute{}, attrs);
        return mlir::success();
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_MLIRRTTIHELPERVCLINUX_H_
