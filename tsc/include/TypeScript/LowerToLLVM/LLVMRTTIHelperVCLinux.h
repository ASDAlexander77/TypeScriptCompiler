#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUX_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUX_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include <sstream>

using namespace mlir;
namespace mlir_ts = mlir::typescript;

namespace typescript
{

struct TypeNames
{
    std::string typeName;
    std::string typeInfoRef;
    std::string catchableTypeInfoRef;
};

class LLVMRTTIHelperVCLinux
{
    Operation *op;
    PatternRewriter &rewriter;
    ModuleOp parentModule;
    TypeHelper th;
    LLVMCodeHelper ch;

    SmallVector<TypeNames> types;

  public:
    std::string catchableTypeInfoArrayRef;
    std::string throwInfoRef;

    LLVMRTTIHelperVCLinux(Operation *op, PatternRewriter &rewriter, TypeConverter &typeConverter)
        : op(op), rewriter(rewriter), parentModule(op->getParentOfType<ModuleOp>()), th(rewriter), ch(op, rewriter, &typeConverter)
    {
    }

    LogicalResult setPersonality(mlir::FuncOp newFuncOp)
    {
        auto name = "__gxx_personality_v0";
        auto cxxFrameHandler3 = ch.getOrInsertFunction(name, th.getFunctionType(th.getI32Type(), {}, true));

        newFuncOp->setAttr(rewriter.getIdentifier("personality"), FlatSymbolRefAttr::get(rewriter.getContext(), name));
        return success();
    }
};
} // namespace typescript

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_LLVMRTTIHELPERVCLINUX_H_
