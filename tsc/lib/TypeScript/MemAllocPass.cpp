//#define DEBUG_TYPE "pass"

#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptPassContext.h"
#include "TypeScript/ModulePass.h"

#include "TypeScript/LowerToLLVMLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

class MemAllocPass : public mlir::PassWrapper<MemAllocPass, ModulePass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MemAllocPass)

    TSContext tsContext;

    void runOnModule() override
    {
        auto m = getModule();

        m.walk([&](mlir::Operation *op) {
            if (auto funcOp = dyn_cast_or_null<LLVM::LLVMFuncOp>(op))
            {
                auto symbolAttr = funcOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
                if (!symbolAttr)
                {
                    return;
                }

                auto name = std::string(symbolAttr.getValue());
                if (!funcOp.getBody().empty())
                {
                    return;
                }

                renameFunction(name, funcOp);
            }

            if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(op))
            {
                if (!callOp.getCallee().has_value())
                {
                    return;
                }

                auto name = callOp.getCallee().value();
                renameCall(name, callOp);
            }
        });
    }

    bool mapName(StringRef name, StringRef &newName)
    {
        if (name == "ts_malloc")
        {
            newName = "malloc";
        }
        else if (name == "ts_realloc")
        {
            newName = "realloc";
        }
        else if (name == "ts_free")
        {
            newName = "free";
        }
        else
        {
            return false;
        }

        return true;
    }

    void renameFunction(StringRef name, LLVM::LLVMFuncOp funcOp)
    {
        StringRef newName;

        // this is function declaration
        if (!mapName(name, newName))
        {
            return;
        }

        funcOp->setAttr(SymbolTable::getSymbolAttrName(), mlir::StringAttr::get(funcOp->getContext(), newName));
    }

    void renameCall(StringRef name, LLVM::CallOp callOp)
    {
        StringRef newName;
        if (!mapName(name, newName))
        {
            return;
        }

        callOp.setCalleeAttr(::mlir::FlatSymbolRefAttr::get(callOp->getContext(), newName));
    }

    void injectDeclarations(ModuleOp m, mlir::MLIRContext *context)
    {
        ConversionPatternRewriter rewriter(context);

        TypeHelper th(context);
        LLVMCodeHelper ch(m, rewriter, nullptr, tsContext.compileOptions);
        auto i8PtrTy = th.getI8PtrType();
        auto sizeTy = tsContext.compileOptions.sizeBits == 64 ? th.getI64Type() : th.getI32Type();
        auto loc = mlir::UnknownLoc::get(context);
        ch.getOrInsertFunction(loc, m, "malloc", th.getFunctionType(i8PtrTy, mlir::ArrayRef<mlir::Type>{sizeTy}));
        ch.getOrInsertFunction(loc, m, "realloc", th.getFunctionType(i8PtrTy, mlir::ArrayRef<mlir::Type>{i8PtrTy, sizeTy}));
        ch.getOrInsertFunction(loc, m, "free", th.getFunctionType(th.getVoidType(), mlir::ArrayRef<mlir::Type>{i8PtrTy}));
    }
};
} // end anonymous namespace

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createMemAllocPass(CompileOptions compileOptions)
{
    auto ptr = std::make_unique<MemAllocPass>();
    ptr.get()->tsContext.compileOptions = compileOptions;
    return ptr;
}
