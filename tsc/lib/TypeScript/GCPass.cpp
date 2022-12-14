#define DEBUG_TYPE "pass"

#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"

#include "TypeScript/LowerToLLVMLogic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

class ModulePass : public OperationPass<mlir::ModuleOp>
{
  public:
    using OperationPass<mlir::ModuleOp>::OperationPass;

    /// The polymorphic API that runs the pass over the currently held function.
    virtual void runOnModule() = 0;

    /// The polymorphic API that runs the pass over the currently held operation.
    void runOnOperation() final
    {
        runOnModule();
    }

    /// Return the current function being transformed.
    mlir::ModuleOp getModule()
    {
        return this->getOperation();
    }
};

class GCPass : public mlir::PassWrapper<GCPass, ModulePass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GCPass)

    void runOnModule() override
    {
        auto f = getModule();

        f.walk([&](mlir::Operation *op) {
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
                    if (name == "main")
                    {
                        injectInit(funcOp);
                    }

                    return;
                }

                renameFunction(name, funcOp);
            }

            if (auto callOp = dyn_cast_or_null<LLVM::CallOp>(op))
            {
                if (!callOp.getCallee().hasValue())
                {
                    return;
                }

                auto name = callOp.getCallee().getValue();
                if (name == "memset")
                {
                    removeRedundantMemSet(callOp);

                    return;
                }

                renameCall(name, callOp);
            }
        });
    }

    bool mapName(StringRef name, StringRef &newName)
    {
        if (name == "malloc" || name == "calloc")
        {
            newName = "GC_malloc";
        }
        else if (name == "realloc")
        {
            newName = "GC_realloc";
        }
        else if (name == "free")
        {
            newName = "GC_free";
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

    void injectInit(LLVM::LLVMFuncOp funcOp)
    {
        ConversionPatternRewriter rewriter(funcOp.getContext());
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());

        TypeHelper th(rewriter.getContext());
        LLVMCodeHelper ch(funcOp, rewriter, nullptr);
        auto i8PtrTy = th.getI8PtrType();
        auto gcInitFuncOp = ch.getOrInsertFunction("GC_init", th.getFunctionType(th.getVoidType(), mlir::ArrayRef<mlir::Type>{}));
        rewriter.create<LLVM::CallOp>(funcOp->getLoc(), gcInitFuncOp, ValueRange{});
    }

    void removeRedundantMemSet(LLVM::CallOp memSetCallOp)
    {
        // this is memset, find out if it is used by GC_malloc
        LLVM_DEBUG(llvm::dbgs() << "DBG: " << memSetCallOp.getOperand(0) << "\n";);
        if (auto probMemAllocCall = dyn_cast_or_null<LLVM::CallOp>(memSetCallOp.getOperand(0).getDefiningOp()))
        {
            if (!probMemAllocCall.getCallee().hasValue())
            {
                return;
            }

            auto name = probMemAllocCall.getCallee().getValue();
            if (name == "GC_malloc")
            {
                ConversionPatternRewriter rewriter(memSetCallOp.getContext());
                rewriter.replaceOp(memSetCallOp, ValueRange{probMemAllocCall.getResult(0)});
            }
        }
    }
};
} // end anonymous namespace

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createGCPass()
{
    return std::make_unique<GCPass>();
}
