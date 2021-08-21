#define DEBUG_TYPE "pass"

#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir_ts = mlir::typescript;

using namespace mlir;
using namespace typescript;

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
    void runOnModule() override
    {
        auto f = getModule();

        f.walk([&](mlir::Operation *op) {
            if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op))
            {
                if (!funcOp.getBody().empty())
                {
                    return;
                }

                auto symbolAttr = funcOp->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
                if (!symbolAttr)
                {
                    return;
                }

                StringRef newName;
                auto name = std::string(symbolAttr.getValue());
                if (!mapName(symbolAttr.getValue(), newName))
                {
                    return;
                }

                funcOp->setAttr(SymbolTable::getSymbolAttrName(), mlir::StringAttr::get(op->getContext(), newName));
            }

            if (auto callOp = dyn_cast<LLVM::CallOp>(op))
            {
                StringRef newName;
                if (!mapName(callOp.callee().getValue(), newName))
                {
                    return;
                }

                callOp.calleeAttr(::mlir::FlatSymbolRefAttr::get(op->getContext(), newName));
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
};
} // end anonymous namespace

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createGCPass()
{
    return std::make_unique<GCPass>();
}
