#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Pass/ModulePass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "pass"

using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

namespace
{

// Is there a file:line in this location, found the way DIScopeForLLVMFuncOpPass looks for one?
static bool hasFileLoc(mlir::Location loc)
{
    if (isa<mlir::FileLineColLoc>(loc))
    {
        return true;
    }

    if (auto nameLoc = dyn_cast<mlir::NameLoc>(loc))
    {
        return hasFileLoc(nameLoc.getChildLoc());
    }

    if (auto opaqueLoc = dyn_cast<mlir::OpaqueLoc>(loc))
    {
        return hasFileLoc(opaqueLoc.getFallbackLocation());
    }

    if (auto fusedLoc = dyn_cast<mlir::FusedLoc>(loc))
    {
        return llvm::any_of(fusedLoc.getLocations(), hasFileLoc);
    }

    if (auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc))
    {
        return hasFileLoc(callSiteLoc.getCaller());
    }

    return false;
}

// An inlined op whose own location has no file:line stands for the call it was inlined from.
static mlir::Location repair(mlir::Location loc)
{
    auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc);
    if (!callSiteLoc)
    {
        return loc;
    }

    auto callee = repair(callSiteLoc.getCallee());
    if (!hasFileLoc(callee))
    {
        return callSiteLoc.getCaller();
    }

    return callee == callSiteLoc.getCallee() ? loc : mlir::CallSiteLoc::get(callee, callSiteLoc.getCaller());
}

// Outside a function there is no inlining to describe: the location the op was written at.
static mlir::Location written(mlir::Location loc)
{
    while (auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc))
    {
        loc = callSiteLoc.getCallee();
    }

    return loc;
}

// Under `--di --opt`, ahead of DIScopeForLLVMFuncOpPass. The inliner gives every op it moves
// `callsite(<its location> at <the call>)`, and DIScopeForLLVMFuncOpPass takes two things for
// granted about an op with such a location, both of which fail here and crashed tslang for most
// programs:
// - that its callee has a file name: an op made without a location of its own - there are some -
//   becomes `callsite(unknown at <the call>)`;
// - that it is inside a function: the lowering makes a string literal's global with the location
//   of the op that used it, which is a call site when that op was inlined.
class InlinedLocationPass : public mlir::PassWrapper<InlinedLocationPass, ModulePass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InlinedLocationPass)

    void runOnModule() override
    {
        getModule().walk([](mlir::Operation *op) {
            auto loc = op->getLoc();
            auto repaired = isa<mlir::LLVM::LLVMFuncOp>(op) || op->getParentOfType<mlir::LLVM::LLVMFuncOp>()
                                ? repair(loc)
                                : written(loc);
            if (repaired != loc)
            {
                op->setLoc(repaired);
            }
        });
    }
};
} // end anonymous namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createInlinedLocationPass()
{
    return std::make_unique<InlinedLocationPass>();
}
