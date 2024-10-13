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

#define DEBUG_TYPE "pass"

namespace mlir_ts = mlir::typescript;

namespace
{

class RelocateConstantPass : public mlir::PassWrapper<RelocateConstantPass, TypeScriptFunctionPass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelocateConstantPass)

    void runOnFunction() override
    {
        // TODO: get rid of using it
        auto f = getFunction();

        SmallPtrSet<Operation *, 16> workSetConst;
        getOps<mlir_ts::ConstantOp>(f, workSetConst);
        /*auto lastConstOp =*/relocateConst(f, workSetConst);

        LLVM_DEBUG(llvm::dbgs() << "\n!! AFTER CONST RELOC FUNC DUMP: \n" << *getFunction() << "\n";);
    }

    Operation *seekFirstNonConstOp(mlir_ts::FuncOp &f)
    {
        return seekFirstNonOp(f, [](Operation *op) {
            if (auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(op))
                return true;
            if (auto constOp = dyn_cast_or_null<mlir::arith::ConstantOp>(op))
                return true;
            return false;
        });
    }

    Operation *relocateConst(mlir_ts::FuncOp &f, SmallPtrSet<Operation *, 16> &workSet)
    {
        // find first non-constant op
        auto firstNonConstOp = seekFirstNonConstOp(f);
        if (firstNonConstOp)
        {
            ConversionPatternRewriter rewriter(f.getContext());
            rewriter.setInsertionPoint(firstNonConstOp);

            LLVM_DEBUG(llvm::dbgs() << "\nInsert const at: \n" << *firstNonConstOp << "\n";);

            for (auto op : workSet)
            {
                auto constantOp = cast<mlir_ts::ConstantOp>(op);

                LLVM_DEBUG(llvm::dbgs() << "\nconst to insert: \n" << constantOp << "\n";);

                auto newOp = rewriter.create<mlir_ts::ConstantOp>(constantOp->getLoc(), constantOp.getType(), constantOp.getValue());
                constantOp->replaceAllUsesWith(newOp);

                constantOp->erase();
            }
        }

        return firstNonConstOp;
    }

    template <typename T> void getOps(mlir_ts::FuncOp &f, SmallPtrSet<Operation *, 16> &workSet, Operation *startFrom = nullptr)
    {
        auto startFromStage = startFrom != nullptr;
        auto skipFirstOps = true;
        f.walk([&](mlir::Operation *op) {
            if (startFrom != nullptr)
            {
                if (startFromStage)
                {
                    if (op != startFrom)
                    {
                        return;
                    }

                    startFromStage = false;
                }
            }

            if (auto typedOp = dyn_cast_or_null<T>(op))
            {
                if (!skipFirstOps)
                {
                    // if op is not child of function but nested block - ignore it
                    if (typedOp->getParentOp() == f)
                    {
                        // select only those consts which are not at the beginning
                        workSet.insert(typedOp);
                    }
                }
            }
            else
            {
                skipFirstOps = false;
            }
        });
    }

    Operation *seekFirstNonOp(mlir_ts::FuncOp funcOp, std::function<bool(Operation *)> filter)
    {
        auto allowSkipConsts = true;
        auto found = false;
        Operation *foundOp = nullptr;
        // find last string
        auto lastUse = [&](Operation *op) {
            if (found)
            {
                return;
            }

            // we need only first level
            if (op->getParentOp() != funcOp)
            {
                // it is not top anymore
                found = true;
                return;
            }

            if (filter(op))
            {
                if (allowSkipConsts)
                {
                    return;
                }
            }

            if (allowSkipConsts)
            {
                allowSkipConsts = false;
                found = true;
                foundOp = op;
                return;
            }
        };

        funcOp.walk(lastUse);

        return foundOp;
    }
};
} // end anonymous namespace

#undef DEBUG_TYPE

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createRelocateConstantPass()
{
    return std::make_unique<RelocateConstantPass>();
}
