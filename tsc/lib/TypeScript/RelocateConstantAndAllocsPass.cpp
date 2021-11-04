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

namespace mlir_ts = mlir::typescript;

namespace
{

class RelocateConstantAndAllocsPass : public mlir::PassWrapper<RelocateConstantAndAllocsPass, TypeScriptFunctionPass>
{
  public:
    void runOnFunction() override
    {
        relocateConst();
    }

    void relocateConst()
    {
        auto f = getFunction();

        SmallPtrSet<Operation *, 16> workSet;

        auto skipFirstConsts = true;
        f.walk([&](mlir::Operation *op) {
            if (auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(op))
            {
                if (!skipFirstConsts)
                {
                    // if const is not child of function but nested block - ignore it
                    if (constantOp->getParentOp() == f)
                    {
                        // select only those consts which are not at the beginning
                        workSet.insert(constantOp);
                    }
                }
            }
            else
            {
                skipFirstConsts = false;
            }
        });

        // find fist non-constant op
        auto firstNonConstOp = seekFirstNonConstantOp(f);
        if (firstNonConstOp)
        {
            ConversionPatternRewriter rewriter(f.getContext());
            rewriter.setInsertionPoint(firstNonConstOp);

            LLVM_DEBUG(llvm::dbgs() << "\nInsert const at: \n" << *firstNonConstOp << "\n";);

            for (auto op : workSet)
            {
                auto constantOp = cast<mlir_ts::ConstantOp>(op);

                LLVM_DEBUG(llvm::dbgs() << "\nconst to insert: \n" << constantOp << "\n";);

                auto newOp = rewriter.create<mlir_ts::ConstantOp>(constantOp->getLoc(), constantOp.getType(), constantOp.value());
                constantOp->replaceAllUsesWith(newOp);

                rewriter.eraseOp(constantOp);
            }
        }
    }

    void relocateAllocs()
    {
        auto f = getFunction();

        SmallPtrSet<Operation *, 16> workSet;

        auto skipFirstConstsOrVariables = true;
        f.walk([&](mlir::Operation *op) {
            if (auto variableOp = dyn_cast_or_null<mlir_ts::VariableOp>(op))
            {
                if (!skipFirstConstsOrVariables)
                {
                    // if const is not child of function but nested block - ignore it
                    if (variableOp->getParentOp() == f)
                    {
                        // select only those consts which are not at the beginning
                        workSet.insert(variableOp);
                    }
                }
            }
            else
            {
                skipFirstConstsOrVariables = false;
            }
        });

        // find fist non-constant op
        auto firstNonConstAndNonAllocOp = seekFirstNonConstantAndNonAllocOp(f);
        if (firstNonConstAndNonAllocOp)
        {
            ConversionPatternRewriter rewriter(f.getContext());
            rewriter.setInsertionPoint(firstNonConstAndNonAllocOp);

            LLVM_DEBUG(llvm::dbgs() << "\nInsert variable at: \n" << *firstNonConstAndNonAllocOp << "\n";);

            for (auto op : workSet)
            {
                auto varOp = cast<mlir_ts::VariableOp>(op);

                LLVM_DEBUG(llvm::dbgs() << "\nvariable to insert: \n" << varOp << "\n";);

                auto newVar =
                    rewriter.create<mlir_ts::VariableOp>(varOp->getLoc(), varOp.getType(), varOp.initializer(), varOp.capturedAttr());
                varOp->replaceAllUsesWith(newVar);

                rewriter.eraseOp(varOp);
            }
        }
    }

    Operation *seekFirstNonConstantOp(mlir_ts::FuncOp funcOp)
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

            auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(op);
            if (constantOp)
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

    Operation *seekFirstNonConstantAndNonAllocOp(mlir_ts::FuncOp funcOp)
    {
        auto allowSkipConstsAndAllocs = true;
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

            auto constantOp = dyn_cast_or_null<mlir_ts::ConstantOp>(op);
            if (constantOp)
            {
                if (allowSkipConstsAndAllocs)
                {
                    return;
                }
            }
            else if (auto varOp = dyn_cast_or_null<mlir_ts::VariableOp>(op))
            {
                if (allowSkipConstsAndAllocs)
                {
                    return;
                }
            }

            if (allowSkipConstsAndAllocs)
            {
                allowSkipConstsAndAllocs = false;
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

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createRelocateConstantAndAllocsPass()
{
    return std::make_unique<RelocateConstantAndAllocsPass>();
}
