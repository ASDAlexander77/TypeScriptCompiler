#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "own"

namespace mlir_ts = mlir::typescript;

namespace
{

// Ownership inference for -mm=own, phase 0. See
// docs/superpowers/specs/2026-09-24-own-memory-model-design.md, section 10.6.
//
// rc's count for a block is its births plus its retains. Where that provably never exceeds one,
// the block has one owner, the release rc already emits is that owner's destroy, and every
// retain on the way is erased. Anything else is reported: phase 0 knows no other verdict.
class OwnershipInferencePass : public mlir::PassWrapper<OwnershipInferencePass, TypeScriptFunctionPass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipInferencePass)

    void runOnFunction() override
    {
        auto f = getFunction();

        // value -> the retain ops that acquire it (a Retain of the value, a RetainSlot of the
        // variable it initializes)
        llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *, 2>> acquirers;
        llvm::SmallVector<mlir::Operation *> unattributed;

        f.walk([&](mlir::Operation *op) {
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
            {
                acquirers[retainOp.getReference()].push_back(op);
            }
            else if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                // A variable with no initializer had its storage hoisted in front of a try, and
                // its value arrives by a store this phase does not follow.
                auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>();
                if (varOp && varOp.getInitializer())
                {
                    acquirers[varOp.getInitializer()].push_back(op);
                }
                else
                {
                    unattributed.push_back(op);
                }
            }
            else if (mlir::isa<mlir_ts::RetainCellOp>(op))
            {
                op->emitError("closures that capture a heap value are not supported by -mm=own yet");
                signalPassFailure();
            }
        });

        for (auto *op : unattributed)
        {
            reportSecondReference(op);
        }

        llvm::SmallVector<mlir::Operation *> toErase;
        for (auto &[value, retains] : acquirers)
        {
            auto total = birth(value) + retains.size();
            if (isFresh(value) && total == 1)
            {
                toErase.append(retains.begin(), retains.end());
                continue;
            }

            for (auto *op : retains)
            {
                reportSecondReference(op);
            }
        }

        for (auto *op : toErase)
        {
            op->erase();
        }
    }

  private:
    static bool isFresh(mlir::Value value)
    {
        auto *def = value.getDefiningOp();
        if (!def)
        {
            return false; // a block argument: a parameter, or a value merged from branches
        }

        return def->hasAttr(OWNED_RESULT_ATTR_NAME) || isLiteral(def) ||
               mlir::isa<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::StringConcatOp,
                         mlir_ts::CharToStringOp>(def);
    }

    // A string literal or a constant array cast to its value type. The result is either the
    // immortal global itself, which a release skips, or a copy nobody else holds, which a
    // release destroys - so, acquired once, it has one owner either way. Acquired twice it is
    // still an error: which of the two it is depends on the cast.
    static bool isLiteral(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(def);
        return castOp && castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>();
    }

    // 1 when the value arrived with a reference the callee took (the retaining return, rc
    // §9.24); 0 otherwise - including a string marked by markFreshStringOwned, whose birth is
    // the retain MLIRGen emitted beside the mark and is counted as one of its acquisitions.
    static unsigned birth(mlir::Value value)
    {
        auto *def = value.getDefiningOp();
        return def && def->hasAttr(OWNED_RESULT_ATTR_NAME) &&
                       mlir::isa<mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::SymbolCallInternalOp,
                                 mlir_ts::CallInternalOp, mlir_ts::CallHybridInternalOp>(def)
                   ? 1
                   : 0;
    }

    static llvm::StringRef varName(mlir_ts::VariableOp varOp)
    {
        // The declaration's name survives only as debug metadata (`--di`), the way
        // LowerToLLVM's VariableOp lowering reads it. Without it the error still points at the
        // declaration.
        if (auto fused = mlir::dyn_cast<mlir::FusedLocWith<mlir::LLVM::DILocalVariableAttr>>(varOp.getLoc()))
        {
            if (auto name = fused.getMetadata().getName())
            {
                return name.getValue();
            }
        }

        return "this value";
    }

    void reportSecondReference(mlir::Operation *op)
    {
        llvm::StringRef name = "this value";
        if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
        {
            if (auto varOp = retainSlotOp.getSlot().getDefiningOp<mlir_ts::VariableOp>())
            {
                name = varName(varOp);
            }
        }

        op->emitError("'") << name << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
        signalPassFailure();
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipInferencePass()
{
    return std::make_unique<OwnershipInferencePass>();
}
