#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "own"

namespace mlir_ts = mlir::typescript;

namespace
{

// Ownership inference for -mm=own, phase 0. See
// docs/superpowers/specs/2026-09-24-own-memory-model-design.md, sections 10.6 and 11.
//
// A fresh heap value may have at most one owner: one use that takes it - a local's
// declaration, a store into a field, element, global or return slot, an insertion into an
// array, a return - or, with none of those, the temporary itself, which rc gives back with a
// `ts.Release`. That owner's release is then its destroy, and every retain rc planted on the
// way is erased. Owners are counted from the uses themselves, not from rc's retains: rc also
// transfers ownership by *consuming* a value (`__owned_consumed`, a push, a field store),
// which leaves no retain behind to count.
//
// Phase 0 also keeps the move trivially safe: every taking use sits in the block that made the
// value, and nothing uses the value after it. That one rule rules out a move inside a loop
// (spec 2.6), a move on one branch, and use after move, without a liveness analysis. Anything
// else is an error at its location.
class OwnershipInferencePass : public mlir::PassWrapper<OwnershipInferencePass, TypeScriptFunctionPass>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipInferencePass)

    void runOnFunction() override
    {
        auto f = getFunction();

        // every value some ownership operation names, in the order the walk meets them
        llvm::SetVector<mlir::Value> candidates;
        llvm::SmallVector<mlir::Operation *> retains;

        f.walk([&](mlir::Operation *op) {
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
            {
                retains.push_back(op);
                candidates.insert(retainOp.getReference());
            }
            else if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(op))
            {
                retains.push_back(op);
            }
            else if (auto releaseOp = mlir::dyn_cast<mlir_ts::ReleaseOp>(op))
            {
                candidates.insert(releaseOp.getReference());
            }
            else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
            {
                if (varOp.getInitializer() && isOwningVariable(varOp))
                {
                    candidates.insert(varOp.getInitializer());
                }
            }
            else if (mlir::isa<mlir_ts::RetainCellOp>(op))
            {
                op->emitError("closures that capture a variable are not supported by -mm=own yet");
                signalPassFailure();
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                op->emitError("'delete' is not supported by -mm=own yet");
                signalPassFailure();
            }

            for (auto result : op->getResults())
            {
                if (isFresh(result))
                {
                    candidates.insert(result);
                }
            }
        });

        // Every value gets the owner check, fresh or not: rc can hand a value it did not make to
        // two consuming declarations without a single retain (the result of an indirect call
        // taken by `let b = a; let c = a;`). Only a fresh value's retains can be erased.
        llvm::DenseSet<mlir::Value> proven;
        llvm::SmallVector<mlir::Operation *> toErase;
        for (auto value : candidates)
        {
            mlir::Operation *movedRelease = nullptr;
            if (hasOneOwner(value, movedRelease) && isFresh(value))
            {
                proven.insert(value);
                if (movedRelease)
                {
                    toErase.push_back(movedRelease);
                }
            }
        }

        // What is proven loses its retains; any other retain is a second reference nobody
        // proved safe - of a loaded value, a parameter, an unknown producer, or of a fresh value
        // whose own check was reported above.
        for (auto *op : retains)
        {
            auto value = retainedValue(op);
            if (value && proven.contains(value))
            {
                toErase.push_back(op);
            }
            else if (!value || isFresh(value) == false)
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
    // The value a retain acquires: a Retain's operand, or a RetainSlot's variable's initializer.
    // None for a variable with no initializer - its storage was hoisted in front of a try and its
    // value arrives by a store this phase does not follow.
    static mlir::Value retainedValue(mlir::Operation *op)
    {
        if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(op))
        {
            return retainOp.getReference();
        }

        auto varOp = mlir::cast<mlir_ts::RetainSlotOp>(op).getSlot().getDefiningOp<mlir_ts::VariableOp>();
        return varOp ? varOp.getInitializer() : mlir::Value();
    }

    static bool isOwningVariable(mlir_ts::VariableOp varOp)
    {
        return varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) || varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME);
    }

    // Made here and held by nobody else: an allocation, a literal cast to its value type, an
    // operation rc marks as arriving with a reference, or a direct call of a function this module
    // defines (every function returns its result retained, rc 9.24; the call's own mark does not
    // survive the affine lowering). A declared callee, an indirect call, a parameter, a load or a
    // value merged from branches is not fresh.
    static bool isFresh(mlir::Value value)
    {
        auto *def = value.getDefiningOp();
        if (!def)
        {
            return false;
        }

        if (mlir::isa<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::StringConcatOp,
                      mlir_ts::CharToStringOp>(def))
        {
            return true;
        }

        if (auto callOp = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(def))
        {
            auto callee = mlir::SymbolTable::lookupNearestSymbolFrom<mlir_ts::FuncOp>(def, callOp.getCalleeAttr());
            return callee && !callee.isDeclaration();
        }

        if (mlir::isa<mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::CallInternalOp, mlir_ts::CallHybridInternalOp>(def))
        {
            return false;
        }

        return def->hasAttr(OWNED_RESULT_ATTR_NAME) || isLiteral(def);
    }

    // A string literal or a constant array cast to its value type. The result is either the
    // immortal global itself, which a release skips, or a copy nobody else holds, which a
    // release destroys - so, owned once, it has one owner either way.
    static bool isLiteral(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(def);
        return castOp && castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>();
    }

    // A use that reads the value without keeping it. Kept deliberately short: anything not listed
    // is treated as taking the value, which can only turn a program into an error.
    static bool isBorrow(mlir::Operation *user, mlir::Value value)
    {
        if (mlir::isa<mlir_ts::PrintOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp,
                      mlir_ts::ThisSymbolRefOp, mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp,
                      mlir_ts::InterfaceSymbolRefOp, mlir_ts::GetThisOp, mlir_ts::GetMethodOp,
                      mlir_ts::ArithmeticBinaryOp, mlir_ts::LogicalBinaryOp, mlir_ts::StringConcatOp>(user))
        {
            return true;
        }

        // `--di`'s record of a folded const's value for the debugger: names it, keeps nothing
        if (mlir::isa<mlir_ts::DebugVariableOp>(user))
        {
            return true;
        }

        // arguments are borrowed; a callee that keeps one retains it, and that retain is its own
        // error
        if (mlir::isa<mlir_ts::CallOp, mlir_ts::CallIndirectOp, mlir_ts::SymbolCallInternalOp, mlir_ts::CallInternalOp,
                      mlir_ts::CallHybridInternalOp>(user))
        {
            return true;
        }

        // the array being pushed onto is written, not taken; what is inserted is taken
        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return pushOp.getOp() == value && !llvm::is_contained(pushOp.getItems(), value);
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return unshiftOp.getOp() == value && !llvm::is_contained(unshiftOp.getItems(), value);
        }

        if (auto spliceOp = mlir::dyn_cast<mlir_ts::ArraySpliceOp>(user))
        {
            return spliceOp.getOp() == value && !llvm::is_contained(spliceOp.getItems(), value);
        }

        // a declaration that does not own what it holds only reads it
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return !isOwningVariable(varOp);
        }

        return false;
    }

    // One owner, taken where the value was made, and no use after the taking. Reports and returns
    // false otherwise.
    //
    // A taker and the temporary's own release are one owner, not two, when the release is the
    // only one and comes after the taker in the same block: the value moves into the taker, so
    // the release has nothing left to give back and is returned in `movedRelease` to be erased.
    // That is the shape of a folded `const` handed on - `const a = new C(); let b = a;` - where rc
    // retains for the `let` and gives the name's reference back at the end of the block. Only a
    // taker rc gave a reference of its own can receive the move (see takerAcquires).
    bool hasOneOwner(mlir::Value value, mlir::Operation *&movedRelease)
    {
        auto *home = value.getParentBlock();

        mlir::Operation *taker = nullptr;
        llvm::SmallVector<mlir::Operation *> releases;
        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::RetainOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::ReleaseOp>(user))
            {
                releases.push_back(user);
                continue;
            }

            if (isBorrow(user, value))
            {
                continue;
            }

            if (user->getBlock() != home)
            {
                user->emitError("'") << describe(value) << "' is given an owner in another block (a loop or a branch); "
                                                           "-mm=own cannot prove a move here yet";
                signalPassFailure();
                return false;
            }

            if (taker)
            {
                reportSecondReference(taker->isBeforeInBlock(user) ? user : taker);
                return false;
            }

            taker = user;
        }

        if (!taker)
        {
            return true; // the temporary is the one owner, or nobody is and it leaks as under rc
        }

        if (!releases.empty())
        {
            if (releases.size() != 1 || releases.front()->getBlock() != home ||
                !taker->isBeforeInBlock(releases.front()) || !takerAcquires(taker, value))
            {
                reportSecondReference(taker);
                return false;
            }

            movedRelease = releases.front();
        }

        // Every other use must come first. A use in another block runs after the whole of this
        // one, and so after the move.
        for (auto *user : value.getUsers())
        {
            if (user == taker || user == movedRelease || mlir::isa<mlir_ts::RetainOp>(user))
            {
                continue;
            }

            if (user->getBlock() != home || taker->isBeforeInBlock(user))
            {
                user->emitError("'") << describe(value) << "' is used after its value was moved";
                signalPassFailure();
                return false;
            }
        }

        return true;
    }

    // Did rc give this taker a reference of its own - a `ts.Retain` of the value in front of it,
    // or the `ts.RetainSlot` of an owning local declared from it? Only then is there a release on
    // the taker's side to give the value back once it has moved. `ts.NewInterface` is a taker to
    // this pass but a view to rc, which retains nothing for it and relies on the instance's own
    // release; moving the instance into it would leave nothing to release it at all.
    static bool takerAcquires(mlir::Operation *taker, mlir::Value value)
    {
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(taker))
        {
            return llvm::any_of(varOp.getResult().getUsers(),
                                [](mlir::Operation *user) { return mlir::isa<mlir_ts::RetainSlotOp>(user); });
        }

        return llvm::any_of(value.getUsers(), [&](mlir::Operation *user) {
            return mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == taker->getBlock() &&
                   user->isBeforeInBlock(taker);
        });
    }

    // The declaration's name survives only as debug metadata (`--di`), the way LowerToLLVM's
    // VariableOp lowering reads it. Without it the error still points at the right place.
    static llvm::StringRef varName(mlir_ts::VariableOp varOp)
    {
        if (auto fused = mlir::dyn_cast<mlir::FusedLocWith<mlir::LLVM::DILocalVariableAttr>>(varOp.getLoc()))
        {
            if (auto name = fused.getMetadata().getName())
            {
                return name.getValue();
            }
        }

        return "this value";
    }

    static llvm::StringRef describe(mlir::Value value)
    {
        for (auto *user : value.getUsers())
        {
            if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
            {
                return varName(varOp);
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
        else if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(op))
        {
            name = varName(varOp);
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
