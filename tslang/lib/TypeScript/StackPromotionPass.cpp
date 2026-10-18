#include "mlir/Pass/Pass.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Analysis/CFGLoopInfo.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "StackEscapeFacts.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "stack-promotion"

namespace mlir_ts = mlir::typescript;

namespace
{

// The suffix of the static method every class gets for `new C()`: its body allocates the object
// and stores the vtable; the caller then calls the constructor with the result as `this`.
constexpr llvm::StringLiteral NewMethodSuffix("..new");

// Stack promotion (docs/superpowers/specs/2026-10-18-stack-promotion-design.md): an allocation
// whose object provably never outlives its function, and is small enough, goes on the stack.
// Every other allocation stays on the heap, silently. Runs at the affine level, where control
// flow is an ordinary CFG and every allocation and ownership op is still present.
//
// An allocation site is a `ts.New` (a boxed object literal, a class expression) or a call of a
// class's `C..new` defined in this module, which is how `new C()` reaches the caller. A promoted
// `C..new` call is replaced by its body: a `ts.New` on the stack plus the vtable store.
class StackPromotionPass : public mlir::PassWrapper<StackPromotionPass, mlir::OperationPass<mlir::ModuleOp>>
{
    CompileOptions compileOptions;
    mlir_ts::StackPromotionLimits limits;
    stack_escape::StackEscapeFacts *facts = nullptr;
    // shared: a pass is copied when the pass manager clones it
    llvm::DenseMap<mlir::Operation *, std::shared_ptr<mlir::CFGLoopInfo>> loopInfos;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StackPromotionPass)

    StackPromotionPass(CompileOptions &compileOptions, mlir_ts::StackPromotionLimits limits)
        : compileOptions(compileOptions), limits(limits)
    {
    }

    void runOnOperation() override
    {
        auto module = getOperation();
        stack_escape::StackEscapeFacts moduleFacts(module);
        facts = &moduleFacts;

        llvm::SmallVector<mlir::Operation *> sites;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            // `C..new` is the allocation template every `new C()` calls, not a site of its own
            if (funcOp.getName().ends_with(NewMethodSuffix))
            {
                return;
            }

            funcOp.walk([&](mlir::Operation *op) {
                if (isSite(op))
                {
                    sites.push_back(op);
                }
            });
        });

        for (auto *site : sites)
        {
            auto reason = decide(site);
            if (limits.report)
            {
                if (reason.empty())
                {
                    mlir::emitRemark(site->getLoc()) << "stack-promotion: stack";
                }
                else
                {
                    mlir::emitRemark(site->getLoc()) << "stack-promotion: heap: " << reason;
                }
            }

            if (reason.empty())
            {
                promote(site);
            }
        }
    }

  private:
    bool isSite(mlir::Operation *op)
    {
        if (auto newOp = dyn_cast<mlir_ts::NewOp>(op))
        {
            return !(newOp.getStackAlloc().has_value() && newOp.getStackAlloc().value());
        }

        return static_cast<bool>(newMethodOf(op));
    }

    // The `C..new` a call allocates through, when it is one defined here with a body. A class
    // imported as a declaration has none: an unknown callee, so a heap site.
    mlir_ts::FuncOp newMethodOf(mlir::Operation *op)
    {
        auto callOp = dyn_cast<mlir_ts::SymbolCallInternalOp>(op);
        if (!callOp || !callOp.getCallOperands().empty() || callOp->getNumResults() != 1 ||
            !isa<mlir_ts::ClassType>(callOp->getResult(0).getType()) ||
            !callOp.getCallee().ends_with(NewMethodSuffix))
        {
            return {};
        }

        auto funcOp = getOperation().lookupSymbol<mlir_ts::FuncOp>(callOp.getCalleeAttr());
        if (!funcOp || funcOp.getBody().empty())
        {
            return {};
        }

        return funcOp;
    }

    // The storage the object's block holds, which is what goes on the stack.
    static mlir::Type storageTypeOf(mlir::Type objectType)
    {
        if (auto classType = dyn_cast<mlir_ts::ClassType>(objectType))
        {
            return classType.getStorageType();
        }

        if (auto valueRefType = dyn_cast<mlir_ts::ValueRefType>(objectType))
        {
            return valueRefType.getElementType();
        }

        return {};
    }

    // Empty when the site may go on the stack; otherwise why not.
    llvm::StringRef decide(mlir::Operation *site)
    {
        auto value = site->getResult(0);
        auto storageType = storageTypeOf(value.getType());
        if (!storageType)
        {
            return "unclassified";
        }

        // version 1, until what the object owns is given back at scope exit (task 5)
        MLIRTypeHelper mth(&getContext(), compileOptions);
        if (mth.ownsHeapMemory(site->getLoc(), storageType))
        {
            return "unclassified";
        }

        if (isa<mlir_ts::SymbolCallInternalOp>(site) && newMethodBody(newMethodOf(site)).empty())
        {
            return "unclassified";
        }

        stack_escape::StackEscapeFacts::Walk walk;
        auto reason = facts->escapes(value, walk);
        if (!reason.empty())
        {
            return reason;
        }

        return loopCarried(site, walk) ? "loop-carried" : "";
    }

    // One stack slot per site, reused by every iteration of a loop around it: the object of one
    // iteration must be gone by the next. It is when nothing that is the object is used outside the
    // innermost loop holding the site, and every local slot it was stored into is declared in that
    // loop with an initializer - so each iteration re-initialises it before anything reads it.
    bool loopCarried(mlir::Operation *site, stack_escape::StackEscapeFacts::Walk &walk)
    {
        auto funcOp = site->getParentOfType<mlir_ts::FuncOp>();
        auto &body = funcOp.getBody();
        // one block has no loop, and has no dominator tree to ask
        if (body.hasOneBlock())
        {
            return false;
        }

        auto *siteOp = body.findAncestorOpInRegion(*site);
        auto *loop = loopInfoOf(funcOp).getLoopFor(siteOp->getBlock());
        if (!loop)
        {
            return false;
        }

        auto inLoop = [&](mlir::Operation *op) {
            auto *ancestor = body.findAncestorOpInRegion(*op);
            return ancestor && loop->contains(ancestor->getBlock());
        };

        for (auto alias : walk.aliases)
        {
            for (auto *user : alias.getUsers())
            {
                if (!inLoop(user))
                {
                    return true;
                }
            }
        }

        for (auto *slot : walk.slots)
        {
            if (!inLoop(slot) || slot->getNumOperands() == 0)
            {
                return true;
            }

            for (auto *user : slot->getUsers())
            {
                if (!inLoop(user))
                {
                    return true;
                }
            }
        }

        return false;
    }

    mlir::CFGLoopInfo &loopInfoOf(mlir_ts::FuncOp funcOp)
    {
        auto &entry = loopInfos[funcOp];
        if (!entry)
        {
            auto &domInfo = getAnalysis<mlir::DominanceInfo>();
            entry = std::make_shared<mlir::CFGLoopInfo>(domInfo.getDomTree(&funcOp.getBody()));
        }

        return *entry;
    }

    // The ops of `C..new` that make the object: its `ts.New` and what follows it up to the point
    // the object is handed to the return slot - the vtable store. Empty when the body is not of
    // that shape, which keeps the site on the heap.
    llvm::SmallVector<mlir::Operation *> newMethodBody(mlir_ts::FuncOp newMethod)
    {
        llvm::SmallVector<mlir::Operation *> ops;
        mlir_ts::NewOp newOp;
        for (auto &op : newMethod.getBody().front())
        {
            if (!newOp)
            {
                newOp = dyn_cast<mlir_ts::NewOp>(&op);
                if (newOp)
                {
                    ops.push_back(&op);
                }

                continue;
            }

            if (isa<mlir_ts::RetainOp, mlir_ts::ReturnInternalOp>(&op))
            {
                return ops;
            }

            if (auto storeOp = dyn_cast<mlir_ts::StoreOp>(&op); storeOp && storeOp.getValue() == newOp.getResult())
            {
                return ops;
            }

            if (!isa<mlir_ts::PropertyRefOp, mlir_ts::AddressOfOp, mlir_ts::StoreOp, mlir_ts::ConstantOp>(&op))
            {
                return {};
            }

            // everything it reads is made in this run of ops: nothing to map from elsewhere
            for (auto operand : op.getOperands())
            {
                if (!llvm::is_contained(ops, operand.getDefiningOp()))
                {
                    return {};
                }
            }

            ops.push_back(&op);
        }

        return {};
    }

    void promote(mlir::Operation *site)
    {
        if (auto newOp = dyn_cast<mlir_ts::NewOp>(site))
        {
            newOp.setStackAllocAttr(mlir::BoolAttr::get(&getContext(), true));
            return;
        }

        // `new C()`: the body of `C..new` in place of the call, its `ts.New` on the stack
        auto body = newMethodBody(newMethodOf(site));
        mlir::OpBuilder builder(site);
        mlir::IRMapping mapping;
        mlir::Value stackObject;
        for (auto *op : body)
        {
            auto *clone = builder.clone(*op, mapping);
            clone->setLoc(site->getLoc());
            if (auto newOp = dyn_cast<mlir_ts::NewOp>(clone))
            {
                newOp.setStackAllocAttr(builder.getBoolAttr(true));
                // the call's ownership marks (`__owned_result`) describe the object it made
                for (auto attr : site->getDiscardableAttrs())
                {
                    newOp->setAttr(attr.getName(), attr.getValue());
                }

                stackObject = newOp.getResult();
            }
        }

        site->getResult(0).replaceAllUsesWith(stackObject);
        site->erase();
    }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir_ts::createStackPromotionPass(CompileOptions &compileOptions,
                                                              mlir_ts::StackPromotionLimits limits)
{
    return std::make_unique<StackPromotionPass>(compileOptions, limits);
}
