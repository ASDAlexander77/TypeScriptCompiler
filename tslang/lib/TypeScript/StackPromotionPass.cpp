#include "mlir/Pass/Pass.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Analysis/CFGLoopInfo.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "TypeScript/ObjDumper.h"

#include "StackEscapeFacts.h"

#include "llvm/ADT/MapVector.h"
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
// An allocation site is a `ts.New`: a boxed object literal, a class expression, or `new C()` of a
// class defined in this module, which MLIRGen allocates in place rather than through `C..new`.
// `new C()` of a class from elsewhere calls that module's `C..new`, an unknown callee: a heap site.
class StackPromotionPass : public mlir::PassWrapper<StackPromotionPass, mlir::OperationPass<mlir::ModuleOp>>
{
    CompileOptions compileOptions;
    mlir_ts::StackPromotionLimits limits;
    stack_escape::StackEscapeFacts *facts = nullptr;
    // what the libraries this program is linked against published (CompileOptions::importedBorrowFacts)
    stack_escape::BorrowFactsMap importedFacts;
    // shared: a pass is copied when the pass manager clones it
    llvm::DenseMap<mlir::Operation *, std::shared_ptr<mlir::CFGLoopInfo>> loopInfos;
    llvm::DenseMap<mlir::Operation *, bool> tryRegions;
    llvm::DenseMap<mlir::Operation *, bool> coroutines;
    bool moduleUsesShared = false;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StackPromotionPass)

    StackPromotionPass(CompileOptions &compileOptions, mlir_ts::StackPromotionLimits limits)
        : compileOptions(compileOptions), limits(limits),
          importedFacts(stack_escape::parseBorrowFacts(compileOptions.importedBorrowFacts))
    {
        // A module the JIT cache compiles apart from the modules it imports declares their
        // functions, and one of them may have a name a library published facts for.
        if (compileOptions.isJit && compileOptions.jitCache && !compileOptions.sourceImports.empty())
        {
            importedFacts.clear();
        }

        // A -shared import defines what it exports: a fact for one of those names may describe
        // another function.
        for (auto &library : compileOptions.sharedLibraryImports)
        {
            for (auto &symbol : Dump::definedSymbols(library))
            {
                importedFacts.erase(symbol);
            }
        }
    }

    void runOnOperation() override
    {
        auto module = getOperation();
        // keyed by operation: one left from an earlier module could name an op of this one
        loopInfos.clear();
        tryRegions.clear();
        coroutines.clear();
        stack_escape::StackEscapeFacts moduleFacts(module, /*trustLocalVTables=*/true, &importedFacts);
        facts = &moduleFacts;
        moduleUsesShared = module
                               .walk([](mlir::Operation *op) {
                                   auto isShared = [](mlir::Type type) { return isa<mlir_ts::SharedType>(type); };
                                   return isa<mlir_ts::SharedNewOp, mlir_ts::SharedCountOp, mlir_ts::SharedValueRefOp>(op) ||
                                                  llvm::any_of(op->getResultTypes(), isShared) ||
                                                  llvm::any_of(op->getOperandTypes(), isShared)
                                              ? mlir::WalkResult::interrupt()
                                              : mlir::WalkResult::advance();
                               })
                               .wasInterrupted();

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

        // every site decided before any is promoted: a promotion rewrites the IR the others read
        llvm::SmallVector<llvm::StringRef> reasons;
        for (auto *site : sites)
        {
            reasons.push_back(decide(site));
        }

        applyFrameBudget(sites, reasons);

        for (auto [site, reason] : llvm::zip(sites, reasons))
        {
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

        if (isa<mlir_ts::CreateArrayOp>(op))
        {
            return !op->hasAttr(STACK_ALLOC_ATTR_NAME);
        }

        if (isa<mlir_ts::VariableOp>(op) && op->hasAttr(CAPTURE_BOX_ATTR_NAME))
        {
            return !op->hasAttr(STACK_ALLOC_ATTR_NAME);
        }

        return false;
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

    // What every kind of site is refused for by where it is, not by what it does.
    llvm::StringRef frameReason(mlir_ts::FuncOp funcOp)
    {
        // version 1: under rc and own an unwind path would need its own ts.ReleaseContents; under
        // gc and none an object gives nothing back, and the unwind edges are in the CFG the loop
        // rule reads
        if (compileOptions.tracksOwnership() && hasTryRegion(funcOp))
        {
            return "try-region";
        }

        // version 1: a frame that is left and re-entered - at an await, at a yield - is not one
        // stack frame
        if (isCoroutine(funcOp))
        {
            return "coroutine";
        }

        // every frame of a recursion would carry the object
        if (facts->isRecursive(funcOp))
        {
            return "recursive";
        }

        return "";
    }

    // An array literal: its header and its elements on the stack, when the elements are known now,
    // the array never changes length, and - under rc and own - its elements own nothing, so that
    // nothing has to be given back when the function returns (version 1).
    llvm::StringRef decideArray(mlir_ts::CreateArrayOp arrayOp)
    {
        auto reason = frameReason(arrayOp->getParentOfType<mlir_ts::FuncOp>());
        if (!reason.empty())
        {
            return reason;
        }

        auto elementType = cast<mlir_ts::ArrayType>(arrayOp.getType()).getElementType();
        MLIRTypeHelper mth(&getContext(), compileOptions);
        if (compileOptions.tracksOwnership() && mth.ownsHeapMemory(arrayOp.getLoc(), elementType))
        {
            return "owning-elements";
        }

        if (arrayDataBytes(arrayOp) > limits.maxObjectBytes)
        {
            return "too-big";
        }

        stack_escape::StackEscapeFacts::Walk walk;
        reason = facts->escapes(arrayOp.getResult(), walk);
        if (!reason.empty())
        {
            return reason;
        }

        return loopCarried(arrayOp, walk) ? "loop-carried" : "";
    }

    // A closure's capture box: on the stack when the closure never leaves the function. Only where
    // nothing counts (version 1): under rc and own the box's release routine is what gives back
    // the cells it holds, and an immortal box would never run it.
    llvm::StringRef decideBox(mlir_ts::VariableOp boxOp)
    {
        if (compileOptions.tracksOwnership())
        {
            return "counted-capture";
        }

        auto reason = frameReason(boxOp->getParentOfType<mlir_ts::FuncOp>());
        if (!reason.empty())
        {
            return reason;
        }

        auto storageType = boxOp.getType().getElementType();
        if (blockBytes(storageType) - headerBytes() > limits.maxObjectBytes)
        {
            return "too-big";
        }

        // Filled through its fields, then made into closures: the closures are what may escape.
        //
        // A closure's body - and every closure nested in it - reaches the captured variables
        // through the cells the box's fields hold: a nested closure gets a box of its own, filled
        // with those cells, and never holds the outer box. That is why running a closure lets
        // nothing of the box out (StackEscapeFacts, a ts.CallInternal's callee). It is checked,
        // not assumed: what is stored into a field must not be a capture box, and the body of
        // every closure made from the box must keep its `this` - the box - inside it.
        stack_escape::StackEscapeFacts::Walk walk;
        for (auto &use : boxOp.getResult().getUses())
        {
            auto *user = use.getOwner();
            if (auto propertyRefOp = dyn_cast<mlir_ts::PropertyRefOp>(user))
            {
                for (auto *fieldUser : propertyRefOp.getResult().getUsers())
                {
                    auto storeOp = dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!isa<mlir_ts::LoadOp>(fieldUser) && !(storeOp && storeOp.getReference() == propertyRefOp.getResult()))
                    {
                        return "unclassified";
                    }

                    auto *stored = storeOp ? storeOp.getValue().getDefiningOp() : nullptr;
                    if (stored && stored->hasAttr(CAPTURE_BOX_ATTR_NAME))
                    {
                        return "captured-box";
                    }
                }

                continue;
            }

            if (auto boundOp = dyn_cast<mlir_ts::CreateBoundFunctionOp>(user); boundOp && use.getOperandNumber() == 0)
            {
                auto symbolRefOp = boundOp->getOperand(1).getDefiningOp<mlir_ts::SymbolRefOp>();
                if (!symbolRefOp)
                {
                    return "unclassified";
                }

                reason = facts->argumentEscapes({symbolRefOp.getIdentifierAttr()}, 0);
                if (reason.empty())
                {
                    reason = facts->escapes(boundOp.getResult(), walk);
                }

                if (!reason.empty())
                {
                    return reason;
                }

                continue;
            }

            // the closure called by name (`const g = () => ...; g()`): the box is its `this`
            if (auto callOp = dyn_cast<mlir_ts::SymbolCallInternalOp>(user))
            {
                reason = facts->argumentEscapes({callOp.getCalleeAttr()}, use.getOperandNumber());
                if (!reason.empty())
                {
                    return reason;
                }

                continue;
            }

            return "unclassified";
        }

        // the box is used where it is filled and called by name as well as through its closures
        walk.aliases.insert(boxOp.getResult());
        return loopCarried(boxOp, walk) ? "loop-carried" : "";
    }

    uint64_t arrayDataBytes(mlir_ts::CreateArrayOp arrayOp)
    {
        llvm::SmallPtrSet<mlir::Type, 8> visiting;
        auto elementType = cast<mlir_ts::ArrayType>(arrayOp.getType()).getElementType();
        return arrayOp.getItems().size() * ((estimateBytes(elementType, visiting) + 7) / 8 * 8);
    }

    // What a site takes in the frame, headers included.
    uint64_t siteBytes(mlir::Operation *site)
    {
        if (auto arrayOp = dyn_cast<mlir_ts::CreateArrayOp>(site))
        {
            // the header block { data, length, capacity }, then the element block
            return headerBytes() + 3 * headerBytes() + headerBytes() + arrayDataBytes(arrayOp);
        }

        if (auto boxOp = dyn_cast<mlir_ts::VariableOp>(site))
        {
            return blockBytes(boxOp.getType().getElementType());
        }

        return blockBytes(storageTypeOf(site->getResult(0).getType()));
    }

    // Empty when the site may go on the stack; otherwise why not.
    llvm::StringRef decide(mlir::Operation *site)
    {
        if (auto arrayOp = dyn_cast<mlir_ts::CreateArrayOp>(site))
        {
            return decideArray(arrayOp);
        }

        if (auto boxOp = dyn_cast<mlir_ts::VariableOp>(site))
        {
            return decideBox(boxOp);
        }

        auto value = site->getResult(0);
        auto storageType = storageTypeOf(value.getType());
        if (!storageType)
        {
            return "unclassified";
        }

        // -mm=own: a borrowing record's box (own spec 32) is released through its interface's tag,
        // whose routine leaves the fields it borrows; ts.ReleaseContents would give back every field
        if (site->hasAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME))
        {
            return "borrowing-record";
        }

        // A promoted object gives back what it owns when its function returns, later than the heap
        // object it replaces would have been destroyed. Only a Shared<T> handle's count can see
        // that, so under the models that count, a module using Shared keeps every object that
        // owns memory on the heap: a field of class type may hold a subclass with a handle in it.
        if (compileOptions.tracksOwnership() && moduleUsesShared)
        {
            MLIRTypeHelper mth(&getContext(), compileOptions);
            if (mth.ownsHeapMemory(site->getLoc(), storageType))
            {
                return "shared-handle";
            }
        }

        auto frame = frameReason(site->getParentOfType<mlir_ts::FuncOp>());
        if (!frame.empty())
        {
            return frame;
        }

        if (blockBytes(storageType) - headerBytes() > limits.maxObjectBytes)
        {
            return "too-big";
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
    // iteration must be gone by the next. It is when every local slot it was stored into is
    // declared in the innermost loop holding the site, with an initializer - so each iteration
    // re-initialises it before anything reads it - and nothing that is the object is used outside
    // that loop where the site can run again afterwards. A use on the way out - a `return`, a
    // `break` out of a search - comes after the last iteration and sees its object.
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

        // whether the site may run again after `op`, along every edge of the CFG - unwind edges too
        auto *siteBlock = siteOp->getBlock();
        auto siteReachableFrom = [&](mlir::Operation *op) {
            auto *ancestor = body.findAncestorOpInRegion(*op);
            if (!ancestor)
            {
                return true;
            }

            llvm::SmallPtrSet<mlir::Block *, 16> seen;
            llvm::SmallVector<mlir::Block *> work{ancestor->getBlock()};
            while (!work.empty())
            {
                auto *block = work.pop_back_val();
                if (block == siteBlock)
                {
                    return true;
                }

                for (auto *successor : block->getSuccessors())
                {
                    if (seen.insert(successor).second)
                    {
                        work.push_back(successor);
                    }
                }
            }

            return false;
        };

        auto carriedUse = [&](mlir::Operation *user) { return !inLoop(user) && siteReachableFrom(user); };

        for (auto alias : walk.aliases)
        {
            if (llvm::any_of(alias.getUsers(), carriedUse))
            {
                return true;
            }
        }

        for (auto *slot : walk.slots)
        {
            if (!inLoop(slot) || slot->getNumOperands() == 0)
            {
                return true;
            }

            if (llvm::any_of(slot->getUsers(), carriedUse))
            {
                return true;
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

    unsigned headerBytes()
    {
        return compileOptions.sizeBits() / 8;
    }

    // What a site's block takes in the frame: the header word and the payload. An estimate - every
    // field rounded up to 8 bytes - because the exact layout is only known once lowered to LLVM;
    // it bounds how much stack promotion takes, which needs no more precision than that.
    uint64_t blockBytes(mlir::Type storageType)
    {
        llvm::SmallPtrSet<mlir::Type, 8> visiting;
        return headerBytes() + estimateBytes(storageType, visiting);
    }

    uint64_t estimateBytes(mlir::Type type, llvm::SmallPtrSetImpl<mlir::Type> &visiting)
    {
        auto word = static_cast<uint64_t>(headerBytes());
        auto roundUp = [](uint64_t bytes) { return (bytes + 7) / 8 * 8; };
        if (!visiting.insert(type).second)
        {
            return word;
        }

        uint64_t bytes = word;
        if (isa<mlir_ts::TupleType, mlir_ts::ClassStorageType, mlir_ts::ObjectStorageType>(type))
        {
            MLIRTypeHelper mth(&getContext(), compileOptions);
            bytes = 0;
            for (auto fieldType : mth.getOwnershipFieldTypes(type))
            {
                bytes += roundUp(estimateBytes(fieldType, visiting));
            }
        }
        else if (auto constTupleType = dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            bytes = 0;
            for (auto &field : constTupleType.getFields())
            {
                bytes += roundUp(estimateBytes(field.type, visiting));
            }
        }
        else if (auto unionType = dyn_cast<mlir_ts::UnionType>(type))
        {
            // the tag, then the largest member
            uint64_t largest = 0;
            for (auto memberType : unionType.getTypes())
            {
                largest = std::max(largest, estimateBytes(memberType, visiting));
            }

            bytes = 8 + roundUp(largest);
        }
        else if (auto optionalType = dyn_cast<mlir_ts::OptionalType>(type))
        {
            bytes = 8 + roundUp(estimateBytes(optionalType.getElementType(), visiting));
        }
        else if (auto literalType = dyn_cast<mlir_ts::LiteralType>(type))
        {
            bytes = estimateBytes(literalType.getElementType(), visiting);
        }
        else if (isa<mlir_ts::InterfaceType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType>(type))
        {
            bytes = 2 * word;
        }

        visiting.erase(type);
        return bytes;
    }

    // The frame budget: per function, sites in loops first - where a heap allocation costs most -
    // then in source order, until the next would not fit.
    void applyFrameBudget(llvm::ArrayRef<mlir::Operation *> sites, llvm::MutableArrayRef<llvm::StringRef> reasons)
    {
        llvm::MapVector<mlir::Operation *, llvm::SmallVector<unsigned>> byFunction;
        for (auto [index, site] : llvm::enumerate(sites))
        {
            if (reasons[index].empty())
            {
                byFunction[site->getParentOfType<mlir_ts::FuncOp>()].push_back(index);
            }
        }

        for (auto &entry : byFunction)
        {
            auto &indexes = entry.second;
            llvm::stable_sort(indexes, [&](unsigned left, unsigned right) {
                return inLoop(sites[left]) && !inLoop(sites[right]);
            });

            uint64_t used = 0;
            for (auto index : indexes)
            {
                auto bytes = siteBytes(sites[index]);
                if (used + bytes > limits.maxFrameBytes)
                {
                    reasons[index] = "frame-budget";
                    continue;
                }

                used += bytes;
            }
        }
    }

    bool inLoop(mlir::Operation *site)
    {
        auto funcOp = site->getParentOfType<mlir_ts::FuncOp>();
        auto &body = funcOp.getBody();
        if (body.hasOneBlock())
        {
            return false;
        }

        return loopInfoOf(funcOp).getLoopFor(body.findAncestorOpInRegion(*site)->getBlock()) != nullptr;
    }

    // A function whose frame is left and re-entered: an async one, which holds async-dialect ops
    // once `await` is expanded, or a generator's step function, which dispatches on its state.
    bool isCoroutine(mlir_ts::FuncOp funcOp)
    {
        auto it = coroutines.find(funcOp);
        if (it != coroutines.end())
        {
            return it->second;
        }

        auto found = funcOp
                         .walk([](mlir::Operation *op) {
                             return isa<mlir_ts::SwitchStateInternalOp, mlir_ts::SwitchStateOp>(op) ||
                                            op->getDialect()->getNamespace() == "async"
                                        ? mlir::WalkResult::interrupt()
                                        : mlir::WalkResult::advance();
                         })
                         .wasInterrupted();
        coroutines[funcOp] = found;
        return found;
    }

    // A function with a try region: after the affine lowering, one with a landing pad.
    bool hasTryRegion(mlir_ts::FuncOp funcOp)
    {
        auto it = tryRegions.find(funcOp);
        if (it != tryRegions.end())
        {
            return it->second;
        }

        auto found = funcOp.walk([](mlir_ts::LandingPadOp) { return mlir::WalkResult::interrupt(); }).wasInterrupted();
        tryRegions[funcOp] = found;
        return found;
    }

    // The object goes to the function's entry block, where it is made once per call and dominates
    // every use and every return; where the allocation was, ts.ReleaseContents `reset` gives back
    // what the previous iteration's object owned and makes a fresh one. Under rc and own what the
    // object owns is given back at every return too.
    void promote(mlir::Operation *site)
    {
        // an array or a box gives nothing back at a return (decideArray, decideBox): its blocks
        // only change where they are allocated
        if (isa<mlir_ts::CreateArrayOp, mlir_ts::VariableOp>(site))
        {
            site->setAttr(STACK_ALLOC_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            return;
        }

        auto funcOp = site->getParentOfType<mlir_ts::FuncOp>();
        auto &entryBlock = funcOp.getBody().front();

        auto newOp = cast<mlir_ts::NewOp>(site);
        mlir::OpBuilder builder(site);
        newOp.setStackAllocAttr(builder.getBoolAttr(true));
        mlir::Value stackObject = newOp.getResult();
        // where the allocation was - so before it moves, which the builder's position follows
        builder.create<mlir_ts::ReleaseContentsOp>(site->getLoc(), stackObject, builder.getUnitAttr());
        newOp->moveBefore(&entryBlock, entryBlock.begin());

        if (!compileOptions.tracksOwnership())
        {
            return;
        }

        // nothing counts or destroys an immortal block: the object's own retains and releases are
        // no-ops, and they go
        for (auto *user : llvm::make_early_inc_range(stackObject.getUsers()))
        {
            if (isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user))
            {
                user->erase();
            }
        }

        MLIRTypeHelper mth(&getContext(), compileOptions);
        if (!mth.ownsHeapMemory(stackObject.getLoc(), storageTypeOf(stackObject.getType())))
        {
            return;
        }

        // A returned value was retained where the return stored it, before the branch to the
        // return block: giving back the object's fields here cannot take it away.
        funcOp.walk([&](mlir_ts::ReturnInternalOp returnOp) {
            mlir::OpBuilder returnBuilder(returnOp);
            returnBuilder.create<mlir_ts::ReleaseContentsOp>(returnOp.getLoc(), stackObject, mlir::UnitAttr());
        });
    }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir_ts::createStackPromotionPass(CompileOptions &compileOptions,
                                                              mlir_ts::StackPromotionLimits limits)
{
    return std::make_unique<StackPromotionPass>(compileOptions, limits);
}
