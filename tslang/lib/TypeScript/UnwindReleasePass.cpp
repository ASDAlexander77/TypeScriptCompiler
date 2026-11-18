#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace ::typescript;

#include "TypeScript/MLIRLogic/MLIRHelper.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#include "TypeScript/MLIRLogic/MLIRRTTIHelperVC.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <optional>

#define DEBUG_TYPE "pass"

namespace mlir_ts = mlir::typescript;

// Under -mm=rc and own, what a frame owns - its locals, the temporaries and folded `const`s it holds,
// a parameter's copy - is given back where the code that owns it ends: at a scope exit, a `return`,
// a `break`, a `throw`. An exception that unwinds out of a call skips all of those, and outside a
// `try` (whose cleanup region gives back its body's scope) nothing gave it back: each throw through
// such a frame leaked what the frame held (`const s = "x" + i; f(i);` with `f` throwing, 2M times:
// 178 MB under rc, 49 MB under gc).
//
// Here - after own's inference and stack promotion, before the inliner (which leaves alone a function
// with landing pads), so what is released is what the code finally releases - a call that may throw
// and is followed, on every path out of it, by the release of something it already holds gets a
// landing pad that does those releases. The pad does what the normal path would have done next, and
// nothing else: a release is copied only when no path from the call to it retains, releases or
// overwrites the same thing first (or, under own, may move it), so the pad cannot free what the
// normal path would not.
//
// - A plain call - in the function's own code or in a catch clause - becomes an invoke into a
//   cleanup pad that does the releases and lets the exception go on out of the function. In a catch
//   clause the catch is ended on the way (ItaniumEndCatchPass; on Windows the pad is a cleanup
//   funclet inside the clause's catch funclet, EH_PAD_PARENT_ATTR_NAME).
// - An invoke - a call in a `try`, or in a catch clause with a cleanup around it - keeps its pad
//   but goes through the releases first. Only what nothing reachable from that pad uses again is
//   released: the handler there, the code after it and a loop back round must be done with it.
//   On Windows the releases are a cleanup funclet beside that pad, unwinding to it (and an invoke
//   that leaves its funclet is left alone: every way out of a funclet must go to one pad). On
//   Linux they follow a copy of the pad's landing pad, then join the rest of the pad.
//
// Which calls may throw is found over the module: a function may throw when its body has a `throw`,
// or calls one that may. Two kinds of call are taken not to, by ruling (2026-11-18): the methods of a
// null interface's vtable (NULL_INTERFACE_VTABLE_ATTR_NAME), whose TypeError is a fault like a crash
// rather than an exception a frame has to give its memory back for - which keeps the ray tracer's
// interface calls free of pads; and a declared function whose name has no `.` (a C function such as
// `sqrt`, the runtime's). An indirect call may reach any function whose address is taken and whose
// signature matches, `this` aside. The analysis only decides where pads are added, never removes one:
// a call wrongly taken not to throw keeps the leak it had.
//
// Not covered: calls inside a cleanup (a `finally`, a `using`'s disposal: an exception thrown while
// another unwinds), and releases that differ between the paths out of a call (each path's own).
namespace
{

class UnwindReleasePass : public mlir::PassWrapper<UnwindReleasePass, mlir::OperationPass<mlir::ModuleOp>>
{
    CompileOptions compileOptions;

    llvm::StringMap<mlir_ts::FuncOp> functions;
    llvm::DenseSet<mlir::Operation *> addressTaken;
    llvm::DenseSet<mlir::Operation *> mayThrow;
    int nextTryId = 0x40000000;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnwindReleasePass)

    explicit UnwindReleasePass(CompileOptions compileOptions) : compileOptions(compileOptions)
    {
    }

    llvm::StringRef getArgument() const final
    {
        return "unwind-release";
    }

    void runOnOperation() final
    {
        if (!compileOptions.tracksOwnership() || compileOptions.isWasm)
        {
            return;
        }

        auto module = getOperation();
        computeMayThrow(module);

        llvm::SmallVector<mlir_ts::FuncOp> funcs;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            if (!funcOp.isDeclaration())
            {
                funcs.push_back(funcOp);
            }
        });

        for (auto funcOp : funcs)
        {
            processFunction(module, funcOp);
        }
    }

  private:
    // ---- which calls may throw ----

    static bool isRethrow(mlir::Operation *throwOp)
    {
        auto exception = throwOp->getOperand(0);
        return exception.getDefiningOp<mlir_ts::NullOp>() != nullptr;
    }

    static mlir::ArrayRef<mlir::Type> inputsOf(mlir::Type type)
    {
        if (auto funcType = mlir::dyn_cast<mlir_ts::FunctionType>(type))
        {
            return funcType.getInputs();
        }

        if (auto hybridType = mlir::dyn_cast<mlir_ts::HybridFunctionType>(type))
        {
            return hybridType.getInputs();
        }

        if (auto boundType = mlir::dyn_cast<mlir_ts::BoundFunctionType>(type))
        {
            return boundType.getInputs();
        }

        return {};
    }

    static mlir::ArrayRef<mlir::Type> resultsOf(mlir::Type type)
    {
        if (auto funcType = mlir::dyn_cast<mlir_ts::FunctionType>(type))
        {
            return funcType.getResults();
        }

        if (auto hybridType = mlir::dyn_cast<mlir_ts::HybridFunctionType>(type))
        {
            return hybridType.getResults();
        }

        if (auto boundType = mlir::dyn_cast<mlir_ts::BoundFunctionType>(type))
        {
            return boundType.getResults();
        }

        return {};
    }

    static bool isObjectLike(mlir::Type type)
    {
        return mlir::isa<mlir_ts::OpaqueType, mlir_ts::ClassType, mlir_ts::InterfaceType, mlir_ts::ObjectType,
                         mlir_ts::RefType, mlir_ts::ValueRefType>(type);
    }

    // the signature a function can be reached by through a pointer of `calleeType`: the same
    // parameters and results, `this` aside (a vtable types it opaque, the method its own class)
    static bool signatureMatches(mlir::Type calleeType, mlir_ts::FuncOp funcOp)
    {
        auto callInputs = inputsOf(calleeType);
        auto callResults = resultsOf(calleeType);
        auto funcType = funcOp.getFunctionType();
        auto funcInputs = funcType.getInputs();
        auto funcResults = funcType.getResults();
        if (callInputs.size() != funcInputs.size() || callResults.size() != funcResults.size())
        {
            return false;
        }

        for (auto [index, pair] : llvm::enumerate(llvm::zip(callInputs, funcInputs)))
        {
            auto [callInput, funcInput] = pair;
            if (callInput != funcInput && !(index == 0 && isObjectLike(callInput) && isObjectLike(funcInput)))
            {
                return false;
            }
        }

        return llvm::equal(callResults, funcResults);
    }

    bool calleeMayThrow(llvm::StringRef name)
    {
        auto it = functions.find(name);
        return it != functions.end() && mayThrow.contains(it->second);
    }

    bool callMayThrow(mlir::Operation *op)
    {
        if (auto symbolCall = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(op))
        {
            return calleeMayThrow(symbolCall.getCallee());
        }

        if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(op))
        {
            if (auto callee = invokeOp.getCallee())
            {
                return calleeMayThrow(*callee);
            }
        }

        if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::InvokeOp>(op) && op->getNumOperands() > 0)
        {
            auto calleeType = op->getOperand(0).getType();
            for (auto *candidate : addressTaken)
            {
                auto funcOp = mlir::cast<mlir_ts::FuncOp>(candidate);
                if (mayThrow.contains(candidate) && signatureMatches(calleeType, funcOp))
                {
                    return true;
                }
            }
        }

        return false;
    }

    void computeMayThrow(mlir::ModuleOp module)
    {
        functions.clear();
        addressTaken.clear();
        mayThrow.clear();

        module.walk([&](mlir_ts::FuncOp funcOp) { functions[funcOp.getName()] = funcOp; });
        module.walk([&](mlir_ts::SymbolRefOp symbolRefOp) {
            auto it = functions.find(symbolRefOp.getIdentifier());
            if (it != functions.end())
            {
                addressTaken.insert(it->second);
            }
        });

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp->hasAttr(NULL_INTERFACE_VTABLE_ATTR_NAME))
            {
                continue;
            }

            if (funcOp.isDeclaration())
            {
                if (funcOp.getName().contains('.'))
                {
                    mayThrow.insert(funcOp);
                }

                continue;
            }

            auto throws = funcOp->walk([](mlir::Operation *op) {
                                    auto *dialect = op->getDialect();
                                    if ((mlir::isa<mlir_ts::ThrowCallOp, mlir_ts::ThrowUnwindOp>(op) && !isRethrow(op)) ||
                                        (dialect && dialect->getNamespace() == "async"))
                                    {
                                        return mlir::WalkResult::interrupt();
                                    }

                                    return mlir::WalkResult::advance();
                                }).wasInterrupted();
            if (throws)
            {
                mayThrow.insert(funcOp);
            }
        }

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                if (funcOp.isDeclaration() || mayThrow.contains(funcOp) || funcOp->hasAttr(NULL_INTERFACE_VTABLE_ATTR_NAME))
                {
                    continue;
                }

                auto calls = funcOp->walk([&](mlir::Operation *op) {
                                       return callMayThrow(op) ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
                                   }).wasInterrupted();
                if (calls)
                {
                    mayThrow.insert(funcOp);
                    changed = true;
                }
            }
        }
    }

    // ---- the releases a call that throws owes ----

    // what a release gives back, and how
    static mlir::Value resourceOf(mlir::Operation *op)
    {
        if (mlir::isa<mlir_ts::ReleaseOp, mlir_ts::ReleaseSlotOp, mlir_ts::ReleaseCellOp>(op))
        {
            return op->getOperand(0);
        }

        return mlir::Value();
    }

    static bool isRelease(mlir::Operation *op)
    {
        return mlir::isa<mlir_ts::ReleaseOp, mlir_ts::ReleaseSlotOp, mlir_ts::ReleaseCellOp>(op);
    }

    // a release that is the giving-up half of an overwrite: a store to the same slot follows in the block
    static bool isHandOver(mlir::Operation *releaseOp)
    {
        auto slot = resourceOf(releaseOp);
        for (auto *op = releaseOp->getNextNode(); op; op = op->getNextNode())
        {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
            {
                if (storeOp.getReference() == slot)
                {
                    return true;
                }
            }

            if (isRelease(op) && resourceOf(op) == slot)
            {
                return false;
            }
        }

        return false;
    }

    // Does `op`, on the way from the call to the release of `resource`, change what the release
    // would give back?
    bool touches(mlir::Operation *op, mlir::Value resource)
    {
        if (isRelease(op) || mlir::isa<mlir_ts::RetainOp, mlir_ts::RetainSlotOp, mlir_ts::RetainCellOp>(op))
        {
            return op->getOperand(0) == resource;
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
        {
            if (storeOp.getReference() == resource)
            {
                return true;
            }

            // under own a value stored somewhere is moved there
            return compileOptions.memoryModel == MemoryModelOwn && storeOp.getValue() == resource;
        }

        // under own a value passed to a call may be moved into it
        if (compileOptions.memoryModel == MemoryModelOwn &&
            mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallInternalOp, mlir_ts::InvokeOp>(op))
        {
            return llvm::is_contained(op->getOperands(), resource);
        }

        return false;
    }

    // The release of `resource` every normal path out of `call` reaches, with nothing on the way
    // changing it - the first one found stands for them all - or none.
    mlir::Operation *releaseAfter(mlir::Operation *call, mlir::Value resource)
    {
        mlir::Operation *found = nullptr;
        auto *callBlock = call->getBlock();
        llvm::SmallPtrSet<mlir::Block *, 16> visited;
        auto callBlockReentered = false;
        llvm::SmallVector<mlir::Operation *> worklist;
        if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(call))
        {
            worklist.push_back(&invokeOp.getNormalDest()->front());
        }
        else
        {
            worklist.push_back(call->getNextNode());
        }

        auto enqueue = [&](mlir::Block *successor) {
            if (successor == callBlock)
            {
                // back round a loop: its start up to the call is all that is not walked yet
                if (!callBlockReentered)
                {
                    callBlockReentered = true;
                    worklist.push_back(&successor->front());
                }

                return;
            }

            if (visited.insert(successor).second)
            {
                worklist.push_back(&successor->front());
            }
        };

        while (!worklist.empty())
        {
            for (auto *op = worklist.pop_back_val(); op; op = op->getNextNode())
            {
                if (op == call)
                {
                    break;
                }

                if (isRelease(op) && resourceOf(op) == resource && !isHandOver(op))
                {
                    found = found ? found : op;
                    break;
                }

                if (touches(op, resource))
                {
                    return nullptr;
                }

                if (!op->hasTrait<mlir::OpTrait::IsTerminator>())
                {
                    continue;
                }

                // a way out with the resource not yet given back
                if (op->getNumSuccessors() == 0)
                {
                    if (mlir::isa<mlir_ts::UnreachableOp>(op))
                    {
                        break;
                    }

                    return nullptr;
                }

                if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(op))
                {
                    // only the normal way on: an exception out of a later call is its own business
                    enqueue(invokeOp.getNormalDest());
                    break;
                }

                for (auto *successor : op->getSuccessors())
                {
                    enqueue(successor);
                }

                break;
            }
        }

        return found;
    }

    // ---- where a call is ----

    static mlir_ts::LandingPadOp landingPadOf(mlir::Block *block)
    {
        if (block->empty() || block->getNumArguments() > 0)
        {
            return mlir_ts::LandingPadOp();
        }

        return mlir::dyn_cast<mlir_ts::LandingPadOp>(block->front());
    }

    static int intAttrOf(mlir::Operation *op, llvm::StringRef name)
    {
        auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
        return attr ? (int)attr.getInt() : 0;
    }

    // the blocks a normal path out of `block` goes on to: an invoke's unwind edge, a cleanup's
    // or a throw's are not ones
    static void normalSuccessors(mlir::Block *block, llvm::SmallVectorImpl<mlir::Block *> &successors)
    {
        auto *terminator = block->getTerminator();
        if (auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(terminator))
        {
            successors.push_back(invokeOp.getNormalDest());
            return;
        }

        if (mlir::isa<mlir_ts::EndCleanupOp, mlir_ts::ThrowUnwindOp>(terminator))
        {
            return;
        }

        for (auto *successor : terminator->getSuccessors())
        {
            successors.push_back(successor);
        }
    }

    // What runs each block: the landing pad whose code it is - a catch clause, up to the end of
    // the catch, or a cleanup - reached from the pad along normal paths; none for the
    // function's own code. A block two pads reach is `ambiguous`.
    struct Handlers
    {
        llvm::DenseMap<mlir::Block *, mlir::Block *> padOf;
        llvm::DenseSet<mlir::Block *> ambiguous;
    };

    static void findHandlers(mlir::Region &region, Handlers &handlers)
    {
        for (auto &padBlock : region)
        {
            if (!landingPadOf(&padBlock))
            {
                continue;
            }

            llvm::SmallPtrSet<mlir::Block *, 16> visited;
            llvm::SmallVector<mlir::Block *> worklist{&padBlock};
            while (!worklist.empty())
            {
                auto *block = worklist.pop_back_val();
                if (!visited.insert(block).second)
                {
                    continue;
                }

                auto [it, inserted] = handlers.padOf.try_emplace(block, &padBlock);
                if (!inserted && it->second != &padBlock)
                {
                    handlers.ambiguous.insert(block);
                }

                auto ended = llvm::any_of(*block, [](mlir::Operation &op) { return mlir::isa<mlir_ts::EndCatchOp>(op); });
                if (!ended)
                {
                    normalSuccessors(block, worklist);
                }
            }
        }
    }

    // A call that may throw, with the pad its unwind goes to now: none for a plain call, whose
    // exception leaves the function.
    struct Site
    {
        mlir::Operation *call;
        mlir::Block *dest;
        // Windows: the funclet the new pad is made in (EH_PAD_PARENT_ATTR_NAME), 0 for none
        int parentPadId;
    };

    // Where a pad can be put for `call`, given what runs its block. Not in a cleanup (an
    // exception thrown while one unwinds), nor where the code's handler is not clear.
    std::optional<Site> siteOf(mlir::Operation *call, Handlers &handlers)
    {
        auto *block = call->getBlock();
        if (handlers.ambiguous.contains(block))
        {
            return std::nullopt;
        }

        // after the end of a catch, in the block that ends it: the code after the clause
        for (auto *op = call->getPrevNode(); op; op = op->getPrevNode())
        {
            if (mlir::isa<mlir_ts::EndCatchOp>(op))
            {
                return std::nullopt;
            }
        }

        // Windows: the funclet the call runs in - its catch clause's pad, or none
        auto funcletPadId = 0;
        if (auto *padBlock = handlers.padOf.lookup(block))
        {
            auto landingPadOp = landingPadOf(padBlock);

            // A cleanup's code - a finally's unwind copy, a disposal - runs while an exception
            // unwinds, and one thrown from it takes over: a plain call there gets a pad as anywhere
            // else (on Windows, nested in the cleanup's funclet when it has an id to name). An
            // invoke there goes to the cleanup's retry machinery, and is left alone.
            if (landingPadOp.getCleanup() && mlir::isa<mlir_ts::InvokeOp>(call))
            {
                return std::nullopt;
            }

            if (compileOptions.isWindows)
            {
                funcletPadId = intAttrOf(landingPadOp, EH_PAD_ID_ATTR_NAME);
                if (!funcletPadId)
                {
                    return std::nullopt;
                }
            }
        }

        auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(call);
        if (!invokeOp)
        {
            return Site{call, nullptr, funcletPadId};
        }

        auto *dest = invokeOp.getUnwindDest();
        auto destPadOp = landingPadOf(dest);
        if (!destPadOp || !invokeOp.getUnwindDestOperands().empty())
        {
            return std::nullopt;
        }

        if (compileOptions.isWindows)
        {
            // the new pad is the destination's sibling, cleanupret unwinding to it; an invoke that
            // leaves its funclet is left as it is - every way out of a funclet must go to the same pad
            auto parentPadId = intAttrOf(destPadOp, EH_PAD_PARENT_ATTR_NAME);
            if (parentPadId != funcletPadId)
            {
                return std::nullopt;
            }

            return Site{call, dest, parentPadId};
        }

        return Site{call, dest, 0};
    }

    // The values used on any path from `dest`, the pad an existing invoke unwinds to: what the
    // handler there, the code after it, or a loop back round may still use.
    llvm::DenseSet<mlir::Value> &usedFrom(mlir::Block *dest)
    {
        auto [it, inserted] = usedFromDest.try_emplace(dest);
        if (!inserted)
        {
            return it->second;
        }

        auto &used = it->second;
        llvm::SmallPtrSet<mlir::Block *, 16> visited;
        llvm::SmallVector<mlir::Block *> worklist{dest};
        while (!worklist.empty())
        {
            auto *block = worklist.pop_back_val();
            if (!visited.insert(block).second)
            {
                continue;
            }

            for (auto &op : *block)
            {
                op.walk([&](mlir::Operation *nested) {
                    for (auto operand : nested->getOperands())
                    {
                        used.insert(operand);
                    }
                });
            }

            for (auto *successor : block->getSuccessors())
            {
                worklist.push_back(successor);
            }
        }

        return used;
    }

    // The names a resource goes by: a value, the slots it is stored into and what is loaded from
    // them; a slot, what is stored into it and loaded from it. A hold kept both as a value and in a
    // slot of its own (OwnedReturnConsumptionPass::holdGlobalArguments) is released through either.
    static llvm::SmallVector<mlir::Value> aliasesOf(mlir::Value resource)
    {
        llvm::SmallVector<mlir::Value> aliases{resource};
        llvm::SmallVector<mlir::Value> slots;
        if (mlir::isa<mlir_ts::RefType>(resource.getType()))
        {
            slots.push_back(resource);
        }

        for (auto *user : resource.getUsers())
        {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
            {
                if (storeOp.getValue() == resource)
                {
                    slots.push_back(storeOp.getReference());
                    aliases.push_back(storeOp.getReference());
                }
                else
                {
                    aliases.push_back(storeOp.getValue());
                }
            }
        }

        for (auto slot : slots)
        {
            for (auto *user : slot.getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    aliases.push_back(loadOp.getResult());
                }
            }
        }

        return aliases;
    }

    // ---- the ways a handler lets the exception go on ----

    // A handler that reads what the frame holds - a `finally` reading a local, its unwind copy -
    // keeps it (usedFrom), and so does every way out of it that lets the exception go on out of the
    // function: nothing gave the local back there. `const s = ...; try { f(); } finally { log(s); }`
    // lost `s` on every exception through it, and on every one a call in the finally threw.
    //
    // So where a pad's handler leaves the function by unwinding - a cleanup's end with nowhere
    // further to unwind to, a throw - the frame's own release of the value is copied in front of
    // it, when:
    // - every way into the pad is an invoke the value is held across: every normal path out of it
    //   releases the value, with nothing touching it on the way (releaseAfter);
    // - the handler's code is entered by that pad only, and ends only so: no catch that ends - the
    //   exception taken, the frame going on - and no return;
    // - nothing in it touches the value or a name of it (a slot it is stored in): no release,
    //   retain or overwrite, and under own no call or store that may move it.
    // A value the handler does not read is the chained pad's (processFunction), not this.
    void releaseAtUnwindExits(mlir_ts::FuncOp funcOp, llvm::ArrayRef<mlir::Value> resources)
    {
        auto &region = funcOp.getBody();
        mlir::DominanceInfo dominance(funcOp);

        llvm::SmallVector<mlir::Block *> pads;
        for (auto &block : region)
        {
            if (landingPadOf(&block))
            {
                pads.push_back(&block);
            }
        }

        for (auto *pad : pads)
        {
            llvm::SmallVector<mlir::Operation *> invokes;
            auto onlyInvokes = !pad->hasNoPredecessors();
            for (auto *pred : pad->getPredecessors())
            {
                auto invokeOp = mlir::dyn_cast<mlir_ts::InvokeOp>(pred->getTerminator());
                if (!invokeOp || invokeOp.getUnwindDest() != pad || invokeOp.getNormalDest() == pad)
                {
                    onlyInvokes = false;
                    break;
                }

                if (!llvm::is_contained(invokes, invokeOp.getOperation()))
                {
                    invokes.push_back(invokeOp);
                }
            }

            if (!onlyInvokes)
            {
                continue;
            }

            // the handler's blocks, and its ways out of the function
            llvm::SetVector<mlir::Block *> blocks;
            llvm::SmallVector<mlir::Operation *> exits;
            auto closed = true;
            llvm::SmallVector<mlir::Block *> worklist{pad};
            while (!worklist.empty() && closed)
            {
                auto *block = worklist.pop_back_val();
                if (!blocks.insert(block))
                {
                    continue;
                }

                for (auto &op : *block)
                {
                    if (mlir::isa<mlir_ts::EndCatchOp>(op))
                    {
                        closed = false;
                    }
                }

                auto *terminator = block->getTerminator();
                if (terminator->getNumSuccessors() == 0)
                {
                    if (mlir::isa<mlir_ts::EndCleanupOp, mlir_ts::ThrowCallOp>(terminator))
                    {
                        exits.push_back(terminator);
                    }
                    else if (!mlir::isa<mlir_ts::UnreachableOp>(terminator))
                    {
                        closed = false;
                    }
                }

                for (auto *successor : terminator->getSuccessors())
                {
                    worklist.push_back(successor);
                }
            }

            // entered by the pad only
            for (auto *block : blocks)
            {
                if (block == pad)
                {
                    continue;
                }

                for (auto *pred : block->getPredecessors())
                {
                    if (!blocks.contains(pred))
                    {
                        closed = false;
                    }
                }
            }

            if (!closed || exits.empty())
            {
                continue;
            }

            for (auto resource : resources)
            {
                auto aliases = aliasesOf(resource);

                // read there, or the chained pad gives it back
                auto read = false;
                auto touched = false;
                for (auto *block : blocks)
                {
                    for (auto &op : *block)
                    {
                        op.walk([&](mlir::Operation *nested) {
                            for (auto alias : aliases)
                            {
                                read |= llvm::is_contained(nested->getOperands(), alias);
                                touched |= touches(nested, alias);
                            }
                        });
                    }
                }

                if (!read || touched)
                {
                    continue;
                }

                auto *definingOp = resource.getDefiningOp();
                auto dominatesAll = [&](llvm::ArrayRef<mlir::Operation *> ops) {
                    return llvm::all_of(ops, [&](mlir::Operation *op) {
                        return definingOp ? dominance.properlyDominates(definingOp, op)
                                          : dominance.dominates(resource.getParentBlock(), op->getBlock());
                    });
                };

                if (!dominatesAll(invokes) || !dominatesAll(exits))
                {
                    continue;
                }

                mlir::Operation *release = nullptr;
                for (auto *invoke : invokes)
                {
                    auto *found = releaseAfter(invoke, resource);
                    if (!found)
                    {
                        release = nullptr;
                        break;
                    }

                    release = release ? release : found;
                }

                if (!release)
                {
                    continue;
                }

                mlir::OpBuilder builder(funcOp.getContext());
                for (auto *exit : exits)
                {
                    builder.setInsertionPoint(exit);
                    builder.clone(*release);
                }
            }
        }
    }

    // ---- making the pads ----

    struct FunctionState
    {
        mlir::Value undefArray;
        int tryId = 0;
        // Linux: a pad an invoke unwinds to, split after its landing pad, and the rest of it
        llvm::DenseMap<mlir::Block *, mlir::Block *> restOf;
    };

    llvm::DenseMap<mlir::Block *, llvm::DenseSet<mlir::Value>> usedFromDest;

    void processFunction(mlir::ModuleOp module, mlir_ts::FuncOp funcOp)
    {
        auto &region = funcOp.getBody();
        if (region.empty())
        {
            return;
        }

        // the releases in the function, by what they release
        llvm::SmallVector<mlir::Value> resources;
        llvm::DenseSet<mlir::Value> seen;
        funcOp.walk([&](mlir::Operation *op) {
            if (auto resource = resourceOf(op))
            {
                if (seen.insert(resource).second)
                {
                    resources.push_back(resource);
                }
            }
        });

        if (resources.empty())
        {
            return;
        }

        releaseAtUnwindExits(funcOp, resources);

        Handlers handlers;
        findHandlers(region, handlers);

        llvm::SmallVector<Site> sites;
        for (auto &block : region)
        {
            for (auto &op : block)
            {
                if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallInternalOp, mlir_ts::InvokeOp>(op) &&
                    callMayThrow(&op))
                {
                    if (auto site = siteOf(&op, handlers))
                    {
                        sites.push_back(*site);
                    }
                }
            }
        }

        if (sites.empty())
        {
            return;
        }

        usedFromDest.clear();
        mlir::DominanceInfo dominance(funcOp);
        FunctionState state;

        struct Plan
        {
            Site site;
            llvm::SmallVector<mlir::Operation *> releases;
        };

        llvm::SmallVector<Plan> plans;
        for (auto &site : sites)
        {
            Plan plan{site, {}};
            for (auto resource : resources)
            {
                auto *definingOp = resource.getDefiningOp();
                auto available = definingOp ? dominance.properlyDominates(definingOp, site.call)
                                            : dominance.dominates(resource.getParentBlock(), site.call->getBlock());
                if (!available)
                {
                    continue;
                }

                // the handler an invoke unwinds to now must have nothing more to do with it: the
                // pad gives it back for good
                if (site.dest && llvm::any_of(aliasesOf(resource), [&](mlir::Value alias) {
                        return usedFrom(site.dest).contains(alias);
                    }))
                {
                    continue;
                }

                if (auto *release = releaseAfter(site.call, resource))
                {
                    plan.releases.push_back(release);
                }
            }

            if (!plan.releases.empty())
            {
                plans.push_back(std::move(plan));
            }
        }

        for (auto &plan : plans)
        {
            if (plan.site.dest)
            {
                chainPad(module, funcOp, state, plan.site, plan.releases);
            }
            else
            {
                makePad(module, funcOp, state, plan.site, plan.releases);
            }
        }
    }

    // the cleanup marker of a cleanup pad (see TryOpLowering): an array-typed operand, made once
    // in the entry block; and on Linux the try the pads are ItaniumEndCatchPass's for, entered at
    // the function's start
    void prepareFunction(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, FunctionState &state, mlir::Location location)
    {
        if (state.undefArray)
        {
            return;
        }

        mlir::OpBuilder builder(funcOp.getContext());
        MLIRTypeHelper mth(funcOp.getContext(), compileOptions);

        auto &entryBlock = funcOp.getBody().front();
        builder.setInsertionPointToStart(&entryBlock);
        if (auto entryOp = mlir::dyn_cast<mlir_ts::EntryOp>(entryBlock.front()))
        {
            builder.setInsertionPointAfter(entryOp);
        }

        auto arrTy = mth.getConstArrayValueType(mth.getOpaqueType(), 1);
        mlir::Value undefArray = builder.create<mlir_ts::UndefOp>(location, arrTy);
        auto nullVal = builder.create<mlir_ts::NullOp>(location, mth.getNullType());
        state.undefArray = builder.create<mlir_ts::InsertPropertyOp>(location, undefArray.getType(), nullVal, undefArray,
                                                                     MLIRHelper::getStructIndex(builder, 0));

        if (!compileOptions.isWindows)
        {
            state.tryId = nextTryId++;
            builder.create<mlir_ts::TryEntryOp>(location, builder.getI32IntegerAttr(state.tryId));
        }

        funcOp.setPersonalityAttr(builder.getBoolAttr(true));
    }

    // a cleanup pad: the releases, then the unwind goes on - to `dest` (Windows) or out of the
    // function
    mlir::Block *cleanupPad(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, FunctionState &state, mlir::Location location,
                            mlir::Block *after, mlir::Block *dest, int parentPadId, llvm::ArrayRef<mlir::Operation *> releases)
    {
        mlir::OpBuilder builder(funcOp.getContext());
        MLIRRTTIHelperVC rttih(builder, module, compileOptions);

        auto *padBlock = new mlir::Block();
        funcOp.getBody().getBlocks().insertAfter(after->getIterator(), padBlock);
        builder.setInsertionPointToStart(padBlock);
        auto landingPadOp = builder.create<mlir_ts::LandingPadOp>(location, rttih.getLandingPadType(), builder.getBoolAttr(true),
                                                                  mlir::ValueRange{state.undefArray});
        if (state.tryId)
        {
            landingPadOp->setAttr(ITANIUM_TRY_ID_ATTR_NAME, builder.getI32IntegerAttr(state.tryId));
        }

        if (parentPadId)
        {
            landingPadOp->setAttr(EH_PAD_PARENT_ATTR_NAME, builder.getI32IntegerAttr(parentPadId));
        }

        builder.create<mlir_ts::BeginCleanupOp>(location);
        for (auto *release : releases)
        {
            builder.clone(*release);
        }

        llvm::SmallVector<mlir::Block *> dests;
        if (dest)
        {
            dests.push_back(dest);
        }

        builder.create<mlir_ts::EndCleanupOp>(location, landingPadOp, dests);
        return padBlock;
    }

    // a plain call: it becomes an invoke into a cleanup pad that gives the releases back and lets
    // the exception go on out of the function
    void makePad(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, FunctionState &state, Site &site,
                 llvm::ArrayRef<mlir::Operation *> releases)
    {
        auto *call = site.call;
        auto location = call->getLoc();
        prepareFunction(module, funcOp, state, location);

        // the call's block, cut after it
        auto *callBlock = call->getBlock();
        auto *continuation = callBlock->splitBlock(call->getNextNode());

        auto *padBlock = cleanupPad(module, funcOp, state, location, continuation, nullptr, site.parentPadId, releases);

        // the call becomes an invoke into it
        mlir::OpBuilder builder(call);
        mlir_ts::InvokeOp invokeOp;
        if (auto symbolCall = mlir::dyn_cast<mlir_ts::SymbolCallInternalOp>(call))
        {
            invokeOp = builder.create<mlir_ts::InvokeOp>(location, call->getResultTypes(), symbolCall.getCalleeAttr(),
                                                         symbolCall.getArgOperands(), continuation, mlir::ValueRange{}, padBlock,
                                                         mlir::ValueRange{});
        }
        else
        {
            invokeOp = builder.create<mlir_ts::InvokeOp>(location, call->getResultTypes(), call->getOperands(), continuation,
                                                         mlir::ValueRange{}, padBlock, mlir::ValueRange{});
        }

        for (auto attr : call->getAttrs())
        {
            if (!invokeOp->hasAttr(attr.getName()))
            {
                invokeOp->setAttr(attr.getName(), attr.getValue());
            }
        }

        call->replaceAllUsesWith(invokeOp->getResults());
        call->erase();
    }

    // An invoke: its unwind goes through the releases first, then on to the pad it went to.
    // Windows: a cleanup pad beside that one, whose cleanupret unwinds to it. Linux, where a
    // landing pad is reached only by an unwind edge: a copy of that pad's landing pad - the same
    // clauses, the same try - then the releases, then a branch into the rest of the pad, which
    // takes what the landing pad gave as an argument. Nothing is caught and thrown again, so it
    // does not depend on ItaniumEndCatchPass ending a catch.
    void chainPad(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, FunctionState &state, Site &site,
                  llvm::ArrayRef<mlir::Operation *> releases)
    {
        auto invokeOp = mlir::cast<mlir_ts::InvokeOp>(site.call);
        auto location = invokeOp.getLoc();
        auto *dest = site.dest;

        if (compileOptions.isWindows)
        {
            prepareFunction(module, funcOp, state, location);
            auto *padBlock =
                cleanupPad(module, funcOp, state, location, invokeOp->getBlock(), dest, site.parentPadId, releases);
            invokeOp->setSuccessor(padBlock, 1);
            return;
        }

        auto destPadOp = landingPadOf(dest);
        mlir::OpBuilder builder(funcOp.getContext());

        auto *&rest = state.restOf[dest];
        if (!rest)
        {
            rest = dest->splitBlock(destPadOp->getNextNode());
            auto padValue = rest->addArgument(destPadOp.getType(), destPadOp.getLoc());
            builder.setInsertionPointToEnd(dest);
            auto branchOp = builder.create<mlir::cf::BranchOp>(destPadOp.getLoc(), rest, mlir::ValueRange{destPadOp});
            destPadOp.getResult().replaceAllUsesExcept(padValue, branchOp);
        }

        auto *padBlock = new mlir::Block();
        funcOp.getBody().getBlocks().insertAfter(invokeOp->getBlock()->getIterator(), padBlock);
        builder.setInsertionPointToStart(padBlock);
        auto *padCopy = builder.clone(*destPadOp);
        for (auto *release : releases)
        {
            builder.clone(*release);
        }

        builder.create<mlir::cf::BranchOp>(location, rest, padCopy->getResults());
        invokeOp->setSuccessor(padBlock, 1);
    }
};

} // namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createUnwindReleasePass(CompileOptions &compileOptions)
{
    return std::make_unique<UnwindReleasePass>(compileOptions);
}
