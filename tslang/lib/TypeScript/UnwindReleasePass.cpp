#include "TypeScript/Config.h"
#include "TypeScript/DataStructs.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

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

#define DEBUG_TYPE "pass"

namespace mlir_ts = mlir::typescript;

// Under -mm=rc and own, what a frame owns - its locals, the temporaries and folded `const`s it holds,
// a parameter's copy - is given back where the code that owns it ends: at a scope exit, a `return`,
// a `break`, a `throw`. An exception that unwinds out of a call skips all of those, and outside a
// `try` (whose cleanup region gives back its body's scope) nothing gave it back: each throw through
// such a frame leaked what the frame held (`const s = "x" + i; f(i);` with `f` throwing, 2M times:
// 178 MB under rc, 49 MB under gc).
//
// Here, last before the lowering to LLVM - after own's inference, stack promotion and the inliner, so
// what is released is what the code finally releases - a call that may throw and is followed, on every
// path out of it, by the release of something it already holds becomes an invoke whose landing pad
// does those releases and resumes the unwind. The pad does what the normal path would have done next,
// and nothing else: a release is copied only when no path from the call to it retains, releases or
// overwrites the same thing first (or, under own, may move it), so the pad cannot free what the
// normal path would not.
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
// Not covered: calls already in a `try` (an invoke), calls inside a catch clause or a cleanup, and
// releases that differ between the paths out of a call (each path's own release).
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
        worklist.push_back(call->getNextNode());

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

    // ---- making the pads ----

    struct FunctionState
    {
        mlir::Value undefArray;
        int tryId = 0;
    };

    void processFunction(mlir::ModuleOp module, mlir_ts::FuncOp funcOp)
    {
        auto &region = funcOp.getBody();
        if (region.empty())
        {
            return;
        }

        // what a catch clause or a cleanup runs: reached from a landing pad without the catch ended
        llvm::DenseSet<mlir::Block *> handlerBlocks;
        {
            llvm::SmallVector<mlir::Block *> worklist;
            for (auto &block : region)
            {
                if (!block.empty() && mlir::isa<mlir_ts::LandingPadOp>(block.front()))
                {
                    worklist.push_back(&block);
                }
            }

            while (!worklist.empty())
            {
                auto *block = worklist.pop_back_val();
                if (!handlerBlocks.insert(block).second)
                {
                    continue;
                }

                auto ended = llvm::any_of(*block, [](mlir::Operation &op) { return mlir::isa<mlir_ts::EndCatchOp>(op); });
                if (ended)
                {
                    continue;
                }

                for (auto *successor : block->getSuccessors())
                {
                    worklist.push_back(successor);
                }
            }
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

        llvm::SmallVector<mlir::Operation *> calls;
        for (auto &block : region)
        {
            if (handlerBlocks.contains(&block))
            {
                continue;
            }

            for (auto &op : block)
            {
                if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallInternalOp>(op) && callMayThrow(&op))
                {
                    calls.push_back(&op);
                }
            }
        }

        if (calls.empty())
        {
            return;
        }

        mlir::DominanceInfo dominance(funcOp);
        FunctionState state;

        struct Plan
        {
            mlir::Operation *call;
            llvm::SmallVector<mlir::Operation *> releases;
        };

        llvm::SmallVector<Plan> plans;
        for (auto *call : calls)
        {
            Plan plan{call, {}};
            for (auto resource : resources)
            {
                auto *definingOp = resource.getDefiningOp();
                auto available = definingOp ? dominance.properlyDominates(definingOp, call)
                                            : dominance.dominates(resource.getParentBlock(), call->getBlock());
                if (!available)
                {
                    continue;
                }

                if (auto *release = releaseAfter(call, resource))
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
            makePad(module, funcOp, state, plan.call, plan.releases);
        }
    }

    void makePad(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, FunctionState &state, mlir::Operation *call,
                 llvm::ArrayRef<mlir::Operation *> releases)
    {
        auto location = call->getLoc();
        mlir::OpBuilder builder(call->getContext());
        MLIRTypeHelper mth(call->getContext(), compileOptions);
        MLIRRTTIHelperVC rttih(builder, module, compileOptions);

        auto &entryBlock = funcOp.getBody().front();
        if (!state.undefArray)
        {
            builder.setInsertionPointToStart(&entryBlock);
            if (auto entryOp = mlir::dyn_cast<mlir_ts::EntryOp>(entryBlock.front()))
            {
                builder.setInsertionPointAfter(entryOp);
            }

            // the cleanup marker of a cleanup pad (see TryOpLowering): an array-typed operand
            auto arrTy = mth.getConstArrayValueType(mth.getOpaqueType(), 1);
            mlir::Value undefArray = builder.create<mlir_ts::UndefOp>(location, arrTy);
            auto nullVal = builder.create<mlir_ts::NullOp>(location, mth.getNullType());
            state.undefArray = builder.create<mlir_ts::InsertPropertyOp>(location, undefArray.getType(), nullVal, undefArray,
                                                                         MLIRHelper::getStructIndex(builder, 0));

            // the try the pads are ItaniumEndCatchPass's for, entered at the function's start
            if (!compileOptions.isWindows)
            {
                state.tryId = nextTryId++;
                builder.create<mlir_ts::TryEntryOp>(location, builder.getI32IntegerAttr(state.tryId));
            }

            funcOp.setPersonalityAttr(builder.getBoolAttr(true));
        }

        // the call's block, cut after it
        auto *callBlock = call->getBlock();
        auto *continuation = callBlock->splitBlock(call->getNextNode());

        // the pad
        auto *padBlock = new mlir::Block();
        funcOp.getBody().getBlocks().insertAfter(continuation->getIterator(), padBlock);
        builder.setInsertionPointToStart(padBlock);
        auto landingPadOp = builder.create<mlir_ts::LandingPadOp>(location, rttih.getLandingPadType(), builder.getBoolAttr(true),
                                                                  mlir::ValueRange{state.undefArray});
        if (state.tryId)
        {
            landingPadOp->setAttr(ITANIUM_TRY_ID_ATTR_NAME, builder.getI32IntegerAttr(state.tryId));
        }

        builder.create<mlir_ts::BeginCleanupOp>(location);
        for (auto *release : releases)
        {
            builder.clone(*release);
        }

        builder.create<mlir_ts::EndCleanupOp>(location, landingPadOp, mlir::SmallVector<mlir::Block *>{});

        // the call becomes an invoke into it
        builder.setInsertionPoint(call);
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
};

} // namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createUnwindReleasePass(CompileOptions &compileOptions)
{
    return std::make_unique<UnwindReleasePass>(compileOptions);
}
