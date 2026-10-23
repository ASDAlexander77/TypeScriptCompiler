#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "OwnershipFacts.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <map>

#define DEBUG_TYPE "own"

namespace mlir_ts = mlir::typescript;

namespace
{

using namespace own_facts;

// The callee facts of -mm=own (spec 2.4, 3.2), computed over the whole module before the
// per-function inference runs. Each call the pass can resolve - a direct call, a virtual call
// whose candidates are all defined here, the `.instanceOf` slot - gets the facts its candidates
// agree on, pinned on the call op, so the inference reads them without looking anything up.
//
// `__own_no_drops` goes one way: a call with no facts may destroy anything, which can only turn a
// program into an error. The facts a callee's own body relies on - a parameter it keeps, a result
// that borrows an argument - go the other way: a caller that does not know them borrows the
// argument and releases it, or releases a result it does not own, a double free either way. So a
// function has those only when every call that can reach it is one this pass resolved (the closed
// world, see `open`).
//
// `__own_no_drops` alone crosses a module boundary: a library built under own exports the names of
// its exported functions that have it (SHARED_LIB_OWN_FACTS), and a module importing it calls them
// knowing it. The other two cannot: a module linking the library statically re-parses its source
// and sees no facts at all, and a caller that does not know them frees twice.
class OwnershipSignaturePass : public mlir::PassWrapper<OwnershipSignaturePass, mlir::OperationPass<mlir::ModuleOp>>
{
  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipSignaturePass)

    // `executable`: the module is built into a program (`--emit=exe`, or the JIT's entry module),
    // whose exports no other module sees (USER decision, spec 31). Unknown is a library.
    explicit OwnershipSignaturePass(bool executable = false) : executable(executable)
    {
    }

    bool executable = false;

    // A global is a root (rc 5ai): what its initializer makes moves into it, and nothing gives it
    // back. rc retains a fresh value for its `ts.GlobalResult`, and that retain is the move: it
    // goes. The per-function inference never sees a global's region, so any other retain there is
    // reported here, by the global's name, rather than left to the lowering's "left a retain
    // behind".
    // Every use of `value` ends in the global's `ts.GlobalResult`: given to it, or an element of a
    // fresh array literal that is (`const nested = [[1, 2], [3]]`), or the payload of a
    // `new Shared(x)` (`let g = new Shared<S>({...})`). Its retains are judged on their own.
    static bool movesIntoResult(mlir::Value value)
    {
        return llvm::all_of(value.getUses(), [](mlir::OpOperand &use) {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::GlobalResultOp>(user))
            {
                return true;
            }

            // given to `new Shared(x)`: it moves into the handle's block, which owns it from then on
            // whatever becomes of the handle - rc counts the handle (spec 23.2, 26)
            if (auto sharedNewOp = mlir::dyn_cast<mlir_ts::SharedNewOp>(user))
            {
                return sharedNewOp.getValue() == use.get();
            }

            if ((isView(user) && use.getOperandNumber() == 0) || mlir::isa<mlir_ts::CreateArrayOp>(user))
            {
                return movesIntoResult(user->getResult(0));
            }

            // copied into a block `ts.New` made, which itself moves in: an object literal's box
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && use.getOperandNumber() == 0)
            {
                auto block = storeOp.getReference();
                return block.getDefiningOp<mlir_ts::NewOp>() &&
                       llvm::all_of(block.getUses(), [&](mlir::OpOperand &blockUse) {
                           return blockUse.getOwner() == storeOp.getOperation() ||
                                  (isView(blockUse.getOwner()) && movesIntoResult(blockUse.getOwner()->getResult(0)));
                       });
            }

            return false;
        });
    }

    // A record copied out of a local a `ts.Constant` initializes, whose owning fields are given
    // nothing that holds a block: it holds constants only - numbers, immortal string literals,
    // functions - and nobody else's reference.
    static bool holdsOnlyConstants(mlir::Value value)
    {
        auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || !varOp.getInitializer() || !varOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>())
        {
            return false;
        }

        return llvm::all_of(varOp->getUsers(), [](mlir::Operation *user) {
            if (mlir::isa<mlir_ts::LoadOp>(user))
            {
                return true;
            }

            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            return propertyRefOp && llvm::all_of(propertyRefOp->getUsers(), [&](mlir::Operation *fieldUser) {
                       auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                       return mlir::isa<mlir_ts::LoadOp>(fieldUser) ||
                              (storeOp && storeOp.getReference() == propertyRefOp.getResult() &&
                               holdsNoBlock(storeOp.getValue()));
                   });
        });
    }

    // Whether `op` may run more than once in one run of its function: it is inside a loop op, or its
    // block - or the block of a region op holding it - is on a cycle of its region's CFG, a path from
    // the block's successors back to it. (By this pass the affine lowering has made every loop a CFG
    // cycle of the function body; the loop op is kept for safety.)
    static bool mayRunAgain(mlir::Operation *op)
    {
        for (auto *at = op; at && !mlir::isa<mlir_ts::FuncOp>(at); at = at->getParentOp())
        {
            if (mlir::isa<mlir::LoopLikeOpInterface>(at))
            {
                return true;
            }

            auto *block = at->getBlock();
            if (!block)
            {
                return true; // not in a function: nothing is known of how often it runs
            }

            llvm::SmallPtrSet<mlir::Block *, 16> seen;
            llvm::SmallVector<mlir::Block *> work(block->getSuccessors().begin(), block->getSuccessors().end());
            while (!work.empty())
            {
                auto *next = work.pop_back_val();
                if (next == block)
                {
                    return true;
                }

                if (seen.insert(next).second)
                {
                    work.append(next->getSuccessors().begin(), next->getSuccessors().end());
                }
            }
        }

        return false;
    }

    // The immortal globals (OWN_IMMORTAL_GLOBAL_ATTR_NAME, spec 25). A global of a class type,
    // not exported, is immortal when only its initializer writes it, once, and gives it a fresh
    // object, nothing, or another immortal global's value. Its initializer is either its own region
    // (`const c = new C()`) or, for a class static, the one store into it in the class's static
    // constructor, which a `ts.GlobalConstructor` runs (the region then yields only the default).
    // Top-level script code is such a constructor too, so a module-level `let c: C;` it stores
    // into once is one as well. "Once" is how often the store runs, not how many stores there are:
    // a store in a loop (`static { for (...) K.w = new V(i); }`) would mark every block it is given
    // immortal, and each old one would leak, so a store that may run again leaves the global mortal,
    // and each store releases the old value, as for any other global.
    //
    // The store - the region's, which the lowering makes - marks the block HEAP_BLOCK_IMMORTAL, so
    // every block the global ever holds is one no release frees. The value test makes sure the block
    // is the global's to keep: an object made right there and given back by nobody. A call's result
    // is not one, whatever its callee: these facts are found before the callees' (computeFacts), and
    // a global's region is no function the inference checks, so a result that borrows its argument -
    // a temporary the region frees, an element of another global's array - would be marked, and
    // stored, as the global's own. Such a global stays mortal, and a read of it keeps today's errors.
    void findImmortalGlobals(mlir::ModuleOp module)
    {
        // the static constructors, which run once before the program
        llvm::StringSet<> constructors;
        module.walk([&](mlir_ts::GlobalConstructorOp constructorOp) { constructors.insert(constructorOp.getGlobalName()); });

        // a static's address taken out of a class's vtable (`k.w = x`, `k: typeof K`) and written
        // through: it may be any class's static the vtable slot holds, so no static is immortal.
        // MLIRGen makes a `ts.ThisVirtualSymbolRef` only for a method (a bound function, never a
        // `!ts.ref`), so no program reaches this through one today; it is listed for an IR that would.
        auto vtableWritten = false;
        module.walk([&](mlir::Operation *op) {
            if (mlir::isa<mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(op) &&
                mlir::isa<mlir_ts::RefType>(op->getResult(0).getType()) &&
                !llvm::all_of(op->getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::LoadOp>(user); }))
            {
                vtableWritten = true;
            }
        });

        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            return; // an op this pass cannot read symbols through: nothing is known never written
        }

        llvm::StringMap<llvm::SmallVector<mlir::Operation *>> usesOf;
        for (auto &use : *uses)
        {
            usesOf[use.getSymbolRef().getRootReference().getValue()].push_back(use.getUser());
        }

        // each candidate with the value its initializer gives it, and the store that does, if any
        llvm::MapVector<mlir_ts::GlobalOp, std::pair<mlir::Value, mlir_ts::StoreOp>> candidates;
        auto mth = typeHelper(&getContext());
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            if (!mlir::isa<mlir_ts::ClassType>(globalOp.getType()) || !mth.ownsHeapMemory(globalOp.getLoc(), globalOp.getType()) ||
                globalOp->hasAttr("export") || globalOp->hasAttr("import") || globalOp->hasAttr(DLL_EXPORT) ||
                globalOp->hasAttr(DLL_IMPORT) || !globalOp.getInitializerRegion().hasOneBlock())
            {
                return;
            }

            auto resultOp = mlir::dyn_cast<mlir_ts::GlobalResultOp>(globalOp.getInitializerBlock()->getTerminator());
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return;
            }

            mlir_ts::StoreOp initStore;
            for (auto *user : usesOf.lookup(globalOp.getSymName()))
            {
                // its vtable's entry, which only a `ts.VirtualSymbolRef` reads (vtableWritten)
                if (mlir::isa<mlir_ts::SymbolRefOp>(user) && user->getParentOfType<mlir_ts::GlobalOp>())
                {
                    if (vtableWritten)
                    {
                        return;
                    }

                    continue;
                }

                if (mlir::isa<mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(user))
                {
                    continue; // only loaded, or vtableWritten
                }

                auto addressOfOp = mlir::dyn_cast<mlir_ts::AddressOfOp>(user);
                if (!addressOfOp)
                {
                    return;
                }

                auto released = false;
                auto stored = false;
                for (auto *addressUser : addressOfOp->getUsers())
                {
                    if (mlir::isa<mlir_ts::LoadOp>(addressUser))
                    {
                        continue;
                    }

                    // the old value, the default, given up before the store
                    if (mlir::isa<mlir_ts::ReleaseSlotOp>(addressUser))
                    {
                        released = true;
                        continue;
                    }

                    // the one store, in a static constructor, run once
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(addressUser);
                    auto funcOp = storeOp ? storeOp->getParentOfType<mlir_ts::FuncOp>() : mlir_ts::FuncOp();
                    if (initStore || !storeOp || storeOp.getReference() != addressOfOp.getResult() ||
                        storeOp.getValue() == addressOfOp.getResult() || !funcOp || !constructors.contains(funcOp.getSymName()) ||
                        mayRunAgain(storeOp))
                    {
                        return;
                    }

                    initStore = storeOp;
                    stored = true;
                }

                if (released && !stored)
                {
                    return;
                }
            }

            // a static's region yields only its default, which the store replaces
            if (initStore && !regionYieldsDefault(globalOp))
            {
                return;
            }

            candidates[globalOp] = {initStore ? initStore.getValue() : resultOp->getOperand(0), initStore};
        });

        // given a fresh object, nothing, or another immortal global's value: in module order, until
        // no more are found
        llvm::StringSet<> immortal;
        auto madeHere = [&](mlir::Value value) {
            auto root = rootOf(value);
            if (!root.getDefiningOp() || isCall(root.getDefiningOp()) || !isFresh(root))
            {
                return false;
            }

            // and not given back before the global takes it
            auto released = false;
            forEachUse(root, [&](mlir::Operation *user, mlir::Value) { released = released || mlir::isa<mlir_ts::ReleaseOp>(user); });
            return !released;
        };
        // isImmortalGlobalRead, of the globals found so far: none is marked yet
        auto readsImmortal = [&](mlir::Value value) {
            auto addressOfOp = globalReadOf(value);
            return addressOfOp && immortal.contains(addressOfOp.getGlobalName());
        };

        for (auto found = true; found;)
        {
            found = false;
            for (auto &[globalOp, init] : candidates)
            {
                auto value = init.first;
                if (!immortal.contains(globalOp.getSymName()) && (madeHere(value) || holdsNoBlock(value) || readsImmortal(value)))
                {
                    immortal.insert(globalOp.getSymName());
                    found = true;
                }
            }
        }

        auto unit = mlir::UnitAttr::get(&getContext());
        for (auto &[globalOp, init] : candidates)
        {
            if (!immortal.contains(globalOp.getSymName()))
            {
                continue;
            }

            globalOp->setAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME, unit);
            if (init.second)
            {
                init.second->setAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME, unit);
            }

            for (auto *user : usesOf.lookup(globalOp.getSymName()))
            {
                if (mlir::isa<mlir_ts::AddressOfOp>(user))
                {
                    user->setAttr(OWN_IMMORTAL_GLOBAL_ATTR_NAME, unit);
                }
            }
        }
    }

    void moveIntoGlobals(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            llvm::SmallVector<mlir_ts::RetainOp> retains;
            globalOp.walk([&](mlir_ts::RetainOp retainOp) { retains.push_back(retainOp); });
            for (auto retainOp : retains)
            {
                // a handle's retain is rc's: the global's count, which own leaves in place (spec 23.2)
                if (touchesHandle(retainOp))
                {
                    continue;
                }

                auto value = retainOp.getReference();

                // an immortal global's value (spec 25): owned by no one, held by any number
                if (isImmortalGlobalRead(value))
                {
                    retainOp.erase();
                    continue;
                }

                // nothing in the region gives the block back, or takes a second reference to it:
                // the global is its one owner
                auto released = false;
                auto retained = 0;
                forEachUse(rootOf(value), [&](mlir::Operation *user, mlir::Value) {
                    released = released || mlir::isa<mlir_ts::ReleaseOp, mlir_ts::ReleaseSlotOp>(user);
                    retained += mlir::isa<mlir_ts::RetainOp>(user);
                });

                if (retained == 1 && !released && (isFresh(value) || holdsOnlyConstants(value) || isConstantData(value)) && movesIntoResult(value))
                {
                    retainOp.erase();
                    continue;
                }

                retainOp.emitError("the initializer of global '")
                    << globalOp.getSymName() << "' takes a second reference; -mm=own cannot prove a move or a borrow here yet";
                signalPassFailure();
            }
        });
    }

    void runOnOperation() override
    {
        auto module = getOperation();
        module.walk([&](mlir_ts::FuncOp funcOp) { functions[funcOp.getSymName()] = funcOp; });
        module.walk([&](mlir_ts::GlobalOp globalOp) { globals[globalOp.getSymName()] = globalOp; });
        module.walk([&](mlir_ts::AddressOfOp addressOfOp) {
            auto onlyLoaded = llvm::all_of(addressOfOp->getUsers(), [](mlir::Operation *user) {
                return mlir::isa<mlir_ts::LoadOp>(user);
            });
            if (!onlyLoaded)
            {
                writtenGlobals.insert(addressOfOp.getGlobalName());
            }
        });
        if (auto names = module->getAttrOfType<mlir::ArrayAttr>(SHARED_LIB_OWN_NO_DROPS_ATTR_NAME))
        {
            for (auto name : names.getAsRange<mlir::StringAttr>())
            {
                importedNoDrops.insert(name.getValue());
            }
        }

        findImmortalGlobals(module);
        moveIntoGlobals(module);
        rejectBorrowedGlobalInitializers(module);
        collectClassVTables(module);
        collectClassStorages(module);
        collectInterfaceFamilies(module);
        fieldFunctions.collect(module);
        // after the class storages (`@cpp`) and the families (holdsFunction reads the interface vtables)
        findEscapingFunctions(module);
        // before the calls are resolved: a call through a callback parameter reaches its family
        collectCallbackFamilies(module);

        module.walk([&](mlir::Operation *op) {
            // in any function: a `ts.Func`, or the `func.func` the async lowering outlines a body into
            if (isCall(op) && op->getParentOfType<mlir::FunctionOpInterface>())
            {
                calls.push_back({op, resolve(op)});
            }
        });

        findOpen(module);
        computeFacts();
        computeDrops();
        pinAssignedGlobals();
        rejectGlobalArgumentsDropped();
        findBorrowingStates(module);
        findBoundedForwarders();
        findYieldedFields(module);
        findImportedStates(module);
        checkDeclared();

        for (auto &call : calls)
        {
            // a generator another module defines, whose signature says what its state borrows
            if (auto bounded = declaredByCallees(call.callees, OWN_DECLARED_BORROWS_ATTR_NAME);
                !bounded.empty() && call.op->getNumResults() == 1)
            {
                call.op->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(&getContext(), bounded));
                // and the caller owns it (the generator's state), as `@borrows` says: fresh, though
                // the function may be another module's
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // an iterator's `next()`, whoever implements it, and a generator's state, whoever makes
            // it: see isIteratorNext. Not a `next()` through a closed interface: its members keep
            // their facts (spec 31), and the call gets theirs like any other. Nor one whose result is
            // a borrowing record (spec 32): the caller owns the box, but what it borrows bounds it.
            if (((isIteratorNext(call.op) && !call.callees.viaInterface) || makesGeneratorState(call.op)) &&
                !isRecordResult(call.op))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            if (!call.callees.known)
            {
                continue;
            }

            if (call.callees.instanceOf || call.callees.imported ||
                llvm::all_of(call.callees.funcs, [&](mlir_ts::FuncOp callee) { return noDrops.contains(callee); }))
            {
                call.op->setAttr(OWN_NO_DROPS_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // `new C()` of a class from a library built under own: see OWN_FRESH_RESULT_ATTR_NAME
            if (call.callees.imported && call.callees.importedName.ends_with("." NEW_METHOD_NAME) &&
                call.op->getNumResults() == 1)
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // a generator's `next`, read out of its state object: a function defined here, which
            // returns its result retained like any other (its symbol sits in a constant, so it is
            // open, and its body may not return a borrow)
            if (call.callees.fieldFunction && call.op->getNumResults() == 1 && !isRecordResult(call.op))
            {
                call.op->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }

            // a generator whose state object borrows from its arguments (not through a callback
            // family, whose member's parameters may be shifted)
            if (call.callees.funcs.size() == 1 && !call.callees.imported && !call.callees.viaCallback)
            {
                if (auto bounded = call.callees.funcs.front()->getAttrOfType<mlir::DenseI32ArrayAttr>(OWN_RESULT_BOUNDED_ATTR_NAME))
                {
                    call.op->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, bounded);
                }
            }
        }

        exportNoDrops();
        ownGlobalInitializerCalls(module);

        module.walk([&](mlir_ts::CreateBoundFunctionOp boundOp) {
            if (!boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME))
            {
                return;
            }

            auto assigned = assignedCaptures(closureBody(boundOp));
            if (!assigned.empty())
            {
                boundOp->setAttr(OWN_ASSIGNS_CAPTURES_ATTR_NAME,
                                 mlir::DenseI32ArrayAttr::get(&getContext(), assigned.getArrayRef()));
            }
        });
    }

  private:
    // The function a closure runs, when it names one defined here.
    mlir_ts::FuncOp closureBody(mlir_ts::CreateBoundFunctionOp boundOp)
    {
        auto symbolRefOp = boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>();
        if (!symbolRefOp)
        {
            return {};
        }

        auto found = functions.find(symbolRefOp.getIdentifier());
        return found != functions.end() && !found->second.isDeclaration() ? found->second : mlir_ts::FuncOp();
    }

    // The fields of a closure body's capture box whose cell it assigns (OWN_ASSIGNS_CAPTURES_ATTR_NAME),
    // directly or through a closure it builds over the cell. Every field, when the body is unknown or
    // a cell goes anywhere this does not follow.
    llvm::SetVector<int32_t> assignedCaptures(mlir_ts::FuncOp funcOp, int depth = 0)
    {
        llvm::SetVector<int32_t> assigned;
        auto boxType = funcOp && funcOp.getNumArguments() > 0 && !funcOp.getBody().empty()
                           ? mlir::dyn_cast<mlir_ts::RefType>(funcOp.getArgument(0).getType())
                           : mlir_ts::RefType();
        auto tupleType = boxType ? mlir::dyn_cast<mlir_ts::TupleType>(boxType.getElementType()) : mlir_ts::TupleType();
        if (!tupleType || depth > 16)
        {
            // unknown: a box of any shape may have had any of its cells assigned
            for (int32_t index = 0; index < 64; ++index)
            {
                assigned.insert(index);
            }

            return assigned;
        }

        if (auto found = assignedCache.find(funcOp); found != assignedCache.end())
        {
            return found->second;
        }

        for (auto *user : funcOp.getArgument(0).getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            auto field = propertyRefOp.getPosition();
            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(fieldUser);
                if (!loadOp || !isLoadedCell(loadOp.getResult()))
                {
                    continue; // a field captured by value: a copy, not a variable
                }

                if (cellAssigned(loadOp.getResult(), depth))
                {
                    assigned.insert(field);
                }
            }
        }

        assignedCache[funcOp] = assigned;
        return assigned;
    }

    // Is the variable behind this cell assigned - here, or by a closure built over it?
    bool cellAssigned(mlir::Value cell, int depth)
    {
        for (auto &use : cell.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::LoadOp, mlir_ts::RetainCellOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::ReleaseSlotOp>(user))
            {
                return true;
            }

            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (!storeOp)
            {
                return true; // the cell goes somewhere this does not follow
            }

            if (storeOp.getReference() == cell)
            {
                return true;
            }

            // captured again, by a closure built here: the field it fills in that closure's box
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto boundOp = propertyRefOp ? closureOfBox(propertyRefOp.getObjectRef()) : mlir_ts::CreateBoundFunctionOp();
            if (!boundOp || assignedCaptures(closureBody(boundOp), depth + 1).contains(propertyRefOp.getPosition()))
            {
                return true;
            }
        }

        return false;
    }

    llvm::DenseMap<mlir::Operation *, llvm::SetVector<int32_t>> assignedCache;

    // What a call may reach: known, when every candidate is a function defined in this module (or
    // the call goes through the `.instanceOf` slot, or reaches a function another module says
    // destroys nothing); else anything.
    struct Callees
    {
        bool known = false;
        bool instanceOf = false;
        bool imported = false;
        // a function read out of a field that only ever holds it (fieldFunctions)
        bool fieldFunction = false;
        // the slot of a closed interface (resolveInterfaceSlot)
        bool viaInterface = false;
        // a parameter of a closed function that callers pass only functions of this module
        // (callbackFamilies), with each member's shift: a closure's body takes its capture box
        // first, so its parameter k is the call's argument k - 1
        bool viaCallback = false;
        llvm::SmallVector<int32_t, 2> shifts;
        // the name the imported function is exported under
        llvm::StringRef importedName;
        llvm::SmallVector<mlir_ts::FuncOp, 2> funcs;
        // the symbols the call names, whether or not it is known: a direct callee, or every
        // member of a virtual slot's family (for the facts a declaration says, declaredFacts)
        llvm::SmallVector<mlir::StringAttr, 2> named;
    };

    struct Call
    {
        mlir::Operation *op;
        Callees callees;
    };

    llvm::StringMap<mlir_ts::FuncOp> functions;
    llvm::StringMap<mlir_ts::GlobalOp> globals;
    // Globals whose address is used other than to load from it: what they hold may change.
    llvm::StringSet<> writtenGlobals;
    llvm::SmallVector<Call> calls;

    // Functions of imported libraries that destroy nothing a caller can reach, by the name they
    // are exported under (SHARED_LIB_OWN_NO_DROPS_ATTR_NAME).
    llvm::StringSet<> importedNoDrops;

    // (tuple type, field) -> the one function an object of that type holds in that field. A
    // generator's state object holds its `next` so.
    own_facts::FieldFunctions fieldFunctions;

    // Class names, from their vtables, to split a method's symbol into class and method name.
    llvm::StringSet<> classNames;
    // (vtable position, method name) -> the functions at that position under that name, in every
    // class vtable of the module: what a virtual call through that slot may reach.
    std::map<std::pair<int64_t, std::string>, llvm::SmallVector<mlir::StringAttr, 2>> families;

    // Functions whose callers are not all known - a caller outside the module, or a reference to
    // the symbol that is not a call this pass resolves - and why, for the error in their body.
    llvm::DenseMap<mlir::Operation *, const char *> open;

    // Methods an interface's vtable holds (`<class>.<interface>..vtbl`, or `<interface>.<n>..vtbl`
    // for an object literal) whose calls through the interface are not resolved: the slot's family
    // is open (interfaceFamilies), or the vtable is one this pass cannot tie to an interface. Such
    // a call (`ts.InterfaceSymbolRef`, then an indirect call) is given none of the method's facts: a
    // caller that hands it an argument it keeps still releases it, and a caller of one whose result
    // borrows an argument or `this` releases the borrow as its own (a double free either way). So a
    // method reached this way that keeps a parameter or returns a borrow - by its body or by `@keeps`
    // / `@lends` - loses its facts, and the body's error says why (`is a parameter, kept here, but
    // -mm=own cannot make the callers of ... give it up`, `is a parameter, returned here, but -mm=own
    // cannot tell the callers of ... that the result borrows it`). A method reached only through
    // closed interfaces keeps its facts: their calls get them (spec 31).
    llvm::DenseSet<mlir::Operation *> interfaceReached;

    // ---- Interfaces closed in this module (G9, spec 31) ----
    //
    // A call through an interface reaches whatever function the slot of the interface value's vtable
    // holds. When every such vtable is one this module defines and every interface value is made
    // from one of them, the slot holds one of a known set of functions - the slot's family - and the
    // call is resolved to it like a virtual call of a private class family: it gets the facts its
    // members agree on (meetFacts demotes a family that disagrees), and `__own_no_drops` when every
    // member destroys nothing. Anything this cannot see opens the interface, and then the call is
    // what it was: no facts, and its members reached that way lose theirs (interfaceReached).
    //
    // Open, for an interface I:
    // - another module can see it: its name in an exported declaration (`__decls`), or `!ts.iface<@I>`
    //   in the type of a function or global another module can reach - an exported or declared
    //   function, a global that is not private, or is exported or imported;
    // - an I value made other than by `ts.NewInterface` over a vtable this pass reads (the global, a
    //   heap copy of it, or the one an object's class vtable holds for I): a cast from another
    //   interface (`J` to `I`, which opens `J` too: its members are reached through `I`), from
    //   `any` or an opaque pointer (which opens every interface put into one), from a tuple or array
    //   that does not hold I already (castMakesInterface), a `ts.InterfaceRef` used as a value;
    // - a vtable of I written after its initializer, or read other than to make an interface value;
    // - a class vtable that holds a vtable of I and is not private (an exported class: another
    //   module may extend it);
    // - and, per slot, a member that is not a function this module defines, a slot that holds no
    //   function (a field's offset, a thunk's cell), or a member that says `@keeps` or `@lends` (its
    //   facts are its signature's, which a family cannot change). A member that is not private (a
    //   `public` method) is open (findOpen): it has no facts of its own to give the family.
    struct InterfaceFamily
    {
        // the slot's name, which every vtable of the interface agrees on
        mlir::StringAttr field;
        llvm::SmallVector<mlir::StringAttr, 2> members;
        bool closed = true;
    };

    // (interface, vtable position) -> the functions every vtable of the interface holds there
    std::map<std::pair<std::string, int64_t>, InterfaceFamily> interfaceFamilies;

    // The words of the module's exported declarations (none in an executable), the classes a class
    // vtable is the own vtable of, and the classes another module can name: exported, or in the
    // type of an exported or declared function or of an exported or imported global (findOpen).
    llvm::StringSet<> exportWords;
    llvm::StringSet<> ownClasses;
    llvm::StringSet<> nameableClasses;

    // A field of function type (`f: (x: number) => number`), called through the interface: the slot
    // holds the offset of a field of each implementing object literal, and the call goes to whatever
    // function that field holds. When every vtable's slot is such an offset and every function ever
    // stored into those fields is one this module defines - the literal's own (`{ f: function ... }`,
    // in its `ts.Constant`), or one stored into the field later, through the object or through the
    // interface - the call is resolved to them (spec 31). A store of anything else (a parameter, a
    // call's result, a closure), a field reached other than to be read or stored, or an object of
    // the literal's type made other than by the literal opens the slot. A class implementing the
    // field holds a thunk's cell, which this does not follow: open.
    struct InterfaceFieldFamily
    {
        mlir::StringAttr field;
        // (object literal's storage tuple, field index) of each vtable
        llvm::SmallVector<std::pair<mlir::Type, int64_t>, 2> targets;
        llvm::SmallVector<mlir::StringAttr, 2> members;
        bool closed = true;
    };

    std::map<std::pair<std::string, int64_t>, InterfaceFieldFamily> interfaceFieldFamilies;

    // interface name -> the tuple types of the vtables its values are made with, when every one of
    // them is read here (holdsFunction: what a value of the interface carries)
    llvm::StringMap<llvm::SmallVector<mlir_ts::TupleType, 2>> interfaceVTableTypes;

    // `ts.PropertyRef(ts.Cast(null), i)`, seen through casts: the offset of field `i` of an object of
    // that type, as an object literal's interface vtable holds it. Null for anything else.
    std::pair<mlir::Type, int64_t> slotField(mlir::Value value)
    {
        while (auto castOp = value ? value.getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp())
        {
            value = castOp.getIn();
        }

        auto propertyRefOp = value ? value.getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto nullObject = propertyRefOp ? propertyRefOp.getObjectRef().getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp();
        if (!nullObject || !nullObject.getIn().getDefiningOp<mlir_ts::NullOp>() ||
            !mlir::isa<mlir_ts::ObjectType>(nullObject.getType()))
        {
            return {};
        }

        auto tupleType = storageTupleOf(nullObject.getType());
        if (!tupleType)
        {
            return {};
        }

        return {tupleType, static_cast<int64_t>(propertyRefOp.getPosition())};
    }

    // What every object-literal field an interface's field slot may be holds (InterfaceFieldFamily).
    void collectFieldFunctions(mlir::ModuleOp module)
    {
        llvm::DenseMap<std::pair<mlir::Type, int64_t>, llvm::SetVector<mlir::StringAttr>> stored;
        llvm::DenseSet<std::pair<mlir::Type, int64_t>> unknownFields;
        llvm::DenseSet<mlir::Type> unknownTypes;
        // a function of unknown origin stored through an interface this pass cannot follow
        auto unknownAnywhere = false;

        auto store = [&](std::pair<mlir::Type, int64_t> key, mlir::Value value) {
            auto symbol = slotFunction(value);
            auto funcOp = symbol ? functions.lookup(symbol.getValue()) : mlir_ts::FuncOp();
            if (funcOp && !funcOp.isDeclaration())
            {
                stored[key].insert(symbol);
            }
            else
            {
                unknownFields.insert(key);
            }
        };

        auto isFunctionType = [](mlir::Type type) {
            return mlir::isa<mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType>(type);
        };

        module.walk([&](mlir::Operation *op) {
            // the literal's own values
            if (auto constantOp = mlir::dyn_cast<mlir_ts::ConstantOp>(op))
            {
                auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(constantOp.getType());
                auto values = mlir::dyn_cast<mlir::ArrayAttr>(constantOp.getValue());
                if (!constTupleType || !values)
                {
                    return;
                }

                mlir::Type tupleType = mlir_ts::TupleType::get(&getContext(), constTupleType.getFields());
                for (auto [index, value] : llvm::enumerate(values))
                {
                    std::pair<mlir::Type, int64_t> key{tupleType, static_cast<int64_t>(index)};
                    if (auto symbol = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(value))
                    {
                        auto funcOp = functions.lookup(symbol.getValue());
                        if (funcOp && !funcOp.isDeclaration())
                        {
                            stored[key].insert(symbol.getAttr());
                        }
                        else
                        {
                            unknownFields.insert(key);
                        }
                    }
                    else if (!mlir::isa<mlir::UnitAttr>(value) && index < constTupleType.getFields().size() &&
                             isFunctionType(constTupleType.getFields()[index].type))
                    {
                        unknownFields.insert(key);
                    }
                }

                return;
            }

            // a store into the field through the object
            if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(op))
            {
                auto tupleType = storageTupleOf(propertyRefOp.getObjectRef().getType());
                if (!tupleType)
                {
                    return;
                }

                std::pair<mlir::Type, int64_t> key{tupleType, static_cast<int64_t>(propertyRefOp.getPosition())};
                for (auto &use : propertyRefOp->getUses())
                {
                    auto *user = use.getOwner();
                    if (mlir::isa<mlir_ts::LoadOp, mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                    {
                        continue;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == propertyRefOp.getResult() && storeOp.getValue() != propertyRefOp.getResult())
                    {
                        store(key, storeOp.getValue());
                        continue;
                    }

                    // an interface vtable's offset of the field (slotField) reads nothing
                    if (user->getParentOfType<mlir_ts::GlobalOp>() && mlir::isa<mlir_ts::CastOp, mlir_ts::InsertPropertyOp>(user))
                    {
                        continue;
                    }

                    unknownFields.insert(key);
                }

                return;
            }

            // a store into the field through an interface: into each literal its vtables reach
            if (auto interfaceRefOp = mlir::dyn_cast<mlir_ts::InterfaceSymbolRefOp>(op))
            {
                auto refType = mlir::dyn_cast<mlir_ts::RefType>(interfaceRefOp.getType());
                if (!refType || !isFunctionType(refType.getElementType()))
                {
                    return;
                }

                auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(interfaceRefOp.getInterfaceVal().getType());
                auto found = interfaceType ? interfaceFieldFamilies.find({interfaceType.getName().getValue().str(),
                                                                         static_cast<int64_t>(interfaceRefOp.getIndex()) +
                                                                             INTERFACE_VTABLE_HEADER_SLOTS})
                                           : interfaceFieldFamilies.end();
                for (auto &use : interfaceRefOp->getUses())
                {
                    auto *user = use.getOwner();
                    if (mlir::isa<mlir_ts::LoadOp, mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                    {
                        continue;
                    }

                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == interfaceRefOp.getResult() && found != interfaceFieldFamilies.end() &&
                        found->second.closed)
                    {
                        for (auto target : found->second.targets)
                        {
                            store(target, storeOp.getValue());
                        }

                        continue;
                    }

                    unknownAnywhere = true;
                }

                return;
            }

            // an object of the literal's type made other than by the literal: its fields may hold
            // anything. A cast keeps the type's own value (the literal's box seen as the object, an
            // object, a reference to its storage) or makes the null object a vtable's offsets are
            // taken from; from anything else it reinterprets.
            if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(op))
            {
                auto tupleType = storageTupleOf(castOp.getType());
                if (tupleType && !mlir::isa<mlir_ts::NullType>(castOp.getIn().getType()) &&
                    storageTupleOf(castOp.getIn().getType()) != tupleType)
                {
                    unknownTypes.insert(tupleType);
                }

                return;
            }

            if (mlir::isa<mlir_ts::LoadOp>(op) || isCall(op) || isView(op) || mlir::isa<mlir_ts::ExtractPropertyOp>(op))
            {
                return;
            }

            for (auto result : op->getResults())
            {
                if (mlir::isa<mlir_ts::TupleType, mlir_ts::ObjectType>(result.getType()))
                {
                    if (auto tupleType = storageTupleOf(result.getType()))
                    {
                        unknownTypes.insert(tupleType);
                    }
                }
            }
        });

        for (auto &[key, family] : interfaceFieldFamilies)
        {
            if (!family.closed)
            {
                continue;
            }

            for (auto target : family.targets)
            {
                if (unknownAnywhere || unknownTypes.contains(target.first) || unknownFields.contains(target))
                {
                    family.closed = false;
                    break;
                }

                for (auto member : stored.lookup(target))
                {
                    if (!llvm::is_contained(family.members, member))
                    {
                        family.members.push_back(member);
                    }
                }
            }

            family.closed = family.closed && !family.members.empty();
        }
    }

    // The interfaces mentioned by a type as values: the interface itself, a union or optional, a
    // tuple's fields, an array's elements, what a reference points to. Not a class's or an object's
    // fields (a reference to a block made elsewhere), nor a function's parameters and result.
    static void valueInterfaces(mlir::Type type, llvm::StringSet<> &found, int depth = 0)
    {
        if (!type || depth > 16)
        {
            return;
        }

        if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(type))
        {
            found.insert(interfaceType.getName().getValue());
            return;
        }

        auto fields = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            for (auto &info : infos)
            {
                valueInterfaces(info.type, found, depth + 1);
            }
        };

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            fields(tupleType.getFields());
        }
        else if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            fields(constTupleType.getFields());
        }
        else if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(type))
        {
            for (auto member : unionType.getTypes())
            {
                valueInterfaces(member, found, depth + 1);
            }
        }
        else if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
        {
            valueInterfaces(optionalType.getElementType(), found, depth + 1);
        }
        else if (auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(type))
        {
            valueInterfaces(arrayType.getElementType(), found, depth + 1);
        }
        else if (auto constArrayType = mlir::dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            valueInterfaces(constArrayType.getElementType(), found, depth + 1);
        }
        else if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            valueInterfaces(refType.getElementType(), found, depth + 1);
        }
        else if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
        {
            valueInterfaces(valueRefType.getElementType(), found, depth + 1);
        }
        else if (auto boundRefType = mlir::dyn_cast<mlir_ts::BoundRefType>(type))
        {
            valueInterfaces(boundRefType.getElementType(), found, depth + 1);
        }
        else if (auto sharedType = mlir::dyn_cast<mlir_ts::SharedType>(type))
        {
            valueInterfaces(sharedType.getElementType(), found, depth + 1);
        }
    }

    // Every interface a type mentions anywhere, a class's fields and a function's parameters
    // included: what another module handed a value of this type could reach. With `classes`, the
    // classes it mentions too (a base class is its derived class's first field).
    static void anyInterfaces(mlir::Type type, llvm::StringSet<> &found, llvm::DenseSet<mlir::Type> &seen,
                              llvm::StringSet<> *classes = nullptr)
    {
        if (!type || !seen.insert(type).second)
        {
            return;
        }

        if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(type))
        {
            found.insert(interfaceType.getName().getValue());
            return;
        }

        auto fields = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            for (auto &info : infos)
            {
                anyInterfaces(info.type, found, seen, classes);
            }
        };

        // the types whose fields are not parameters of the type
        if (auto classStorageType = mlir::dyn_cast<mlir_ts::ClassStorageType>(type))
        {
            if (classes)
            {
                classes->insert(classStorageType.getName().getValue());
            }

            fields(classStorageType.getFields());
            return;
        }

        if (auto objectStorageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(type))
        {
            fields(objectStorageType.getFields());
            return;
        }

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            fields(tupleType.getFields());
            return;
        }

        if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            fields(constTupleType.getFields());
            return;
        }

        type.walkImmediateSubElements([](mlir::Attribute) {}, [&](mlir::Type sub) { anyInterfaces(sub, found, seen, classes); });
    }

    // The function values a value of this type holds: the type itself, a union's or an optional's
    // member, a tuple's field, an array's element. Not what a reference points to (a vtable's
    // address seen as an opaque pointer is the compiler's own, not a value).
    static void functionValues(mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &found, int depth = 0)
    {
        if (!type || depth > 16)
        {
            return;
        }

        if (mlir::isa<mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType,
                      mlir_ts::ExtensionFunctionType, mlir_ts::ConstructFunctionType>(type))
        {
            found.push_back(type);
            return;
        }

        auto fields = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            for (auto &info : infos)
            {
                functionValues(info.type, found, depth + 1);
            }
        };

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            fields(tupleType.getFields());
        }
        else if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            fields(constTupleType.getFields());
        }
        else if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(type))
        {
            for (auto member : unionType.getTypes())
            {
                functionValues(member, found, depth + 1);
            }
        }
        else if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(type))
        {
            functionValues(optionalType.getElementType(), found, depth + 1);
        }
        else if (auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(type))
        {
            functionValues(arrayType.getElementType(), found, depth + 1);
        }
        else if (auto constArrayType = mlir::dyn_cast<mlir_ts::ConstArrayType>(type))
        {
            functionValues(constArrayType.getElementType(), found, depth + 1);
        }
        else if (auto sharedType = mlir::dyn_cast<mlir_ts::SharedType>(type))
        {
            functionValues(sharedType.getElementType(), found, depth + 1);
        }
    }

    // A function type's parameters and results, whatever kind of function it is.
    static std::pair<llvm::SmallVector<mlir::Type>, llvm::SmallVector<mlir::Type>> signatureOf(mlir::Type type)
    {
        llvm::SmallVector<mlir::Type> inputs;
        llvm::SmallVector<mlir::Type> results;
        if (auto functionType = mlir::dyn_cast<mlir_ts::FunctionType>(type))
        {
            inputs.append(functionType.getInputs().begin(), functionType.getInputs().end());
            results.append(functionType.getResults().begin(), functionType.getResults().end());
        }
        else if (auto boundType = mlir::dyn_cast<mlir_ts::BoundFunctionType>(type))
        {
            inputs.append(boundType.getInputs().begin(), boundType.getInputs().end());
            results.append(boundType.getResults().begin(), boundType.getResults().end());
        }
        else if (auto hybridType = mlir::dyn_cast<mlir_ts::HybridFunctionType>(type))
        {
            inputs.append(hybridType.getInputs().begin(), hybridType.getInputs().end());
            results.append(hybridType.getResults().begin(), hybridType.getResults().end());
        }
        else
        {
            // a kind this does not read: never the same signature as another
            inputs.push_back(type);
        }

        return {inputs, results};
    }

    // Does a cast reinterpret a function value - from `any` or `unknown`, or as a function of another
    // signature? Then an interface value its parameters or
    // results hold crosses unseen (a `(o: Op2) => ...` called as `(o: Op) => ...`): `interfaces`
    // gets every interface either side's signatures mention, and `boxed` the input's when it is put
    // into `any`.
    static bool castReinterpretsFunction(mlir::Type in, mlir::Type out, llvm::StringSet<> &interfaces, bool &boxing)
    {
        // `!ts.opaque` is the compiler's own (a vtable slot read as the `.instanceOf` it holds), not
        // a value a program can put a function into
        auto isOpaque = [](mlir::Type type) { return mlir::isa<mlir_ts::AnyType, mlir_ts::UnknownType>(type); };
        llvm::SmallVector<mlir::Type> inFunctions;
        llvm::SmallVector<mlir::Type> outFunctions;
        functionValues(in, inFunctions);
        functionValues(out, outFunctions);
        boxing = false;
        if ((inFunctions.empty() && outFunctions.empty()) || in == out)
        {
            return false;
        }

        auto mention = [&](llvm::ArrayRef<mlir::Type> types) {
            for (auto type : types)
            {
                llvm::DenseSet<mlir::Type> seen;
                anyInterfaces(type, interfaces, seen);
            }
        };

        if (isOpaque(out))
        {
            boxing = true;
            mention(inFunctions);
            return false;
        }

        // the same signatures, but for `this` seen as an opaque pointer on one side, or bound on one
        // side only (a closure's box, a bound function's object: the caller supplies neither)
        auto sameSignature = [](mlir::Type a, mlir::Type b) {
            auto [aInputs, aResults] = signatureOf(a);
            auto [bInputs, bResults] = signatureOf(b);
            auto dropBoundThis = [](mlir::Type type, llvm::SmallVector<mlir::Type> &inputs) {
                if (!inputs.empty() && (mlir::isa<mlir_ts::BoundFunctionType>(type) || mlir::isa<mlir_ts::OpaqueType>(inputs.front())))
                {
                    inputs.erase(inputs.begin());
                }
            };

            if (aInputs.size() == bInputs.size() + 1)
            {
                dropBoundThis(a, aInputs);
            }
            else if (bInputs.size() == aInputs.size() + 1)
            {
                dropBoundThis(b, bInputs);
            }

            if (aResults != bResults || aInputs.size() != bInputs.size())
            {
                return false;
            }

            for (auto [index, pair] : llvm::enumerate(llvm::zip(aInputs, bInputs)))
            {
                auto [aInput, bInput] = pair;
                if (aInput != bInput &&
                    !(index == 0 && (mlir::isa<mlir_ts::OpaqueType>(aInput) || mlir::isa<mlir_ts::OpaqueType>(bInput))))
                {
                    return false;
                }
            }

            return true;
        };

        // a function out of `any`, or one function seen as another; a function seen as something
        // that is no function (a vtable slot's reference to it) is the compiler's own
        auto reinterprets = isOpaque(in) && !outFunctions.empty();
        if (!inFunctions.empty() && !outFunctions.empty())
        {
            reinterprets = reinterprets || inFunctions.size() != outFunctions.size();
            for (auto [inFunction, outFunction] : llvm::zip(inFunctions, outFunctions))
            {
                reinterprets = reinterprets || !sameSignature(inFunction, outFunction);
            }
        }

        if (reinterprets)
        {
            mention(inFunctions);
            mention(outFunctions);
        }

        return reinterprets;
    }

    // The interfaces an interface member's type mentions: a method's or a function field's
    // parameters and result, a field's type, anything they hold (a class's fields included).
    static void memberTypeInterfaces(mlir::Type type, llvm::StringSet<> &found)
    {
        llvm::DenseSet<mlir::Type> seen;
        anyInterfaces(type, found, seen);
    }

    static bool isNullish(mlir::Type type)
    {
        return mlir::isa<mlir_ts::NullType, mlir_ts::UndefinedType>(type);
    }

    // May a cast from `in` to `out` make an interface value out of something that was not that
    // interface's value already? A union or optional narrowed to its interface member (whose other
    // members are null or undefined) or widened from it, null given a nullable interface type, a
    // class or object seen as another (the same block, its fields as they are) and a function seen as
    // another are not; a cast from another interface, `any`, an opaque pointer, or a tuple or an
    // array whose element is not the same interface is.
    static bool castMakesInterface(mlir::Type in, mlir::Type out, int depth = 0)
    {
        // null seen as an interface value has no vtable: a call through it reaches nothing
        if (in == out || depth > 16 || isNullish(in))
        {
            return false;
        }

        llvm::StringSet<> made;
        valueInterfaces(out, made);
        if (made.empty())
        {
            return false;
        }

        if (mlir::isa<mlir_ts::InterfaceType>(out))
        {
            return !holdsAsIs(in, out, /*onlyNullable=*/true);
        }

        if (mlir::isa<mlir_ts::UnionType, mlir_ts::OptionalType>(out))
        {
            if (isNullish(in) || holdsAsIs(out, in))
            {
                return false;
            }

            llvm::SmallVector<mlir::Type> members;
            if (auto unionType = mlir::dyn_cast<mlir_ts::UnionType>(in))
            {
                members.append(unionType.getTypes().begin(), unionType.getTypes().end());
            }
            else if (auto optionalType = mlir::dyn_cast<mlir_ts::OptionalType>(in))
            {
                members.push_back(optionalType.getElementType());
            }
            else
            {
                return true;
            }

            return !llvm::all_of(members, [&](mlir::Type member) { return isNullish(member) || holdsAsIs(out, member); });
        }

        auto tupleFields = [](mlir::Type type) -> std::optional<llvm::ArrayRef<mlir_ts::FieldInfo>> {
            if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
            {
                return tupleType.getFields();
            }

            if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
            {
                return constTupleType.getFields();
            }

            return std::nullopt;
        };

        if (auto outFields = tupleFields(out))
        {
            auto inFields = tupleFields(in);
            if (!inFields || inFields->size() != outFields->size())
            {
                return true;
            }

            for (auto [inField, outField] : llvm::zip(*inFields, *outFields))
            {
                if (castMakesInterface(inField.type, outField.type, depth + 1))
                {
                    return true;
                }
            }

            return false;
        }

        auto elementOf = [](mlir::Type type) -> mlir::Type {
            if (auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(type))
            {
                return arrayType.getElementType();
            }

            if (auto constArrayType = mlir::dyn_cast<mlir_ts::ConstArrayType>(type))
            {
                return constArrayType.getElementType();
            }

            return {};
        };

        if (auto outElement = elementOf(out))
        {
            auto inElement = elementOf(in);
            return !inElement || castMakesInterface(inElement, outElement, depth + 1);
        }

        // a reference seen as another (`!ts.bound_ref<T>` as `!ts.ref<T>`): what it points to
        auto pointee = [](mlir::Type type) -> mlir::Type {
            if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
            {
                return refType.getElementType();
            }

            if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
            {
                return valueRefType.getElementType();
            }

            if (auto boundRefType = mlir::dyn_cast<mlir_ts::BoundRefType>(type))
            {
                return boundRefType.getElementType();
            }

            return {};
        };

        if (auto outPointee = pointee(out))
        {
            auto inPointee = pointee(in);
            return !inPointee || castMakesInterface(inPointee, outPointee, depth + 1);
        }

        // anything else reinterpreted: what is read out of it was never checked
        return true;
    }

    // The function a vtable slot holds: `ts.SymbolRef`, seen through casts. Null for anything else.
    static mlir::StringAttr slotFunction(mlir::Value value)
    {
        while (auto castOp = value ? value.getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp())
        {
            value = castOp.getIn();
        }

        auto symbolRefOp = value ? value.getDefiningOp<mlir_ts::SymbolRefOp>() : mlir_ts::SymbolRefOp();
        return symbolRefOp ? symbolRefOp.getIdentifierAttr().getAttr() : mlir::StringAttr();
    }

    static mlir_ts::AddressOfOp addressOfThroughCasts(mlir::Value value)
    {
        while (auto castOp = value ? value.getDefiningOp<mlir_ts::CastOp>() : mlir_ts::CastOp())
        {
            value = castOp.getIn();
        }

        return value ? value.getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
    }

    // The words of the module's exported declarations (`__decls*` globals): what another module
    // can name.
    static llvm::StringSet<> exportedWords(mlir::ModuleOp module)
    {
        llvm::StringSet<> words;
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            if (!globalOp.getSymName().starts_with("__decls"))
            {
                return;
            }

            globalOp.getInitializerRegion().walk([&](mlir_ts::ConstantOp constantOp) {
                auto text = mlir::dyn_cast<mlir::StringAttr>(constantOp.getValue());
                if (!text)
                {
                    return;
                }

                llvm::StringRef rest = text.getValue();
                while (!rest.empty())
                {
                    auto isWordChar = [](char c) { return llvm::isAlnum(c) || c == '_' || c == '$'; };
                    rest = rest.drop_while([&](char c) { return !isWordChar(c); });
                    auto word = rest.take_while(isWordChar);
                    if (!word.empty())
                    {
                        words.insert(word);
                    }

                    rest = rest.drop_front(word.size());
                }
            });
        });

        return words;
    }

    void collectInterfaceFamilies(mlir::ModuleOp module)
    {
        llvm::StringSet<> open;
        auto openAll = [&](const llvm::StringSet<> &names) {
            for (auto &entry : names)
            {
                open.insert(entry.getKey());
            }
        };

        // the vtables: an interface's (`ts.InsertProperty` of each slot, slot 0 `.instanceOf` of a
        // class that is not the vtable's owner, or `.instanceOf.none` of an object literal), and the
        // slots of a class's own vtable that hold the address of one of its interface vtables
        struct InterfaceVTable
        {
            mlir_ts::TupleType type;
            llvm::DenseMap<int64_t, mlir::StringAttr> slots;
            // a field's slot: the object literal's field it is the offset of
            llvm::DenseMap<int64_t, std::pair<mlir::Type, int64_t>> fields;
            bool written = false;
        };

        llvm::StringMap<InterfaceVTable> vtables;
        // `..vtbl` globals that are neither (no `.instanceOf` this reads in slot 0)
        llvm::SmallVector<llvm::StringRef> unread;
        // interface name -> the interface vtables class vtables hold for it, read through an
        // object's class at run time (`ts.VTableOffsetRef` of the object's vtable)
        llvm::StringMap<llvm::SetVector<llvm::StringRef>> classHeld;
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto name = globalOp.getSymName();
            if (!name.ends_with("..vtbl"))
            {
                return;
            }

            auto owner = name.drop_back(llvm::StringRef("..vtbl").size());
            llvm::DenseMap<int64_t, mlir::Value> values;
            globalOp.getInitializerRegion().walk([&](mlir_ts::InsertPropertyOp insertOp) {
                if (insertOp.getPosition().size() == 1)
                {
                    values[insertOp.getPosition()[0]] = insertOp.getValue();
                }
            });

            auto header = slotFunction(values.lookup(0));
            auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(globalOp.getType());
            if (!header || !tupleType)
            {
                unread.push_back(name);
                return;
            }

            if (header.getValue() == (owner + "..instanceOf").str())
            {
                // an exported class (not in an executable, whose exports no module sees)
                auto exposed = !executable && (!globalOp.isPrivate() || globalOp->hasAttr("export") || globalOp->hasAttr(DLL_EXPORT));
                ownClasses.insert(owner);
                if (exposed)
                {
                    nameableClasses.insert(owner);
                }

                for (auto &[position, value] : values)
                {
                    auto addressOfOp = addressOfThroughCasts(value);
                    auto id = position < static_cast<int64_t>(tupleType.size())
                                  ? mlir::dyn_cast_or_null<mlir::StringAttr>(tupleType.getFieldInfo(position).id)
                                  : mlir::StringAttr();
                    if (!addressOfOp || !id)
                    {
                        continue;
                    }

                    classHeld[id.getValue()].insert(addressOfOp.getGlobalName());
                    if (exposed)
                    {
                        open.insert(id.getValue()); // another module may extend the class
                    }
                }

                return;
            }

            if (!header.getValue().ends_with("..instanceOf") && header.getValue() != ".instanceOf.none")
            {
                unread.push_back(name);
                return;
            }

            auto &vtable = vtables[name];
            vtable.type = tupleType;
            for (auto &[position, value] : values)
            {
                if (position > 0)
                {
                    vtable.slots[position] = slotFunction(value);
                    if (auto field = slotField(value); field.first)
                    {
                        vtable.fields[position] = field;
                    }
                }
            }
        });

        // interface name -> the vtables its values are made with
        llvm::StringMap<llvm::SetVector<llvm::StringRef>> madeWith;
        auto vtableOf = [&](mlir::Value value) -> llvm::StringRef {
            // the global itself
            if (auto addressOfOp = addressOfThroughCasts(value))
            {
                return vtables.count(addressOfOp.getGlobalName()) ? addressOfOp.getGlobalName() : llvm::StringRef();
            }

            // a copy of it on the heap, its absent members' slots patched
            while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
            {
                value = castOp.getIn();
            }

            if (auto newOp = value.getDefiningOp<mlir_ts::NewOp>())
            {
                llvm::StringRef copied;
                for (auto *user : newOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    auto loadOp = storeOp && storeOp.getReference() == newOp.getResult()
                                      ? storeOp.getValue().getDefiningOp<mlir_ts::LoadOp>()
                                      : mlir_ts::LoadOp();
                    auto addressOfOp = loadOp ? addressOfThroughCasts(loadOp.getReference()) : mlir_ts::AddressOfOp();
                    if (addressOfOp && vtables.count(addressOfOp.getGlobalName()))
                    {
                        if (!copied.empty())
                        {
                            return {};
                        }

                        copied = addressOfOp.getGlobalName();
                    }
                }

                return copied;
            }

            return {};
        };

        // an object's class vtable read at run time: `ts.VTableOffsetRef(ts.Load(ts.PropertyRef(object, 0)), k)`
        auto readFromClass = [](mlir::Value value) {
            auto offsetRefOp = value.getDefiningOp<mlir_ts::VTableOffsetRefOp>();
            auto loadOp = offsetRefOp ? offsetRefOp.getVtable().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
            auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
            return propertyRefOp && propertyRefOp.getPosition() == 0 &&
                   mlir::isa<mlir_ts::ClassType>(propertyRefOp.getObjectRef().getType());
        };

        // what reads a vtable's address: making an interface value, a class vtable's slot, the copy
        std::function<bool(mlir::Value)> onlyMakesValues = [&](mlir::Value address) {
            return llvm::all_of(address.getUses(), [&](mlir::OpOperand &use) {
                auto *user = use.getOwner();
                if (auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(user))
                {
                    return newInterfaceOp.getInterfaceVTable() == use.get();
                }

                if (mlir::isa<mlir_ts::CastOp>(user))
                {
                    return onlyMakesValues(user->getResult(0));
                }

                if (mlir::isa<mlir_ts::InsertPropertyOp>(user))
                {
                    return use.getOperandNumber() == 0 && user->getParentOfType<mlir_ts::GlobalOp>() != nullptr;
                }

                // the copy: read whole, and stored into a block `ts.New` made
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    return llvm::all_of(loadOp->getUsers(), [&](mlir::Operation *loadUser) {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(loadUser);
                        return storeOp && storeOp.getValue() == loadOp.getResult() &&
                               storeOp.getReference().getDefiningOp<mlir_ts::NewOp>();
                    });
                }

                return false;
            });
        };

        // An interface value made out of another one reaches the other's members: both are open. One
        // made out of `any` or an opaque pointer may be any interface value put into one: every
        // interface so boxed is open with it, and every interface at all when it is untraceable.
        auto allOpen = false;
        llvm::StringSet<> boxed;
        llvm::StringSet<> unboxedInto;
        auto unboxedFunction = false;
        // the interfaces a cast makes a value of out of another interface's or an opaque one, and
        // the ones it came from: closed over their members' types below
        llvm::StringSet<> castOpened;
        // interface -> the interfaces its members' types mention: its vtables' slots and what is
        // read out of its values (`ts.InterfaceSymbolRef`), function signatures and class fields
        // included
        llvm::StringMap<llvm::StringSet<>> memberInterfaces;
        auto isOpaque = [](mlir::Type type) { return mlir::isa<mlir_ts::AnyType, mlir_ts::UnknownType, mlir_ts::OpaqueType>(type); };
        module.walk([&](mlir::Operation *op) {
            if (auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(op))
            {
                auto name = mlir::cast<mlir_ts::InterfaceType>(newInterfaceOp.getType()).getName().getValue();
                auto vtable = newInterfaceOp.getInterfaceVTable();
                if (auto found = vtableOf(vtable); !found.empty())
                {
                    madeWith[name].insert(found);
                }
                else if (readFromClass(vtable))
                {
                    // any class's vtable for the interface (classHeld, all in its families below)
                    if (!classHeld.count(name))
                    {
                        open.insert(name);
                    }
                }
                else if (auto extractOp = vtable.getDefiningOp<mlir_ts::ExtractInterfaceVTableOp>())
                {
                    castOpened.insert(name);
                    valueInterfaces(extractOp.getInterfaceVal().getType(), castOpened);
                }
                else
                {
                    allOpen = true;
                }

                return;
            }

            if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(op))
            {
                auto in = castOp.getIn().getType();
                auto out = castOp.getType();
                if (isOpaque(out))
                {
                    valueInterfaces(in, boxed);
                }

                if (castMakesInterface(in, out))
                {
                    valueInterfaces(out, castOpened);
                    valueInterfaces(in, castOpened);
                    if (isOpaque(in))
                    {
                        valueInterfaces(out, unboxedInto);
                    }
                }

                // a function value reinterpreted carries the interface values of its parameters and
                // results across unseen (review6 x1c, x1r)
                llvm::StringSet<> signatures;
                auto boxing = false;
                if (castReinterpretsFunction(in, out, signatures, boxing))
                {
                    openAll(signatures);
                    if (isOpaque(in))
                    {
                        unboxedFunction = true;
                    }
                }
                else if (boxing)
                {
                    for (auto &entry : signatures)
                    {
                        boxed.insert(entry.getKey());
                    }
                }

                return;
            }

            if (auto interfaceRefOp = mlir::dyn_cast<mlir_ts::InterfaceRefOp>(op))
            {
                if (!interfaceRefOp->use_empty())
                {
                    open.insert(interfaceRefOp.getIdentifier());
                }

                return;
            }

            if (auto symbolRefOp = mlir::dyn_cast<mlir_ts::InterfaceSymbolRefOp>(op))
            {
                if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(symbolRefOp.getInterfaceVal().getType()))
                {
                    memberTypeInterfaces(symbolRefOp.getType(), memberInterfaces[interfaceType.getName().getValue()]);
                }

                return;
            }

            if (auto addressOfOp = mlir::dyn_cast<mlir_ts::AddressOfOp>(op))
            {
                if (auto found = vtables.find(addressOfOp.getGlobalName()); found != vtables.end() && !onlyMakesValues(addressOfOp.getResult()))
                {
                    found->second.written = true;
                }
            }
        });

        // seen from another module: a word of its exported declarations, or in a reachable type. An
        // executable's exports are seen by no other module (USER decision, spec 31): only what it
        // imports - a declared function, an imported global - comes from outside.
        if (!executable)
        {
            exportWords = exportedWords(module);
        }

        auto &words = exportWords;
        llvm::StringSet<> exposed;
        llvm::DenseSet<mlir::Type> seen;
        for (auto &entry : functions)
        {
            // a `public` method of a class no other module can name is not one: only a declaration
            // re-parsed from `__decls` names a function of another module
            auto exported = !executable && (entry.second->hasAttr("export") || entry.second->hasAttr(DLL_EXPORT));
            if (entry.second.isDeclaration() || exported)
            {
                anyInterfaces(entry.second.getFunctionType(), exposed, seen, &nameableClasses);
            }
        }

        for (auto &entry : globals)
        {
            auto globalOp = entry.second;
            // another module's global, declared here with no initializer (an import of a .ts module
            // carries `export`)
            auto imported = globalOp->hasAttr("import") || globalOp->hasAttr(DLL_IMPORT) ||
                            globalOp.getInitializerRegion().empty() || globalOp.getInitializerRegion().front().empty();
            auto exported = !globalOp.isPrivate() || globalOp->hasAttr("export") || globalOp->hasAttr(DLL_EXPORT);
            if (imported || (exported && !executable))
            {
                anyInterfaces(globalOp.getType(), exposed, seen, &nameableClasses);
            }
        }

        openAll(exposed);
        if (!unboxedInto.empty() || unboxedFunction)
        {
            openAll(boxed);
            for (auto &entry : boxed)
            {
                castOpened.insert(entry.getKey());
            }
        }

        // An interface value made out of another one (or out of `any`) is called with the other's
        // members: `HasF2 { f: (o: Op2, ...) }` seen as `HasF { f: (o: Op, ...) }` hands an `Op` to a
        // function that calls through `Op2` (review6r1 v5, v11f). So every interface either side's
        // members mention is open too, and theirs, recursively (controller ruling, spec 31).
        //
        // A class's vtables for its interfaces are its members' families whether or not this module
        // makes an interface value of the class (a method called directly then keeps its facts), and
        // their slots' types are the interface's members' types: an interface whose values are all
        // made from class objects is read through them (review6r2 d1-d3).
        for (auto &[name, held] : classHeld)
        {
            madeWith[name].insert(held.begin(), held.end());
        }

        for (auto &[name, made] : madeWith)
        {
            for (auto vtableName : made)
            {
                if (auto found = vtables.find(vtableName); found != vtables.end() && found->second.type)
                {
                    for (auto [position, field] : llvm::enumerate(found->second.type.getFields()))
                    {
                        if (position > 0)
                        {
                            memberTypeInterfaces(field.type, memberInterfaces[name]);
                        }
                    }
                }
            }
        }

        llvm::SmallVector<std::string> work;
        for (auto &entry : castOpened)
        {
            work.push_back(entry.getKey().str());
        }

        llvm::StringSet<> expanded;
        while (!work.empty())
        {
            auto name = work.pop_back_val();
            if (!expanded.insert(name).second)
            {
                continue;
            }

            open.insert(name);
            for (auto &member : memberInterfaces[name])
            {
                work.push_back(member.getKey().str());
            }
        }

        auto isOpen = [&](llvm::StringRef name) {
            // a namespaced interface is declared by its last name
            auto last = name.rsplit('.').second;
            return allOpen || open.contains(name) || words.contains(name) || (!last.empty() && words.contains(last));
        };

        // A vtable this pass could not read, or one no value of any interface is made with, that is
        // named for an interface (`<class>.<I>..vtbl`, `<I>.<n>..vtbl`): the interface may have an
        // implementation outside its families. Belt and braces - every way to make an interface
        // value is traced above - but a missed member would get its family's facts.
        llvm::DenseSet<llvm::StringRef> made;
        for (auto &entry : madeWith)
        {
            made.insert(entry.second.begin(), entry.second.end());
        }

        llvm::SmallVector<llvm::StringRef> loose(unread.begin(), unread.end());
        for (auto &entry : vtables)
        {
            if (!made.contains(entry.getKey()))
            {
                loose.push_back(entry.getKey());
            }
        }

        for (auto &entry : madeWith)
        {
            auto name = entry.getKey().str();
            if (llvm::any_of(loose, [&](llvm::StringRef vtableName) {
                    return vtableName.ends_with("." + name + "..vtbl") || vtableName.starts_with(name + ".");
                }))
            {
                open.insert(entry.getKey());
            }
        }

        // the families
        llvm::DenseSet<llvm::StringRef> tied;
        for (auto &[name, made] : madeWith)
        {
            // a class vtable's slot holding a vtable this pass did not read (another module's)
            auto allRead = llvm::all_of(made, [&](llvm::StringRef vtableName) { return vtables.count(vtableName); });
            auto closed = !isOpen(name) && allRead;
            for (auto vtableName : made)
            {
                if (!vtables.count(vtableName))
                {
                    continue;
                }

                if (allRead)
                {
                    interfaceVTableTypes[name].push_back(vtables[vtableName].type);
                }

                tied.insert(vtableName);
                auto &vtable = vtables[vtableName];
                for (auto [position, field] : llvm::enumerate(vtable.type.getFields()))
                {
                    if (position == 0)
                    {
                        continue;
                    }

                    auto &family = interfaceFamilies[{name.str(), static_cast<int64_t>(position)}];
                    auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id);
                    auto member = vtable.slots.lookup(position);
                    auto funcOp = member ? functions.lookup(member.getValue()) : mlir_ts::FuncOp();
                    if (!family.field)
                    {
                        family.field = id;
                    }

                    // a member that is not private (a `public` method) is open (findOpen): it has no
                    // facts to give, and gives the family none
                    family.closed = family.closed && closed && !vtable.written && id && family.field == id && funcOp &&
                                    !funcOp.isDeclaration() && !declaresCallerFacts(funcOp);
                    if (member && !llvm::is_contained(family.members, member))
                    {
                        family.members.push_back(member);
                    }

                    // the same slot as a field: an object literal's field of function type
                    auto &fieldFamily = interfaceFieldFamilies[{name.str(), static_cast<int64_t>(position)}];
                    auto target = vtable.fields.lookup(position);
                    auto refType = mlir::dyn_cast<mlir_ts::RefType>(field.type);
                    if (!fieldFamily.field)
                    {
                        fieldFamily.field = id;
                    }

                    fieldFamily.closed = fieldFamily.closed && closed && !vtable.written && id && fieldFamily.field == id &&
                                         target.first && refType && mlir::isa<mlir_ts::FunctionType>(refType.getElementType());
                    if (target.first && !llvm::is_contained(fieldFamily.targets, target))
                    {
                        fieldFamily.targets.push_back(target);
                    }
                }
            }
        }

        collectFieldFunctions(module);

        // what an open family holds, and a vtable no interface value is made with, is reached
        // through calls that are not resolved
        auto reached = [&](mlir::StringAttr member) {
            if (auto funcOp = member ? functions.lookup(member.getValue()) : mlir_ts::FuncOp())
            {
                interfaceReached.insert(funcOp);
            }
        };

        for (auto &[key, family] : interfaceFamilies)
        {
            if (!family.closed)
            {
                llvm::for_each(family.members, reached);
            }
        }

        for (auto &entry : vtables)
        {
            if (!tied.contains(entry.getKey()))
            {
                for (auto &slot : entry.second.slots)
                {
                    reached(slot.second);
                }
            }
        }
    }
    static constexpr const char *INTERFACE_REACHED_WHY =
        "it is reached through an interface, whose calls are not given its facts";

    // The parameters a function's result borrows and the parameters it keeps (spec 2.4); absent is
    // rc's convention. A plain borrow names one parameter; a borrowing record (`record`, spec 32)
    // any number, and its caller owns the box.
    struct Facts
    {
        llvm::SmallVector<int32_t> borrows;
        llvm::SmallVector<int32_t> owned;
        bool record = false;

        bool empty() const
        {
            return borrows.empty() && owned.empty();
        }

        bool operator==(const Facts &other) const
        {
            return borrows == other.borrows && owned == other.owned && record == other.record;
        }

        bool operator!=(const Facts &other) const
        {
            return !(*this == other);
        }
    };

    // Each function's facts, from its body (local) and as its callers see them (effective: closed,
    // and agreed by every family it is in).
    llvm::DenseMap<mlir::Operation *, Facts> factsLocal;
    llvm::DenseMap<mlir::Operation *, Facts> factsEffective;
    // The parameters each body takes into something - a field, an element, a global, a keeping call,
    // an owning local it assigns (localOwnedParams). A kept parameter that is not one of them is
    // released on the paths that do not move it (OWN_PARAMS_RELEASED_ATTR_NAME, spec 33).
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int32_t>> takenParams;

    // ---- Facts a signature says (`@keeps`, `@borrows`, `@lends`) ----
    //
    // A function that says them has them whoever calls it: a caller in another module reads them
    // from the declaration it re-parses. Its body is checked against them instead (checkDeclared).

    static llvm::ArrayRef<int32_t> declared(mlir::Operation *funcOp, llvm::StringRef name)
    {
        if (auto attr = funcOp->getAttrOfType<mlir::DenseI32ArrayAttr>(name))
        {
            return attr.asArrayRef();
        }

        return {};
    }

    static bool declaresFacts(mlir::Operation *funcOp)
    {
        return funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) ||
               funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME);
    }

    // The facts `@keeps` and `@lends` say (Facts): what callers move in, and what the result is a
    // borrow of.
    static bool declaresCallerFacts(mlir::Operation *funcOp)
    {
        return funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME) || funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME);
    }

    // The parameters `@lends` names: one for a plain borrow, any number for a borrowing record
    // (spec 32).
    static llvm::SmallVector<int32_t> declaredLends(mlir::Operation *funcOp)
    {
        auto lends = declared(funcOp, OWN_DECLARED_LENDS_ATTR_NAME);
        llvm::SmallVector<int32_t> sorted(lends.begin(), lends.end());
        llvm::sort(sorted);
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        return sorted;
    }

    // What every function a call names declares under `name`, when they all declare the same;
    // nothing when one of them does not.
    llvm::SmallVector<int32_t> declaredByCallees(const Callees &callees, llvm::StringRef name)
    {
        llvm::SmallVector<int32_t> agreed;
        for (auto [index, symbol] : llvm::enumerate(callees.named))
        {
            // a function, or the global holding the address of one a library exports (a dynamic
            // import), which MLIRGen gave the declaration's markers
            mlir::Operation *declaring = functions.lookup(symbol.getValue());
            if (!declaring)
            {
                declaring = globals.lookup(symbol.getValue());
            }

            if (!declaring || !declaring->hasAttr(name))
            {
                return {};
            }

            auto indices = declared(declaring, name);
            if (index == 0)
            {
                agreed.assign(indices.begin(), indices.end());
            }
            else if (llvm::ArrayRef<int32_t>(agreed) != indices)
            {
                return {};
            }
        }

        return agreed;
    }

    // The parameters, for an error, by the names MLIRGen recorded beside the marker.
    static std::string describeParams(mlir_ts::FuncOp funcOp, llvm::ArrayRef<int32_t> indices)
    {
        if (indices.empty())
        {
            return "nothing";
        }

        auto names = funcOp->getAttrOfType<mlir::ArrayAttr>(OWN_DECLARED_PARAM_NAMES_ATTR_NAME);
        std::string text;
        llvm::raw_string_ostream os(text);
        for (auto [position, index] : llvm::enumerate(indices))
        {
            os << (position ? ", " : "") << "'";
            if (names && static_cast<size_t>(index) < names.size())
            {
                os << mlir::cast<mlir::StringAttr>(names[index]).getValue();
            }
            else
            {
                os << "#" << index;
            }

            os << "'";
        }

        return os.str();
    }

    // Functions this module defines that `@borrows`, and those whose state was found to borrow.
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<int32_t>> borrowsFound;

    // The argument of `funcOp` that `value` is, or a field of: its own value, its slot's, or a
    // read out of either. -1 for anything else.
    static int32_t parameterOf(mlir::Value value, mlir_ts::FuncOp funcOp)
    {
        auto root = rootOf(value);
        if (auto loadOp = root.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto propertyRefOp = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                return parameterOf(propertyRefOp->getOperand(0), funcOp);
            }

            auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            int index = -1;
            if (varOp && isParameterSlot(varOp, index) && varOp->getParentOfType<mlir_ts::FuncOp>() == funcOp)
            {
                return index;
            }

            return -1;
        }

        auto argument = mlir::dyn_cast<mlir::BlockArgument>(root);
        if (argument && argument.getOwner()->isEntryBlock() && argument.getOwner()->getParentOp() == funcOp.getOperation())
        {
            return static_cast<int32_t>(argument.getArgNumber());
        }

        return -1;
    }

    // ---- The iterator protocol ----
    //
    // `it.next()` returns a `{value, done}` record, and under own the caller owns what it holds,
    // whoever implements it: a generator's `next` moves what it yields into the record (a yield of
    // a borrow is an error in its body), and any other `next` is a function whose callers this pass
    // cannot all see, which may not return a borrow. So the result is fresh even when the call goes
    // through the `Iterator` interface, or into another module - the default library's `matchAll`.

    // A call of a function named `next` - read through an interface, out of a field, or by its
    // symbol - whose one result is a `{value, done}` record.
    static bool isIteratorNext(mlir::Operation *op)
    {
        if (op->getNumResults() != 1)
        {
            return false;
        }

        auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(op->getResult(0).getType());
        if (!tupleType || tupleType.getFields().size() != 2)
        {
            return false;
        }

        auto fieldName = [&](int index) {
            auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(tupleType.getFields()[index].id);
            return id ? id.getValue() : llvm::StringRef();
        };
        if (fieldName(0) != "value" || fieldName(1) != "done")
        {
            return false;
        }

        auto isNext = [](llvm::StringRef name) { return name == "next" || name.ends_with(".next"); };
        if (auto callee = directCallee(op))
        {
            return isNext(callee.getValue());
        }

        auto function = calleeValue(op);
        if (!function)
        {
            return false;
        }

        if (auto getMethodOp = function.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            function = getMethodOp.getBoundFunc();
        }

        if (auto boundOp = function.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            function = boundOp.getFunc();
        }

        if (auto loadOp = function.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto refOp = loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>())
            {
                auto identifier = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
                return identifier && isNext(identifier.getValue());
            }

            if (auto propertyRefOp = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>())
            {
                auto storageType = storageTupleOfType(propertyRefOp->getOperand(0).getType());
                if (!storageType || propertyRefOp.getPosition() >= storageType.getFields().size())
                {
                    return false;
                }

                auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(storageType.getFields()[propertyRefOp.getPosition()].id);
                return id && isNext(id.getValue());
            }

            return false;
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::ThisVirtualSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::VirtualSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        if (auto refOp = function.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            return isNext(refOp.getIdentifier());
        }

        return false;
    }

    // A generator another module made: its state object's type is only its shape here, and what each
    // field owns - a local, a yielded value now the caller's, what it borrows from its arguments - is
    // known in that module alone. A release routine made here from the shape would give back what is
    // not the state's. So this module's gives back none of it: only the `.captured` box's cells, and
    // their values too unless the maker says it borrows them (`@borrows`). What the generator's own
    // locals still hold when it is given up early leaks rather than being freed twice.
    void findImportedStates(mlir::ModuleOp module)
    {
        llvm::MapVector<mlir::Type, bool> states;
        for (auto &call : calls)
        {
            if (!makesGeneratorState(call.op) || (call.callees.known && !call.callees.funcs.empty()))
            {
                continue;
            }

            auto tupleType = storageTupleOfType(call.op->getResult(0).getType());
            auto borrows = !declaredByCallees(call.callees, OWN_DECLARED_BORROWS_ATTR_NAME).empty();
            states[tupleType] = states.lookup(tupleType) || borrows;
        }

        if (states.empty())
        {
            return;
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        if (auto existing = module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME))
        {
            entries.append(existing.begin(), existing.end());
        }

        for (auto &[type, borrows] : states)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            auto captured = fieldIndex(tupleType, CAPTURED_NAME);
            // the box first, when it borrows; -1 names no field, so a box that owns is released
            llvm::SmallVector<int32_t> fields{borrows && captured >= 0 ? captured : -1};
            for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
            {
                if (static_cast<int32_t>(index) != captured && ownsHeap(module.getLoc(), field.type))
                {
                    fields.push_back(static_cast<int32_t>(index));
                }
            }

            entries.push_back(mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, fields)}));
        }

        module->setAttr(OWN_BORROWING_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    // A call whose result is a generator's state object (an object whose fields are `.step` and
    // `next`): a generator function made it, a `ts.New` block it hands over (spec 18.1), which the
    // caller owns - in another module too, as the iteration that follows needs.
    static bool makesGeneratorState(mlir::Operation *op)
    {
        if (op->getNumResults() != 1 || !mlir::isa<mlir_ts::ObjectType>(op->getResult(0).getType()))
        {
            return false;
        }

        auto tupleType = storageTupleOfType(op->getResult(0).getType());
        if (!tupleType)
        {
            return false;
        }

        auto hasStep = false;
        auto hasNext = false;
        for (auto field : tupleType.getFields())
        {
            if (auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id))
            {
                hasStep = hasStep || id.getValue() == ".step";
                hasNext = hasNext || id.getValue() == "next";
            }
        }

        return hasStep && hasNext;
    }

    // storageTupleOf, without the pass's context: an object's storage, a reference's or a tuple
    static mlir_ts::TupleType storageTupleOfType(mlir::Type type)
    {
        if (auto objectType = mlir::dyn_cast<mlir_ts::ObjectType>(type))
        {
            if (auto storageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(objectType.getStorageType()))
            {
                return mlir_ts::TupleType::get(type.getContext(), storageType.getFields());
            }

            // an object a declaration names by its shape (another module's generator)
            return mlir::dyn_cast<mlir_ts::TupleType>(objectType.getStorageType());
        }

        if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
        {
            type = refType.getElementType();
        }
        else if (auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(type))
        {
            type = valueRefType.getElementType();
        }

        return mlir::dyn_cast<mlir_ts::TupleType>(type);
    }

    // ---- Fields a generator yields ----
    //
    // A generator's locals are fields of its state object, and `yield groups` puts what one holds into
    // the `{value, done}` record its `next` returns: a borrow of the state, which the caller cannot
    // keep. When the field is read for nothing but yields, and each value stored into it is yielded
    // before the generator can stop (later in the same block), the yield moves the value instead:
    // the read is fresh (OWN_FRESH_RESULT_ATTR_NAME) and the field owns nothing
    // (OWN_YIELDED_FIELDS_ATTR_NAME).

    // `ts.Store(value, ts.PropertyRef(record, i))`, where `record` is a local a `ts.Constant`
    // initializes: a field of a record built to be returned, a yield's.
    static bool isRecordFieldStore(mlir::Operation *user, mlir::Value value)
    {
        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
        auto propertyRefOp = storeOp && storeOp.getValue() == value
                                 ? storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>()
                                 : mlir_ts::PropertyRefOp();
        auto recordOp = propertyRefOp ? propertyRefOp->getOperand(0).getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        return recordOp && recordOp.getInitializer() && recordOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>();
    }

    void findYieldedFields(mlir::ModuleOp module)
    {
        // (state type, field) -> its reads and its stores
        struct FieldUses
        {
            llvm::SmallVector<mlir_ts::LoadOp> reads;
            llvm::SmallVector<mlir_ts::StoreOp> stores;
            bool other = false;
        };

        llvm::MapVector<std::pair<mlir::Type, int32_t>, FieldUses> fields;
        module.walk([&](mlir_ts::PropertyRefOp propertyRefOp) {
            auto tupleType = storageTupleOf(propertyRefOp->getOperand(0).getType());
            auto position = static_cast<int32_t>(propertyRefOp.getPosition());
            if (!tupleType || fieldIndex(tupleType, ".step") < 0 || position == fieldIndex(tupleType, CAPTURED_NAME) ||
                !mlir::isa<mlir_ts::RefType>(propertyRefOp.getType()) ||
                !ownsHeap(propertyRefOp.getLoc(), mlir::cast<mlir_ts::RefType>(propertyRefOp.getType()).getElementType()))
            {
                return;
            }

            auto &uses = fields[{tupleType, position}];
            for (auto *user : propertyRefOp->getUsers())
            {
                if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
                {
                    uses.reads.push_back(loadOp);
                }
                else if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == propertyRefOp.getResult())
                {
                    uses.stores.push_back(storeOp);
                }
                else if (!mlir::isa<mlir_ts::ReleaseSlotOp, mlir_ts::RetainSlotOp>(user))
                {
                    uses.other = true;
                }
            }
        });

        // a field the signature pass already found borrowing is the caller's, not the generator's
        llvm::DenseSet<std::pair<mlir::Type, int32_t>> borrowing;
        if (auto entries = module->getAttrOfType<mlir::ArrayAttr>(OWN_BORROWING_FIELDS_ATTR_NAME))
        {
            for (auto entry : entries.getAsRange<mlir::ArrayAttr>())
            {
                for (auto index : mlir::cast<mlir::DenseI32ArrayAttr>(entry[1]).asArrayRef())
                {
                    borrowing.insert({mlir::cast<mlir::TypeAttr>(entry[0]).getValue(), index});
                }
            }
        }

        llvm::MapVector<mlir::Type, llvm::SmallVector<int32_t>> yielded;
        for (auto &[key, uses] : fields)
        {
            if (uses.other || uses.reads.empty() || uses.stores.empty() || borrowing.contains(key))
            {
                continue;
            }

            // read only to be put into a yield's record
            auto onlyYielded = llvm::all_of(uses.reads, [](mlir_ts::LoadOp loadOp) {
                auto users = llvm::make_filter_range(loadOp->getUsers(), [](mlir::Operation *user) {
                    return !mlir::isa<mlir_ts::RetainOp, mlir_ts::DebugVariableOp>(user);
                });
                return llvm::hasSingleElement(users) && isRecordFieldStore(*users.begin(), loadOp.getResult());
            });

            // each value stored is yielded later in its block: nothing between stops the generator
            auto eachYielded = llvm::all_of(uses.stores, [&](mlir_ts::StoreOp storeOp) {
                return llvm::any_of(uses.reads, [&](mlir_ts::LoadOp loadOp) {
                    return loadOp->getBlock() == storeOp->getBlock() && storeOp->isBeforeInBlock(loadOp);
                });
            });

            if (!onlyYielded || !eachYielded)
            {
                continue;
            }

            yielded[key.first].push_back(key.second);
            for (auto loadOp : uses.reads)
            {
                loadOp->setAttr(OWN_FRESH_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        }

        if (yielded.empty())
        {
            return;
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        for (auto &[type, indices] : yielded)
        {
            entries.push_back(mlir::ArrayAttr::get(context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, indices)}));
        }

        module->setAttr(OWN_YIELDED_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    // A function that says `@borrows` and makes no state itself, but returns what a call that has
    // a bounded result gives it (`return this.value.matchAll(expr)`): it borrows the parameters
    // the call's bounded arguments are, or are read out of.
    void findBoundedForwarders()
    {
        llvm::DenseMap<mlir::Operation *, const Callees *> calleesOf;
        for (auto &call : calls)
        {
            calleesOf[call.op] = &call.callees;
        }

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() || !funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME) || borrowsFound.count(funcOp))
            {
                continue;
            }

            llvm::SmallVector<int32_t> found;
            auto forwards = true;
            for (auto value : returnedValues(funcOp))
            {
                if (holdsNoBlock(value))
                {
                    continue;
                }

                auto *call = rootOf(value).getDefiningOp();
                auto known = call ? calleesOf.find(call) : calleesOf.end();
                if (known == calleesOf.end())
                {
                    forwards = false;
                    break;
                }

                llvm::SmallVector<int32_t> bounded = declaredByCallees(*known->second, OWN_DECLARED_BORROWS_ATTR_NAME);
                if (bounded.empty() && known->second->funcs.size() == 1 && !known->second->viaCallback)
                {
                    auto made = declared(known->second->funcs.front(), OWN_RESULT_BOUNDED_ATTR_NAME);
                    bounded.assign(made.begin(), made.end());
                }

                if (bounded.empty())
                {
                    forwards = false;
                    break;
                }

                auto args = callArgs(call);
                for (auto index : bounded)
                {
                    auto param = static_cast<size_t>(index) < args.size() ? parameterOf(args[index], funcOp) : -1;
                    if (param < 0)
                    {
                        forwards = false;
                        break;
                    }

                    if (!llvm::is_contained(found, param))
                    {
                        found.push_back(param);
                    }
                }
            }

            if (forwards && !found.empty())
            {
                llvm::sort(found);
                borrowsFound[funcOp] = found;
            }
        }
    }

    // A body that does other than its signature says, and an override that says other than the
    // method it overrides: either way a caller that trusts the signature frees twice or leaks.
    void checkDeclared()
    {
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration())
            {
                continue;
            }

            if (funcOp->hasAttr(OWN_DECLARED_LENDS_ATTR_NAME) && funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
            {
                funcOp.emitError("'") << funcOp.getSymName()
                                      << "' cannot say both '@" LENDS_DECORATOR "' and '@" BORROWS_DECORATOR
                                         "': its caller owns a result that borrows, and nothing of one it is lent";
                signalPassFailure();
                continue;
            }

            if (declaresCallerFacts(funcOp))
            {
                auto local = factsLocal.lookup(funcOp);
                auto keeps = declared(funcOp, OWN_DECLARED_KEEPS_ATTR_NAME);
                auto lends = declaredLends(funcOp);
                if (local.borrows != lends)
                {
                    if (lends.empty())
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' returns a borrow of "
                                              << describeParams(funcOp, local.borrows)
                                              << ", which it has to say with '@" LENDS_DECORATOR "'";
                    }
                    else if (local.borrows.empty())
                    {
                        funcOp.emitError("'@" LENDS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, lends)
                            << ", but its body does not return a borrow of it";
                    }
                    else
                    {
                        funcOp.emitError("'@" LENDS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, lends)
                            << ", but its body returns a borrow of " << describeParams(funcOp, local.borrows);
                    }

                    signalPassFailure();
                }
                else if (llvm::ArrayRef<int32_t>(local.owned) != keeps)
                {
                    if (funcOp->hasAttr(OWN_DECLARED_KEEPS_ATTR_NAME))
                    {
                        funcOp.emitError("'@" KEEPS_DECORATOR "' of '")
                            << funcOp.getSymName() << "' names " << describeParams(funcOp, keeps) << ", but its body keeps "
                            << describeParams(funcOp, local.owned);
                    }
                    else
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' keeps " << describeParams(funcOp, local.owned)
                                              << ", which it has to say with '@" KEEPS_DECORATOR "'";
                    }

                    signalPassFailure();
                }
            }

            if (funcOp->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME))
            {
                auto says = declared(funcOp, OWN_DECLARED_BORROWS_ATTR_NAME);
                auto found = borrowsFound.find(funcOp);
                if (found == borrowsFound.end())
                {
                    funcOp.emitError("'@" BORROWS_DECORATOR "' of '")
                        << funcOp.getSymName() << "' names " << describeParams(funcOp, says)
                        << ", but it neither makes a generator whose state borrows its arguments nor returns one";
                    signalPassFailure();
                }
                else if (llvm::ArrayRef<int32_t>(found->second) != says)
                {
                    funcOp.emitError("'@" BORROWS_DECORATOR "' of '")
                        << funcOp.getSymName() << "' names " << describeParams(funcOp, says)
                        << ", but the state it makes borrows " << describeParams(funcOp, found->second);
                    signalPassFailure();
                }
            }
        }

        // the members of a family a declaring method is in: each says the same
        for (auto &[key, members] : families)
        {
            mlir_ts::FuncOp declaring;
            for (auto member : members)
            {
                if (auto funcOp = functions.lookup(member.getValue()); funcOp && declaresFacts(funcOp))
                {
                    declaring = funcOp;
                    break;
                }
            }

            if (!declaring)
            {
                continue;
            }

            for (auto member : members)
            {
                auto funcOp = functions.lookup(member.getValue());
                if (!funcOp || funcOp == declaring)
                {
                    continue;
                }

                for (auto *name : {OWN_DECLARED_KEEPS_ATTR_NAME, OWN_DECLARED_BORROWS_ATTR_NAME, OWN_DECLARED_LENDS_ATTR_NAME})
                {
                    if (funcOp->hasAttr(name) != declaring->hasAttr(name) || declared(funcOp, name) != declared(declaring, name))
                    {
                        funcOp.emitError("'") << funcOp.getSymName() << "' overrides '" << declaring.getSymName()
                                              << "', and has to say the same '@" KEEPS_DECORATOR "', '@" BORROWS_DECORATOR
                                                 "' and '@" LENDS_DECORATOR "'";
                        signalPassFailure();
                        break;
                    }
                }
            }
        }
    }

    // Functions that destroy nothing their caller can reach.
    llvm::DenseSet<mlir::Operation *> noDrops;

    // A class's vtable is a global named `<class>..vtbl` that holds its `..instanceOf`. An
    // interface's vtable for a class (`B.I..vtbl`) holds only the methods and is not one: a call
    // through an interface does not name what it reaches.
    void collectClassVTables(mlir::ModuleOp module)
    {
        llvm::SmallVector<std::pair<llvm::StringRef, llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>>>> vtables;
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto entries = classVTableEntries(globalOp);
            if (!entries.empty())
            {
                auto name = globalOp.getSymName();
                classNames.insert(name.drop_back(llvm::StringRef("..vtbl").size()));
                vtables.push_back({name, std::move(entries)});
            }
        });

        for (auto &[name, entries] : vtables)
        {
            for (auto [position, symbol] : entries)
            {
                families[{position, methodName(symbol.getValue())}].push_back(symbol);
            }
        }
    }

    // Does a method reached through an open interface keep a parameter or return a borrow
    // (interfaceReached)? Then its callers through the interface cannot be told, and it has no facts
    // at all.
    bool factsThroughInterface(mlir::Operation *funcOp, const Facts &facts)
    {
        return !facts.empty() && interfaceReached.contains(funcOp);
    }

    mlir_ts::TupleType storageTupleOf(mlir::Type type)
    {
        return own_facts::storageTupleOf(&getContext(), type);
    }

    // The functions a class vtable holds, by position; empty for any other global.
    static llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>> classVTableEntries(mlir_ts::GlobalOp globalOp)
    {
        llvm::SmallVector<std::pair<int64_t, mlir::StringAttr>> entries;
        if (!globalOp.getSymName().ends_with("..vtbl"))
        {
            return entries;
        }

        auto isClass = false;
        globalOp.getInitializerRegion().walk([&](mlir_ts::InsertPropertyOp insertOp) {
            auto symbolRefOp = insertOp.getValue().getDefiningOp<mlir_ts::SymbolRefOp>();
            if (!symbolRefOp || insertOp.getPosition().size() != 1)
            {
                return;
            }

            auto symbol = symbolRefOp.getIdentifierAttr().getAttr();
            isClass = isClass || symbol.getValue().ends_with("..instanceOf");
            entries.push_back({insertOp.getPosition()[0], symbol});
        });

        if (!isClass)
        {
            entries.clear();
        }

        return entries;
    }

    static bool isClassVTableEntry(mlir::Operation *op)
    {
        auto globalOp = op->getParentOfType<mlir_ts::GlobalOp>();
        return globalOp && !classVTableEntries(globalOp).empty();
    }

    // `B.get` -> `get`: the symbol with the longest class name that prefixes it taken off. An
    // override keeps its base's method name, which is what puts both in one family.
    std::string methodName(llvm::StringRef symbol)
    {
        size_t best = 0;
        for (auto &entry : classNames)
        {
            auto name = entry.getKey();
            if (name.size() > best && symbol.size() > name.size() && symbol.starts_with(name) &&
                symbol[name.size()] == '.')
            {
                best = name.size() + 1;
            }
        }

        return symbol.drop_front(best).str();
    }

    llvm::SmallVector<mlir::StringAttr, 2> familyOf(int64_t index, mlir::StringAttr identifier)
    {
        llvm::SmallVector<mlir::StringAttr, 2> members;
        if (auto found = families.find({index, methodName(identifier.getValue())}); found != families.end())
        {
            members = found->second;
        }

        if (!llvm::is_contained(members, identifier))
        {
            members.push_back(identifier);
        }

        return members;
    }

    // Adds `symbol`'s function to `callees`; false when it is not defined in this module.
    bool addDefined(mlir::StringAttr symbol, Callees &callees)
    {
        auto funcOp = functions.lookup(symbol.getValue());
        if (!funcOp || funcOp.isDeclaration())
        {
            return false;
        }

        if (!llvm::is_contained(callees.funcs, funcOp))
        {
            callees.funcs.push_back(funcOp);
        }

        return true;
    }

    // A function declared here and defined in a library that exported its `__own_no_drops`: one
    // linked with this module, through its import library. Or a C function declared
    // `@nodrops` (OWN_DECLARED_NO_DROPS_ATTR_NAME), or one whose parameters and result hold no
    // heap block (scalarDeclaration).
    bool addImported(mlir::StringAttr symbol, Callees &callees)
    {
        auto funcOp = functions.lookup(symbol.getValue());
        if (!funcOp || !funcOp.isDeclaration() ||
            !(importedNoDrops.contains(symbol.getValue()) || funcOp->hasAttr(OWN_DECLARED_NO_DROPS_ATTR_NAME) ||
              (scalarDeclaration(funcOp) && !functionsEscape)))
        {
            return false;
        }

        callees.imported = true;
        callees.importedName = symbol.getValue();
        return true;
    }

    // A number, a boolean, a character, or an enum or a literal of one: a value that holds no block
    // and no function of this module, so given to foreign code it hands it neither. Not an opaque
    // pointer, a reference, a string, a class, an interface, an object, a tuple, an array, a union
    // with such a member, a function, `any` or `unknown`.
    static bool scalarType(mlir::Type type)
    {
        if (auto literalType = mlir::dyn_cast<mlir_ts::LiteralType>(type))
        {
            return scalarType(literalType.getElementType());
        }

        if (auto enumType = mlir::dyn_cast<mlir_ts::EnumType>(type))
        {
            return scalarType(enumType.getElementType());
        }

        return mlir::isa<mlir_ts::NumberType, mlir_ts::BooleanType, mlir_ts::ByteType, mlir_ts::CharType, mlir::IntegerType,
                         mlir::FloatType, mlir::IndexType>(type);
    }

    // Every parameter and result scalar (a void result too), and no rest parameter.
    static bool scalarSignature(mlir_ts::FunctionType functionType)
    {
        return !functionType.isVarArg() && llvm::all_of(functionType.getInputs(), scalarType) &&
               llvm::all_of(functionType.getResults(),
                            [](mlir::Type type) { return mlir::isa<mlir_ts::VoidType>(type) || scalarType(type); });
    }

    // A body-less `declare function` of C (no `export` or import marker: not another TypeScript
    // module's, whose declarations carry `export`) whose parameters and result are scalar
    // (scalarType): given no block of this module, it can destroy none (controller ruling, spec 31:
    // `sqrt`, `floor`, `pow`, `fabs`). The risk is a C function that calls back into the module
    // (`signal(15, f as Opaque)` then `raise(15)`, review6r1 c3), which it can only do with a
    // function it was given - so the rule is off in a module whose foreign surface could hand it
    // one (functionsEscape) - or an exported function reached by name, which a `@nodrops`
    // declaration takes on as well.
    static bool scalarDeclaration(mlir_ts::FuncOp funcOp)
    {
        return funcOp.isDeclaration() && !funcOp->hasAttr("export") && !funcOp->hasAttr("import") &&
               !funcOp->hasAttr(DLL_EXPORT) && !funcOp->hasAttr(DLL_IMPORT) && scalarSignature(funcOp.getFunctionType());
    }

    // `ts.Load(ts.AddressOf @f)`, where the global `f` holds the address of a function exported
    // by a library loaded at run time, looked up by name: that function, if the library exported
    // its `__own_no_drops`.
    bool addImportedByAddress(mlir::Value callee, Callees &callees)
    {
        auto loadOp = callee.getDefiningOp<mlir_ts::LoadOp>();
        auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
        if (!addressOfOp)
        {
            return false;
        }

        auto globalOp = globals.lookup(addressOfOp.getGlobalName());
        if (!globalOp || writtenGlobals.contains(addressOfOp.getGlobalName()))
        {
            return false;
        }

        // the global is never anything but that address: `Cast(SearchForAddressOfSymbol(name))`
        auto &region = globalOp.getInitializerRegion();
        if (!region.hasOneBlock())
        {
            return false;
        }

        auto resultOp = mlir::dyn_cast<mlir_ts::GlobalResultOp>(region.front().getTerminator());
        auto castOp = resultOp && resultOp->getNumOperands() == 1
                          ? resultOp->getOperand(0).getDefiningOp<mlir_ts::CastOp>()
                          : mlir_ts::CastOp();
        auto searchOp = castOp ? castOp.getIn().getDefiningOp<mlir_ts::SearchForAddressOfSymbolOp>()
                               : mlir_ts::SearchForAddressOfSymbolOp();
        auto nameOp = searchOp ? searchOp->getOperand(0).getDefiningOp<mlir_ts::ConstantOp>() : mlir_ts::ConstantOp();
        auto name = nameOp ? mlir::dyn_cast<mlir::StringAttr>(nameOp.getValue()) : mlir::StringAttr();
        if (!name || !importedNoDrops.contains(name.getValue()))
        {
            return false;
        }

        callees.imported = true;
        callees.importedName = name.getValue();
        return true;
    }

    Callees resolve(mlir::Operation *op)
    {
        Callees callees;
        if (auto callee = directCallee(op))
        {
            callees.named.push_back(callee.getAttr());
            callees.known = addDefined(callee.getAttr(), callees) || addImported(callee.getAttr(), callees);
        }
        else if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::InvokeOp>(op))
        {
            resolveValue(calleeValue(op), callees);
        }

        if (!callees.known)
        {
            callees.funcs.clear();
        }

        return callees;
    }

    // The function value a call goes through: a method of a class, virtual or not, a symbol, or
    // the `.instanceOf` slot. Anything else - a closure, a function read from a field, an
    // interface's method - is unknown.
    void resolveValue(mlir::Value callee, Callees &callees)
    {
        // `ts.Load(ts.AddressOf @g)`: a dynamic import's global, which may say what the function
        // keeps; a method of a class imported so is bound to its object first
        auto function = callee;
        if (auto getMethodOp = function.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            function = getMethodOp.getBoundFunc();
        }

        if (auto boundOp = function.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            function = boundOp.getFunc();
        }

        auto loadOp = function.getDefiningOp<mlir_ts::LoadOp>();
        if (auto addressOfOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
            addressOfOp && !writtenGlobals.contains(addressOfOp.getGlobalName()))
        {
            callees.named.push_back(addressOfOp.getGlobalNameAttr().getAttr());
        }

        if (addImportedByAddress(callee, callees))
        {
            callees.known = true;
            return;
        }

        if (auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            callee = getMethodOp.getBoundFunc();
        }

        auto *def = callee.getDefiningOp();
        mlir::StringAttr identifier;
        int64_t index = -1;
        if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
            index = refOp.getIndex();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::VirtualSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
            index = refOp.getIndex();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (auto refOp = mlir::dyn_cast_or_null<mlir_ts::SymbolRefOp>(def))
        {
            identifier = refOp.getIdentifierAttr().getAttr();
        }
        else if (isInstanceOfSlot(def))
        {
            callees.known = true;
            callees.instanceOf = true;
            return;
        }
        else if (auto symbol = fieldFunctions.of(def))
        {
            callees.known = addDefined(symbol, callees);
            callees.fieldFunction = callees.known;
            return;
        }
        else if (auto interfaceRefOp = mlir::dyn_cast_or_null<mlir_ts::InterfaceSymbolRefOp>(def))
        {
            resolveInterfaceSlot(interfaceRefOp, interfaceFamilies, callees);
            return;
        }
        else if (auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def))
        {
            resolveCallback(loadOp, callees);
            return;
        }
        else if (auto boundOp = mlir::dyn_cast_or_null<mlir_ts::CreateBoundFunctionOp>(def))
        {
            // a function-typed field of an interface, called: `ts.CreateBoundFunction(
            // ts.ExtractInterfaceThis(i), ts.Load(ts.InterfaceSymbolRef(i, f)))`
            auto loadOp = boundOp.getFunc().getDefiningOp<mlir_ts::LoadOp>();
            auto refOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>() : mlir_ts::InterfaceSymbolRefOp();
            auto thisOp = boundOp.getThisVal().getDefiningOp<mlir_ts::ExtractInterfaceThisOp>();
            if (refOp && isInterfaceField(refOp.getResult()) && thisOp && thisOp.getInterfaceVal() == refOp.getInterfaceVal())
            {
                resolveInterfaceSlot(refOp, interfaceFieldFamilies, callees);
            }

            return;
        }

        if (!identifier)
        {
            return;
        }

        if (index < 0)
        {
            callees.named.push_back(identifier);
            callees.known = addDefined(identifier, callees) || addImported(identifier, callees);
            return;
        }

        callees.named = familyOf(index, identifier);
        resolveFamily(index, identifier, callees);
    }

    // A method of a closed interface (interfaceFamilies), `ts.InterfaceSymbolRef(value, index)`, or a
    // function-typed field of one (interfaceFieldFamilies), read and called: the family of the slot
    // after the vtable's header - every function the vtables hold there, or every function the
    // implementing literals' fields are ever given. Open, the call is unknown, as before.
    template <typename Family>
    void resolveInterfaceSlot(mlir_ts::InterfaceSymbolRefOp refOp, const std::map<std::pair<std::string, int64_t>, Family> &families,
                              Callees &callees)
    {
        auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(refOp.getInterfaceVal().getType());
        auto identifier = mlir::dyn_cast<mlir::StringAttr>(refOp.getIdentifier());
        if (!interfaceType || !identifier)
        {
            return;
        }

        auto position = static_cast<int64_t>(refOp.getIndex()) + INTERFACE_VTABLE_HEADER_SLOTS;
        auto found = families.find({interfaceType.getName().getValue().str(), position});
        if (found == families.end() || !found->second.closed || found->second.field != identifier ||
            found->second.members.empty())
        {
            return;
        }

        callees.named.assign(found->second.members.begin(), found->second.members.end());
        callees.viaInterface = true;
        callees.known = true;
        for (auto member : found->second.members)
        {
            callees.known = addDefined(member, callees) && callees.known;
        }
    }

    // Can a module other than this one call the function? A `public` method (not private, not
    // exported itself) of a class no other module can name - not exported, not a word of the
    // exported declarations, not in the type of an exported or declared function or of an exported
    // or imported global (nameableClasses) - no: another module calls a function of this one through
    // the declaration `__decls` gives it (controller ruling, spec 31). But a method of a class that
    // extends a class another module can name, or one another module defines, yes, private or not:
    // that module calls it through the base's vtable.
    bool callableFromOutside(mlir_ts::FuncOp funcOp)
    {
        if (!funcOp.isPrivate() && (funcOp.isDeclaration() || funcOp->hasAttr("export") || funcOp->hasAttr(DLL_EXPORT)))
        {
            return true;
        }

        // its class: the longest own class name that prefixes its symbol
        auto symbol = funcOp.getSymName();
        llvm::StringRef owner;
        for (auto &entry : ownClasses)
        {
            auto name = entry.getKey();
            if (name.size() > owner.size() && symbol.size() > name.size() && symbol.starts_with(name) && symbol[name.size()] == '.')
            {
                owner = name;
            }
        }

        if (owner.empty())
        {
            return !funcOp.isPrivate();
        }

        auto nameable = [&](llvm::StringRef name) {
            auto last = name.rsplit('.').second;
            return nameableClasses.contains(name) || exportWords.contains(name) || (!last.empty() && exportWords.contains(last));
        };

        if (!funcOp.isPrivate() && nameable(owner))
        {
            return true;
        }

        // An override of a method of a nameable class, or of one another module defines (no own
        // vtable here), fills a slot of that class's vtable: another module calls it through the
        // base (review6r1 p5, p6, p7; controller ruling: any base of that kind on the chain). The
        // base is the first field of a derived class's storage.
        auto storage = classStorages.lookup(owner);
        if (!storage)
        {
            // its storage is not known here: no telling what it extends
            return !funcOp.isPrivate();
        }

        for (auto depth = 0; depth < 64; ++depth)
        {
            auto fields = storage.getFields();
            auto base = fields.empty() ? mlir_ts::ClassStorageType() : mlir::dyn_cast<mlir_ts::ClassStorageType>(fields.front().type);
            if (!base)
            {
                return false;
            }

            auto name = base.getName().getValue();
            if (nameable(name) || !ownClasses.contains(name))
            {
                return true;
            }

            storage = base;
        }

        return true;
    }

    // Can foreign code learn of a function of this module, to call it back during a call of a
    // scalar declaration (scalarDeclaration)? It can only through the module's foreign surface, or
    // through a function turned into data (controller ruling, spec 31, fix round 5: a closed-world
    // test, in place of listing the routes a function value takes out). So the rule is on only when:
    // - every foreign symbol is scalar (scalarType): every body-less function (C's, another
    //   module's - `import`, a DLL's - a `declare class`'s, a `@cpp` class's slot; a rest parameter
    //   is not scalar), every global declared with no initializer or imported (a `declare var`, a
    //   `declare class`'s vtable and type name), every `@cpp` class's fields and its bases', with
    //   no method of its own (C++ calls one through the C++ vtable), every function found in a
    //   library at run time (`SearchForAddressOfSymbol`, seen through its cast); an opaque pointer
    //   or a reference anywhere in such a type is a value of foreign memory, not scalar;
    // - no cast makes an opaque pointer or a reference out of a number (`<Opaque>0x1234`, then a
    //   store through it reaches foreign memory);
    // - no cast makes a value of a type that holds no function (`any`, `unknown`, an opaque
    //   pointer, an integer, a reference; a boolean cannot carry one and does not count) out of one
    //   that holds one (`onSig as Opaque`, `k.m as u64`, `<Reference<u64>>k`; a class or object
    //   takes its fields along unless seen as a class or an interface);
    // - no `inline_asm` or `call_intrinsic` is given a value whose type holds a function.
    bool functionsEscape = false;

    // The types a holdsFunction walk has been through, with or without their fields.
    using SeenTypes = llvm::DenseSet<std::pair<mlir::Type, int>>;

    // Does a value of this type hold a function: the type itself, a union's or an optional's
    // member, a tuple's field, an array's element, what a reference points to, an interface whose
    // vtable holds one past its header - a method, a function-typed field, or a field whose type
    // holds one, recursively (an interface with methods, a class with a function field, an array of
    // them) - or whose vtable this module does not make values with (a foreign one: `declare
    // function makeW(): W`), and with `fields` a class's or an object's fields. The header slot
    // (`.instanceOf`, the compiler's own, which destroys nothing) does not count.
    bool holdsFunction(mlir::Type type, bool fields, SeenTypes &seen)
    {
        if (!type || !seen.insert({type, fields}).second)
        {
            return false;
        }

        if (mlir::isa<mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType,
                      mlir_ts::ExtensionFunctionType, mlir_ts::ConstructFunctionType>(type))
        {
            return true;
        }

        if (auto interfaceType = mlir::dyn_cast<mlir_ts::InterfaceType>(type))
        {
            auto found = interfaceVTableTypes.find(interfaceType.getName().getValue());
            if (found == interfaceVTableTypes.end())
            {
                return true;
            }

            return llvm::any_of(found->second, [&](mlir_ts::TupleType vtableType) {
                return llvm::any_of(llvm::drop_begin(vtableType.getFields(), 1),
                                    [&](const mlir_ts::FieldInfo &info) { return holdsFunction(info.type, /*fields=*/true, seen); });
            });
        }

        auto any = [&](llvm::ArrayRef<mlir_ts::FieldInfo> infos) {
            return llvm::any_of(infos, [&](const mlir_ts::FieldInfo &info) { return holdsFunction(info.type, fields, seen); });
        };

        if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type))
        {
            return any(tupleType.getFields());
        }

        if (auto constTupleType = mlir::dyn_cast<mlir_ts::ConstTupleType>(type))
        {
            return any(constTupleType.getFields());
        }

        if (auto classStorageType = mlir::dyn_cast<mlir_ts::ClassStorageType>(type))
        {
            return fields && any(classStorageType.getFields());
        }

        if (auto objectStorageType = mlir::dyn_cast<mlir_ts::ObjectStorageType>(type))
        {
            return fields && any(objectStorageType.getFields());
        }

        if (mlir::isa<mlir_ts::ClassType, mlir_ts::ObjectType>(type) && !fields)
        {
            return false;
        }

        auto found = false;
        type.walkImmediateSubElements([](mlir::Attribute) {}, [&](mlir::Type sub) { found = found || holdsFunction(sub, fields, seen); });
        return found;
    }

    // The `@cpp` class storage a method's `this` is of, if any: the storage of its first
    // parameter's class, when the method's symbol is `Class.name`.
    mlir_ts::ClassStorageType cppClassOfMethod(mlir_ts::FuncOp funcOp)
    {
        auto inputs = funcOp.getFunctionType().getInputs();
        auto classType = inputs.empty() ? mlir_ts::ClassType() : mlir::dyn_cast<mlir_ts::ClassType>(inputs.front());
        auto storage = classType ? mlir::dyn_cast<mlir_ts::ClassStorageType>(classType.getStorageType()) : mlir_ts::ClassStorageType();
        if (!storage || !storage.isCpp())
        {
            return {};
        }

        auto name = storage.getName().getValue();
        auto symbol = funcOp.getSymName();
        return symbol.size() > name.size() && symbol.starts_with(name) && symbol[name.size()] == '.' ? storage : mlir_ts::ClassStorageType();
    }

    // Does a value of this type carry an address of memory this module made - so a cast of it to
    // an opaque pointer or a reference points into this module, not into foreign memory?
    static bool addressType(mlir::Type type)
    {
        return mlir::isa<mlir_ts::OpaqueType, mlir_ts::RefType, mlir_ts::ValueRefType, mlir_ts::BoundRefType, mlir_ts::ClassType,
                         mlir_ts::ClassStorageType, mlir_ts::ObjectType, mlir_ts::ObjectStorageType, mlir_ts::InterfaceType,
                         mlir_ts::ArrayType, mlir_ts::ConstArrayType, mlir_ts::StringType, mlir_ts::TupleType,
                         mlir_ts::ConstTupleType, mlir_ts::FunctionType, mlir_ts::BoundFunctionType, mlir_ts::HybridFunctionType,
                         mlir_ts::ExtensionFunctionType, mlir_ts::ConstructFunctionType, mlir_ts::NullType,
                         mlir_ts::UndefinedType>(type);
    }

    void findEscapingFunctions(mlir::ModuleOp module)
    {
        // every body-less function: C's, another module's, a `declare class`'s (its methods, its
        // `..instanceOf`, its `..new`), a `@cpp` class's slot
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() && !scalarSignature(funcOp.getFunctionType()))
            {
                functionsEscape = true;
                return;
            }

            // a method of a `@cpp` class: C++ calls it through the class's C++ vtable
            if (cppClassOfMethod(funcOp))
            {
                functionsEscape = true;
                return;
            }
        }

        // every global declared with no initializer or imported: a `declare var`, a `declare
        // class`'s vtable (functions) and type name (a string)
        for (auto &entry : globals)
        {
            auto globalOp = entry.second;
            auto foreign = globalOp->hasAttr("import") || globalOp->hasAttr(DLL_IMPORT) || globalOp.getInitializerRegion().empty() ||
                           globalOp.getInitializerRegion().front().empty();
            if (foreign && !scalarType(globalOp.getType()))
            {
                functionsEscape = true;
                return;
            }
        }

        // every `@cpp` class's fields; a base (the first field, a storage) is a `@cpp` class of its
        // own, judged on its own
        for (auto &entry : classStorages)
        {
            auto storage = entry.second;
            if (!storage.isCpp())
            {
                continue;
            }

            for (auto [index, field] : llvm::enumerate(storage.getFields()))
            {
                auto base = index == 0 ? mlir::dyn_cast<mlir_ts::ClassStorageType>(field.type) : mlir_ts::ClassStorageType();
                if (base ? !base.isCpp() : !scalarType(field.type))
                {
                    functionsEscape = true;
                    return;
                }
            }
        }

        module.walk([&](mlir::Operation *op) {
            if (functionsEscape)
            {
                return;
            }

            // a function found in a library at run time: scalar through every cast of its address
            if (mlir::isa<mlir_ts::SearchForAddressOfSymbolOp>(op))
            {
                functionsEscape = !llvm::all_of(op->getResult(0).getUsers(), [](mlir::Operation *user) {
                    auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
                    auto functionType = castOp ? mlir::dyn_cast<mlir_ts::FunctionType>(castOp.getType()) : mlir_ts::FunctionType();
                    return functionType && scalarSignature(functionType);
                });
                return;
            }

            // `inline_asm<T>(...)` and `call_intrinsic<T>(...)`: a function given to either is data
            // of no type (review6r5 h1, h2: `inline_asm<u64>("mov $1, $0", "=r,r", onSig)`)
            if (mlir::isa<mlir_ts::InlineAsmOp, mlir_ts::CallIntrinsicOp>(op))
            {
                functionsEscape = llvm::any_of(op->getOperands(), [&](mlir::Value operand) {
                    SeenTypes seen;
                    return holdsFunction(operand.getType(), /*fields=*/true, seen);
                });
                return;
            }

            auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(op);
            if (!castOp)
            {
                return;
            }

            // a vtable's address (a class's, an interface's, a thunk's cell) seen as an opaque
            // pointer or an integer: the compiler's own, inside a vtable or storing an object's
            // vtable, never a value a program gives C
            auto vtableOp = op->getParentOfType<mlir_ts::GlobalOp>();
            auto addressOfOp = addressOfThroughCasts(castOp.getIn());
            auto isVTable = [](llvm::StringRef name) { return name.ends_with("..vtbl") || name.contains("..vtbl."); };
            if ((vtableOp && isVTable(vtableOp.getSymName())) || (addressOfOp && isVTable(addressOfOp.getGlobalName())))
            {
                return;
            }

            // an opaque pointer or a reference made out of a number: an address of foreign memory
            if (mlir::isa<mlir_ts::OpaqueType, mlir_ts::RefType>(castOp.getType()) && !addressType(castOp.getIn().getType()))
            {
                functionsEscape = true;
                return;
            }

            // a function seen as something that holds none; a boolean cannot carry one. A class or
            // object takes its fields along, seen as anything (`any`, an opaque pointer, an integer,
            // a reference: `<Reference<u64>>k`, then a load of the field, review6r5 g2) but a class
            // (its base) or an interface (a view of it, read through its vtable only)
            if (mlir::isa<mlir_ts::BooleanType>(castOp.getType()))
            {
                return;
            }

            auto fields = !mlir::isa<mlir_ts::ClassType, mlir_ts::InterfaceType>(castOp.getType());
            SeenTypes seenIn;
            SeenTypes seenOut;
            functionsEscape = holdsFunction(castOp.getIn().getType(), fields, seenIn) &&
                              !holdsFunction(castOp.getType(), /*fields=*/true, seenOut);
        });
    }

    // class name -> its storage, from every class type a function's signature or a global mentions
    llvm::StringMap<mlir_ts::ClassStorageType> classStorages;

    void collectClassStorages(mlir::ModuleOp module)
    {
        llvm::DenseSet<mlir::Type> seen;
        std::function<void(mlir::Type)> visit = [&](mlir::Type type) {
            if (!type || !seen.insert(type).second)
            {
                return;
            }

            if (auto storage = mlir::dyn_cast<mlir_ts::ClassStorageType>(type))
            {
                classStorages[storage.getName().getValue()] = storage;
                for (auto &field : storage.getFields())
                {
                    visit(field.type);
                }

                return;
            }

            type.walkImmediateSubElements([](mlir::Attribute) {}, visit);
        };

        for (auto &entry : functions)
        {
            visit(entry.second.getFunctionType());
        }

        module.walk([&](mlir::Operation *op) {
            for (auto result : op->getResults())
            {
                if (mlir::isa<mlir_ts::ClassType>(result.getType()))
                {
                    visit(result.getType());
                }
            }
        });
    }

    // A class another module can see may be extended there, and its override is a candidate this
    // module never sees: every member has to be private and defined here.
    void resolveFamily(int64_t index, mlir::StringAttr identifier, Callees &callees)
    {
        callees.known = true;
        for (auto member : familyOf(index, identifier))
        {
            callees.known = addDefined(member, callees) && callees.known;
        }

        callees.known = callees.known && llvm::all_of(callees.funcs, [&](mlir_ts::FuncOp funcOp) { return !callableFromOutside(funcOp); });
    }

    // `ts.Cast(ts.VTableOffsetRef(vtable, 0))`, where `vtable` is the one an object's first word
    // points to or an interface value's own: slot 0 of either is `.instanceOf` - a class's
    // generated one, a string compare, or `.instanceOf.none` for an object literal behind an
    // interface. Only mlirGenInstanceOfThroughVTable builds a call through it. Neither destroys
    // anything.
    static bool isInstanceOfSlot(mlir::Operation *def)
    {
        auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(def);
        auto slot = castOp ? castOp.getIn().getDefiningOp<mlir_ts::VTableOffsetRefOp>() : mlir_ts::VTableOffsetRefOp();
        if (!slot || slot.getIndex() != INTERFACE_VTABLE_INSTANCEOF_SLOT)
        {
            return false;
        }

        // a class vtable, read out of an object's first word, or an interface's own vtable,
        // whose slot 0 is its implementer's `.instanceOf` (INTERFACE_VTABLE_HEADER_SLOTS)
        auto *vtable = slot.getVtable().getDefiningOp();
        if (auto classVTable = mlir::dyn_cast_or_null<mlir_ts::VTableOffsetRefOp>(vtable))
        {
            return classVTable.getIndex() == 0;
        }

        return mlir::isa_and_nonnull<mlir_ts::ExtractInterfaceVTableOp>(vtable);
    }

    // Is every use of this function value the callee of a call - through `ts.GetMethod`, with
    // `ts.GetThis` reading the object beside it?
    static bool onlyCalled(mlir::Value value)
    {
        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::CallInternalOp, mlir_ts::CallIndirectOp, mlir_ts::InvokeOp>(user) &&
                calleeValue(user) == use.get() && use.getOperandNumber() == 0)
            {
                continue;
            }

            if (mlir::isa<mlir_ts::GetThisOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::GetMethodOp>(user) && onlyCalled(user->getResult(0)))
            {
                continue;
            }

            return false;
        }

        return true;
    }

    // The closed world. A function is open when a caller this pass cannot see may reach it: it is
    // not private (another module may call it), or its symbol is used other than by a direct call,
    // a class vtable's entry, or a method reference used only as a callee. A virtual reference
    // that escapes opens its whole family.
    void findOpen(mlir::ModuleOp module)
    {
        for (auto &entry : functions)
        {
            if (callableFromOutside(entry.second))
            {
                open.try_emplace(entry.second, "it can be called from another module; '@" LENDS_DECORATOR "' or '@" KEEPS_DECORATOR
                                               "' would say it in its signature");
            }
        }

        // the module is a symbol table, which getSymbolUses does not look inside: walk its body
        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            // an op this pass cannot read symbols through: nothing is closed
            for (auto &entry : functions)
            {
                open.try_emplace(entry.second, "its callers cannot all be found");
            }

            return;
        }

        for (auto &use : *uses)
        {
            auto symbol = use.getSymbolRef().getRootReference();
            auto funcOp = functions.lookup(symbol.getValue());
            if (!funcOp)
            {
                continue;
            }

            auto *user = use.getUser();
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::InvokeOp>(user))
            {
                continue;
            }

            if (mlir::isa<mlir_ts::SymbolRefOp>(user) && isClassVTableEntry(user))
            {
                continue;
            }

            // passed only to a parameter whose family is closed: its calls are resolved (spec 33)
            if (callbackPassing.contains(user))
            {
                continue;
            }

            int64_t index = -1;
            if (auto refOp = mlir::dyn_cast<mlir_ts::ThisVirtualSymbolRefOp>(user))
            {
                index = refOp.getIndex();
            }
            else if (auto refOp = mlir::dyn_cast<mlir_ts::VirtualSymbolRefOp>(user))
            {
                index = refOp.getIndex();
            }

            auto inFunction = !!user->getParentOfType<mlir::FunctionOpInterface>();
            if (inFunction && mlir::isa<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::VirtualSymbolRefOp,
                                        mlir_ts::ThisSymbolRefOp, mlir_ts::SymbolRefOp>(user) &&
                onlyCalled(user->getResult(0)))
            {
                // a virtual call this pass cannot resolve reaches its family without the facts
                if (index >= 0)
                {
                    Callees callees;
                    resolveFamily(index, symbol, callees);
                    if (!callees.known)
                    {
                        openFamily(index, symbol, "an override may be defined in another module");
                    }
                }

                continue;
            }

            open.try_emplace(funcOp, "it is used other than by a call");
            if (index >= 0)
            {
                openFamily(index, symbol, "it is used other than by a call");
            }
        }
    }

    void openFamily(int64_t index, mlir::StringAttr symbol, const char *why)
    {
        for (auto member : familyOf(index, symbol))
        {
            if (auto memberOp = functions.lookup(member.getValue()))
            {
                open.try_emplace(memberOp, why);
            }
        }
    }

    // ---- Callbacks (spec 33) ----
    //
    // `___array_reduce<T, R>(arr, f, init)` calls its parameter `f`: an indirect call, which reaches
    // whatever function its callers pass. When the function is closed - private, and every use of its
    // symbol a direct call - the parameter is only ever called, and every caller passes there a
    // function this module defines, by its symbol or as a closure over a capture box made right
    // there, a call through the parameter reaches exactly those functions: the parameter's family.
    // Its members are closed too, when their symbols are used only to be called directly or passed
    // so; then the call gets the facts the members agree on (callFacts), and `__own_no_drops` when
    // none of them destroys anything. A closure's body takes its capture box first: its parameter k
    // is the call's argument k - 1 (CallbackMember::shift).
    //
    // A kept parameter is agreed on the other way round from a class family's (meetCallbacks): a
    // member that only borrows a parameter another member keeps is given it to keep too, and releases
    // it on every path (OWN_PARAMS_RELEASED_ATTR_NAME) - `reduce` with a callback that returns its
    // accumulator on some paths and one that never does, in one `___array_reduce<V, V>`. Anything
    // else the members disagree on demotes them all, as a class family's members are.

    struct CallbackMember
    {
        mlir::StringAttr symbol;
        int32_t shift = 0;
    };

    struct CallbackFamily
    {
        llvm::SmallVector<CallbackMember, 2> members;
        // the ops that name a member's symbol to pass it here
        llvm::SmallVector<mlir::Operation *, 2> passing;
        bool closed = true;
    };

    // (function, parameter index) -> the functions its callers pass there
    std::map<std::pair<mlir::Operation *, int32_t>, CallbackFamily> callbackFamilies;
    // the members whose families disagreed (meetFacts), for the error in their bodies
    llvm::DenseSet<mlir::Operation *> demotedCallbacks;
    // uses of a member's symbol that only pass it to a closed family: no escape (findOpen)
    llvm::DenseSet<mlir::Operation *> callbackPassing;

    static bool isFunctionValueType(mlir::Type type)
    {
        return mlir::isa<mlir_ts::FunctionType, mlir_ts::HybridFunctionType, mlir_ts::BoundFunctionType>(type);
    }

    // Is every use of this parameter - through its slot, never assigned or captured - the function
    // a call goes through?
    static bool onlyCalledParam(mlir::BlockArgument argument)
    {
        for (auto *user : argument.getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            auto index = -1;
            if (!varOp || !isParameterSlot(varOp, index))
            {
                return false;
            }

            for (auto *slotUser : varOp->getUsers())
            {
                if (mlir::isa<mlir_ts::DebugVariableOp>(slotUser))
                {
                    continue;
                }

                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser);
                if (!loadOp)
                {
                    return false;
                }

                for (auto &use : loadOp->getUses())
                {
                    auto *call = use.getOwner();
                    if (!isCall(call) || calleeValue(call) != use.get() ||
                        llvm::is_contained(callArgs(call), use.get()))
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    // The function a caller passes as `value`: its symbol, seen as a function value through casts
    // between function types, or a closure over a capture box built of it. Null for anything else.
    static mlir_ts::SymbolRefOp passedFunction(mlir::Value value, int32_t &shift)
    {
        while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
        {
            if (!isFunctionValueType(castOp.getIn().getType()) || !isFunctionValueType(castOp.getType()))
            {
                return {};
            }

            value = castOp.getIn();
        }

        if (auto refOp = value.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            shift = 0;
            return refOp;
        }

        auto boundOp = value.getDefiningOp<mlir_ts::CreateBoundFunctionOp>();
        if (!boundOp || closureOfBox(boundOp.getThisVal()) != boundOp)
        {
            return {};
        }

        shift = 1;
        return boundOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>();
    }

    // Does the function value `value` go only where `passes` allows - as an argument of a call - and
    // through casts between function types, a closure built of it, and that closure's own retains
    // and releases?
    static bool passedOnly(mlir::Value value, llvm::function_ref<bool(mlir::OpOperand &)> passes, int depth = 0)
    {
        if (depth > 8)
        {
            return false;
        }

        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp, mlir_ts::DebugVariableOp>(user) || passes(use))
            {
                continue;
            }

            auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
            auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(user);
            if (castOp && isFunctionValueType(castOp.getType()) && passedOnly(castOp.getResult(), passes, depth + 1))
            {
                continue;
            }

            if (boundOp && boundOp.getFunc() == use.get() && closureOfBox(boundOp.getThisVal()) == boundOp &&
                passedOnly(boundOp.getResult(), passes, depth + 1))
            {
                continue;
            }

            return false;
        }

        return true;
    }

    void collectCallbackFamilies(mlir::ModuleOp module)
    {
        auto uses = mlir::SymbolTable::getSymbolUses(&module.getBodyRegion());
        if (!uses)
        {
            return;
        }

        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>> usersOf;
        for (auto &use : *uses)
        {
            if (auto funcOp = functions.lookup(use.getSymbolRef().getRootReference().getValue()))
            {
                usersOf[funcOp].push_back(use.getUser());
            }
        }

        auto callsOnly = [&](mlir_ts::FuncOp funcOp) {
            auto &users = usersOf[funcOp];
            return !users.empty() && llvm::all_of(users, [&](mlir::Operation *user) {
                auto callee = directCallee(user);
                return callee && callee.getValue() == funcOp.getSymName();
            });
        };

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (funcOp.isDeclaration() || callableFromOutside(funcOp) || declaresFacts(funcOp) || !callsOnly(funcOp))
            {
                continue;
            }

            for (auto argument : funcOp.getBody().front().getArguments())
            {
                if (!isFunctionValueType(argument.getType()) || !onlyCalledParam(argument))
                {
                    continue;
                }

                auto index = static_cast<int32_t>(argument.getArgNumber());
                CallbackFamily family;
                for (auto *user : usersOf[funcOp])
                {
                    auto args = callArgs(user);
                    auto shift = 0;
                    auto refOp = static_cast<size_t>(index) < args.size() ? passedFunction(args[index], shift)
                                                                          : mlir_ts::SymbolRefOp();
                    auto member = refOp ? functions.lookup(refOp.getIdentifier()) : mlir_ts::FuncOp();
                    if (!member || member.isDeclaration() || callableFromOutside(member) || declaresFacts(member))
                    {
                        family.closed = false;
                        break;
                    }

                    auto symbol = member.getSymNameAttr();
                    auto same = llvm::find_if(family.members, [&](const CallbackMember &known) { return known.symbol == symbol; });
                    if (same != family.members.end() && same->shift != shift)
                    {
                        family.closed = false;
                        break;
                    }

                    if (same == family.members.end())
                    {
                        family.members.push_back({symbol, shift});
                    }

                    family.passing.push_back(refOp);
                }

                callbackFamilies[{funcOp, index}] = std::move(family);
            }
        }

        // A member's symbol used anywhere else - a call through another value, a field, a vtable, a
        // global - reaches callers this cannot see; so does one passed to a family that is not closed.
        // Each family that has such a member is open, until nothing changes.
        auto passesToClosed = [&](mlir::OpOperand &use) {
            auto *call = use.getOwner();
            auto callee = directCallee(call);
            auto calleeOp = callee ? functions.lookup(callee.getValue()) : mlir_ts::FuncOp();
            if (!calleeOp)
            {
                return false;
            }

            // the argument this use is, by its operand's position
            auto args = callArgs(call);
            if (args.empty() || use.getOperandNumber() < args.getBeginOperandIndex() ||
                use.getOperandNumber() >= args.getBeginOperandIndex() + args.size())
            {
                return false;
            }

            auto index = static_cast<int32_t>(use.getOperandNumber() - args.getBeginOperandIndex());
            auto found = callbackFamilies.find({calleeOp, index});
            return found != callbackFamilies.end() && found->second.closed;
        };

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &[key, family] : callbackFamilies)
            {
                if (!family.closed)
                {
                    continue;
                }

                for (auto &member : family.members)
                {
                    auto memberOp = functions.lookup(member.symbol.getValue());
                    auto escapes = llvm::any_of(usersOf[memberOp], [&](mlir::Operation *user) {
                        if (auto callee = directCallee(user); callee && callee.getValue() == member.symbol.getValue())
                        {
                            return false;
                        }

                        auto refOp = mlir::dyn_cast<mlir_ts::SymbolRefOp>(user);
                        return !refOp || !user->getParentOfType<mlir::FunctionOpInterface>() ||
                               !passedOnly(refOp.getResult(), passesToClosed);
                    });

                    if (escapes)
                    {
                        family.closed = false;
                        changed = true;
                        break;
                    }
                }
            }
        }

        for (auto &[key, family] : callbackFamilies)
        {
            if (family.closed)
            {
                callbackPassing.insert(family.passing.begin(), family.passing.end());
            }
        }
    }

    // A call through a parameter whose family is closed: its members.
    void resolveCallback(mlir_ts::LoadOp loadOp, Callees &callees)
    {
        auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
        auto funcOp = loadOp->getParentOfType<mlir_ts::FuncOp>();
        auto index = -1;
        if (!varOp || !funcOp || !isParameterSlot(varOp, index))
        {
            return;
        }

        auto found = callbackFamilies.find({funcOp.getOperation(), index});
        if (found == callbackFamilies.end() || !found->second.closed || found->second.members.empty())
        {
            return;
        }

        callees.viaCallback = true;
        callees.known = true;
        for (auto &member : found->second.members)
        {
            callees.named.push_back(member.symbol);
            callees.known = addDefined(member.symbol, callees) && callees.known;
            callees.shifts.push_back(member.shift);
        }

        callees.known = callees.known && callees.shifts.size() == callees.funcs.size();
    }

    // A member's facts as a call through its family sees them: each parameter k as argument
    // k - shift. False when a fact is about the capture box itself, which the call does not pass.
    static bool shiftFacts(const Facts &facts, int32_t shift, Facts &shifted)
    {
        shifted = Facts();
        shifted.record = facts.record;
        for (auto [from, to] : {std::make_pair(&facts.owned, &shifted.owned), std::make_pair(&facts.borrows, &shifted.borrows)})
        {
            for (auto index : *from)
            {
                if (index < shift)
                {
                    return false;
                }

                to->push_back(index - shift);
            }
        }

        return true;
    }

    // Can the body of `funcOp` be given parameter `index` to keep and release it where it does not
    // move it: a plain slot (isParameterSlot), a block, not a handle, not a string?
    bool canReleaseParam(mlir_ts::FuncOp funcOp, int32_t index)
    {
        if (funcOp.getBody().empty() || static_cast<unsigned>(index) >= funcOp.getNumArguments())
        {
            return false;
        }

        auto argument = funcOp.getBody().front().getArgument(index);
        if (isHandle(argument) || isCopyableString(argument.getType()) || holdsNoBlock(argument))
        {
            return false;
        }

        return llvm::any_of(argument.getUsers(), [&](mlir::Operation *user) {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            auto position = -1;
            return varOp && isParameterSlot(varOp, position) && position == index;
        });
    }

    // The members of each closed callback family agree (the section above), by promotion of a kept
    // parameter or by demotion. True when any member's facts changed.
    bool meetCallbacks(llvm::DenseSet<mlir::Operation *> &demoted)
    {
        auto changed = false;
        for (auto &[key, family] : callbackFamilies)
        {
            if (!family.closed)
            {
                continue;
            }

            llvm::SmallVector<mlir_ts::FuncOp, 2> members;
            llvm::SmallVector<Facts, 2> seen;
            auto shiftable = true;
            for (auto &member : family.members)
            {
                auto funcOp = functions.lookup(member.symbol.getValue());
                Facts shifted;
                shiftable = shiftable && funcOp && shiftFacts(factsEffective.lookup(funcOp), member.shift, shifted);
                members.push_back(funcOp);
                seen.push_back(shifted);
            }

            if (shiftable && llvm::all_equal(seen))
            {
                continue;
            }

            // a parameter some member keeps, and each other one only borrows: kept by all
            llvm::SmallVector<int32_t> kept;
            for (auto &facts : seen)
            {
                kept.append(facts.owned.begin(), facts.owned.end());
            }

            llvm::sort(kept);
            kept.erase(std::unique(kept.begin(), kept.end()), kept.end());
            auto promotable = shiftable && llvm::all_of(seen, [&](const Facts &facts) {
                return facts.borrows == seen.front().borrows && facts.record == seen.front().record &&
                       llvm::none_of(kept, [&](int32_t index) { return llvm::is_contained(facts.borrows, index); });
            });

            for (auto [position, member] : llvm::enumerate(family.members))
            {
                auto funcOp = members[position];
                // a member another caller reaches without the facts is given none (findOpen)
                for (auto index : kept)
                {
                    promotable = promotable && !demoted.contains(funcOp) && !open.contains(funcOp) &&
                                 !interfaceReached.contains(funcOp) &&
                                 (llvm::is_contained(seen[position].owned, index) ||
                                  canReleaseParam(funcOp, index + member.shift));
                }
            }

            if (promotable)
            {
                for (auto [position, member] : llvm::enumerate(family.members))
                {
                    auto funcOp = members[position];
                    auto facts = factsEffective.lookup(funcOp);
                    for (auto index : kept)
                    {
                        auto param = index + member.shift;
                        if (!llvm::is_contained(facts.owned, param))
                        {
                            facts.owned.insert(llvm::upper_bound(facts.owned, param), param);
                        }
                    }

                    if (facts != factsEffective.lookup(funcOp))
                    {
                        factsEffective[funcOp] = facts;
                        changed = true;
                    }
                }

                continue;
            }

            for (auto funcOp : members)
            {
                if (funcOp && !declaresCallerFacts(funcOp))
                {
                    demoted.insert(funcOp);
                    if (factsEffective.erase(funcOp))
                    {
                        changed = true;
                    }
                }
            }
        }

        return changed;
    }

    // ---- Results that borrow an argument ----

    // The values a function returns: what is stored into the local its returns read (MLIRGen's
    // result slot), or what a return hands back directly.
    static llvm::SmallVector<mlir::Value> returnedValues(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<mlir::Value> values;
        funcOp.walk([&](mlir_ts::ReturnInternalOp returnOp) {
            for (auto operand : returnOp.getRetOperands())
            {
                auto loadOp = operand.getDefiningOp<mlir_ts::LoadOp>();
                auto slot = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (!slot || slot.getInitializer() || isOwningVariable(slot))
                {
                    values.push_back(operand);
                    continue;
                }

                for (auto *user : slot.getResult().getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == slot.getResult())
                    {
                        values.push_back(storeOp.getValue());
                    }
                }
            }
        });

        return values;
    }

    struct ResultBorrows
    {
        llvm::SmallVector<int32_t> borrows;
        bool record = false;
    };

    // What the function's result borrows (spec 2.4, 32): the one parameter every heap value it returns
    // is a borrow of, or, when every such value is a borrowing record, the parameters the records
    // borrow. A result that holds no block (`null`, a number) agrees with any; a record on one path
    // and anything else on another is nothing a caller can be told.
    ResultBorrows localResultBorrows(mlir_ts::FuncOp funcOp)
    {
        auto found = -1;
        auto record = false;
        llvm::SmallVector<int32_t> params;
        for (auto value : returnedValues(funcOp))
        {
            if (holdsNoBlock(value))
            {
                continue;
            }

            // a handle returned is a counted copy, never a borrow (spec 23.2)
            if (isHandle(value))
            {
                return {};
            }

            switch (recordParamsOf(value, params))
            {
            case RecordFlow::Bad:
                return {};
            case RecordFlow::Record:
                record = true;
                continue;
            case RecordFlow::None:
                break;
            }

            auto index = borrowedParam(value);
            if (index < 0 || (found >= 0 && index != found))
            {
                return {};
            }

            found = index;
        }

        if (record && found >= 0)
        {
            return {};
        }

        if (record)
        {
            llvm::sort(params);
            params.erase(std::unique(params.begin(), params.end()), params.end());
            return {params, true};
        }

        if (found >= 0)
        {
            return {{found}, false};
        }

        return {};
    }

    // The parameter a function returns as it is on some paths, where every other value it returns
    // that holds a block is one it owns (spec 33): `return v.x > 2 ? acc : V.plus(acc, v)`. Such a
    // parameter is kept: the callers move the argument in, the path that returns it moves it out,
    // and every other path releases it. -1 for none - a parameter returned on every path is a
    // borrowed result; two parameters returned, or a part of one (`acc.next`), or a borrowing
    // record, have no such answer; a string is copied instead (spec 22).
    int32_t mixedReturnParam(mlir_ts::FuncOp funcOp)
    {
        auto param = -1;
        auto other = false;
        auto returned = returnedValues(funcOp);
        llvm::SmallVector<mlir::Value> work(returned.begin(), returned.end());
        llvm::DenseSet<mlir::Value> seen;
        while (!work.empty())
        {
            auto value = work.pop_back_val();
            if (!seen.insert(value).second || holdsNoBlock(value))
            {
                continue;
            }

            if (isHandle(value) || isCopyableString(value.getType()))
            {
                return -1;
            }

            auto root = rootOf(value);
            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(root); argument && !argument.getOwner()->isEntryBlock())
            {
                llvm::SmallVector<mlir::Value> merged;
                if (!mergedInto(argument, merged))
                {
                    return -1;
                }

                work.append(merged.begin(), merged.end());
                continue;
            }

            int32_t index = -1;
            if (readsParameterSlot(root, funcOp, index))
            {
                if ((param >= 0 && param != index) || isHandle(root) || isCopyableString(root.getType()))
                {
                    return -1;
                }

                param = index;
                continue;
            }

            llvm::SmallVector<int32_t> recordParams;
            if (borrowedParam(root) >= 0 || recordParamsOf(root, recordParams) != RecordFlow::None)
            {
                return -1;
            }

            other = true;
        }

        return other ? param : -1;
    }

    // ---- Borrowing records (spec 32) ----
    //
    // `{ thing: this, ray: ray, dist: d }` given as an interface: an object literal's box holding values
    // it does not own beside values it does. Rust's `struct Hit<'a> { thing: &'a Th, .. }`: the box is
    // its owner's - the caller's, when it is returned - and it may not outlive what it borrows. Each box
    // is classified where it is made: the fields it borrows (OWN_RECORD_BORROWED_FIELDS_ATTR_NAME, on the
    // box and on every interface made of it, whose tag then names a release routine that leaves them),
    // and, when every one of them is a parameter, `this`, or a read out of one, those parameters - what
    // a function returning it lends (OWN_RESULT_RECORD_ATTR_NAME). The inference bounds every use.

    struct RecordBox
    {
        llvm::SmallVector<int32_t> params;
        // every borrowed field is a parameter's, `this`'s, a read out of one, or immortal
        bool returnable = true;
    };

    llvm::DenseMap<mlir::Operation *, RecordBox> recordBoxes;

    void classifyRecordBoxes()
    {
        recordBoxes.clear();
        getOperation().walk([&](mlir_ts::NewOp newOp) {
            newOp->removeAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME);
            for (auto *user : newOp->getUsers())
            {
                for (auto *viewUser : user->getUsers())
                {
                    if (mlir::isa<mlir_ts::NewInterfaceOp>(viewUser))
                    {
                        viewUser->removeAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME);
                    }
                }
            }

            classifyRecordBox(newOp);
        });
    }

    void classifyRecordBox(mlir_ts::NewOp newOp)
    {
        auto boxOp = recordBoxOf(newOp.getResult());
        auto varOp = boxOp == newOp ? recordLocalOf(newOp.getResult()) : mlir_ts::VariableOp();
        if (!varOp)
        {
            return;
        }

        // the record read once, and only into its box (rc's retain of it beside)
        mlir_ts::LoadOp read;
        for (auto *user : varOp->getUsers())
        {
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(user))
            {
                if (read)
                {
                    return;
                }

                read = loadOp;
            }
        }

        for (auto *user : read->getUsers())
        {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            if (!mlir::isa<mlir_ts::RetainOp>(user) && !(storeOp && storeOp.getReference() == newOp.getResult()))
            {
                return;
            }
        }

        // the box seen only as interfaces: every release of it reads their tag
        llvm::SmallVector<mlir::Operation *> interfaces;
        for (auto &use : newOp->getUses())
        {
            auto *user = use.getOwner();
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == newOp.getResult())
            {
                continue;
            }

            auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user);
            if (!castOp || !isView(castOp))
            {
                return;
            }

            for (auto *castUser : castOp->getUsers())
            {
                auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(castUser);
                if (!newInterfaceOp || newInterfaceOp.getThisVal() != castOp.getResult())
                {
                    return;
                }

                interfaces.push_back(castUser);
            }
        }

        if (interfaces.empty())
        {
            return;
        }

        // each field that holds a block: owned (made here, or nothing) or borrowed, never both
        RecordBox info;
        llvm::SmallVector<int32_t> borrowed;
        auto mth = typeHelper(&getContext());
        for (auto *user : varOp->getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp)
            {
                continue;
            }

            auto field = propertyRefOp.getResult();
            auto refType = mlir::dyn_cast<mlir_ts::RefType>(field.getType());
            if (!refType || !mth.ownsHeapMemory(field.getLoc(), refType.getElementType()))
            {
                continue;
            }

            auto owns = false;
            auto borrows = false;
            for (auto *fieldUser : field.getUsers())
            {
                if (mlir::isa<mlir_ts::LoadOp>(fieldUser))
                {
                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != field)
                {
                    return;
                }

                auto value = storeOp.getValue();
                switch (recordFieldSource(value))
                {
                case RecordFieldSource::NoBlock:
                    continue;
                case RecordFieldSource::Fresh:
                    owns = true;
                    continue;
                case RecordFieldSource::Unknown:
                    return;
                case RecordFieldSource::Immortal:
                    borrows = true;
                    continue;
                case RecordFieldSource::Param:
                case RecordFieldSource::Place:
                    if (auto param = borrowedParam(value); param >= 0)
                    {
                        info.params.push_back(param);
                    }
                    else
                    {
                        info.returnable = false;
                    }

                    borrows = true;
                    continue;
                case RecordFieldSource::Slot:
                case RecordFieldSource::Cell:
                    info.returnable = false;
                    borrows = true;
                    continue;
                }
            }

            // a string is copied, not borrowed (spec 22)
            if ((owns && borrows) || (borrows && isCopyableString(refType.getElementType())))
            {
                return;
            }

            if (borrows)
            {
                borrowed.push_back(propertyRefOp.getPosition());
            }
        }

        if (borrowed.empty())
        {
            return;
        }

        llvm::sort(borrowed);
        borrowed.erase(std::unique(borrowed.begin(), borrowed.end()), borrowed.end());
        llvm::sort(info.params);
        info.params.erase(std::unique(info.params.begin(), info.params.end()), info.params.end());

        auto attr = mlir::DenseI32ArrayAttr::get(&getContext(), borrowed);
        newOp->setAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME, attr);
        for (auto *newInterfaceOp : interfaces)
        {
            newInterfaceOp->setAttr(OWN_RECORD_BORROWED_FIELDS_ATTR_NAME, attr);
        }

        recordBoxes[newOp.getOperation()] = std::move(info);
    }

    enum class RecordFlow
    {
        None,   // no borrowing record
        Record, // borrowing records only (or values that hold no block)
        Bad     // a record that borrows what is not a parameter, or one beside anything else
    };

    // Whether `value` - through views, merges and the owning locals that hold it - is a borrowing
    // record, and the parameters it borrows: a box made here, or a call's result that is one, whose
    // arguments are this function's parameters or reads out of them.
    RecordFlow recordParamsOf(mlir::Value value, llvm::SmallVectorImpl<int32_t> &params)
    {
        llvm::SmallVector<mlir::Value> work{value};
        llvm::SmallPtrSet<mlir::Value, 8> seen;
        auto record = false;
        auto other = false;
        while (!work.empty())
        {
            // what holds no block, by its own type (a box's `value_ref` owns nothing by type, but
            // the object its view shows does)
            auto next = work.pop_back_val();
            auto current = rootOf(next);
            if (!seen.insert(current).second || holdsNoBlock(next))
            {
                continue;
            }

            auto *def = current.getDefiningOp();
            if (auto found = def ? recordBoxes.find(def) : recordBoxes.end(); found != recordBoxes.end())
            {
                if (!found->second.returnable)
                {
                    return RecordFlow::Bad;
                }

                params.append(found->second.params.begin(), found->second.params.end());
                record = true;
                continue;
            }

            if (def && isCall(def) && isRecordResult(def))
            {
                auto args = callArgs(def);
                for (auto index : resultBorrowsAll(def))
                {
                    if (static_cast<size_t>(index) >= args.size())
                    {
                        return RecordFlow::Bad;
                    }

                    auto arg = args[index];
                    if (argumentHoldsNoBlock(arg) || isImmortalGlobalRead(arg))
                    {
                        continue;
                    }

                    auto param = borrowedParam(arg);
                    if (param < 0)
                    {
                        return RecordFlow::Bad;
                    }

                    params.push_back(param);
                }

                record = true;
                continue;
            }

            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(current))
            {
                llvm::SmallVector<mlir::Value> merged;
                if (mergedInto(argument, merged))
                {
                    work.append(merged.begin(), merged.end());
                    continue;
                }

                other = true;
                continue;
            }

            // what an owning local holds: its initializer and every value stored into it
            auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(def);
            auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            if (varOp && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false))
            {
                if (varOp.getInitializer())
                {
                    work.push_back(varOp.getInitializer());
                }

                for (auto *user : varOp->getUsers())
                {
                    if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == varOp.getResult())
                    {
                        work.push_back(storeOp.getValue());
                    }
                }

                continue;
            }

            other = true;
        }

        if (!record)
        {
            return RecordFlow::None;
        }

        return other ? RecordFlow::Bad : RecordFlow::Record;
    }

    // ---- Parameters the callee keeps ----

    // Does `user` take `used` - a read of a parameter - into something that outlives the call: a
    // store into a field, an element or a global, an insertion into an array, or a call that keeps
    // it in turn? A local that holds it, and the result it is returned as, do not count: the first
    // is a borrow, the second a borrowed result.
    static bool keeps(mlir::Operation *user, mlir::Value used)
    {
        // a number, a boolean, a string literal: nothing to keep
        if (holdsNoBlock(used))
        {
            return false;
        }

        if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
        {
            // a field a borrowing record borrows (spec 32): the record keeps nothing
            if (isBorrowedRecordFieldStore(user))
            {
                return false;
            }

            auto ref = storeOp.getReference();
            return storeOp.getValue() == used && (isPlace(ref) || ref.getDefiningOp<mlir_ts::AddressOfOp>());
        }

        if (auto pushOp = mlir::dyn_cast<mlir_ts::ArrayPushOp>(user))
        {
            return llvm::is_contained(pushOp.getItems(), used);
        }

        // `new Shared(x)` moves `x` into the block, as a store into a place does (spec 23.1)
        if (auto sharedNewOp = mlir::dyn_cast<mlir_ts::SharedNewOp>(user))
        {
            return sharedNewOp.getValue() == used;
        }

        if (auto unshiftOp = mlir::dyn_cast<mlir_ts::ArrayUnshiftOp>(user))
        {
            return llvm::is_contained(unshiftOp.getItems(), used);
        }

        if (auto spliceOp = mlir::dyn_cast<mlir_ts::ArraySpliceOp>(user))
        {
            return llvm::is_contained(spliceOp.getItems(), used);
        }

        if (isCall(user))
        {
            auto args = callArgs(user);
            return llvm::any_of(ownedParams(user), [&](int32_t index) {
                return static_cast<size_t>(index) < args.size() && args[index] == used;
            });
        }

        // `let r = init; ... r = f(r, v)`: a local that owns what it is declared from and is given
        // another value is no borrow of the parameter - the assignment releases what it holds - so
        // the parameter moves into it (spec 33: `___array_reduce`'s accumulator). Not a string: the
        // local copies it (spec 22)
        if (auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user))
        {
            return varOp.getInitializer() == used && isOwningVariable(varOp) && !varOp.getCaptured().value_or(false) &&
                   isAssigned(varOp) && !isCopyableString(used.getType());
        }

        return false;
    }

    // The parameters the body keeps (spec 2.4, owned-by-callee): a read of the parameter's slot -
    // never assigned - that something keeps.
    static llvm::SmallVector<int32_t> localOwnedParams(mlir_ts::FuncOp funcOp)
    {
        llvm::SmallVector<int32_t> owned;
        if (funcOp.getBody().empty())
        {
            return owned;
        }

        for (auto argument : funcOp.getBody().front().getArguments())
        {
            // a handle kept is a counted copy: its caller gives nothing up (spec 23.2)
            if (isHandle(argument))
            {
                continue;
            }

            auto kept = false;
            for (auto *user : argument.getUsers())
            {
                auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
                auto index = -1;
                if (!varOp || !isParameterSlot(varOp, index))
                {
                    continue;
                }

                for (auto *slotUser : varOp.getResult().getUsers())
                {
                    if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(slotUser))
                    {
                        forEachUse(loadOp.getResult(), [&](mlir::Operation *use, mlir::Value used) {
                            kept = kept || keeps(use, used);
                        });
                    }
                }
            }

            if (kept)
            {
                owned.push_back(argument.getArgNumber());
            }
        }

        return owned;
    }

    // ---- The facts callers must know ----

    // Owned parameters and a result that borrows only hold where every caller knows them, so the
    // facts survive only on a closed function whose every family agrees, and families are
    // demoted until they do: a member that loses its facts can make another family disagree. A
    // caller that returns what such a call returns, or passes a parameter on to a kept one, gets
    // the fact too, so the facts are recomputed with the calls pinned until nothing changes; if
    // that does not settle, nobody gets any.
    void computeFacts()
    {
        llvm::DenseMap<mlir::Operation *, Facts> pinned;
        for (auto round = 0; round < 16; ++round)
        {
            // which object literals borrow depends on what the calls stored into them return, and a
            // function's facts on which literals borrow (spec 32)
            classifyRecordBoxes();

            for (auto &entry : functions)
            {
                if (!entry.second.isDeclaration())
                {
                    auto result = localResultBorrows(entry.second);
                    Facts facts;
                    facts.borrows = std::move(result.borrows);
                    facts.record = result.record;
                    facts.owned = localOwnedParams(entry.second);
                    // what the body takes into something on every path; a parameter it returns on
                    // some paths only is kept too, and released on the others (spec 33)
                    takenParams[entry.second] = facts.owned;
                    if (auto mixed = mixedReturnParam(entry.second); mixed >= 0 && !llvm::is_contained(facts.owned, mixed))
                    {
                        facts.owned.insert(llvm::upper_bound(facts.owned, mixed), mixed);
                    }

                    factsLocal[entry.second] = std::move(facts);
                }
            }

            meetFacts();

            llvm::DenseMap<mlir::Operation *, Facts> next;
            for (auto &call : calls)
            {
                auto facts = callFacts(call.callees);
                if (facts.empty())
                {
                    // a call this pass cannot resolve still has what its callees' signatures say
                    // A declaration's `@lends` is a plain borrow to its callers: its body, which would
                    // say whether the result is a record the caller owns (spec 32), is not here. A
                    // record's box then leaks, and nothing is freed twice.
                    auto keeps = declaredByCallees(call.callees, OWN_DECLARED_KEEPS_ATTR_NAME);
                    facts.owned.assign(keeps.begin(), keeps.end());
                    if (auto lends = declaredByCallees(call.callees, OWN_DECLARED_LENDS_ATTR_NAME); !lends.empty())
                    {
                        llvm::sort(lends);
                        lends.erase(std::unique(lends.begin(), lends.end()), lends.end());
                        facts.borrows = lends;
                    }
                }

                if (!facts.empty())
                {
                    next[call.op] = facts;
                }
            }

            auto settled = next == pinned;
            pinned = std::move(next);
            pinFacts(pinned);
            if (settled)
            {
                return;
            }
        }

        factsEffective.clear();
        pinFacts(llvm::DenseMap<mlir::Operation *, Facts>());
    }

    void meetFacts()
    {
        factsEffective.clear();
        for (auto &[funcOp, facts] : factsLocal)
        {
            if (!facts.empty() && !open.contains(funcOp) && !factsThroughInterface(funcOp, facts))
            {
                factsEffective[funcOp] = facts;
            }
        }

        // what a signature says stands, open or not; checkDeclared holds the body to it. But not a
        // `@keeps` or `@lends` through an interface: the interface's callers are not told
        // (interfaceReached).
        for (auto &entry : functions)
        {
            if (declaresCallerFacts(entry.second))
            {
                auto keeps = declared(entry.second, OWN_DECLARED_KEEPS_ATTR_NAME);
                Facts said;
                said.borrows = declaredLends(entry.second);
                said.owned.assign(keeps.begin(), keeps.end());
                // a body here that returns a borrowing record of what `@lends` names: callers in this
                // module own the box (spec 32); checkDeclared holds the body to the parameters
                auto local = factsLocal.lookup(entry.second);
                said.record = !said.borrows.empty() && local.record && local.borrows == said.borrows;
                if (factsThroughInterface(entry.second, said))
                {
                    continue;
                }

                factsEffective[entry.second] = said;
            }
        }

        // the members of a family that disagree lose their facts: a call through the family is
        // given only what all of them say. A closed interface's slot is such a family too.
        auto demote = [&](llvm::ArrayRef<mlir::StringAttr> members) {
            llvm::SmallVector<Facts> seen;
            for (auto member : members)
            {
                auto funcOp = functions.lookup(member.getValue());
                seen.push_back(funcOp ? factsEffective.lookup(funcOp) : Facts());
            }

            auto changed = false;
            if (llvm::all_equal(seen))
            {
                return changed;
            }

            for (auto member : members)
            {
                if (auto funcOp = functions.lookup(member.getValue());
                    funcOp && !declaresCallerFacts(funcOp) && factsEffective.erase(funcOp))
                {
                    changed = true;
                }
            }

            return changed;
        };

        // a member a callback family demoted is never promoted again: each step only adds or only
        // takes away, so this settles
        demotedCallbacks.clear();
        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &[key, members] : families)
            {
                changed = demote(members) || changed;
            }

            changed = meetCallbacks(demotedCallbacks) || changed;

            for (auto &[key, family] : interfaceFamilies)
            {
                if (family.closed)
                {
                    changed = demote(family.members) || changed;
                }
            }

            for (auto &[key, family] : interfaceFieldFamilies)
            {
                if (family.closed)
                {
                    changed = demote(family.members) || changed;
                }
            }
        }
    }

    Facts callFacts(const Callees &callees)
    {
        if (!callees.known || callees.instanceOf || callees.funcs.empty())
        {
            return {};
        }

        // each member's facts as the call's arguments see them (a closure's body takes its box first)
        if (callees.viaCallback)
        {
            std::optional<Facts> agreed;
            for (auto [funcOp, shift] : llvm::zip(callees.funcs, callees.shifts))
            {
                Facts shifted;
                if (!shiftFacts(factsEffective.lookup(funcOp), shift, shifted) || (agreed && *agreed != shifted))
                {
                    return {};
                }

                agreed = shifted;
            }

            return agreed.value_or(Facts());
        }

        auto facts = factsEffective.lookup(callees.funcs.front());
        auto agree = llvm::all_of(callees.funcs, [&](mlir_ts::FuncOp callee) {
            return factsEffective.lookup(callee) == facts;
        });
        return agree ? facts : Facts();
    }

    static void setFacts(mlir::Operation *op, const Facts &facts)
    {
        auto *context = op->getContext();
        op->removeAttr(OWN_RESULT_BORROWS_ATTR_NAME);
        op->removeAttr(OWN_RESULT_RECORD_ATTR_NAME);
        op->removeAttr(OWN_PARAMS_ATTR_NAME);
        if (!facts.borrows.empty())
        {
            op->setAttr(OWN_RESULT_BORROWS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, facts.borrows));
            if (facts.record)
            {
                op->setAttr(OWN_RESULT_RECORD_ATTR_NAME, mlir::UnitAttr::get(context));
            }
        }

        if (!facts.owned.empty())
        {
            op->setAttr(OWN_PARAMS_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, facts.owned));
        }
    }

    // The calls get the facts their callees agree on; each function gets its own, or the reason it
    // lost those its body has, for the error there.
    void pinFacts(const llvm::DenseMap<mlir::Operation *, Facts> &pinned)
    {
        for (auto &call : calls)
        {
            setFacts(call.op, pinned.lookup(call.op));
        }

        for (auto &[funcOp, local] : factsLocal)
        {
            auto effective = factsEffective.lookup(funcOp);
            setFacts(funcOp, effective);
            funcOp->removeAttr(OWN_FACTS_LOST_ATTR_NAME);
            funcOp->removeAttr(OWN_PARAMS_RELEASED_ATTR_NAME);
            // a signature's `@keeps` is checked against the body as it is (checkDeclared)
            llvm::SmallVector<int32_t> released;
            for (auto index : effective.owned)
            {
                if (!declaresCallerFacts(funcOp) && !llvm::is_contained(takenParams.lookup(funcOp), index))
                {
                    released.push_back(index);
                }
            }

            if (!released.empty())
            {
                funcOp->setAttr(OWN_PARAMS_RELEASED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(&getContext(), released));
            }

            if (effective.empty() && !local.empty())
            {
                // through an interface first: a marker would not help there (interfaceReached)
                auto *why = factsThroughInterface(funcOp, local) ? INTERFACE_REACHED_WHY : open.lookup(funcOp);
                if (!why && demotedCallbacks.contains(funcOp))
                {
                    why = "a function passed in its place disagrees";
                }

                funcOp->setAttr(OWN_FACTS_LOST_ATTR_NAME,
                                mlir::StringAttr::get(&getContext(), why ? why : "an override in its class family disagrees"));
            }
        }
    }

    // ---- Generators that borrow (phase 7b) ----
    //
    // A generator over a parameter that holds a block - `function* each(a: number[])` - keeps the
    // parameter's cell in a box in its state object's `.captured` field. Under 5b's rule that is an
    // escaping closure over a value that is the caller's, an error. Instead the state object
    // borrows: the caller owns it, it may not outlive the argument (OWN_RESULT_BOUNDED_ATTR_NAME),
    // and its release gives back neither the parameter's value nor any field that only ever holds
    // what was read out of it (OWN_BORROWING_FIELDS_ATTR_NAME). A state type is unique to its
    // generator - its `next` field's type names a per-site `object_storage` - so a fact per type is
    // a fact per generator, which is what the per-type release routine needs.

    struct BorrowingState
    {
        mlir_ts::FuncOp maker;
        int32_t captured = -1;
        llvm::SmallVector<int32_t> bounded;
        llvm::SetVector<int32_t> fields;
    };

    static int32_t fieldIndex(mlir_ts::TupleType tupleType, llvm::StringRef name)
    {
        for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
        {
            if (auto id = mlir::dyn_cast_or_null<mlir::StringAttr>(field.id); id && id.getValue() == name)
            {
                return static_cast<int32_t>(index);
            }
        }

        return -1;
    }

    // The box the maker stores into the `.captured` field of the initial state it copies into
    // `newOp`'s block: `ts.Store(box, ts.PropertyRef(state, captured))`, `state` a local whose only
    // read fills the block.
    static mlir::Value boxStoredIntoState(mlir_ts::NewOp newOp, int32_t captured)
    {
        mlir_ts::StoreOp fill;
        for (auto *user : newOp->getUsers())
        {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getReference() == newOp.getResult())
            {
                fill = storeOp;
            }
        }

        auto loadOp = fill ? fill.getValue().getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
        auto stateOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!stateOp)
        {
            return {};
        }

        mlir::Value box;
        for (auto *user : stateOp->getUsers())
        {
            auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!propertyRefOp || propertyRefOp.getPosition() != captured)
            {
                continue;
            }

            for (auto *fieldUser : propertyRefOp->getUsers())
            {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || box)
                {
                    return {};
                }

                box = storeOp.getValue();
            }
        }

        auto boxOp = box ? box.getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        return boxOp && boxOp->hasAttr(CAPTURE_BOX_ATTR_NAME) ? box : mlir::Value();
    }

    // Is every caller of `funcOp` a direct call this pass sees? It is private, and each use of its
    // symbol is a call, or a closure over it that is only retained and released - `.map(f)` builds
    // one and calls its body straight through the box.
    bool calledOnlyDirectly(mlir::ModuleOp module, mlir_ts::FuncOp funcOp, bool publicToo = false)
    {
        if (!funcOp.isPrivate() && !publicToo)
        {
            return false;
        }

        auto uses = mlir::SymbolTable::getSymbolUses(funcOp.getSymNameAttr(), &module.getBodyRegion());
        if (!uses)
        {
            return false;
        }

        for (auto &use : *uses)
        {
            auto *user = use.getUser();
            if (mlir::isa<mlir_ts::SymbolCallInternalOp, mlir_ts::CallOp, mlir_ts::InvokeOp>(user))
            {
                continue;
            }

            // a method whose signature gives its callers the bound: called through its class's
            // vtable or a reference to it, which the facts reach too (declaredByCallees)
            if (publicToo &&
                (isClassVTableEntry(user) ||
                 (mlir::isa<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::VirtualSymbolRefOp, mlir_ts::ThisSymbolRefOp,
                            mlir_ts::SymbolRefOp>(user) &&
                  onlyCalled(user->getResult(0)))))
            {
                continue;
            }

            auto symbolRefOp = mlir::dyn_cast<mlir_ts::SymbolRefOp>(user);
            if (!symbolRefOp)
            {
                return false;
            }

            for (auto *refUser : symbolRefOp->getUsers())
            {
                auto boundOp = mlir::dyn_cast<mlir_ts::CreateBoundFunctionOp>(refUser);
                if (!boundOp || !llvm::all_of(boundOp->getUsers(), [](mlir::Operation *boundUser) {
                        return mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp>(boundUser);
                    }))
                {
                    return false;
                }
            }
        }

        return true;
    }

    // A parameter's cell: a captured local declared from an argument of `maker`'s entry block, that
    // owns nothing (its value is the caller's).
    static bool isParameterCell(mlir_ts::VariableOp cellOp, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto argument = cellOp.getInitializer() ? mlir::dyn_cast<mlir::BlockArgument>(cellOp.getInitializer())
                                                : mlir::BlockArgument();
        if (!cellOp.getCaptured().value_or(false) || isOwningVariable(cellOp) || !argument ||
            !argument.getOwner()->isEntryBlock() || argument.getOwner()->getParentOp() != maker.getOperation())
        {
            return false;
        }

        index = static_cast<int32_t>(argument.getArgNumber());
        return true;
    }

    // `ts.Load` of one of `maker`'s parameter slots (isParameterSlot): `index` is its position.
    static bool readsParameterSlot(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        int position = -1;
        if (!varOp || !isParameterSlot(varOp, position) || varOp->getParentOfType<mlir_ts::FuncOp>() != maker)
        {
            return false;
        }

        index = static_cast<int32_t>(position);
        return true;
    }

    // `ts.Load(ts.PropertyRef(argument, i))` of one of `maker`'s parameters: `index` is its position.
    static bool readsParameterField(mlir::Value value, mlir_ts::FuncOp maker, int32_t &index)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
        auto argument = propertyRefOp ? mlir::dyn_cast<mlir::BlockArgument>(propertyRefOp->getOperand(0)) : mlir::BlockArgument();
        if (!argument || !argument.getOwner()->isEntryBlock() || argument.getOwner()->getParentOp() != maker.getOperation())
        {
            return false;
        }

        index = static_cast<int32_t>(argument.getArgNumber());
        return true;
    }

    bool ownsHeap(mlir::Location location, mlir::Type type)
    {
        return own_facts::ownsHeap(location, type);
    }

    void findBorrowingStates(mlir::ModuleOp module)
    {
        llvm::MapVector<mlir::Type, BorrowingState> states;
        llvm::DenseMap<mlir::Type, int> made;
        module.walk([&](mlir_ts::NewOp newOp) {
            auto valueRefType = mlir::dyn_cast<mlir_ts::ValueRefType>(newOp.getType());
            auto tupleType = valueRefType ? mlir::dyn_cast<mlir_ts::TupleType>(valueRefType.getElementType())
                                          : mlir_ts::TupleType();
            if (!tupleType)
            {
                return;
            }

            ++made[tupleType];
            auto captured = fieldIndex(tupleType, CAPTURED_NAME);
            auto maker = newOp->getParentOfType<mlir_ts::FuncOp>();
            // an open maker has callers this pass cannot give the bound to, unless its signature
            // gives it to them (`@borrows`)
            if (captured < 0 || !maker ||
                !calledOnlyDirectly(module, maker, /*publicToo=*/maker->hasAttr(OWN_DECLARED_BORROWS_ATTR_NAME)))
            {
                return;
            }

            auto box = boxStoredIntoState(newOp, captured);
            if (!box)
            {
                return;
            }

            BorrowingState state{maker, captured};
            for (auto *user : box.getUsers())
            {
                auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                if (!propertyRefOp)
                {
                    continue;
                }

                for (auto *fieldUser : propertyRefOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                    if (!storeOp)
                    {
                        return;
                    }

                    auto value = storeOp.getValue();
                    int32_t index = -1;
                    if (auto cellOp = value.getDefiningOp<mlir_ts::VariableOp>(); cellOp && isParameterCell(cellOp, maker, index))
                    {
                        auto cellType = mlir::cast<mlir_ts::RefType>(cellOp.getType());
                        // a handle parameter's cell holds a count of its own (spec 23.2), which only
                        // the box's release gives back: a box holding one owns what it holds
                        if (MLIRTypeHelper::isSharedHandleType(cellType.getElementType()))
                        {
                            return;
                        }

                        if (ownsHeap(cellOp.getLoc(), cellType.getElementType()))
                        {
                            state.bounded.push_back(index);
                        }

                        continue;
                    }

                    // a copy of what a parameter holds in a field: `.map`'s maker is a closure body
                    // that copies its own box's fields into the generator's
                    if (!ownsHeap(value.getLoc(), value.getType()))
                    {
                        continue;
                    }

                    // `this`, which the box holds by value (`.this`), not in a cell: read out of
                    // the parameter's slot
                    if (readsParameterSlot(value, maker, index))
                    {
                        if (MLIRTypeHelper::isSharedHandleType(value.getType()))
                        {
                            return;
                        }

                        if (!llvm::is_contained(state.bounded, index))
                        {
                            state.bounded.push_back(index);
                        }

                        continue;
                    }

                    if (!readsParameterField(value, maker, index))
                    {
                        return; // a box with anything but what the parameters hold owns what it holds
                    }

                    if (!llvm::is_contained(state.bounded, index))
                    {
                        state.bounded.push_back(index);
                    }
                }
            }

            if (!state.bounded.empty())
            {
                states[tupleType] = state;
            }
        });

        // made in one place only: the maker is the one function a bound can come from
        states.remove_if([&](auto &entry) { return made[entry.first] != 1; });
        for (auto &[type, state] : states)
        {
            auto bounded = state.bounded;
            llvm::sort(bounded);
            borrowsFound[state.maker] = bounded;
        }
        if (states.empty())
        {
            return;
        }

        // Each store into a field of a state type, by field: what the field may be given.
        llvm::DenseMap<std::pair<mlir::Type, int64_t>, llvm::SmallVector<mlir::Value>> stored;
        module.walk([&](mlir_ts::StoreOp storeOp) {
            auto propertyRefOp = storeOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
            auto tupleType = propertyRefOp ? storageTupleOf(propertyRefOp->getOperand(0).getType()) : mlir_ts::TupleType();
            if (tupleType && states.count(tupleType))
            {
                stored[{tupleType, propertyRefOp.getPosition()}].push_back(storeOp.getValue());
            }
        });

        for (auto &[type, state] : states)
        {
            auto tupleType = mlir::cast<mlir_ts::TupleType>(type);
            // least fixpoint: a field borrows once everything stored into it is a read of a borrowing
            // cell or field
            for (auto changed = true; changed;)
            {
                changed = false;
                for (auto [index, field] : llvm::enumerate(tupleType.getFields()))
                {
                    auto position = static_cast<int32_t>(index);
                    auto found = stored.find({type, position});
                    if (position == state.captured || state.fields.contains(position) || found == stored.end() ||
                        !ownsHeap(state.maker.getLoc(), field.type))
                    {
                        continue;
                    }

                    if (llvm::all_of(found->second, [&](mlir::Value value) { return readsBorrowed(value, tupleType, state); }))
                    {
                        state.fields.insert(position);
                        changed = true;
                    }
                }
            }
        }

        auto *context = &getContext();
        llvm::SmallVector<mlir::Attribute> entries;
        for (auto &[type, state] : states)
        {
            llvm::SmallVector<int32_t> fields{state.captured};
            fields.append(state.fields.begin(), state.fields.end());
            entries.push_back(mlir::ArrayAttr::get(
                context, {mlir::TypeAttr::get(type), mlir::DenseI32ArrayAttr::get(context, fields)}));
            state.maker->setAttr(OWN_RESULT_BOUNDED_ATTR_NAME, mlir::DenseI32ArrayAttr::get(context, state.bounded));
        }

        module->setAttr(OWN_BORROWING_FIELDS_ATTR_NAME, mlir::ArrayAttr::get(context, entries));
    }

    // Is `value` a read of what a state of `tupleType` borrows: the value in one of its box's cells
    // (`ts.Load` of `ts.Load(ts.PropertyRef(ts.Load(ts.PropertyRef(state, captured)), i))`), or a
    // borrowing field's (`ts.Load(ts.PropertyRef(state, p))`)?
    bool readsBorrowed(mlir::Value value, mlir_ts::TupleType tupleType, const BorrowingState &state)
    {
        auto loadOp = rootOf(value).getDefiningOp<mlir_ts::LoadOp>();
        if (!loadOp)
        {
            return false;
        }

        auto isStateField = [&](mlir::Value ref, int32_t &position) {
            auto propertyRefOp = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
            if (!propertyRefOp || storageTupleOf(propertyRefOp->getOperand(0).getType()) != tupleType)
            {
                return false;
            }

            position = propertyRefOp.getPosition();
            return true;
        };

        int32_t position = -1;
        if (isStateField(loadOp.getReference(), position))
        {
            return state.fields.contains(position);
        }

        // a field of the box the state holds, copied in by value, or a cell read out of it
        auto isBoxField = [&](mlir::Value ref) {
            auto fieldRef = ref.getDefiningOp<mlir_ts::PropertyRefOp>();
            auto boxLoad = fieldRef ? fieldRef->getOperand(0).getDefiningOp<mlir_ts::LoadOp>() : mlir_ts::LoadOp();
            return boxLoad && isStateField(boxLoad.getReference(), position) && position == state.captured;
        };

        if (isBoxField(loadOp.getReference()))
        {
            return true;
        }

        auto cellLoad = loadOp.getReference().getDefiningOp<mlir_ts::LoadOp>();
        return cellLoad && isBoxField(cellLoad.getReference());
    }

    // ---- Globals given a borrow ----

    // A global initialized with a read of another global, a field or an element (`let k = a`,
    // `let k = h.c`) would be a second owner of a block another owner keeps: the first overwrite
    // of either frees what the other still reads, and both give it back at exit. rc counts the
    // global's own reference (OwnedReturnConsumptionPass, claimGlobalInitializers); own cannot
    // move the value out of its owner, and a global cannot hold a borrow. A handle is counted
    // under own too.
    void rejectBorrowedGlobalInitializers(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            auto *block = globalOp.getInitializerBlock();
            auto resultOp = block && !block->empty() ? mlir::dyn_cast<mlir_ts::GlobalResultOp>(block->getTerminator())
                                                     : mlir_ts::GlobalResultOp();
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return;
            }

            auto value = resultOp->getOperand(0);
            while (value.getDefiningOp() && isView(value.getDefiningOp()))
            {
                value = value.getDefiningOp()->getOperand(0);
            }

            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            auto *place = loadOp ? loadOp.getReference().getDefiningOp() : nullptr;
            // an immortal global's value (spec 25) is owned by no one: any number of globals hold it
            if (!place || !mlir::isa<mlir_ts::AddressOfOp, mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(place) ||
                isHandle(value) || isConstantData(value) || isImmortalGlobalRead(value) ||
                !ownsHeap(value.getLoc(), value.getType()))
            {
                return;
            }

            resultOp.emitError("the initializer of global '")
                << globalOp.getSymName()
                << "' reads a value another owner keeps; -mm=own cannot make a global a second owner";
            signalPassFailure();
        });
    }

    // ---- Drops ----

    // Least fixpoint: a function drops nothing until its body or a call it makes shows it may.
    void computeDrops()
    {
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (!funcOp.isDeclaration() && !dropsInBody(funcOp))
            {
                noDrops.insert(funcOp);
            }
        }

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                if (!noDrops.contains(funcOp))
                {
                    continue;
                }

                for (auto *call : callsIn[funcOp])
                {
                    if (!callMayDrop(call->callees))
                    {
                        continue;
                    }

                    noDrops.erase(funcOp);
                    changed = true;
                    break;
                }
            }
        }
    }

    bool callMayDrop(const Callees &callees)
    {
        if (!callees.known)
        {
            return true;
        }

        return !callees.instanceOf && !callees.imported &&
               llvm::any_of(callees.funcs, [&](mlir_ts::FuncOp callee) { return !noDrops.contains(callee); });
    }

    // ---- Globals a call may assign ----

    // The globals each call may give a new value, pinned on the call for the inference
    // (OWN_ASSIGNS_GLOBALS_ATTR_NAME, OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME): a store to a global, or its
    // release, in a callee's body, or in a call it makes. A call that may drop and whose callees are
    // not known may assign any; one that drops nothing assigns none. A read of a global ends where
    // the global is given a new value (checkGlobalRead), and a call that does so ends it as surely as
    // a store in the same function: `const c = G; reset(); print(c.x)`.
    void pinAssignedGlobals()
    {
        llvm::DenseMap<mlir::Operation *, llvm::SmallVector<Call *>> callsIn;
        for (auto &call : calls)
        {
            callsIn[call.op->getParentOfType<mlir_ts::FuncOp>()].push_back(&call);
        }

        llvm::DenseMap<mlir::Operation *, llvm::SetVector<mlir::StringRef>> assigns;
        llvm::DenseSet<mlir::Operation *> assignsAny;
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            funcOp.walk([&](mlir::Operation *op) {
                mlir::Value slot;
                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
                {
                    slot = storeOp.getReference();
                }
                else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
                {
                    slot = releaseSlotOp.getSlot();
                }

                if (auto addressOfOp = slot ? slot.getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp())
                {
                    assigns[funcOp].insert(addressOfOp.getGlobalName());
                }
            });
        }

        // what a call may assign, from what its callees do so far
        auto assignedBy = [&](Call &call, llvm::SetVector<mlir::StringRef> &globals) {
            if (!callMayDrop(call.callees))
            {
                return false;
            }

            if (!call.callees.known)
            {
                return true;
            }

            auto any = false;
            for (auto callee : call.callees.funcs)
            {
                any |= assignsAny.contains(callee);
                if (auto found = assigns.find(callee); found != assigns.end())
                {
                    globals.insert(found->second.begin(), found->second.end());
                }
            }

            return any;
        };

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : functions)
            {
                auto funcOp = entry.second;
                for (auto *call : callsIn[funcOp])
                {
                    llvm::SetVector<mlir::StringRef> globals;
                    if (assignedBy(*call, globals))
                    {
                        changed |= assignsAny.insert(funcOp).second;
                    }

                    for (auto global : globals)
                    {
                        changed |= assigns[funcOp].insert(global);
                    }
                }
            }
        }

        for (auto &call : calls)
        {
            llvm::SetVector<mlir::StringRef> globals;
            if (assignedBy(call, globals))
            {
                call.op->setAttr(OWN_ASSIGNS_ANY_GLOBAL_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                continue;
            }

            if (globals.empty())
            {
                continue;
            }

            llvm::SmallVector<mlir::Attribute> names;
            for (auto global : globals)
            {
                names.push_back(mlir::StringAttr::get(&getContext(), global));
            }

            call.op->setAttr(OWN_ASSIGNS_GLOBALS_ATTR_NAME, mlir::ArrayAttr::get(&getContext(), names));
        }
    }

    // An argument read out of a global (`run(G)`, `G.m()`) is lent to the callee, and the callee may
    // free it under its own parameter: `function run(h: C) { G = mk(); print(h.x); }`. The caller's
    // borrow ends at the call that takes it, so nothing in the caller sees that, and in the callee
    // the parameter is not known to be the global's. A call that may drop is an error for such an
    // argument, when something assigns that global. (A field of a global given that way is a
    // borrow of a place, which the inference already bounds by the call.)
    void rejectGlobalArgumentsDropped()
    {
        for (auto &call : calls)
        {
            if (!callMayDrop(call.callees))
            {
                continue;
            }

            for (auto argument : callArgs(call.op))
            {
                // a method's object: `this` of the method bound to it
                if (auto getThisOp = argument.getDefiningOp<mlir_ts::GetThisOp>())
                {
                    auto *boundOp = getThisOp->getOperand(0).getDefiningOp();
                    if (mlir::isa_and_nonnull<mlir_ts::ThisSymbolRefOp, mlir_ts::ThisVirtualSymbolRefOp>(boundOp))
                    {
                        argument = boundOp->getOperand(0);
                    }
                }

                // an immortal global (spec 25) is written by its initializer only, and that runs once
                // (a static's is a store in its static constructor, outside any loop, which
                // writtenGlobals counts), and no release frees its block
                if (isHandle(argument) || isImmortalGlobalRead(argument) || !ownsHeap(call.op->getLoc(), argument.getType()))
                {
                    continue;
                }

                auto loadOp = rootOf(argument).getDefiningOp<mlir_ts::LoadOp>();
                auto global = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>() : mlir_ts::AddressOfOp();
                if (!global || !writtenGlobals.contains(global.getGlobalName()))
                {
                    continue;
                }

                call.op->emitError("a value read out of the global '")
                    << global.getGlobalName()
                    << "' is lent to a call that may release or overwrite it; -mm=own cannot prove it outlives the call";
                signalPassFailure();
            }
        }
    }

    // ---- Calls in a global's initializer ----

    // A global's region is no function: its calls get no facts pinned, and the inference never sees
    // it, so its calls were read with the defaults - an owned result, borrowed arguments. A result
    // that borrows an argument (`const c = child(n)`, `id(new V(1))`) then made the global a second
    // owner of a block its argument's owner keeps, or kept one the region freed; and a fresh
    // argument a callee keeps (`new N(new V(4))`, the constructor keeping `c`) was released at the
    // end of the region, under the object that kept it. A kept argument made in the region and used
    // by nothing after the call moves into the callee: its release goes. Anything else is an error.
    void ownGlobalInitializerCalls(mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            // collected first: a move erases a release the walk has not reached yet
            llvm::SmallVector<mlir::Operation *> calls;
            globalOp.walk([&](mlir::Operation *op) {
                if (isCall(op))
                {
                    calls.push_back(op);
                }
            });

            for (auto *op : calls)
            {
                auto callees = resolve(op);
                auto facts = callFacts(callees);
                auto calleeName = callees.funcs.empty() ? llvm::StringRef("the callee") : callees.funcs.front().getSymName();
                if (!facts.borrows.empty())
                {
                    op->emitError("the initializer of global '")
                        << globalOp.getSymName() << "' holds the result of '" << calleeName
                        << "', which borrows its argument; a global cannot hold a borrow under -mm=own";
                    signalPassFailure();
                    continue;
                }

                auto args = callArgs(op);
                llvm::SmallPtrSet<mlir::Value, 4> given;
                for (auto index : facts.owned)
                {
                    if (index < 0 || index >= static_cast<int32_t>(args.size()) ||
                        !given.insert(rootOf(args[index])).second || !moveIntoCallee(op, args[index]))
                    {
                        op->emitError("the initializer of global '")
                            << globalOp.getSymName() << "' gives '" << calleeName
                            << "' a value it keeps, but -mm=own cannot move it there";
                        signalPassFailure();
                        break;
                    }
                }
            }
        });
    }

    // `value` was made in the call's block and nothing but its release, if it has one, comes after
    // the call: the release goes, and the callee owns the block. Constant data (a string literal)
    // is nobody's, and is given as it is.
    static bool moveIntoCallee(mlir::Operation *call, mlir::Value value)
    {
        auto root = rootOf(value);
        if (isConstantData(root))
        {
            return true;
        }

        if (!isFresh(root) || root.getDefiningOp()->getBlock() != call->getBlock())
        {
            return false;
        }

        mlir::Operation *release = nullptr;
        auto usedAfter = false;
        forEachUse(root, [&](mlir::Operation *user, mlir::Value) {
            if (user == call)
            {
                return;
            }

            if (user->getBlock() != call->getBlock())
            {
                usedAfter = true;
            }
            else if (call->isBeforeInBlock(user))
            {
                usedAfter = usedAfter || release || !mlir::isa<mlir_ts::ReleaseOp>(user);
                release = mlir::isa<mlir_ts::ReleaseOp>(user) ? user : release;
            }
        });

        if (usedAfter)
        {
            return false;
        }

        if (release)
        {
            release->erase();
        }

        return true;
    }

    // ---- Export ----

    // A library says, beside its memory-model marker, which of its exported functions destroy
    // nothing (SHARED_LIB_OWN_FACTS). The global is a copy of the marker `__tsmm_own_<module>`
    // under another name and with another value, so it is exported exactly as the marker is, and
    // MLIRGen emits the same under own as under rc. No marker, nothing is exported.
    void exportNoDrops()
    {
        auto markerPrefix = std::string(SHARED_LIB_MEMORY_MODEL) + "own_";
        mlir_ts::GlobalOp marker;
        for (auto &entry : globals)
        {
            if (entry.getKey().starts_with(markerPrefix))
            {
                marker = entry.second;
                break;
            }
        }

        if (!marker)
        {
            return;
        }

        llvm::SmallVector<llvm::StringRef> names;
        for (auto &entry : functions)
        {
            auto funcOp = entry.second;
            if (!funcOp.isDeclaration() && !funcOp.isPrivate() && noDrops.contains(funcOp))
            {
                names.push_back(funcOp.getSymName());
            }
        }

        llvm::sort(names);
        std::string text;
        for (auto name : names)
        {
            text += name.str();
            text += '\n';
        }

        auto suffix = marker.getSymName().drop_front(markerPrefix.size());
        auto name = (llvm::Twine(SHARED_LIB_OWN_FACTS) + suffix).str();

        mlir::OpBuilder builder(marker);
        builder.setInsertionPointAfter(marker);
        auto factsOp = mlir::cast<mlir_ts::GlobalOp>(builder.clone(*marker.getOperation()));
        factsOp.setSymName(name);
        factsOp.getInitializerRegion().walk([&](mlir_ts::ConstantOp constantOp) {
            if (mlir::isa<mlir::StringAttr>(constantOp.getValue()))
            {
                constantOp.setValueAttr(builder.getStringAttr(text));
            }
        });
    }

    // Does the body itself destroy something a caller may reach: overwrite a field or an element
    // of a block it did not make, assign a global, remove elements from an array it did not make,
    // or `delete`? A local's own releases are not drops, nor is the constructor's filling of the
    // object it is building.
    bool dropsInBody(mlir_ts::FuncOp funcOp)
    {
        auto drops = false;
        funcOp.walk([&](mlir::Operation *op) {
            if (drops)
            {
                return;
            }

            // A handle given back may be the last one, and its payload goes with it, and with that
            // anything the payload owns (spec 23.3). One held by a field or an element is a place's
            // overwrite, judged below like any other: a constructor filling its own `this` is not
            // a drop.
            if (mlir::isa<mlir_ts::ReleaseOp>(op) && touchesHandle(op))
            {
                drops = true;
            }
            else if (auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(op))
            {
                // a captured variable assigned through its cell, by a closure's body: the variable
                // is its declaring function's
                auto slot = releaseSlotOp.getSlot();
                drops = slot.getDefiningOp<mlir_ts::AddressOfOp>() || (isPlace(slot) && reachesOutside(slot, funcOp)) ||
                        isLoadedCell(slot) || (!isPlace(slot) && touchesHandle(op));
            }
            else if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op))
            {
                drops = reachesOutside(op->getOperand(0), funcOp);
            }
            else if (mlir::isa<mlir_ts::ArrayPushOp, mlir_ts::ArrayUnshiftOp>(op))
            {
                // through a handle, it may move the elements another handle's borrow reads (spec 23.3)
                drops = reachesSharedValue(op->getOperand(0));
            }
            else if (mlir::isa<mlir_ts::DeleteOp>(op))
            {
                drops = true;
            }
        });

        return drops;
    }

    // Can the block this reference points into be one the function's caller can reach? Walks up
    // from a place to what holds it. Not when every way up ends in a block the function made (an
    // allocation, a call's result it owns) or in an owning local that only ever held such blocks,
    // or, in a constructor, in the object being built.
    bool reachesOutside(mlir::Value ref, mlir_ts::FuncOp funcOp)
    {
        auto isConstructor = funcOp.getSymName().ends_with(".constructor");
        llvm::SmallVector<mlir::Value> work{ref};
        llvm::DenseSet<mlir::Value> seen;
        while (!work.empty())
        {
            auto value = rootOf(work.pop_back_val());
            if (!seen.insert(value).second)
            {
                continue;
            }

            if (auto object = boundThis(value))
            {
                work.push_back(object);
                continue;
            }

            if (auto place = placeBase(value))
            {
                // a handle's payload: any other handle may reach it (spec 23.3)
                if (place.kind == PlaceKind::SharedValue)
                {
                    return true;
                }

                work.push_back(place.base);
                continue;
            }

            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
            {
                llvm::SmallVector<mlir::Value> merged;
                if (mergedInto(argument, merged))
                {
                    work.append(merged.begin(), merged.end());
                    continue;
                }

                if (isConstructor && argument.getOwner()->isEntryBlock() && argument.getArgNumber() == 0)
                {
                    continue; // the object being built
                }

                return true;
            }

            if (auto varOp = value.getDefiningOp<mlir_ts::VariableOp>())
            {
                if (holdsOnlyFresh(varOp))
                {
                    continue;
                }

                // a parameter's slot: what the caller passed
                if (auto init = varOp.getInitializer())
                {
                    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(init);
                        argument && isConstructor && argument.getOwner()->isEntryBlock() && argument.getArgNumber() == 0 &&
                        !isAssigned(varOp))
                    {
                        continue;
                    }
                }

                return true;
            }

            if (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
            {
                work.push_back(loadOp.getReference());
                continue;
            }

            if (isFresh(value))
            {
                continue;
            }

            return true;
        }

        return false;
    }

    static bool isAssigned(mlir_ts::VariableOp varOp)
    {
        return llvm::any_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            return storeOp && storeOp.getReference() == varOp.getResult();
        });
    }

    // An owning, uncaptured local whose every value - its initializer and each assignment - is one
    // the function made.
    static bool holdsOnlyFresh(mlir_ts::VariableOp varOp)
    {
        if (!isOwningVariable(varOp) || varOp.getCaptured().value_or(false))
        {
            return false;
        }

        if (auto init = varOp.getInitializer(); init && !isFresh(rootOf(init)))
        {
            return false;
        }

        return llvm::all_of(varOp.getResult().getUsers(), [&](mlir::Operation *user) {
            auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
            return !storeOp || storeOp.getReference() != varOp.getResult() || isFresh(rootOf(storeOp.getValue()));
        });
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

std::unique_ptr<mlir::Pass> mlir_ts::createOwnershipSignaturePass(bool executable)
{
    return std::make_unique<OwnershipSignaturePass>(executable);
}
