#include "mlir/Pass/Pass.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"
#include "TypeScript/MLIRLogic/MLIROwnedReference.h"

#include "OwnershipFacts.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <set>
#include <tuple>

#define DEBUG_TYPE "pass"

namespace mlir_ts = mlir::typescript;

namespace
{

// Lets a call take over the reference its callee returned, instead of retaining a second one.
//
// Every function retains its result on the way out (§9.24), so a receiver that retains it again
// - `let y = f()`, `h.item = f()` - is one owner above the truth and nothing is ever freed. The
// receiving sites already know how to consume such a value (§9.25); what they cannot know, at
// the point MLIRGen builds the call, is whether *this* callee is one that retains. Three things
// stop that being answerable there: the retain lives in the return statement, so other paths to
// a return reach it differently; the callee's FuncOp need not exist yet, which would make a
// lookup depend on declaration order; and a declared, imported or runtime callee looks identical
// to a local one while having no retaining return at all.
//
// All three dissolve once every function is present, which is why this is a pass rather than
// another marking site. It does not predict which callees retain - it looks. A function counts
// as returning owned only when every `ts.ReturnVal` (and, in a generator under rc, every
// `ts.YieldReturnVal`) of a heap-owning value in it is preceded by a `ts.Retain` of that same
// value. A callee with no body, a generator under any other model, and anything else that does
// not match are simply not marked, and their
// callers go on retaining - which leaks rather than frees something live. The whole design puts
// the uncertain case on the leaking side.
//
// It runs in every memory model. The ops it removes erase on the way to LLVM under `gc` and
// `none` anyway, so removing them early keeps the IR the same shape in all three and keeps the
// ownership verifier checking one thing rather than two.
class OwnedReturnConsumptionPass
    : public mlir::PassWrapper<OwnedReturnConsumptionPass, mlir::OperationPass<mlir::ModuleOp>>
{
    CompileOptions compileOptions;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnedReturnConsumptionPass)

    OwnedReturnConsumptionPass(CompileOptions &compileOptions) : compileOptions(compileOptions)
    {
    }

    void runOnOperation() override
    {
        auto module = getOperation();
        MLIRTypeHelper mth(module->getContext(), compileOptions);

        copyEscapingStringViews(module);

        retainYieldedRecords(mth, module);

        holdGlobalArguments(mth, module);

        own_facts::FieldFunctions fieldFunctionsOfModule;
        const own_facts::FieldFunctions *fieldFunctions = nullptr;
        if (compileOptions.isRefCounted())
        {
            fieldFunctionsOfModule.collect(module);
            fieldFunctions = &fieldFunctionsOfModule;
        }

        llvm::DenseSet<mlir::StringRef> returnsOwned;
        // Every method in the module, grouped by the name after the last dot, with the class it
        // takes first: the candidate set for a virtual call, see virtualCallReturnsOwned.
        llvm::StringMap<llvm::SmallVector<std::pair<mlir::StringRef, mlir::Type>>> methodsByMemberName;
        // Is there any function here at all that hands back a heap value without retaining it?
        // See callThroughValueReturnsOwned.
        auto anyUnclassifiedOwningReturn = false;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            // a method of an interface's null vtable throws, and returns nothing to anyone
            // (mlirGenNullInterfaceVTable)
            if (funcOp->hasAttr(NULL_INTERFACE_VTABLE_ATTR_NAME))
            {
                return;
            }

            auto name = funcOp.getName();
            auto classified = functionReturnsOwned(mth, funcOp, compileOptions.isRefCounted());
            if (classified)
            {
                returnsOwned.insert(name);
            }

            if (!classified)
            {
                for (auto resultType : funcOp.getFunctionType().getResults())
                {
                    if (mth.ownsHeapMemory(funcOp.getLoc(), resultType))
                    {
                        anyUnclassifiedOwningReturn = true;
                        LLVM_DEBUG(llvm::dbgs() << "\n!! unclassified owning return: " << name << "\n";);                    }
                }
            }

            // a class's method, which takes its object first: what a virtual call can reach. An
            // object literal's method, or a generator's `next`, shares the member name only.
            auto inputs = funcOp.getFunctionType().getInputs();
            auto isClassMethod = !inputs.empty() && mlir::isa<mlir_ts::ClassType, mlir_ts::ClassStorageType>(inputs.front());
            auto dot = name.rfind('.');
            if (isClassMethod && dot != mlir::StringRef::npos && dot + 1 < name.size())
            {
                methodsByMemberName[name.substr(dot + 1)].push_back({name, inputs.front()});
            }
        });

        // What each vtable global puts in each of its slots, for the interface half of
        // virtualCallReturnsOwned. Only slots initialised from a symbol are recorded; a slot that
        // holds anything else - a field's offset, say - is simply absent, and absent means
        // unclassifiable there rather than ignorable.
        // An interface's null vtable is not one: its methods throw, and never return anything
        // (mlirGenNullInterfaceVTable).
        llvm::StringMap<llvm::DenseMap<int64_t, mlir::StringRef>> vtableSlots;
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            if (!globalOp.getSymName().ends_with(VTABLE_NAME) || globalOp->hasAttr(NULL_INTERFACE_VTABLE_ATTR_NAME))
            {
                return;
            }

            auto &slots = vtableSlots[globalOp.getSymName()];
            globalOp.walk([&](mlir_ts::InsertPropertyOp insertOp) {
                auto position = insertOp.getPosition();
                if (position.size() != 1)
                {
                    return;
                }

                if (auto symbolRefOp = insertOp.getValue().getDefiningOp<mlir_ts::SymbolRefOp>())
                {
                    slots[position[0]] = symbolRefOp.getIdentifier();
                }
            });
        });

        // -mm=own: a function another module defines, whose declaration says `@record`, returns a box
        // its caller owns (own spec 32) - the one result of an import this pass can count as owned
        auto own = compileOptions.memoryModel == MemoryModelOwn;
        auto anyDeclaredRecord = false;
        if (own)
        {
            module.walk([&](mlir::Operation *op) {
                if (mlir::isa<mlir_ts::FuncOp, mlir_ts::GlobalOp>(op) && op->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME))
                {
                    anyDeclaredRecord = true;
                }
            });
        }

        // `new C()` is marked where it is built, so there can be discarded temporaries to give
        // back even when no function here is classified as returning owned - and conditionals and
        // pattern defaults to settle: a module with no such function (no class, no function returning
        // a heap value) skipped consumeConditionalResults, and `const xs: number[] = c ? [i] : []`
        // under rc kept the branch's count of the new array and the local's own (one leaked every
        // evaluation); settleOwnPatternDefaults, under own, never ran there either.
        if (returnsOwned.empty() && !anyDeclaredRecord)
        {
            settleModuleLocalResults(mth, module);
            return;
        }

        // Every adapter for calls through `any` here returns owned (they are all made by the same code, as
        // they are in another module): then so does the call of one in `__any_invoke`, whichever it reaches.
        // One that is not turns it off - the leaking side, as for any other call.
        auto everyAnyCallAdapterReturnsOwned = true;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            if (funcOp->hasAttr(ANY_CALL_ADAPTER_ATTR_NAME) && !returnsOwned.contains(funcOp.getName()))
            {
                everyAnyCallAdapterReturnsOwned = false;
            }
        });

        llvm::SmallVector<mlir::Operation *> toErase;
        module.walk([&](mlir_ts::CallIndirectOp callOp) {
            if (callOp.getNumResults() != 1)
            {
                return;
            }

            auto result = callOp.getResult(0);
            if (!mth.ownsHeapMemory(callOp.getLoc(), result.getType()))
            {
                return;
            }

            auto callee = calleeNameOf(callOp, fieldFunctions);
            auto calleeReturnsOwned = (!callee.empty() && returnsOwned.contains(callee)) ||
                                      (anyDeclaredRecord && calleeDeclaresRecord(module, callOp)) ||
                                      // the call of an adapter in `__any_invoke`: one of this module's, or
                                      // another's made by the same code
                                      (callOp->hasAttr(ANY_CALL_ADAPTER_CALL_ATTR_NAME) && everyAnyCallAdapterReturnsOwned);

            // A call through a plain value - a callback handed to `reduce`, a function-typed
            // field - names nothing at all, and neither vtables nor member names help. What does
            // help is that the question has a whole-module answer: if every function here that
            // hands back a heap value retains it first, then every call that returns one hands
            // back a reference, whatever it dispatched to. That is the +1 convention (§9.24)
            // stated over the module rather than over one callee.
            //
            // One unclassified function anywhere turns this off for every indirect call, which is
            // the conservative direction and the reason it is worth so little on its own and so
            // much here: `reduce` calls its argument, and `raytrace.ts` builds its colours through
            // `reduce`. An external function that returns a heap value counts as unclassified,
            // so linking against anything whose returns cannot be seen switches it off too -
            // unless it is an `export`ed import or says `@owned`, as the default library's
            // declarations do (importedFunctionReturnsOwned): otherwise every program with the
            // library had it off.
            auto closedWorld = !anyUnclassifiedOwningReturn;

            if (!calleeReturnsOwned && !closedWorld &&
                !virtualCallReturnsOwned(callOp, returnsOwned, methodsByMemberName) &&
                !interfaceCallReturnsOwned(callOp, returnsOwned, vtableSlots))
            {
                return;
            }

            // Already settled at the point the call was built - `new C()` of another module's
            // class calls its `C..new`, and is marked there, where the callee is known outright.
            // Nothing left to remove.
            if (callOp->hasAttr(OWNED_RESULT_ATTR_NAME))
            {
                return;
            }

            auto retains = findReceiverRetains(result);
            if (!retains.empty())
            {
                callOp->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                callOp->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                toErase.append(retains.begin(), retains.end());
                return;
            }

            // Nobody took it. The +1 stands with no owner, which is the leak §9.30 closes -
            // marked here, released below once every consumer has had its say.
            callOp->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
        });

        // A getter read is a call as well - `h.cc` becomes `H.get_cc(h)` in the affine lowering,
        // and the getter retains what it returns like any other function - so its result is
        // settled the same way. Left out, `h.cc.x` kept the getter's reference and nobody gave
        // it back: every temporary read through a getter leaked.
        module.walk([&](mlir::Operation *op) {
            if (!isGetterRead(op) || op->hasAttr(OWNED_RESULT_ATTR_NAME))
            {
                return;
            }

            auto result = op->getResult(0);
            if (!mth.ownsHeapMemory(op->getLoc(), result.getType()))
            {
                return;
            }

            // Assigning through an accessor builds a getter read first and a setter after it
            // (MLIRGen's assignment rebuilds the op with the value to set); the read is left with
            // no use, for the canonicalizer to delete. A release would be a use, and would keep a
            // call the program never made.
            if (!hasLiveUse(result))
            {
                return;
            }

            if (anyUnclassifiedOwningReturn &&
                !getterReturnsOwned(op, returnsOwned, methodsByMemberName, vtableSlots))
            {
                return;
            }

            auto retains = findReceiverRetains(result);
            if (!retains.empty())
            {
                op->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                op->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                toErase.append(retains.begin(), retains.end());
                return;
            }

            op->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
        });

        for (auto *op : toErase)
        {
            op->erase();
        }

        settleModuleLocalResults(mth, module);
    }

    // What settles results within a function, once calls' results are settled (or none is a call's to settle)
    void settleModuleLocalResults(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        ownNewArrayArguments(mth, module);

        settleOwnPatternDefaults(module);

        consumeConditionalResults(mth, module);

        claimGlobalInitializers(mth, module);

        holdReadOnlyNewArrays(mth, module);

        releaseDiscardedTemporaries(mth, module);
    }

    // An array literal that can own what it holds: each element that owns memory is
    // - a new value moved into it - an owned result its receiver consumed, used after that by nothing but the
    //   literal and its retains (`"x" + i`; a `new C()`, whose constructor's call and the like come before), or
    // - text that something else owns (`s`, a named string): the literal is given a copy of it
    //   (copyBorrowedTextElements), as a place a string is stored into is.
    // (`["x" + i, mk()]`, `[new C(), new C()]`, `["x" + i, s]`.) An object something else owns can't be copied.
    static bool isBorrowedText(mlir::Value element)
    {
        return own_facts::isCopyableString(element.getType());
    }

    static bool isMovedIn(mlir_ts::CreateArrayOp arrayOp, mlir::Value element)
    {
        auto *elementOp = element.getDefiningOp();
        return elementOp && elementOp->hasAttr(OWNED_RESULT_ATTR_NAME) && elementOp->hasAttr(OWNED_RESULT_CONSUMED_ATTR_NAME) &&
               llvm::count(arrayOp->getOperands(), element) == 1 &&
               llvm::all_of(element.getUsers(), [&](mlir::Operation *user) {
                   return user == arrayOp.getOperation() || mlir::isa<mlir_ts::RetainOp>(user) ||
                          (user->getBlock() == arrayOp->getBlock() && user->isBeforeInBlock(arrayOp));
               });
    }

    // ...or, under -mm=own, a literal nested in it (`[[mk(i)], [mk(2)]]`): a new array no one else uses - the outer
    // literal and the retains it took of it - that can own what it holds itself. It moves into the outer one as a
    // call's result does, and goes with it (claimNestedLiterals); before, the outer literal looked as if it
    // borrowed it, and neither was ever given back.
    static bool isNestedLiteral(MLIRTypeHelper &mth, mlir_ts::CreateArrayOp arrayOp, mlir::Value element)
    {
        auto innerOp = element.getDefiningOp<mlir_ts::CreateArrayOp>();
        return innerOp && !innerOp->hasAttr(OWNED_RESULT_ATTR_NAME) && innerOp->getBlock() == arrayOp->getBlock() &&
               llvm::count(arrayOp->getOperands(), element) == 1 &&
               llvm::all_of(element.getUsers(), [&](mlir::Operation *user) {
                   return user == arrayOp.getOperation() ||
                          (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == arrayOp->getBlock() && user->isBeforeInBlock(arrayOp));
               }) &&
               holdsWhatItIsGiven(mth, innerOp, /*givenToCall=*/false);
    }

    // ...or a tuple literal built in place (`[["a" + i, 1]]`, `new Map([["a" + i, 1]])`): a local nothing releases,
    // filled field by field, read whole once into this literal (and the retain the literal took of it). Each field
    // that owns memory is given a new value no one else keeps (`"a" + i`) or a string constant, so the tuple is a
    // new value moved in as `[mk(i)]` is, and goes with the literal. Before, the literal looked as if it borrowed the
    // tuple, was given back by nothing, and each call leaked it and what its fields held.
    static bool isInPlaceTupleLiteral(MLIRTypeHelper &mth, mlir_ts::CreateArrayOp arrayOp, mlir::Value element)
    {
        // (a tuple's fields have no names; an object literal's do, and is a record of its own: classifyRecords)
        auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(element.getType());
        if (!tupleType || llvm::any_of(tupleType.getFields(), [](mlir_ts::FieldInfo field) { return !!field.id; }) ||
            llvm::count(arrayOp->getOperands(), element) != 1)
        {
            return false;
        }

        auto onlyThisLiteral = llvm::all_of(element.getUsers(), [&](mlir::Operation *user) {
            return user == arrayOp.getOperation() ||
                   (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == arrayOp->getBlock() && user->isBeforeInBlock(arrayOp));
        });
        if (!onlyThisLiteral)
        {
            return false;
        }

        // a new value, or a string constant: nothing else owns either
        auto isNewField = [&](mlir::Value stored) {
            if (!mth.ownsHeapMemory(arrayOp->getLoc(), stored.getType()))
            {
                return true;
            }

            auto *storedOp = stored.getDefiningOp();
            if (!storedOp)
            {
                return false;
            }

            if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(storedOp); castOp && castOp.getIn().getDefiningOp<mlir_ts::ConstantOp>())
            {
                return true;
            }

            return storedOp->hasAttr(OWNED_RESULT_ATTR_NAME) && llvm::all_of(stored.getUsers(), [&](mlir::Operation *user) {
                       return mlir::isa<mlir_ts::StoreOp, mlir_ts::RetainOp, mlir_ts::ReleaseOp>(user) &&
                              (!mlir::isa<mlir_ts::StoreOp>(user) || user->getOperand(0) == stored);
                   });
        };

        // `["a", 1]` of values only: one `ts.CreateTuple`
        if (auto createTupleOp = element.getDefiningOp<mlir_ts::CreateTupleOp>())
        {
            return createTupleOp->getBlock() == arrayOp->getBlock() && llvm::all_of(createTupleOp->getOperands(), isNewField);
        }

        auto loadOp = element.getDefiningOp<mlir_ts::LoadOp>();
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        if (!varOp || varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) || varOp.getCaptured().value_or(false) || !varOp.getInitializer() ||
            !varOp.getInitializer().getDefiningOp<mlir_ts::ConstantOp>())
        {
            return false;
        }

        auto reads = 0;
        return llvm::all_of(varOp->getUsers(), [&](mlir::Operation *varUser) {
                   if (varUser == loadOp.getOperation())
                   {
                       return ++reads == 1;
                   }

                   auto fieldOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(varUser);
                   return fieldOp && fieldOp->getBlock() == arrayOp->getBlock() && fieldOp->isBeforeInBlock(loadOp) &&
                          llvm::all_of(fieldOp->getUsers(), [&](mlir::Operation *fieldUser) {
                              if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser))
                              {
                                  return storeOp.getReference() == fieldOp.getResult() && isNewField(storeOp.getValue());
                              }

                              return mlir::isa<mlir_ts::LoadOp>(fieldUser) && fieldUser->use_empty();
                          });
               }) &&
               reads == 1;
    }

    // ...or, under -mm=own, a value something else had, which the literal took a retain of (`[mk(i + 4), o]` of a local
    // `o`, `[mk(i), GA[0]]`). A literal that does not borrow takes every item it is given (callGivenAgain): the
    // inference moves the value in (`o`'s own release goes), counts it with --own-fallback=count (the retain stays,
    // and the literal's release gives that count back), or refuses the program - never leaves it a borrow, which
    // only a borrowing array does (classifyArgumentArrays, classifyLocalBorrowingArrays). Those take a literal with
    // no new value in it, so one holding such a value is owned only with a new value beside it, or when each taken
    // value is a local's own (`[o].length`, `[o][0].x`): an owning `let`, or a `const` folded into the value it was
    // given (OWNED_RESULT_NAMED_ATTR_NAME), which neither borrowing array takes - unless it is given to a call
    // (`g([o])`, borrowed by classifyArgumentArrays).
    // (The same value given more than once, `f([mk(2), o, o])`, takes a retain each time: plain own refuses the second
    // owner, and --own-fallback=count keeps each retain, which the literal's release gives back one by one.)
    static bool isTakenElement(mlir_ts::CreateArrayOp arrayOp, mlir::Value element)
    {
        auto retains = llvm::count_if(element.getUsers(), [&](mlir::Operation *user) {
            return mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == arrayOp->getBlock() && user->isBeforeInBlock(arrayOp);
        });
        auto given = llvm::count(arrayOp->getOperands(), element);
        return retains > 0 && retains >= given;
    }

    static bool isLocalsOwn(mlir::Value element)
    {
        auto *definingOp = element.getDefiningOp();
        if (!definingOp)
        {
            return false;
        }

        if (definingOp->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME) && definingOp->hasAttr(OWNED_RESULT_ATTR_NAME))
        {
            return true;
        }

        auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(definingOp);
        auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
        return varOp && varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) && !varOp.getCaptured().value_or(false);
    }

    static bool holdsWhatItIsGiven(MLIRTypeHelper &mth, mlir_ts::CreateArrayOp arrayOp, bool givenToCall)
    {
        if (arrayOp->getNumOperands() == 0)
        {
            return false;
        }

        auto anyNew = false;
        auto allTakenAreLocals = true;
        // (`po([undefined])`, `[null]`: elements that are no value at all, so nothing to own or to borrow - such a
        // literal given to a call was never given back, own leaked it every call)
        auto onlyNoValues = true;
        // text it holds a copy of (copyBorrowedTextElements), or was given new (`["x" + i]`): its own, as a new value
        // moved in is. Not counted, a literal of strings alone given to a call (`new Set(["a" + i])`, `len(["x" + i])`)
        // was owned by nothing and given back by nothing: each call leaked it and its strings
        // (not beside a value something else had, which the literal would take over: `[s, o]` borrows both)
        auto holdsText = false;
        auto anyTaken = false;
        for (auto element : arrayOp->getOperands())
        {
            if (isNoValue(element))
            {
                continue;
            }

            if (!mth.ownsHeapMemory(arrayOp->getLoc(), element.getType()) || isBorrowedText(element))
            {
                onlyNoValues = onlyNoValues && !isBorrowedText(element);
                holdsText = holdsText || isBorrowedText(element);
                continue;
            }

            onlyNoValues = false;

            if (isMovedIn(arrayOp, element) || isNestedLiteral(mth, arrayOp, element) || isInPlaceTupleLiteral(mth, arrayOp, element))
            {
                anyNew = true;
                continue;
            }

            if (!isTakenElement(arrayOp, element))
            {
                return false;
            }

            anyTaken = true;
            allTakenAreLocals = allTakenAreLocals && isLocalsOwn(element);
        }

        return anyNew || (holdsText && !anyTaken) || onlyNoValues || (allTakenAreLocals && !givenToCall);
    }

    // undefined or null, through the casts and the optional that only widen it
    static bool isNoValue(mlir::Value value)
    {
        for (auto *op = value.getDefiningOp(); op; op = op->getNumOperands() == 1 ? op->getOperand(0).getDefiningOp() : nullptr)
        {
            if (mlir::isa<mlir_ts::OptionalUndefOp, mlir_ts::UndefOp, mlir_ts::NullOp>(op))
            {
                return true;
            }

            if (!mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp>(op))
            {
                return false;
            }
        }

        return false;
    }

    static bool holdsOnlyMovedInElements(MLIRTypeHelper &mth, mlir::Operation *op, bool givenToCall)
    {
        // `f([])`: an empty array holds nothing (the empty constant array cast to an array is made one first,
        // ownNewArrayArguments)
        if (mlir::isa_and_nonnull<mlir_ts::NewEmptyArrayOp>(op))
        {
            return true;
        }

        auto arrayOp = mlir::dyn_cast_or_null<mlir_ts::CreateArrayOp>(op);
        return arrayOp && holdsWhatItIsGiven(mth, arrayOp, givenToCall);
    }

    // The literals nested in one about to be owned (isNestedLiteral), innermost first: each is a new value moved into
    // the one holding it - marked so, its text copied as its holder's is, and the retains its holder took of it gone,
    // as for a call's result (runOnOperation's toErase)
    void claimNestedLiterals(MLIRTypeHelper &mth, mlir::Operation *op)
    {
        auto arrayOp = mlir::dyn_cast_or_null<mlir_ts::CreateArrayOp>(op);
        if (!arrayOp)
        {
            return;
        }

        for (auto element : arrayOp->getOperands())
        {
            if (!mth.ownsHeapMemory(arrayOp->getLoc(), element.getType()) || isBorrowedText(element) || isMovedIn(arrayOp, element))
            {
                continue;
            }

            // a tuple literal built in place moves in whole: the retain the literal took of it goes
            if (isInPlaceTupleLiteral(mth, arrayOp, element))
            {
                for (auto *user : llvm::make_early_inc_range(element.getUsers()))
                {
                    if (mlir::isa<mlir_ts::RetainOp>(user))
                    {
                        user->erase();
                    }
                }

                continue;
            }

            if (!isNestedLiteral(mth, arrayOp, element))
            {
                continue;
            }

            auto *innerOp = element.getDefiningOp();
            claimNestedLiterals(mth, innerOp);
            copyBorrowedTextElements(mth, innerOp);
            innerOp->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            innerOp->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            for (auto *user : llvm::make_early_inc_range(element.getUsers()))
            {
                if (mlir::isa<mlir_ts::RetainOp>(user))
                {
                    user->erase();
                }
            }
        }
    }

    // ...the copies: the retain the literal took of each such element becomes the copy it holds
    static void copyBorrowedTextElements(MLIRTypeHelper &mth, mlir::Operation *op)
    {
        auto arrayOp = mlir::dyn_cast_or_null<mlir_ts::CreateArrayOp>(op);
        if (!arrayOp)
        {
            return;
        }

        for (auto &operand : arrayOp->getOpOperands())
        {
            auto element = operand.get();
            if (!mth.ownsHeapMemory(arrayOp->getLoc(), element.getType()) || isMovedIn(arrayOp, element) || !isBorrowedText(element))
            {
                continue;
            }

            // the literal's own retain of it: the last one before the literal, in its block
            mlir::Operation *retain = nullptr;
            for (auto *user : element.getUsers())
            {
                if (mlir::isa<mlir_ts::RetainOp>(user) && user->getBlock() == arrayOp->getBlock() && user->isBeforeInBlock(arrayOp) &&
                    (!retain || retain->isBeforeInBlock(user)))
                {
                    retain = user;
                }
            }

            mlir::OpBuilder builder(arrayOp);
            auto copy = builder.create<mlir_ts::StringCopyOp>(arrayOp->getLoc(), element.getType(), element);
            copy->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(arrayOp->getContext()));
            copy->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(arrayOp->getContext()));
            operand.set(copy);
            if (retain)
            {
                retain->erase();
            }
        }
    }

    // The array a conversion makes for one call (`f(a)` of an `int[]` given as a `number[]`, an array given as
    // an Iterable<number>): a `ts.NewArray` whose elements are written one by one, then given to the call -
    // its other uses are the element references of that loop
    static bool isConvertedArrayFor(mlir::Value value, mlir::Operation *call)
    {
        if (!value.getDefiningOp<mlir_ts::NewArrayOp>())
        {
            return false;
        }

        auto calls = 0;
        for (auto *user : value.getUsers())
        {
            if (user == call)
            {
                calls++;
                continue;
            }

            auto elementRefOp = mlir::dyn_cast<mlir_ts::ElementRefOp>(user);
            if (!elementRefOp || elementRefOp.getArray() != value ||
                llvm::any_of(elementRefOp->getUsers(), [](mlir::Operation *refUser) { return !mlir::isa<mlir_ts::LoadOp, mlir_ts::StoreOp>(refUser); }))
            {
                return false;
            }
        }

        return calls == 1 && llvm::count(call->getOperands(), value) == 1;
    }

    // A new array, born uncounted: a literal's CreateArray, `new Array(n)`, or the read of a spread literal's slot
    static bool isNewArray(mlir::Operation *op)
    {
        return mlir::isa<mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::NewEmptyArrayOp>(op) ||
               (mlir::isa<mlir_ts::LoadOp>(op) && op->hasAttr(SPREAD_ARRAY_LITERAL_ATTR_NAME));
    }

    // A new array given straight to a call (`g([mk(1)])`) is born at count 0 (bornUncounted), and
    // nothing else owns it: a count the callee took and gave back - `const a = p` in a loop, or a
    // parameter it assigns, copied into an owned local - freed the array the callee still read.
    // Under rc the caller holds it across the call too, and so gives it back after the call,
    // which nothing did before.
    // A literal with a spread (`f(...xs)`, `g([...xs])`) is a new array too, read out of the slot it was
    // filled in (SPREAD_ARRAY_LITERAL_ATTR_NAME) - under rc only: own's handling (ownNewArrayArguments) is
    // for the shapes it was written for.
    static bool arrayMadeFor(mlir::Value value, mlir::Operation *call, bool spreadLiterals)
    {
        auto *definingOp = value.getDefiningOp();
        auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(definingOp);
        auto isNewArray =
            mlir::isa_and_nonnull<mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::NewEmptyArrayOp>(definingOp) ||
            (castOp && mlir::isa<mlir_ts::ConstArrayType>(castOp.getIn().getType()) &&
             mlir::isa<mlir_ts::ArrayType>(value.getType())) ||
            (spreadLiterals && mlir::isa_and_nonnull<mlir_ts::LoadOp>(definingOp) &&
             definingOp->hasAttr(SPREAD_ARRAY_LITERAL_ATTR_NAME));
        if (!isNewArray)
        {
            return false;
        }

        // its one use is the call, through casts, and the optional an optional parameter takes it as (`new
        // Set([a, b])` of a `constructor(values?: V[])`: given back by nothing, each call leaked the literal)
        while (value.hasOneUse())
        {
            auto *user = *value.getUsers().begin();
            if (user == call)
            {
                return true;
            }

            if (!mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp>(user))
            {
                return false;
            }

            value = user->getResult(0);
        }

        return false;
    }

    // The new array an argument given as an optional is (`ts.OptionalValue` of it, its one use), or the argument
    static mlir::Value arrayInsideOptional(mlir::Value argument)
    {
        auto optionalValueOp = argument.getDefiningOp<mlir_ts::OptionalValueOp>();
        if (optionalValueOp && optionalValueOp->getOperand(0).hasOneUse() &&
            mlir::isa<mlir_ts::ArrayType>(optionalValueOp->getOperand(0).getType()))
        {
            return optionalValueOp->getOperand(0);
        }

        return argument;
    }

    // -mm=own: may a new array own what it holds, so that giving it back gives its elements back too? One of
    // elements that own nothing, or a literal that can own what it holds (holdsOnlyMovedInElements: new values
    // moved into it, and text it is given copies of - `["x" + i, mk()]`, `[new C()]`, `["x" + i, s]`). One of
    // strings or objects something else owns - a rest parameter's (`logParts(text)`), `[o]` of a local `o` -
    // borrows them (spec 83), and owned it would take them over.
    static bool ownsWhatItHolds(MLIRTypeHelper &mth, mlir::Operation *arrayOp, bool givenToCall)
    {
        auto arrayType = mlir::dyn_cast<mlir_ts::ArrayType>(arrayOp->getResult(0).getType());
        return arrayType &&
               (!mth.ownsHeapMemory(arrayOp->getLoc(), arrayType.getElementType()) || holdsOnlyMovedInElements(mth, arrayOp, givenToCall));
    }

    // Under own nothing gave a new array given to a call back at all (`len([i, 1, 2])`, 400 bytes a call; the
    // optimizer hid it where the array went nowhere else, and a callee returning a record that borrows it -
    // `mh([...])` returning `{ a }` - leaked it at any level). It is a new value as `new C()` is: marked owned,
    // it is released at the end of the block when nothing takes it (releaseDiscardedTemporaries), and moved
    // into a callee that keeps it - when it owns what it holds (ownsWhatItHolds): nothing else gives the new
    // values back, so every call leaked the array and what it held (the classifyArgumentArrays of the
    // inference leaves such an array as it is).
    // It runs once the results of the calls are settled: a call's result (`f([mk(1), mk(2)])`) is marked an
    // owned result moved into the literal there, so before it the literal looked as if it borrowed them and
    // was never given back.
    // A global's initializer too (`const n = count(1, 2)` of a `count(...a: any[])`, TSLANG 190): its rest
    // array was given back by nothing, and own refused the values boxed into it ("the initializer of global
    // takes a second reference"); releaseDiscardedTemporaries gives it back at the end of the region.
    void ownNewArrayArguments(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        if (compileOptions.memoryModel != MemoryModelOwn)
        {
            return;
        }

        module.walk([&](mlir::Operation *op) {
            if (!own_facts::isCall(op) || op->hasTrait<mlir::OpTrait::IsTerminator>() ||
                !(op->getParentOfType<mlir::FunctionOpInterface>() || op->getParentOfType<mlir_ts::GlobalOp>()))
            {
                return;
            }

            llvm::SmallPtrSet<mlir::Value, 4> seen;
            for (auto argument : own_facts::callArgs(op))
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

                // an optional parameter's (`new Set([a, b])`): the array inside the optional
                argument = arrayInsideOptional(argument);

                auto *definingOp = argument.getDefiningOp();

                // `f([])`: the empty constant array cast to an array - a copy made on the heap there - is the new empty
                // array it is (ts.NewEmptyArray), which the inference moves into a callee that keeps it (`new
                // Order(-1, [])`, the constructor storing it) and gives back after one that does not (`ln([])`, which
                // leaked it). Marked owned as the cast it was, it was given back after a keeping callee as well, which
                // then freed it again (DefaultLib worker_objects under -mm=own on wasm: "free" of a freed block)
                if (auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(definingOp);
                    castOp && !castOp->hasAttr(OWNED_RESULT_ATTR_NAME) && mlir::isa<mlir_ts::ArrayType>(argument.getType()) &&
                    arrayMadeFor(argument, op, false))
                {
                    auto constArrayType = mlir::dyn_cast<mlir_ts::ConstArrayType>(castOp.getIn().getType());
                    if (constArrayType && constArrayType.getSize() == 0)
                    {
                        mlir::OpBuilder builder(castOp);
                        auto emptyOp = builder.create<mlir_ts::NewEmptyArrayOp>(castOp->getLoc(), argument.getType());
                        castOp->getResult(0).replaceAllUsesWith(emptyOp.getResult());
                        castOp->erase();
                        argument = emptyOp.getResult();
                        definingOp = emptyOp;
                    }
                }

                if (!definingOp || !mlir::isa<mlir_ts::ArrayType>(argument.getType()) || !seen.insert(argument).second ||
                    definingOp->hasAttr(OWNED_RESULT_ATTR_NAME) || !ownsWhatItHolds(mth, definingOp, /*givenToCall=*/true) ||
                    !(arrayMadeFor(argument, op, false) || isConvertedArrayFor(argument, op)))
                {
                    continue;
                }

                claimNestedLiterals(mth, definingOp);
                copyBorrowedTextElements(mth, definingOp);
                definingOp->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
        });
    }

    // -mm=own (user ruling 11-23): a `let` given a string read out of a temporary array literal
    // (`let st: string = ["s" + i, ms(i)][1]`) holds a copy of it, made the way the literal's borrowed text
    // is (copyBorrowedTextElements): the literal is given back at the end of its block
    // (holdReadOnlyNewArrays), and a `let` borrowing its element could not be given another value after it
    // (`st = [..][0]` in a loop: "cannot be assigned"), which own accepted only while the literal leaked. The
    // copy is the local's own: it consumes it in place of the retain of its declaration. An object element
    // stays a borrow, as out of a named array.
    // (an element of a literal nested in it, `[["s" + i]][0][0]`, read through the inner literal read out of it, the
    // same)
    static void copyTextElementsIntoLocals(mlir::Operation *arrayOp)
    {
        copyTextElementsIntoLocals(arrayOp->getResult(0));
    }

    static void copyTextElementsIntoLocals(mlir::Value array)
    {
        for (auto *user : array.getUsers())
        {
            auto elementRefOp = mlir::dyn_cast<mlir_ts::ElementRefOp>(user);
            if (!elementRefOp || elementRefOp->getOperand(0) != array)
            {
                continue;
            }

            for (auto *refUser : elementRefOp->getUsers())
            {
                auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(refUser);
                if (loadOp && mlir::isa<mlir_ts::ArrayType>(loadOp.getType()))
                {
                    copyTextElementsIntoLocals(loadOp.getResult());
                    continue;
                }

                if (!loadOp || !own_facts::isCopyableString(loadOp.getType()))
                {
                    continue;
                }

                for (auto &use : llvm::make_early_inc_range(loadOp->getUses()))
                {
                    // the `let` the element reaches, through the casts that only widen it (`let st: string | null`,
                    // an optional, a union), each the one use of the one before
                    mlir::Value value = loadOp.getResult();
                    auto *user = use.getOwner();
                    while (mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp, mlir_ts::CreateUnionInstanceOp>(user) &&
                           user->getNumResults() == 1 && user->getResult(0).hasOneUse())
                    {
                        value = user->getResult(0);
                        user = *value.getUsers().begin();
                    }

                    auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
                    if (!varOp || varOp.getInitializer() != value || !varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) ||
                        varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME) || varOp.getCaptured().value_or(false))
                    {
                        continue;
                    }

                    // the declaration's retain, which the copy stands in for. A declaration that took none has
                    // nothing for a copy to consume: the `let` is left the borrow it was, which fails safe (at
                    // worst own refuses a later assignment; nothing is freed early).
                    mlir::Operation *retainSlot = nullptr;
                    for (auto *slotUser : varOp.getResult().getUsers())
                    {
                        if (mlir::isa<mlir_ts::RetainSlotOp>(slotUser) && slotUser->getBlock() == varOp->getBlock())
                        {
                            retainSlot = slotUser;
                            break;
                        }
                    }

                    if (!retainSlot)
                    {
                        continue;
                    }

                    // right after the read: the widening casts, if any, are of the copy
                    mlir::OpBuilder builder(varOp->getContext());
                    builder.setInsertionPointAfter(loadOp);
                    auto copy = builder.create<mlir_ts::StringCopyOp>(loadOp->getLoc(), loadOp.getType(), loadOp.getResult());
                    copy->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(varOp->getContext()));
                    copy->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(varOp->getContext()));
                    use.set(copy);
                    varOp->setAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(varOp->getContext()));
                    retainSlot->erase();
                }
            }
        }
    }

    // A new array only read where it is made - `[a, b].length`, `[a, b][i]`, `[...xs, y].length` - is born
    // uncounted (bornUncounted), holding a count of each element, and no receiver ever takes one: a local,
    // a call (which holds it, see arrayMadeFor), a field or a return each do, and give it back. Nothing gave
    // this one back, so under rc every evaluation leaked the array and its elements. It takes the count a
    // receiver would and is marked an owned result, for releaseDiscardedTemporaries to give back at the end
    // of its block, as a discarded `new C()` is. Only reads that keep nothing - its length, an element
    // reference - qualify; any other use leaves it to whatever already handles that use. An element a `?:`
    // branch yields (`c ? [s][0] : d`) qualifies once the branch has taken a count of it for the result
    // (consumeConditionalResults, borrowsBranchTemporary), which runs first.
    // Under own the same arrays leaked as well (`["k" + i, s][1]`, `for (const s of ["a" + i])`): nothing holds
    // them, and an element read out of one is a borrow of it, as out of a local's array - a kept string is a
    // copy, a kept object refused (OwnershipInferencePass). So the array is given back at the end of its block,
    // as a local holding it is at the end of its scope, once it owns what it holds (ownsWhatItHolds; a literal
    // of borrowed objects stays as it was). No count to take: own's release frees it.
    void holdReadOnlyNewArrays(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        if (!compileOptions.tracksOwnership())
        {
            return;
        }

        auto own = compileOptions.memoryModel == MemoryModelOwn;
        llvm::SmallVector<mlir::Operation *> arrays;
        llvm::SmallVector<std::pair<mlir::Operation *, int>> yieldedArrays;
        module.walk([&](mlir::Operation *op) {
            if (!isNewArray(op) || op->getNumResults() != 1 || !mlir::isa<mlir_ts::ArrayType>(op->getResult(0).getType()) ||
                op->hasAttr(OWNED_RESULT_ATTR_NAME) || op->use_empty() || !op->getParentOfType<mlir_ts::FuncOp>())
            {
                return;
            }

            // (a spread literal's slot owns what was pushed into it: own takes or copies each value pushed)
            if (own && mlir::isa<mlir_ts::CreateArrayOp>(op) && !ownsWhatItHolds(mth, op, /*givenToCall=*/false))
            {
                return;
            }

            // An element read out and handed on by a terminator - a conditional expression's branch yielding
            // `[s][0]` - leaves the block before its receiver takes a count: given back with the array at the
            // end of the block, it would be freed first. Unless the branch took a count of it for the
            // `ts.If` result (consumeConditionalResults, borrowsBranchTemporary), which then carries it.
            // (seen through the casts that only widen it: `c ? [s][0] : null` yields it as `string | null`)
            // For an optional chain on the literal (`[mk(i)]?.[0]`: the literal's truthiness is tested), a
            // conditional within the literal's block that yields the element on - the chain, and its conversion to
            // `| null` (`ts.Value` of the chain's optional) - hands it to its own result, which is followed the same
            // way: it leaves only if that does (borrowsBranchTemporary has the enclosing branch retain it when it
            // goes on to that branch's `ts.Result`; under own, yieldedByBranch holds the literal past it). Under rc
            // for any literal: array destructuring with defaults (`const [a = 1, b = "x"] = [n, s]`) reads each
            // element through a conditional that yields it or the default, into the local in the same block. Under
            // own for any literal only when nothing then leaves the block (that destructuring): one whose element a
            // branch yields on is left to yieldedByBranch, as before.
            auto isTruthinessTest = [](mlir::Operation *user) {
                return mlir::isa<mlir_ts::CastOp>(user) && mlir::isa<mlir_ts::BooleanType>(user->getResult(0).getType());
            };
            auto chained = llvm::any_of(op->getUsers(), isTruthinessTest);
            auto followsConditionals = chained || !own;
            // (own's destructuring: a conditional yielding the element or the default, not an optional chain's - whose
            // optional result is read on through `ts.Value`, as a field or an element, out of what the literal holds)
            auto destructuringOnly = false;
            std::function<bool(mlir::Value)> leavesUncounted = [&](mlir::Value value) {
                for (auto &use : value.getUses())
                {
                    auto *user = use.getOwner();
                    if (mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp, mlir_ts::CreateUnionInstanceOp>(user) ||
                        (followsConditionals && !destructuringOnly && mlir::isa<mlir_ts::ValueOp>(user)))
                    {
                        if (leavesUncounted(user->getResult(0)))
                        {
                            return true;
                        }

                        continue;
                    }

                    if (!user->hasTrait<mlir::OpTrait::IsTerminator>())
                    {
                        continue;
                    }

                    // (or a value-or-default's region, settled the same way: consumeValueOrDefaultResults)
                    auto *conditional = user->getParentOp();
                    auto yieldsCounted = mlir::isa<mlir_ts::ResultOp>(user) &&
                                         mlir::isa_and_nonnull<mlir_ts::IfOp, mlir_ts::OptionalValueOrDefaultOp>(conditional) &&
                                         conditional->hasAttr(OWNED_RESULT_ATTR_NAME) && user->getPrevNode() &&
                                         mlir::isa<mlir_ts::RetainOp>(user->getPrevNode()) &&
                                         user->getPrevNode()->getOperand(0) == user->getOperand(0);
                    if (!yieldsCounted)
                    {
                        if (followsConditionals && mlir::isa<mlir_ts::ResultOp>(user) && mlir::isa_and_nonnull<mlir_ts::IfOp>(conditional) &&
                            op->getParentRegion()->isAncestor(conditional->getParentRegion()) &&
                            (!destructuringOnly ||
                             !mlir::isa<mlir_ts::OptionalType>(conditional->getResult(use.getOperandNumber()).getType())))
                        {
                            if (leavesUncounted(conditional->getResult(use.getOperandNumber())))
                            {
                                return true;
                            }

                            continue;
                        }

                        return true;
                    }
                }

                return false;
            };

            // (an element of a nested literal, `c ? [[mk(i)]][0][0] : null`, read through the inner literal: the same)
            std::function<bool(mlir::Operation *)> elementLeavesBlock = [&](mlir::Operation *elementRef) {
                return llvm::any_of(elementRef->getUsers(), [&](mlir::Operation *read) {
                    if (read->getNumResults() != 1)
                    {
                        return false;
                    }

                    if (leavesUncounted(read->getResult(0)))
                    {
                        return true;
                    }

                    // (under an optional chain, read on through it: chainedAliases)
                    if (!own || !mlir::isa<mlir_ts::LoadOp>(read) || !mlir::isa<mlir_ts::ArrayType>(read->getResult(0).getType()))
                    {
                        return false;
                    }

                    auto inner = read->getResult(0);
                    return llvm::any_of(chained ? chainedAliases(inner, op->getBlock()) : llvm::SmallVector<mlir::Value>{inner},
                                        [&](mlir::Value array) {
                                            return llvm::any_of(array.getUsers(), [&](mlir::Operation *innerUser) {
                                                return mlir::isa<mlir_ts::ElementRefOp>(innerUser) &&
                                                       innerUser->getOperand(0) == array && elementLeavesBlock(innerUser);
                                            });
                                        });
                });
            };

            // (its truthiness test too: an optional chain on the literal tests it before reading it)
            auto readsOnly = [&]() {
                return llvm::all_of(op->getUsers(), [&](mlir::Operation *user) {
                    return mlir::isa<mlir_ts::LengthOfOp>(user) || (chained && isTruthinessTest(user)) ||
                           (mlir::isa<mlir_ts::ElementRefOp>(user) && !elementLeavesBlock(user));
                });
            };
            auto onlyRead = readsOnly();
            if (!onlyRead && !followsConditionals)
            {
                followsConditionals = destructuringOnly = true;
                onlyRead = readsOnly();
                followsConditionals = destructuringOnly = false;
            }

            if (onlyRead)
            {
                arrays.push_back(op);
                return;
            }

            // -mm=own: an element a conditional expression's branch yields (`c ? [mk(i)][0] : null`) - only
            // that one way out, the branch's own `ts.Result`
            if (auto levels = 0; own && mlir::isa<mlir_ts::CreateArrayOp>(op) && yieldedByBranch(op, leavesUncounted, levels))
            {
                yieldedArrays.push_back({op, levels});
            }
        });

        for (auto *op : arrays)
        {
            if (own)
            {
                claimNestedLiterals(mth, op);
                copyBorrowedTextElements(mth, op);
                copyTextElementsIntoLocals(op);
            }
            else
            {
                mlir::OpBuilder builder(op->getContext());
                builder.setInsertionPointAfter(op);
                builder.create<mlir_ts::RetainOp>(op->getLoc(), op->getResult(0));
            }

            op->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
        }

        for (auto [op, levels] : yieldedArrays)
        {
            claimNestedLiterals(mth, op);
            copyBorrowedTextElements(mth, op);
            holdPastConditional(op, levels);
        }
    }

    // The conditional a branch block's `ts.Result` yields to, when it is a plain one: values for each of its results
    // from both branches, not yet settled as owned (consumeConditionalResults)
    static mlir_ts::IfOp conditionalOf(mlir::Block *block)
    {
        auto resultOp = mlir::dyn_cast<mlir_ts::ResultOp>(block->getTerminator());
        auto ifOp = resultOp ? mlir::dyn_cast<mlir_ts::IfOp>(resultOp->getParentOp()) : mlir_ts::IfOp();
        if (!ifOp || ifOp->getNumResults() == 0 || ifOp->hasAttr(OWNED_RESULT_ATTR_NAME))
        {
            return {};
        }

        for (auto &region : ifOp->getRegions())
        {
            if (region.empty() || !mlir::isa<mlir_ts::ResultOp>(region.back().getTerminator()) ||
                region.back().getTerminator()->getNumOperands() != ifOp->getNumResults())
            {
                return {};
            }
        }

        return ifOp;
    }

    // Does `value` leave its block by `resultOp` only, if at all? (seen through the casts that only widen it; for an
    // element of a literal under an optional chain, `chained`, also through the chain's conditionals in that block
    // and their conversion to `| null`, `ts.Value` of the chain's optional: their results are the value then)
    static bool leavesOnlyBy(mlir::Value value, mlir::Operation *resultOp, bool chained = false)
    {
        for (auto &use : value.getUses())
        {
            auto *user = use.getOwner();
            if (mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp, mlir_ts::CreateUnionInstanceOp>(user) ||
                (chained && mlir::isa<mlir_ts::ValueOp>(user)))
            {
                if (!leavesOnlyBy(user->getResult(0), resultOp, chained))
                {
                    return false;
                }

                continue;
            }

            if (user->hasTrait<mlir::OpTrait::IsTerminator>() && user != resultOp)
            {
                auto *conditional = user->getParentOp();
                if (chained && mlir::isa<mlir_ts::ResultOp>(user) && mlir::isa_and_nonnull<mlir_ts::IfOp>(conditional) &&
                    resultOp->getBlock()->findAncestorOpInBlock(*conditional) &&
                    leavesOnlyBy(conditional->getResult(use.getOperandNumber()), resultOp, chained))
                {
                    continue;
                }

                return false;
            }
        }

        return true;
    }

    // Does `value` stay in `block` but for its terminator? Seen through the casts that only widen it, and through a
    // conditional expression in `block` (or in a statement of it) whose branch yields it: that conditional's result
    // is the value then. `leaves` is set when `block`'s terminator takes it; false for any other terminator.
    bool leavesBlockBy(mlir::Value value, mlir::Block *block, bool &leaves, int depth = 0)
    {
        if (depth > 64)
        {
            return false;
        }

        for (auto *user : value.getUsers())
        {
            if (mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp, mlir_ts::CreateUnionInstanceOp>(user))
            {
                if (!leavesBlockBy(user->getResult(0), block, leaves, depth + 1))
                {
                    return false;
                }

                continue;
            }

            if (!user->hasTrait<mlir::OpTrait::IsTerminator>())
            {
                continue;
            }

            if (user == block->getTerminator())
            {
                leaves = true;
                continue;
            }

            auto handedOn = handedOnByConditional(user, value);
            if (handedOn.empty() || !block->findAncestorOpInBlock(*user->getParentOp()))
            {
                return false;
            }

            for (auto result : handedOn)
            {
                if (!leavesBlockBy(result, block, leaves, depth + 1))
                {
                    return false;
                }
            }
        }

        return true;
    }
    // An inner literal read out of a literal under an optional chain (`[[mk(i)]]?.[0]?.[0]`) is read on through the
    // chain: the chain's conditional yields it as an optional, and the next link reads `ts.Value` of that. The same
    // array, then: `value` and what it is handed on as - through the casts that only widen it, an optional's value,
    // and a conditional in `block` (or in a statement of it) whose branch yields it. At most 64 values are
    // collected (each link of a chain adds a few; a 25-level chain stays well below), which bounds the walk; a chain
    // past that is not followed further and is judged as it was before (refused or released at the branch end).
    static llvm::SmallVector<mlir::Value> chainedAliases(mlir::Value value, mlir::Block *block)
    {
        llvm::SmallVector<mlir::Value> aliases{value};
        for (size_t index = 0; index < aliases.size() && aliases.size() < 64; index++)
        {
            for (auto &use : aliases[index].getUses())
            {
                auto *user = use.getOwner();
                if (mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp, mlir_ts::CreateUnionInstanceOp, mlir_ts::ValueOp>(user))
                {
                    aliases.push_back(user->getResult(0));
                    continue;
                }

                auto *conditional = user->getParentOp();
                if (mlir::isa<mlir_ts::ResultOp>(user) && mlir::isa_and_nonnull<mlir_ts::IfOp>(conditional) &&
                    block->findAncestorOpInBlock(*conditional))
                {
                    aliases.push_back(conditional->getResult(use.getOperandNumber()));
                }
            }
        }

        return aliases;
    }

    // Is a literal only read, but for an element its block - a branch of a conditional expression - yields? (seen
    // through the casts that only widen it). The conditional's result may be yielded on by a branch of another
    // (`c ? (d ? [mk(i)][0] : null) : null`), and so on: `levels` is how many conditionals the literal is held past,
    // to the end of the block the outermost one is in.
    bool yieldedByBranch(mlir::Operation *op, const std::function<bool(mlir::Value)> &leavesUncounted, int &levels)
    {
        auto ifOp = conditionalOf(op->getBlock());
        if (!ifOp || !op->getParentOfType<mlir_ts::FuncOp>())
        {
            return false;
        }

        auto *resultOp = op->getBlock()->getTerminator();
        levels = 1;
        for (auto conditional = ifOp;;)
        {
            // where the conditional's results go on to leave its block: by its block's own terminator only - another
            // conditional's branch reading it (`const q = d ? o : null`, `o ? o : null`) hands it on to that one's
            // result, in the same block. Any other way out (a loop's condition, a `return`) and it is not held.
            auto leaves = false;
            auto other = false;
            for (auto result : conditional->getResults())
            {
                if (!leavesBlockBy(result, conditional->getBlock(), leaves))
                {
                    other = true;
                }
            }

            if (other)
            {
                return false;
            }

            // held to the end of the block the outermost conditional around it is in (threading the literal out is
            // one result per level): through a branch that hands the result on, and any other around it too
            auto outer = conditionalOf(conditional->getBlock());
            if (!outer)
            {
                if (leaves)
                {
                    return false;
                }

                break;
            }

            conditional = outer;
            levels++;
        }

        // (an optional chain on the literal tests its truthiness first: a read)
        auto isTruthinessTest = [](mlir::Operation *user) {
            return mlir::isa<mlir_ts::CastOp>(user) && mlir::isa<mlir_ts::BooleanType>(user->getResult(0).getType());
        };
        auto chained = llvm::any_of(op->getUsers(), isTruthinessTest);
        auto leavesOnlyByResult = [&](mlir::Value value) { return leavesOnlyBy(value, resultOp, chained); };
        // each element read out of it, and out of a literal nested in it (`c ? [[mk(i)]][0][0] : null`): one that
        // leaves, leaves by that way only
        auto yields = false;
        // (a literal nested in one under an optional chain is read on through the chain: chainedAliases)
        std::function<bool(mlir::Value, bool)> readsLeaveOnlyByResult = [&](mlir::Value literal, bool outermost) {
            llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> users;
            for (auto array : !outermost && chained ? chainedAliases(literal, op->getBlock()) : llvm::SmallVector<mlir::Value>{literal})
            {
                for (auto *user : array.getUsers())
                {
                    users.push_back({user, array});
                }
            }

            for (auto [user, array] : users)
            {
                if (mlir::isa<mlir_ts::LengthOfOp>(user) || (chained && isTruthinessTest(user)))
                {
                    continue;
                }

                if (!mlir::isa<mlir_ts::ElementRefOp>(user) || user->getOperand(0) != array)
                {
                    if (outermost)
                    {
                        return false;
                    }

                    continue;
                }

                for (auto *read : user->getUsers())
                {
                    if (read->getNumResults() != 1)
                    {
                        continue;
                    }

                    auto value = read->getResult(0);
                    if (mlir::isa<mlir_ts::LoadOp>(read) && mlir::isa<mlir_ts::ArrayType>(value.getType()) &&
                        !readsLeaveOnlyByResult(value, false))
                    {
                        return false;
                    }

                    if (!leavesUncounted(value))
                    {
                        continue;
                    }

                    if (!leavesOnlyByResult(value))
                    {
                        return false;
                    }

                    yields = true;
                }
            }

            return true;
        };

        return readsLeaveOnlyByResult(op->getResult(0), true) && yields;
    }

    // ...which is held past the conditional expression, as a named array would be (`const t = [mk(i)]; c ? t[0] :
    // null`): the element read out of it borrows it, as out of a local's array, and so must not be freed at the end
    // of the branch, which the conditional's result outlives. The `ts.If` yields the literal as well, as an optional
    // (undefined from a branch that made none), and that result is a new value of the block the conditional is in,
    // given back at its end (releaseDiscardedTemporaries) - where the borrow ends. Before, the literal and what it
    // held were never given back. (A borrow of it can't be stored, returned or captured: own refuses those.) A
    // conditional whose result a branch of another yields (yieldedByBranch's `levels`) yields it on the same way.
    void holdPastConditional(mlir::Operation *op, int levels)
    {
        auto location = op->getLoc();
        auto heldType = mlir_ts::OptionalType::get(op->getResult(0).getType());
        mlir::Value held = op->getResult(0);
        auto *heldBlock = op->getBlock();
        mlir::OpBuilder builder(op->getContext());
        for (auto level = 0; level < levels; level++)
        {
            auto *resultOp = heldBlock->getTerminator();
            auto ifOp = mlir::cast<mlir_ts::IfOp>(resultOp->getParentOp());

            llvm::SmallVector<mlir::Type> resultTypes(ifOp->getResultTypes());
            resultTypes.push_back(heldType);

            builder.setInsertionPoint(ifOp);
            auto newIfOp = builder.create<mlir_ts::IfOp>(ifOp.getLoc(), resultTypes, ifOp.getCondition(), /*withElseRegion=*/true);
            newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
            newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
            newIfOp->setAttrs(ifOp->getAttrs());
            for (auto [index, result] : llvm::enumerate(ifOp->getResults()))
            {
                result.replaceAllUsesWith(newIfOp->getResult(index));
            }

            ifOp->erase();

            for (auto &region : newIfOp->getRegions())
            {
                auto *terminator = region.back().getTerminator();
                builder.setInsertionPoint(terminator);
                mlir::Value yielded = terminator != resultOp ? builder.create<mlir_ts::OptionalUndefOp>(location, heldType).getResult()
                                      : level == 0       ? builder.create<mlir_ts::OptionalValueOp>(location, heldType, held).getResult()
                                                         : held;
                terminator->insertOperands(terminator->getNumOperands(), yielded);
            }

            held = newIfOp->getResult(newIfOp->getNumResults() - 1);
            heldBlock = newIfOp->getBlock();
        }

        // the literal moves into the result; what holds the result is a new value, released at the end of its block
        op->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
        op->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));

        builder.setInsertionPointAfter(held.getDefiningOp());
        auto holder = builder.create<mlir_ts::CastOp>(location, heldType, held);
        holder->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
    }

    // A `yield` hands its `{ value, done }` record to the caller the way a `return` does, with a
    // reference of its own (§9.24) - under rc, where a return statement retains what it returns
    // and a yield, built by MLIRGen without going through it, did not. The record was filled by
    // stores that take no count, so what it held lived only on the +1 it was born with: a fresh
    // `"s" + i` was never given back (the caller retains what it reads and releases that), and a
    // value read out of the state object was the state's own reference. With the retain the
    // record owns its fields, a fresh value made for it is a temporary released before the yield
    // like any other, and `next` returns owned like a plain function, so its caller can take the
    // record over (functionReturnsOwned, fieldFunctionNameOf).
    //
    // Under own the yielded value moves into the record instead, and a retain is an error.
    void retainYieldedRecords(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        if (!compileOptions.isRefCounted())
        {
            return;
        }

        module.walk([&](mlir_ts::YieldReturnValOp yieldOp) {
            auto record = yieldOp.getOperand();
            if (!mth.ownsHeapMemory(yieldOp.getLoc(), record.getType()) || retainPrecedes(yieldOp, record))
            {
                return;
            }

            mlir::OpBuilder builder(yieldOp);
            builder.create<mlir_ts::RetainOp>(yieldOp.getLoc(), record);
        });
    }

    // A call's argument read out of a global (`run(G)`, `run(G.c)`, `G.m()`) is borrowed like any
    // other argument, and the callee may give the global a new value while it still uses the
    // parameter (`function run(h: C) { G = mk(); print(h.x); }`): the global's release freed the
    // block the parameter read. Under rc the caller holds a count across the call - retained where
    // the argument is read, released right after the call - and, when a try around the call takes
    // what it throws, released where that try's handler starts (unwindRegionOf). This runs before
    // the affine lowering, so a call in a try body is still a plain call here (a `ts.Invoke` later).
    //
    // A Shared<T> handle (spec 23.1) is counted under rc and own alike, and a program reads its
    // count (`Shared.count`) as the number of handles - which a parameter, a borrow, is not one of.
    // So a handle is held, under both, only across a call that may give the global it was read
    // from a new value, the one case where the hold is what keeps the block alive.
    //
    // A read of a local is held too, under rc, where the local is given a new value after the read
    // and before the call: by a later argument (`f(n, (n = mk(), 0))`, in a conditional one too),
    // or, for a captured local, by a closure called there. The assignment released the block the
    // argument read. Under own only a handle is held so, being counted there too; own rejects the
    // others (OwnershipInferencePass).
    void holdGlobalArguments(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        if (!compileOptions.tracksOwnership())
        {
            return;
        }

        // the global a read comes out of, or a place (a field, an element, a handle's value) is in,
        // through casts
        auto globalOf = [](mlir::Value value, bool isPlace) -> mlir::StringRef {
            // gone through a field or an element on the way: then a local is looked through
            auto inPlace = false;
            for (auto depth = 0; depth < 16 && value; depth++)
            {
                mlir::Operation *place = nullptr;
                if (isPlace)
                {
                    place = value.getDefiningOp();
                    isPlace = false;
                }
                else
                {
                    while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
                    {
                        value = castOp.getIn();
                    }

                    auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
                    place = loadOp ? loadOp.getReference().getDefiningOp() : nullptr;
                }

                if (auto addressOfOp = mlir::dyn_cast_or_null<mlir_ts::AddressOfOp>(place))
                {
                    return addressOfOp.getGlobalName();
                }

                // A local given a value once and never again - `const a = GA` - names what it was
                // given: a place in it is a place in that global's object (`f(a[0])`, the callee
                // setting `GA[0]`, fuzz 946037). The local holds a count of its own, so a read of
                // the local alone is not one.
                if (auto varOp = mlir::dyn_cast_or_null<mlir_ts::VariableOp>(place); varOp && inPlace)
                {
                    auto storedTo = llvm::any_of(varOp->getUsers(), [&](mlir::Operation *user) {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                        return storeOp && storeOp.getReference() == varOp.getResult();
                    });
                    value = storedTo ? mlir::Value() : varOp.getInitializer();
                    continue;
                }

                inPlace = mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::SharedValueRefOp>(place);
                value = inPlace ? place->getOperand(0) : mlir::Value();
            }

            return {};
        };

        auto readsGlobalItself = [](mlir::Value value) {
            while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
            {
                value = castOp.getIn();
            }

            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            return loadOp && loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>();
        };

        // The function a call names, if it can be named.
        auto calleeOf = [](mlir::Operation *op) -> mlir::StringRef {
            if (auto callee = own_facts::directCallee(op))
            {
                return callee.getValue();
            }

            auto value = own_facts::calleeValue(op);
            auto symbolRefOp = value ? value.getDefiningOp<mlir_ts::SymbolRefOp>() : mlir_ts::SymbolRefOp();
            if (symbolRefOp)
            {
                return symbolRefOp.getIdentifier();
            }

            auto thisSymbolRefOp = value ? value.getDefiningOp<mlir_ts::ThisSymbolRefOp>() : mlir_ts::ThisSymbolRefOp();
            return thisSymbolRefOp ? thisSymbolRefOp.getIdentifier() : mlir::StringRef();
        };

        // The globals each function may give a new value: by a store to the global or into a place
        // in it, its own or a callee's. A call through a value that names no function may assign
        // any of them; a function with no body here assigns none.
        llvm::StringMap<llvm::StringSet<>> assigns;
        llvm::StringMap<llvm::SmallVector<mlir::StringRef>> callees;
        llvm::StringSet<> assignsAny;
        // ...and the places each function gives a new value or takes a value out of (rc, readsAliasedPlace), by kind:
        // a field by its name ("F:" + name), an element ("E": a store, `pop`, `shift`, `splice`), a captured local's
        // cell ("C"), anything else ("*"). Deliberately coarse - a name stands for that field of every class, an
        // element for every array - but not the stores that only make an object: a constructor's into `this`, a
        // function's into the object it just made, a local record's into its own slot. A function that calls through
        // a value may run any function here: it writes what the module writes (moduleKinds).
        llvm::StringMap<llvm::StringSet<>> writtenKinds;
        llvm::StringSet<> moduleKinds;
        auto fieldKindOf = [](mlir_ts::PropertyRefOp propertyRefOp) -> std::string {
            auto type = propertyRefOp->getOperand(0).getType();
            if (auto refType = mlir::dyn_cast<mlir_ts::RefType>(type))
            {
                type = refType.getElementType();
            }

            if (auto classType = mlir::dyn_cast<mlir_ts::ClassType>(type))
            {
                type = classType.getStorageType();
            }

            mlir::Attribute id;
            auto position = propertyRefOp.getPosition();
            if (auto storageType = mlir::dyn_cast<mlir_ts::ClassStorageType>(type);
                storageType && position >= 0 && static_cast<size_t>(position) < storageType.getFields().size())
            {
                id = storageType.getFields()[position].id;
            }
            else if (auto tupleType = mlir::dyn_cast<mlir_ts::TupleType>(type);
                     tupleType && position >= 0 && static_cast<size_t>(position) < tupleType.getFields().size())
            {
                id = tupleType.getFields()[position].id;
            }

            auto name = mlir::dyn_cast_or_null<mlir::StringAttr>(id);
            return name ? ("F:" + name.getValue()).str() : std::string("*");
        };
        // the object a store makes rather than changes: `this` in a constructor, a new object of this function, a
        // local record's slot
        auto initializes = [](mlir::Value object, mlir_ts::FuncOp funcOp) {
            while (auto castOp = object.getDefiningOp<mlir_ts::CastOp>())
            {
                object = castOp.getIn();
            }

            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(object))
            {
                return argument.getArgNumber() == 0 && argument.getOwner()->isEntryBlock() &&
                       funcOp.getSymName().ends_with(".constructor");
            }

            if (auto loadOp = object.getDefiningOp<mlir_ts::LoadOp>())
            {
                // `this` read out of a constructor's parameter slot; or a new box's contents
                if (loadOp.getReference().getDefiningOp<mlir_ts::NewOp>())
                {
                    return true;
                }

                auto *slot = loadOp.getReference().getDefiningOp();
                auto initial = mlir::isa_and_nonnull<mlir_ts::VariableOp, mlir_ts::ParamOp>(slot) && slot->getNumOperands() > 0
                                   ? slot->getOperand(0)
                                   : mlir::Value();
                auto argument = initial ? mlir::dyn_cast<mlir::BlockArgument>(initial) : mlir::BlockArgument();
                return argument && argument.getArgNumber() == 0 && argument.getOwner()->isEntryBlock() &&
                       funcOp.getSymName().ends_with(".constructor");
            }

            return mlir::isa_and_nonnull<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::NewEmptyArrayOp,
                                         mlir_ts::VariableOp>(object.getDefiningOp());
        };
        module.walk([&](mlir_ts::FuncOp funcOp) {
            auto name = funcOp.getSymName();

            // A function with no body here - another module's, the default library's non-generic one - may write any
            // place of an object it is given (TSLANG 190: `clobber(arr[0], h)` from another module, setting
            // `h.items[0]`, freed what its first parameter read). Not one given only values that reach no object:
            // numbers, booleans, strings (immutable), which is most of the runtime's.
            if (funcOp.getBody().empty())
            {
                auto reachesObjects = llvm::any_of(funcOp.getFunctionType().getInputs(), [&](mlir::Type type) {
                    return !mlir::isa<mlir_ts::StringType>(type) && mth.ownsHeapMemory(funcOp.getLoc(), type);
                });
                if (reachesObjects)
                {
                    writtenKinds[name].insert("*");
                }

                return;
            }

            funcOp.walk([&](mlir::Operation *op) {
                // (`a.length = n` too: a shorter array gives back the elements it drops)
                if (mlir::isa<mlir_ts::ArrayPopOp, mlir_ts::ArrayShiftOp, mlir_ts::ArraySpliceOp, mlir_ts::SetLengthOfOp>(op))
                {
                    writtenKinds[name].insert("E");
                }

                // `delete`: what it frees takes every place in it along
                if (mlir::isa<mlir_ts::DeleteOp>(op))
                {
                    writtenKinds[name].insert("*");
                }

                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op))
                {
                    auto reference = storeOp.getReference();
                    if (auto propertyRefOp = reference.getDefiningOp<mlir_ts::PropertyRefOp>())
                    {
                        if (!initializes(propertyRefOp->getOperand(0), funcOp))
                        {
                            writtenKinds[name].insert(fieldKindOf(propertyRefOp));
                        }
                    }
                    else if (auto elementRefOp = reference.getDefiningOp<mlir_ts::ElementRefOp>())
                    {
                        if (!initializes(elementRefOp->getOperand(0), funcOp))
                        {
                            writtenKinds[name].insert("E");
                        }
                    }
                    else if (reference.getDefiningOp<mlir_ts::NewOp>())
                    {
                        // a new box's contents (an object literal boxed for an interface): made, not changed
                    }
                    else if (own_facts::isPlace(reference))
                    {
                        writtenKinds[name].insert("*");
                    }
                    else if (reference.getDefiningOp<mlir_ts::LoadOp>())
                    {
                        writtenKinds[name].insert("C");
                    }

                    if (auto global = globalOf(storeOp.getReference(), true); !global.empty())
                    {
                        assigns[name].insert(global);
                    }
                }
                else if (own_facts::isCall(op))
                {
                    if (auto callee = calleeOf(op); !callee.empty())
                    {
                        callees[name].push_back(callee);
                    }
                    else
                    {
                        assignsAny.insert(name);
                    }
                }
            });
        });

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : callees)
            {
                auto caller = entry.getKey();
                for (auto callee : entry.getValue())
                {
                    if (assignsAny.contains(callee) && assignsAny.insert(caller).second)
                    {
                        changed = true;
                    }


                    if (auto it = assigns.find(callee); it != assigns.end())
                    {
                        // copied first: `assigns[caller]` may move the entry being read
                        llvm::SmallVector<mlir::StringRef> globals;
                        for (auto &global : it->getValue())
                        {
                            globals.push_back(global.getKey());
                        }

                        for (auto global : globals)
                        {
                            changed |= assigns[caller].insert(global).second;
                        }
                    }
                }
            }
        }

        for (auto &entry : writtenKinds)
        {
            for (auto &kind : entry.getValue())
            {
                moduleKinds.insert(kind.getKey());
            }
        }

        // a function calling through a value may run any of them
        for (auto &entry : assignsAny)
        {
            for (auto &kind : moduleKinds)
            {
                writtenKinds[entry.getKey()].insert(kind.getKey());
            }
        }

        for (auto changed = true; changed;)
        {
            changed = false;
            for (auto &entry : callees)
            {
                auto caller = entry.getKey();
                for (auto callee : entry.getValue())
                {
                    auto it = writtenKinds.find(callee);
                    if (it == writtenKinds.end())
                    {
                        continue;
                    }

                    llvm::SmallVector<std::string> kinds;
                    for (auto &kind : it->getValue())
                    {
                        kinds.push_back(kind.getKey().str());
                    }

                    for (auto &kind : kinds)
                    {
                        changed |= writtenKinds[caller].insert(kind).second;
                    }
                }
            }
        }

        auto mayAssignAnyGlobal = [&](mlir::Operation *op) {
            auto callee = calleeOf(op);
            if (callee.empty() || assignsAny.contains(callee))
            {
                return true;
            }

            auto it = assigns.find(callee);
            return it != assigns.end() && !it->getValue().empty();
        };

        // Is this a read out of a field or an element of an object a parameter names (`p.n`, `p.a[0].v`)? The
        // parameter may be a global's object (`f(HG)`), so a call that gives a global's field a new value may
        // free what was read.
        auto readsParameterPlace = [](mlir::Value value) {
            auto inPlace = false;
            for (auto depth = 0; depth < 16 && value; depth++)
            {
                while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
                {
                    value = castOp.getIn();
                }

                auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
                auto *place = loadOp ? loadOp.getReference().getDefiningOp() : nullptr;
                if (mlir::isa_and_nonnull<mlir_ts::ParamOp>(place))
                {
                    return inPlace;
                }

                if (!mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(place))
                {
                    return false;
                }

                inPlace = true;
                value = place->getOperand(0);
            }

            return false;
        };

        // the kinds of places a call may write (writtenKinds; through a value: what any function here writes)
        auto mayWriteKind = [&](mlir::Operation *op, llvm::StringRef kind) {
            auto callee = calleeOf(op);
            const llvm::StringSet<> *kinds = &moduleKinds;
            if (!callee.empty() && !assignsAny.contains(callee))
            {
                // A callee with no body here writes any place of an object it may be given (writtenKinds, "*"),
                // and nothing when it is given none. The default library's mutating array methods are generic:
                // instantiated in this module, their bodies are walked like any function here.
                auto it = writtenKinds.find(callee);
                if (it == writtenKinds.end())
                {
                    return false;
                }

                kinds = &it->getValue();
            }

            return kinds->contains(kind) || kinds->contains("*");
        };

        // Is this a read out of a field or an element of an object some other name may reach, which the call may give
        // a new value? `f(arr[0])` with `arr` captured by a closure that sets `arr[0]`, also stored in a global whose
        // element the callee sets, put into another array, copied to another local that `pop`s it; `f(h.o)` with
        // `arrh = [h]` and a closure setting `arrh[0].o`. The callee - or what it calls - frees the block the argument
        // read while it still reads its parameter. Two facts together, both needed:
        // - some object on the way to the place may be reached by another name: it is not one made here that never
        //   escapes (read only through its places, its length, a comparison or this very call) - a parameter's
        //   object, a captured local's, one read out of somewhere else, one stored or passed on;
        // - the call may write a place of a kind on that way (writtenKinds: the field's name, an element, a captured
        //   local's cell), a write that makes an object aside.
        // A global's object is the global rule's (globalOf, mayAssign), not this one's.
        std::function<bool(mlir::Value, mlir::Operation *, int)> escapes = [&](mlir::Value value, mlir::Operation *call, int depth) {
            if (depth > 6)
            {
                return true;
            }

            for (auto &use : value.getUses())
            {
                auto *user = use.getOwner();
                if (user == call || mlir::isa<mlir_ts::LengthOfOp, mlir_ts::LogicalBinaryOp, mlir_ts::ReleaseOp,
                                              mlir_ts::RetainOp, mlir_ts::ArrayPushOp, mlir_ts::ArrayPopOp,
                                              mlir_ts::ArrayShiftOp, mlir_ts::ArrayUnshiftOp, mlir_ts::ArraySpliceOp>(user))
                {
                    continue;
                }

                // a place in it: reading and writing it is not another name for the object
                if (mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp>(user) && use.getOperandNumber() == 0)
                {
                    continue;
                }

                // a store of something else into it
                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user); storeOp && storeOp.getValue() != value)
                {
                    continue;
                }

                if (mlir::isa<mlir_ts::CastOp>(user))
                {
                    if (escapes(user->getResult(0), call, depth + 1))
                    {
                        return true;
                    }

                    continue;
                }

                return true;
            }

            return false;
        };
        auto isFresh = [](mlir::Value value) {
            auto *def = value.getDefiningOp();
            return mlir::isa_and_nonnull<mlir_ts::NewOp, mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::NewEmptyArrayOp>(def) ||
                   (mlir::isa_and_nonnull<mlir_ts::CastOp>(def) && mlir::isa<mlir_ts::ConstArrayType>(def->getOperand(0).getType()));
        };
        // may another name reach the object a value is? Not one made here that never escapes, nor a local's slot given
        // only such values, not captured, nothing read out of it escaping. (`captured`: the object is a captured
        // local's, whose cell a closure may give a new value)
        auto mayBeAliased = [&](mlir::Value object, mlir::Operation *call, bool &captured) {
            while (auto castOp = object.getDefiningOp<mlir_ts::CastOp>())
            {
                object = castOp.getIn();
            }

            if (auto loadOp = object.getDefiningOp<mlir_ts::LoadOp>())
            {
                auto varOp = loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                if (!varOp)
                {
                    // a global's, or one read out of another place (whose object is looked at in its turn)
                    return false;
                }

                if (varOp.getCaptured().value_or(false))
                {
                    captured = true;
                    return true;
                }

                if (!varOp.getInitializer() || !isFresh(varOp.getInitializer()))
                {
                    return true;
                }

                for (auto *user : varOp->getUsers())
                {
                    if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
                    {
                        if (storeOp.getReference() == varOp.getResult() && !isFresh(storeOp.getValue()))
                        {
                            return true;
                        }

                        continue;
                    }

                    if (auto slotRead = mlir::dyn_cast<mlir_ts::LoadOp>(user); slotRead && escapes(slotRead.getResult(), call, 0))
                    {
                        return true;
                    }
                }

                return false;
            }

            // a parameter's object: its caller's, which other names may reach
            if (mlir::isa<mlir::BlockArgument>(object))
            {
                return true;
            }

            return !isFresh(object) || escapes(object, call, 0);
        };
        auto readsAliasedPlace = [&](mlir::Value value, mlir::Operation *call) {
            while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
            {
                value = castOp.getIn();
            }

            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            if (!loadOp || !own_facts::isPlace(loadOp.getReference()))
            {
                return false;
            }

            // each place on the way, and the object it is in
            auto aliased = false;
            auto written = false;
            auto ref = loadOp.getReference();
            for (auto depth = 0; depth < 16 && ref && own_facts::isPlace(ref); depth++)
            {
                auto *place = ref.getDefiningOp();
                if (auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(place))
                {
                    written = written || mayWriteKind(call, fieldKindOf(propertyRefOp));
                }
                else if (mlir::isa<mlir_ts::ElementRefOp>(place))
                {
                    written = written || mayWriteKind(call, "E");
                }
                else
                {
                    written = true;
                }

                auto object = place->getOperand(0);
                while (auto castOp = object.getDefiningOp<mlir_ts::CastOp>())
                {
                    object = castOp.getIn();
                }

                // (a field of a local tuple: the place is in the slot itself)
                if (mlir::isa_and_nonnull<mlir_ts::VariableOp>(object.getDefiningOp()))
                {
                    return false;
                }

                auto captured = false;
                aliased = mayBeAliased(object, call, captured) || aliased;
                written = written || (captured && mayWriteKind(call, "C"));

                auto innerLoad = object.getDefiningOp<mlir_ts::LoadOp>();
                ref = innerLoad ? innerLoad.getReference() : mlir::Value();
            }

            return aliased && written;
        };

        auto mayAssign = [&](mlir::Operation *op, mlir::StringRef global) {
            auto callee = calleeOf(op);
            if (callee.empty() || assignsAny.contains(callee))
            {
                return true;
            }

            auto it = assigns.find(callee);
            return it != assigns.end() && it->getValue().contains(global);
        };

        // What a hold keeps of an argument: the value read before the call. Under rc a view of it is
        // made at the call, after every argument is evaluated - after a later one may have given
        // the global a new value - so the value read is the one behind it: the object a cast or an
        // interface is made over (`f(G, (G = mk(), x))` with an interface parameter), or the values
        // an object literal was given (`f({ v: G.v }, ..)`), which the literal holds with no count
        // until it is boxed for the view, at the call too.
        auto readsOf = [&](mlir::Operation *call, mlir::Value argument, llvm::SmallVectorImpl<mlir::Value> &reads) {
            if (!compileOptions.isRefCounted())
            {
                reads.push_back(argument);
                return;
            }

            auto value = argument;
            for (;;)
            {
                // a constant array cast to an array is copied to the heap there: the copy is the read
                auto castOp = value.getDefiningOp<mlir_ts::CastOp>();
                if (castOp && !mlir::isa<mlir_ts::ConstArrayType>(castOp.getIn().getType()))
                {
                    value = castOp.getIn();
                }
                else if (auto newInterfaceOp = value.getDefiningOp<mlir_ts::NewInterfaceOp>())
                {
                    value = newInterfaceOp.getThisVal();
                }
                else
                {
                    break;
                }
            }

            // an object literal boxed for an interface: stored into a new value_ref
            auto newOp = value.getDefiningOp<mlir_ts::NewOp>();
            if (newOp && mlir::isa<mlir_ts::ValueRefType>(newOp.getInstance().getType()))
            {
                for (auto *user : newOp->getUsers())
                {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (storeOp && storeOp.getReference() == newOp.getInstance())
                    {
                        value = storeOp.getValue();
                        break;
                    }
                }
            }

            // an object literal: a read of the slot it was built in, in the call's block
            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
            if (varOp && !varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) && varOp->getBlock() == call->getBlock() &&
                mlir::isa<mlir_ts::TupleType>(value.getType()))
            {
                for (auto *user : varOp->getUsers())
                {
                    auto fieldRef = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                    if (!fieldRef)
                    {
                        continue;
                    }

                    for (auto *fieldUser : fieldRef->getUsers())
                    {
                        auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                        if (storeOp && storeOp.getReference() == fieldRef.getResult())
                        {
                            reads.push_back(storeOp.getValue());
                        }
                    }
                }

                return;
            }

            reads.push_back(value);
        };

        llvm::SmallVector<std::pair<mlir::Operation *, mlir::Value>> held;
        module.walk([&](mlir::Operation *op) {
            if (!own_facts::isCall(op) || op->hasTrait<mlir::OpTrait::IsTerminator>() ||
                !op->getParentOfType<mlir::FunctionOpInterface>())
            {
                return;
            }

            llvm::SmallPtrSet<mlir::Value, 4> seen;
            llvm::SmallVector<mlir::Value> reads;
            for (auto argument : own_facts::callArgs(op))
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

                readsOf(op, argument, reads);
            }

            for (auto argument : reads)
            {
                if (!mth.ownsHeapMemory(op->getLoc(), argument.getType()) || !seen.insert(argument).second)
                {
                    continue;
                }

                if (compileOptions.isRefCounted() && arrayMadeFor(arrayInsideOptional(argument), op, true))
                {
                    held.push_back({op, arrayInsideOptional(argument)});
                    continue;
                }

                // (under own a new array given to a call is not held: it is moved into the call, or given
                // back after it - ownNewArrayArguments, once the calls' results are settled)

                // An argument read out of a slot or a place that a later argument of the same call
                // overwrites (`f(n, (n = mk(), n))`), or out of a field or an element of an object
                // the call is given as well (`f(n, n.items[0])`, which `f` may overwrite through
                // its other parameter): either one frees the block the argument read. Under rc the
                // caller holds it, as it holds a global's (A1, R8 of the fuzzer).
                if (compileOptions.isRefCounted() &&
                    (overwrittenBeforeCall(argument, op) || readFromOtherArgument(argument, op)))
                {
                    held.push_back({op, argument});
                    continue;
                }

                // Out of a field of an object a parameter names, to a call that may give a global's field a new
                // value: the object may be that global's (`f(HG)` then `g(p.n)`, `g` setting `HG.n`), and the
                // store freed the block the argument read. Under rc the caller holds it (fuzz seed 71210).
                if (compileOptions.isRefCounted() && readsParameterPlace(argument) && mayAssignAnyGlobal(op))
                {
                    held.push_back({op, argument});
                    continue;
                }

                // Out of a field or an element of an object another name may reach, to a call that may write that
                // place (readsAliasedPlace). Under rc the caller holds it.
                if (compileOptions.isRefCounted() && readsAliasedPlace(argument, op))
                {
                    held.push_back({op, argument});
                    continue;
                }

                auto global = globalOf(argument, false);
                if (global.empty())
                {
                    if (localAssignedBefore(op, argument))
                    {
                        held.push_back({op, argument});
                    }

                    continue;
                }

                auto isHandle = MLIRTypeHelper::isSharedHandleType(argument.getType());
                if (isHandle ? !mayAssign(op, global) : !compileOptions.isRefCounted())
                {
                    continue;
                }

                // under own, a handle read out of a place in a global is a borrow of that place,
                // rejected when the call may overwrite it; only the global itself is held
                if (!compileOptions.isRefCounted() && !readsGlobalItself(argument))
                {
                    continue;
                }

                held.push_back({op, argument});
            }
        });

        for (auto [op, argument] : held)
        {
            auto location = op->getLoc();

            // From the read on, not from the call: an argument after it may give the global a new
            // value first (`f(G, (G = mk(), x))`), and its release freed the block this one read
            // before a retain at the call could keep it. A conditional argument between them puts
            // the call in a block of its own, run once for each read unless a loop leads back.
            mlir::OpBuilder builder(op);
            auto *readOp = argument.getDefiningOp();
            if (readOp && (readOp->getBlock() == op->getBlock() ||
                           (readOp->getParentRegion() == op->getParentRegion() &&
                            !blockReaches(op->getBlock(), readOp->getBlock()))))
            {
                builder.setInsertionPointAfter(readOp);
            }

            builder.create<mlir_ts::RetainOp>(location, argument);

            // A call a try around it takes the exceptions of gives the hold back on that path too,
            // through a slot that holds the value only while the hold lasts: null before the read
            // and after the call, so the handler's release does nothing when something else threw.
            auto *unwindRegion = unwindRegionOf(op);
            mlir::Value slot;
            if (unwindRegion)
            {
                auto type = argument.getType();
                auto &entryBlock = op->getParentOfType<mlir_ts::FuncOp>().getBody().front();
                mlir::OpBuilder entryBuilder(&entryBlock, entryBlock.begin());
                if (auto entryOp = dyn_cast<mlir_ts::EntryOp>(entryBlock.front()))
                {
                    entryBuilder.setInsertionPointAfter(entryOp);
                }

                auto none = entryBuilder.create<mlir_ts::DefaultOp>(location, type);
                slot = entryBuilder.create<mlir_ts::VariableOp>(location, mlir_ts::RefType::get(type), none,
                                                                entryBuilder.getBoolAttr(false),
                                                                entryBuilder.getIndexAttr(0));
                builder.create<mlir_ts::StoreOp>(location, argument, slot);
            }

            builder.setInsertionPointAfter(op);
            builder.create<mlir_ts::ReleaseOp>(location, argument);
            if (!slot)
            {
                continue;
            }

            builder.create<mlir_ts::StoreOp>(location, builder.create<mlir_ts::DefaultOp>(location, argument.getType()),
                                             slot);

            // in a catch clause, after the catch variable is set
            auto &handlerBlock = unwindRegion->front();
            mlir::OpBuilder handlerBuilder(&handlerBlock, handlerBlock.begin());
            for (auto &handlerOp : handlerBlock)
            {
                if (isa<mlir_ts::CatchOp>(handlerOp))
                {
                    handlerBuilder.setInsertionPointAfter(&handlerOp);
                    break;
                }
            }

            handlerBuilder.create<mlir_ts::ReleaseSlotOp>(location, slot);
            handlerBuilder.create<mlir_ts::StoreOp>(
                location, handlerBuilder.create<mlir_ts::DefaultOp>(location, argument.getType()), slot);
        }
    }

    // Whether a read of a local, `argument`, is followed by a new value given to that local (or its
    // release) before `call` takes it (holdGlobalArguments). Under rc, and for a handle under own.
    bool localAssignedBefore(mlir::Operation *call, mlir::Value argument)
    {
        if (!compileOptions.isRefCounted() && !MLIRTypeHelper::isSharedHandleType(argument.getType()))
        {
            return false;
        }

        auto *read = argument.getDefiningOp();
        if (!read)
        {
            return false;
        }

        if (auto local = mlir::dyn_cast_or_null<mlir_ts::VariableOp>(own_facts::slotOf(argument)))
        {
            auto captured = local.getCaptured().value_or(false);
            return own_facts::findBetween(read, call, [&](mlir::Operation *op) {
                       return (captured && own_facts::isCall(op)) ||
                              own_facts::sameSlot(own_facts::slotWritten(op), local);
                   }) != nullptr;
        }

        // a field or an element of an object no local's slot holds (`const h = new H()`, folded):
        // a store into the same field, or any element, of that object (`f(h.v, (h.v = mk(), 0))`)
        auto value = argument;
        while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
        {
            value = castOp.getIn();
        }

        auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
        if (!loadOp || !own_facts::isPlace(loadOp.getReference()))
        {
            return false;
        }

        auto baseOf = [](mlir::Operation *place) {
            auto base = place->getOperand(0);
            while (auto castOp = base.getDefiningOp<mlir_ts::CastOp>())
            {
                base = castOp.getIn();
            }

            return base;
        };

        auto *place = loadOp.getReference().getDefiningOp();
        auto base = baseOf(place);
        auto propertyRef = mlir::dyn_cast<mlir_ts::PropertyRefOp>(place);
        return own_facts::findBetween(read, call, [&](mlir::Operation *op) {
                   auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(op);
                   auto *written = storeOp ? storeOp.getReference().getDefiningOp() : nullptr;
                   if (!written || written->getName() != place->getName() || baseOf(written) != base)
                   {
                       return false;
                   }

                   auto writtenField = mlir::dyn_cast<mlir_ts::PropertyRefOp>(written);
                   return !propertyRef || writtenField.getPosition() == propertyRef.getPosition();
               }) != nullptr;
    }

    static bool blockReaches(mlir::Block *from, mlir::Block *to)
    {
        llvm::SmallPtrSet<mlir::Block *, 16> seen;
        llvm::SmallVector<mlir::Block *> work{from};
        while (!work.empty())
        {
            auto *block = work.pop_back_val();
            if (block == to)
            {
                return true;
            }

            if (seen.insert(block).second)
            {
                work.append(block->getSuccessors().begin(), block->getSuccessors().end());
            }
        }

        return false;
    }

    static mlir::Value stripCasts(mlir::Value value)
    {
        while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
        {
            value = castOp.getIn();
        }

        return value;
    }

    // Do two references name the same place? One SSA reference, one global, or one field of the
    // same value - each access to a field builds a `ts.PropertyRef` of its own.
    static bool samePlace(mlir::Value a, mlir::Value b)
    {
        if (a == b)
        {
            return true;
        }

        auto globalA = a.getDefiningOp<mlir_ts::AddressOfOp>();
        auto globalB = b.getDefiningOp<mlir_ts::AddressOfOp>();
        if (globalA && globalB)
        {
            return globalA.getGlobalName() == globalB.getGlobalName();
        }

        auto fieldA = a.getDefiningOp<mlir_ts::PropertyRefOp>();
        auto fieldB = b.getDefiningOp<mlir_ts::PropertyRefOp>();
        return fieldA && fieldB && fieldA.getPosition() == fieldB.getPosition() &&
               sameValue(fieldA->getOperand(0), fieldB->getOperand(0));
    }

    // Are two values one value: one SSA value, or reads of one place?
    static bool sameValue(mlir::Value a, mlir::Value b)
    {
        a = stripCasts(a);
        b = stripCasts(b);
        if (a == b)
        {
            return true;
        }

        auto loadA = a.getDefiningOp<mlir_ts::LoadOp>();
        auto loadB = b.getDefiningOp<mlir_ts::LoadOp>();
        return loadA && loadB && samePlace(loadA.getReference(), loadB.getReference());
    }

    // Might two references name one place under different names? One field of two objects of one
    // class (`p.n` and `G.n`, where `p` is `G`), or an element of two arrays of one type: which
    // object a value is cannot be known here.
    static bool mayBeSamePlace(mlir::Value a, mlir::Value b)
    {
        if (samePlace(a, b))
        {
            return true;
        }

        auto fieldA = a.getDefiningOp<mlir_ts::PropertyRefOp>();
        auto fieldB = b.getDefiningOp<mlir_ts::PropertyRefOp>();
        if (fieldA && fieldB)
        {
            auto type = stripCasts(fieldA->getOperand(0)).getType();
            return fieldA.getPosition() == fieldB.getPosition() && mlir::isa<mlir_ts::ClassType>(type) &&
                   type == stripCasts(fieldB->getOperand(0)).getType();
        }

        auto elementA = a.getDefiningOp<mlir_ts::ElementRefOp>();
        auto elementB = b.getDefiningOp<mlir_ts::ElementRefOp>();
        if (elementA && elementB)
        {
            auto type = stripCasts(elementA->getOperand(0)).getType();
            return mlir::isa<mlir_ts::ArrayType>(type) && type == stripCasts(elementB->getOperand(0)).getType();
        }

        return false;
    }

    // Is the argument read out of a slot or a place that something between the read and the call
    // overwrites - a later argument (`f(n, (n = mk(), n))`), under its own name or another
    // (`f((p.n = mk(), p.n), (G.n = mk(), 0))`, with `p` being `G`)?
    //
    // So is one read out of a field or an element of a value read so: overwriting `n` in
    // `f({ v: n.v }, (n = mk(), 0))` gives back the object whose field the argument read.
    static bool overwrittenBeforeCall(mlir::Value argument, mlir::Operation *call)
    {
        auto value = stripCasts(argument);
        for (auto depth = 0; depth < 8; depth++)
        {
            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            if (!loadOp || loadOp->getBlock() != call->getBlock())
            {
                return false;
            }

            auto reference = loadOp.getReference();
            for (auto *op = loadOp->getNextNode(); op && op != call; op = op->getNextNode())
            {
                auto overwrites = op->walk([&](mlir::Operation *nested) {
                    auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(nested);
                    auto releaseSlotOp = mlir::dyn_cast<mlir_ts::ReleaseSlotOp>(nested);
                    auto place =
                        storeOp ? storeOp.getReference() : releaseSlotOp ? releaseSlotOp.getSlot() : mlir::Value();
                    return place && mayBeSamePlace(place, reference) ? mlir::WalkResult::interrupt()
                                                                    : mlir::WalkResult::advance();
                });

                if (overwrites.wasInterrupted())
                {
                    return true;
                }
            }

            auto *place = reference.getDefiningOp();
            if (!mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::InterfaceSymbolRefOp>(place))
            {
                return false;
            }

            value = stripCasts(place->getOperand(0));
        }

        return false;
    }

    // Is the argument read out of a field or an element of a value the call is given as another
    // argument (`f(n, n.items[0])`)?
    // Might two values be one object, read from different places? Two objects of one class may be
    // (`f2(HG.n, n11.v)` where the local `n11` was given `HG.n`): which object a value is cannot be
    // known here, as for mayBeSamePlace. Fuzz seed 937072: `f2` set `p3.v`, which freed the block its
    // other parameter, read out of that field, still pointed at.
    static bool mayBeSameObject(mlir::Value a, mlir::Value b)
    {
        auto type = stripCasts(a).getType();
        return mlir::isa<mlir_ts::ClassType>(type) && type == stripCasts(b).getType();
    }

    static bool readFromOtherArgument(mlir::Value argument, mlir::Operation *call)
    {
        auto value = stripCasts(argument);
        for (auto depth = 0; depth < 8; depth++)
        {
            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            auto *place = loadOp ? loadOp.getReference().getDefiningOp() : nullptr;
            if (!mlir::isa_and_nonnull<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::InterfaceSymbolRefOp>(place))
            {
                return false;
            }

            value = place->getOperand(0);
            for (auto other : own_facts::callArgs(call))
            {
                if (stripCasts(other) != stripCasts(argument) && (sameValue(other, value) || mayBeSameObject(other, value)))
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Where an exception out of `op` is taken first within its function: the region of the try
    // around it that runs then, in the order its landing pad takes them (TryOpLowering) - the
    // cleanup, else the catch clause, else the finally, for a call in its body; the finally for a
    // call in its catch clause. Null when the exception leaves the function.
    static mlir::Region *unwindRegionOf(mlir::Operation *op)
    {
        auto hasOps = [](mlir::Region &region) {
            return llvm::any_of(region, [](mlir::Block &block) { return &block.front() != block.getTerminator(); });
        };

        for (auto *region = op->getParentRegion(); region; region = region->getParentOp()->getParentRegion())
        {
            auto *parentOp = region->getParentOp();
            if (isa<mlir::FunctionOpInterface>(parentOp))
            {
                return nullptr;
            }

            auto tryOp = dyn_cast<mlir_ts::TryOp>(parentOp);
            if (!tryOp)
            {
                continue;
            }

            if (region == &tryOp.getBody())
            {
                if (hasOps(tryOp.getCleanup()))
                {
                    return &tryOp.getCleanup();
                }

                if (tryOp->hasAttr(HAS_CATCH_ATTR_NAME) || hasOps(tryOp.getCatches()))
                {
                    return &tryOp.getCatches();
                }

                if (hasOps(tryOp.getFinally()))
                {
                    return &tryOp.getFinally();
                }
            }
            else if (region == &tryOp.getCatches() && hasOps(tryOp.getFinally()))
            {
                return &tryOp.getFinally();
            }
        }

        return nullptr;
    }

    // A global is born holding the reference its initializer carries.
    //
    // The initializer region ends in `ts.GlobalResult`, a terminator, so a value given to it
    // straight is never released as a discarded temporary (§9.30): `let g = mk()` keeps the call's
    // +1. A value given to it through a view of the global's type is: `ts.Cast` to `C | null` or
    // `Shared<T> | null`, `ts.OptionalValue` to `C | undefined`, `ts.CreateUnionInstance`,
    // `ts.NewInterface` over a class object for an interface-typed global (`let g: I = mk()`), and
    // over the class read out of a `C | null` (`let g: I | null = maybe()`) - the view
    // is the one use the producer has, and it is releasable, so the producer's reference was given
    // back at the end of the region and the global was born at count 0, pointing at freed memory
    // (#459; own spec 23.1 for handles).
    //
    // The producer is anything that carries an owned reference by now: `new Shared(..)` or a fresh
    // string, marked where MLIRGen builds it, or a call, marked above once its callee is classified
    // - which is why MLIRGen alone cannot see it. A read out of a field, an element or another
    // global carries none, and gets a retain of its own (under own, a handle only: a retain left in
    // a global's region is an error for any other type).
    void claimGlobalInitializers(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        module.walk([&](mlir_ts::GlobalOp globalOp) {
            if (!mth.ownsHeapMemory(globalOp.getLoc(), globalOp.getType()))
            {
                return;
            }

            auto *block = globalOp.getInitializerBlock();
            if (!block || block->empty())
            {
                return;
            }

            auto resultOp = mlir::dyn_cast<mlir_ts::GlobalResultOp>(block->getTerminator());
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return;
            }

            auto value = resultOp->getOperand(0);
            while (auto *definingOp = value.getDefiningOp())
            {
                if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(definingOp))
                {
                    value = castOp.getIn();
                }
                else if (auto optionalValueOp = mlir::dyn_cast<mlir_ts::OptionalValueOp>(definingOp))
                {
                    value = optionalValueOp.getIn();
                }
                else if (auto createUnionInstanceOp = mlir::dyn_cast<mlir_ts::CreateUnionInstanceOp>(definingOp))
                {
                    value = createUnionInstanceOp.getIn();
                }
                else if (auto newInterfaceOp = mlir::dyn_cast<mlir_ts::NewInterfaceOp>(definingOp))
                {
                    value = newInterfaceOp.getThisVal();
                }
                else if (auto getValueFromUnionOp = mlir::dyn_cast<mlir_ts::GetValueFromUnionOp>(definingOp))
                {
                    value = getValueFromUnionOp.getIn();
                }
                else
                {
                    break;
                }
            }

            if (producesOwnedResult(value))
            {
                value.getDefiningOp()->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                return;
            }

            if (bornUncounted(value))
            {
                retainForGlobal(resultOp, resultOp->getOperand(0));
                return;
            }

            // An optional widened into a union (`let g: C | null | undefined = mkU()`, #462) is a
            // ts.If over the call made before it: each branch yields that call's value through
            // views, or a value that holds nothing (the empty optional). The global takes the call's
            // count, as it would take the call's straight.
            if (auto ifOp = value.getDefiningOp<mlir_ts::IfOp>())
            {
                auto throughViews = [](mlir::Value yielded) {
                    while (auto *definingOp = yielded.getDefiningOp())
                    {
                        if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(definingOp))
                        {
                            yielded = castOp.getIn();
                        }
                        else if (auto optionalValueOp = mlir::dyn_cast<mlir_ts::OptionalValueOp>(definingOp))
                        {
                            yielded = optionalValueOp.getIn();
                        }
                        else if (auto createUnionInstanceOp = mlir::dyn_cast<mlir_ts::CreateUnionInstanceOp>(definingOp))
                        {
                            yielded = createUnionInstanceOp.getIn();
                        }
                        else if (auto valueOp = mlir::dyn_cast<mlir_ts::ValueOp>(definingOp))
                        {
                            yielded = valueOp.getIn();
                        }
                        else
                        {
                            break;
                        }
                    }

                    return yielded;
                };

                mlir::Value producer;
                auto agrees = true;
                for (auto &region : ifOp->getRegions())
                {
                    auto resultOp = region.empty() ? mlir_ts::ResultOp()
                                                   : mlir::dyn_cast<mlir_ts::ResultOp>(region.back().getTerminator());
                    if (!resultOp || resultOp->getNumOperands() != 1)
                    {
                        agrees = false;
                        break;
                    }

                    auto yielded = throughViews(resultOp->getOperand(0));
                    auto *yieldedOp = yielded.getDefiningOp();
                    if (yieldedOp && mlir::isa<mlir_ts::OptionalUndefOp, mlir_ts::NullOp, mlir_ts::UndefOp>(yieldedOp))
                    {
                        continue;
                    }

                    if (producer && producer != yielded)
                    {
                        agrees = false;
                        break;
                    }

                    producer = yielded;
                }

                if (agrees && producer && producesOwnedResult(producer) &&
                    !ifOp->isAncestor(producer.getDefiningOp()))
                {
                    producer.getDefiningOp()->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME,
                                                      mlir::UnitAttr::get(&getContext()));
                    return;
                }
            }

            // An object literal (`let o = { s: mk() }`) is built in a temporary slot and read out
            // whole: what its fields were given goes into the global with it (#461), nested
            // literals included.
            if (auto literalLoad = value.getDefiningOp<mlir_ts::LoadOp>())
            {
                if (auto literalSlot = literalLoad.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    literalSlot && !literalSlot->hasAttr(OWNED_LOCAL_ATTR_NAME))
                {
                    claimStoredIntoLiteral(literalSlot);
                    return;
                }
            }

            // A field or element read (`holder.child`, `new P().child`, `mkP().child`,
            // `mkArr()[0]`) carries no reference: the object or array keeps its count, and a
            // temporary one gives it back at the end of the region. Nor does a read of another
            // global (`let k = a`), which keeps its own. The global takes a count of its own,
            // retained right where it is read - before the temporary's release - or the first
            // overwrite of either global frees what the other still reads. Under own only a handle
            // is counted; any other read like these is an error there (OwnershipSignaturePass).
            // Erased under gc and none.
            auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>();
            auto *place = loadOp ? loadOp.getReference().getDefiningOp() : nullptr;
            if (place && (MLIRTypeHelper::isSharedHandleType(value.getType()) || compileOptions.isRefCounted()) &&
                mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::AddressOfOp>(place))
            {
                mlir::OpBuilder builder(loadOp->getContext());
                builder.setInsertionPointAfter(loadOp);
                builder.create<mlir_ts::RetainOp>(loadOp->getLoc(), value);
                return;
            }

            // A method bound to an object (`let c = mk().get`, `let c = G.v.get`, `let c = ix.get`)
            // is born with no count, and only its receivers take one; under rc the bound value counts
            // the object it is bound to. The global is the receiver here: without a count of its
            // own, a temporary object was given back at the end of the region, and the first
            // `c = d` released it a second time.
            auto *boundOp = value.getDefiningOp();
            auto isBoundMethod =
                mlir::isa_and_nonnull<mlir_ts::ThisVirtualSymbolRefOp, mlir_ts::ThisSymbolRefOp>(boundOp) ||
                (mlir::isa_and_nonnull<mlir_ts::InterfaceSymbolRefOp>(boundOp) &&
                 mlir::isa<mlir_ts::BoundFunctionType>(value.getType())) ||
                (mlir::isa_and_nonnull<mlir_ts::CreateBoundFunctionOp>(boundOp) && !boundOp->hasAttr(OWNS_CAPTURE_ATTR_NAME));
            if (isBoundMethod)
            {
                retainForGlobal(resultOp, resultOp->getOperand(0));
            }
        });
    }

    // A string made over memory it does not own - `<string><Opaque>Ref(buffer[0])`, a cast out of
    // `!ts.opaque` - is a view of that memory, not a block of its own (#481). Under rc a receiver
    // used to retain it, counting whatever word sits in front of the pointer: the array's data
    // block for `Ref(buffer[0])`, which then outlived the array and leaked once the array grew,
    // and element data or another block's tail for `Ref(buffer[1])`. Now, as under own, a view
    // keeps no count: where a receiver retains it (a return, a store into a field, an element or
    // a global, a push, a capture), the receiver takes a copy of the string instead, made right
    // there - after the bytes were written through the view (`sprintf_s(s, ...)`). A `let` that
    // only ever holds a view borrows it; one that is assigned again starts from a copy, since the
    // assignment releases what it held.
    //
    // A view still points into the array: used after the array grows or dies, it reads freed
    // memory in every model, as a C pointer would. A view kept inside a tuple or record literal,
    // or by a function it is passed to, is not copied.
    void copyEscapingStringViews(mlir::ModuleOp module)
    {
        if (!compileOptions.isRefCounted())
        {
            return;
        }

        llvm::DenseSet<mlir::Operation *> viewLocals;
        auto isView = [&](mlir::Value value) {
            for (auto depth = 0; depth < 16 && value; depth++)
            {
                auto *definingOp = value.getDefiningOp();
                if (auto castOp = mlir::dyn_cast_or_null<mlir_ts::CastOp>(definingOp))
                {
                    if (mlir::isa<mlir_ts::OpaqueType>(castOp.getIn().getType()))
                    {
                        return true;
                    }

                    value = own_facts::isCopyableString(castOp.getIn().getType()) ? castOp.getIn() : mlir::Value();
                }
                else if (auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(definingOp))
                {
                    auto *slot = loadOp.getReference().getDefiningOp();
                    value = slot && viewLocals.contains(slot)
                                ? mlir::cast<mlir_ts::VariableOp>(slot).getInitializer()
                                : mlir::Value();
                }
                else
                {
                    return false;
                }
            }

            return false;
        };

        // `let l = view`: borrowed when nothing assigns the variable again, a copy when something does.
        // Collected first: the walk has already stepped to the op after a variable - its RetainSlot,
        // which is erased below.
        llvm::SmallVector<mlir_ts::VariableOp> variables;
        module.walk([&](mlir_ts::VariableOp varOp) { variables.push_back(varOp); });
        for (auto varOp : variables)
        {
            auto init = varOp.getInitializer();
            if (!init || !own_facts::isCopyableString(init.getType()) || !isView(init) || varOp.getCaptured().value_or(false))
            {
                continue;
            }

            auto assignedAgain = llvm::any_of(varOp->getUsers(), [&](mlir::Operation *user) {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                return storeOp && storeOp.getReference() == varOp.getResult();
            });

            if (assignedAgain)
            {
                mlir::OpBuilder builder(varOp);
                auto copy = builder.create<mlir_ts::StringCopyOp>(varOp.getLoc(), init.getType(), init);
                varOp.getInitializerMutable().assign(copy);
                continue;
            }

            llvm::SmallVector<mlir::Operation *> slotOps;
            for (auto *user : varOp->getUsers())
            {
                if (mlir::isa<mlir_ts::RetainSlotOp, mlir_ts::ReleaseSlotOp>(user))
                {
                    slotOps.push_back(user);
                }
            }

            for (auto *slotOp : slotOps)
            {
                slotOp->erase();
            }

            varOp->removeAttr(OWNED_LOCAL_ATTR_NAME);
            viewLocals.insert(varOp);
        }

        // a receiver's retain: MLIRGen puts it right in front of the receiver, with nothing but
        // other counting between them (`ts.Retain(v); ts.ReleaseSlot(field); ts.Store(v, field)`)
        llvm::SmallVector<mlir_ts::RetainOp> retains;
        module.walk([&](mlir_ts::RetainOp retainOp) {
            auto value = retainOp.getReference();
            if (own_facts::isCopyableString(value.getType()) && isView(value))
            {
                retains.push_back(retainOp);
            }
        });

        for (auto retainOp : retains)
        {
            auto value = retainOp.getReference();
            auto *receiver = retainOp->getNextNode();
            while (receiver && mlir::isa<mlir_ts::RetainOp, mlir_ts::ReleaseOp, mlir_ts::RetainSlotOp,
                                         mlir_ts::ReleaseSlotOp>(receiver))
            {
                receiver = receiver->getNextNode();
            }

            if (!receiver || !llvm::is_contained(receiver->getOperands(), value))
            {
                continue;
            }

            mlir::OpBuilder builder(retainOp);
            auto copy = builder.create<mlir_ts::StringCopyOp>(retainOp->getLoc(), value.getType(), value);
            retainOp->setOperand(0, copy);
            for (auto &operand : receiver->getOpOperands())
            {
                if (operand.get() == value)
                {
                    operand.set(copy);
                }
            }
        }
    }

    // An array or tuple built in a global's region (`const a = [1, 2]`, `const t = [1, [6, 7]]`, or
    // a field of `const o = { items: [1, 2] }`) is born at count 0 under rc: a local takes its first
    // count with RetainSlot, and nothing took one for the global, so the first local that read it
    // and let go freed it. A global is a root and holds a count of its own (§9.49). Under own the
    // global owns what it holds without a count, and a retain there is an error.
    // A constant array reaches the global through its cast to an array, which copies it to the heap;
    // any other constant (a string literal, an RTTI name) is static and immortal, and its global
    // stays a constant initializer.
    static bool bornUncounted(mlir::Value value)
    {
        auto *definingOp = value.getDefiningOp();
        return definingOp && (mlir::isa<mlir_ts::CreateArrayOp, mlir_ts::NewArrayOp, mlir_ts::NewEmptyArrayOp,
                                        mlir_ts::CreateTupleOp>(definingOp) ||
                              mlir::isa<mlir_ts::ConstArrayType>(value.getType()));
    }

    void retainForGlobal(mlir::Operation *before, mlir::Value value)
    {
        if (!compileOptions.isRefCounted())
        {
            return;
        }

        mlir::OpBuilder builder(before->getContext());
        builder.setInsertionPoint(before);
        builder.create<mlir_ts::RetainOp>(before->getLoc(), value);
    }

    // Each owned producer stored into a field of an object literal's temporary slot, through the
    // views of claimGlobalInitializers, is taken over by whoever takes the literal; a field that is
    // itself a literal read out of its own temporary slot is followed into.
    void claimStoredIntoLiteral(mlir_ts::VariableOp literalSlot)
    {
        for (auto *user : literalSlot->getUsers())
        {
            auto fieldRef = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
            if (!fieldRef)
            {
                continue;
            }

            for (auto *fieldUser : fieldRef->getUsers())
            {
                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                if (!storeOp || storeOp.getReference() != fieldRef.getResult())
                {
                    continue;
                }

                auto stored = storeOp.getValue();
                while (auto *definingOp = stored.getDefiningOp())
                {
                    if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(definingOp))
                    {
                        stored = castOp.getIn();
                    }
                    else if (auto optionalValueOp = mlir::dyn_cast<mlir_ts::OptionalValueOp>(definingOp))
                    {
                        stored = optionalValueOp.getIn();
                    }
                    else if (auto createUnionInstanceOp = mlir::dyn_cast<mlir_ts::CreateUnionInstanceOp>(definingOp))
                    {
                        stored = createUnionInstanceOp.getIn();
                    }
                    else
                    {
                        break;
                    }
                }

                if (producesOwnedResult(stored))
                {
                    stored.getDefiningOp()->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME,
                                                    mlir::UnitAttr::get(&getContext()));
                }
                else if (bornUncounted(stored))
                {
                    retainForGlobal(storeOp, storeOp.getValue());
                }
                else if (auto innerLoad = stored.getDefiningOp<mlir_ts::LoadOp>())
                {
                    if (auto innerSlot = innerLoad.getReference().getDefiningOp<mlir_ts::VariableOp>();
                        innerSlot && !innerSlot->hasAttr(OWNED_LOCAL_ATTR_NAME))
                    {
                        claimStoredIntoLiteral(innerSlot);
                    }
                }
            }
        }
    }

    // A conditional expression whose branches disagree about ownership.
    //
    // `cond ? a : f()` builds a `ts.If` with a result, and the two branches hand back references
    // on different terms: `f()` produces a +1 that somebody must take over, while `a` is
    // borrowed from wherever it already lives. The receiver outside can only do one thing, and
    // what it does is retain - so the +1 from the call side is left with no owner at all.
    //
    // §9.30 cannot reach it, and says so in its own terms: the value's only user is `ts.Result`,
    // a terminator, and releasing a value handed to a successor would free it while it is still
    // live. So the call escapes the region unconsumed and the branch that allocates leaks every
    // time it is taken. This is not a corner of the language - it is `x = cond ? y : f()` - and
    // in `raytrace.ts` one such line held **18 of the 20 MB** reference counting had left on the
    // table. Written as `if`/`else` with an assignment the same program reclaims everything,
    // because then each branch stores into the slot and the store consumes.
    //
    // The fix is to make the branches agree before anyone outside looks: whichever branch
    // borrows takes a reference of its own, so that every branch yields +1, and the `ts.If`
    // result then carries a reference exactly the way a call's result does. From there the
    // ordinary machinery applies - a receiver takes it over, or §9.30 gives it back at the end
    // of the block. Retaining the borrowed side rather than releasing the owned one is what
    // keeps this safe: there is no point inside the region at which the owned value could be
    // released, because the receiver's retain has not happened yet.
    void consumeConditionalResults(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        llvm::SmallVector<mlir::Operation *> toErase;

        module.walk([&](mlir_ts::IfOp ifOp) {
            if (ifOp.getNumResults() != 1)
            {
                return;
            }

            auto result = ifOp.getResult(0);
            if (!mth.ownsHeapMemory(ifOp.getLoc(), result.getType()))
            {
                return;
            }

            // Already settled, either by this pass on an earlier op or where it was built.
            if (ifOp->hasAttr(OWNED_RESULT_ATTR_NAME))
            {
                return;
            }

            // What each branch hands back. Every region has to yield exactly one value, or this
            // is a shape that is not understood and is left alone.
            llvm::SmallVector<mlir_ts::ResultOp> yields;
            for (auto &region : ifOp->getRegions())
            {
                if (region.empty())
                {
                    return;
                }

                auto resultOp = mlir::dyn_cast_or_null<mlir_ts::ResultOp>(region.back().getTerminator());
                if (!resultOp || resultOp->getNumOperands() != 1)
                {
                    return;
                }

                yields.push_back(resultOp);
            }

            if (yields.size() < 2)
            {
                return;
            }

            // Only worth doing when some branch actually carries an unclaimed +1. Where every
            // branch borrows, the receiver's retain is already the right and only answer.
            // What a branch yields, seen through the casts that only widen it: `new C()` yielded as
            // `C | null` or `C | undefined` is still that block, so its reference goes out with it
            // (#465). So is a literal yielded as the @value type it is laid out as (the cast lowers to
            // nothing): `c ? null : { at: d, owner: new Owner(d) }` of `Mark | null`. A cast that makes
            // a new value (a box) carries its own reference instead.
            auto producerOf = [](mlir::Value value) {
                while (auto *definingOp = value.getDefiningOp())
                {
                    if (definingOp->hasAttr(OWNED_RESULT_ATTR_NAME))
                    {
                        break;
                    }

                    if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(definingOp);
                        castOp && (mlir::isa<mlir_ts::UnionType, mlir_ts::OptionalType>(castOp.getType()) ||
                                   (mlir::isa<mlir_ts::ValueType>(castOp.getType()) &&
                                    mlir::isa<mlir_ts::TupleType, mlir_ts::ConstTupleType>(castOp.getIn().getType())) ||
                                   // a bound function retyped as another (an adapter bound over its box, given
                                   // as the method type it stands for): the same value, the same reference
                                   (mlir::isa<mlir_ts::BoundFunctionType>(castOp.getType()) &&
                                    mlir::isa<mlir_ts::BoundFunctionType>(castOp.getIn().getType()))))
                    {
                        value = castOp.getIn();
                    }
                    else if (auto optionalValueOp = mlir::dyn_cast<mlir_ts::OptionalValueOp>(definingOp))
                    {
                        value = optionalValueOp.getIn();
                    }
                    else if (auto createUnionInstanceOp = mlir::dyn_cast<mlir_ts::CreateUnionInstanceOp>(definingOp))
                    {
                        value = createUnionInstanceOp.getIn();
                    }
                    else
                    {
                        break;
                    }
                }

                return value;
            };

            auto carriesOwned = [&](mlir_ts::ResultOp resultOp) {
                auto produced = producerOf(resultOp->getOperand(0));
                auto *definingOp = produced.getDefiningOp();
                return ::typescript::mayTakeOverReference(produced) &&
                       definingOp->getParentRegion() == resultOp->getParentRegion();
            };

            // ...or some branch yields a field read out of a temporary it made (`given ?? pair(n).item`):
            // the temporary is given back before the branch ends, and with it what the field holds,
            // so the receiver's retain after the `ts.If` comes too late. The branch retains it first.
            // An element read out of a new array the branch made (`c ? ["c" + i][0] : d`) is the same under rc:
            // the array is given back at the end of the branch (holdReadOnlyNewArrays), and with it the element.
            // So is what is read out of an element of one, through elements and fields: an element of an inner
            // array (`c ? [[mk(i)]][0][0] : null`), a field of an element (`c ? [mk(i)][0].s : null`, `c ? [{ v:
            // mk(i) }][0].v : null`), and so on - the outer array is given back there, and with it all it holds.
            // A cast that only narrows a union or an optional to a member (`!`, `as N`, `<N>` of an `N | null`
            // element) is the same block, and is looked through on the way.
            auto narrowed = [](mlir::Value value) {
                while (auto castOp = value.getDefiningOp<mlir_ts::CastOp>())
                {
                    if (!mlir::isa<mlir_ts::UnionType, mlir_ts::OptionalType>(castOp.getIn().getType()) ||
                        !mlir::isa<mlir_ts::ClassType, mlir_ts::StringType, mlir_ts::ArrayType, mlir_ts::UnionType>(
                            castOp.getType()))
                    {
                        break;
                    }

                    value = castOp.getIn();
                }

                return value;
            };
            // Under rc, whether a value is read out of a literal made in `region` - an array literal, or the tuple a
            // literal of mixed elements is (`c ? [mk(i), "x"][0] : null`, whose element is the +1 the branch gives
            // back) - through elements and fields, widening and narrowing casts and an optional's value (a read out of
            // it); or read out of a temporary made in `region` that the branch gives back (an owned call result or
            // conditional result). A value a conditional within `region` yields - an optional chain (`c ? [mk(i)][0]?.s : null`,
            // `[[mk(i)]][0]?.[0]`) and its conversion to `| null` - is what its branches yield, looked into the same
            // way, each conditional's result once (`visited`), so the walk is linear in what it looks into.
            // `readOut`: the value is an element, a field or an optional's value, not the literal or temporary itself.
            llvm::DenseSet<std::pair<mlir::Value, unsigned>> visited;
            std::function<bool(mlir::Value, mlir::Region *, bool)> readsBranchLiteral = [&](mlir::Value value,
                                                                                         mlir::Region *region, bool readOut) {
                    for (;;)
                    {
                        auto peeled = narrowed(producerOf(value));
                        if (auto valueOp = peeled.getDefiningOp<mlir_ts::ValueOp>())
                        {
                            peeled = valueOp.getIn();
                            readOut = true;
                        }
                        else if (auto extractPropertyOp = peeled.getDefiningOp<mlir_ts::ExtractPropertyOp>())
                        {
                            peeled = extractPropertyOp.getObject();
                            readOut = true;
                        }

                        if (peeled == value)
                        {
                            break;
                        }

                        value = peeled;
                    }

                    auto *definingOp = value.getDefiningOp();
                    if (!definingOp)
                    {
                        return false;
                    }

                    if (readOut && (isNewArray(definingOp) || mlir::isa<mlir_ts::CreateTupleOp>(definingOp)) &&
                        definingOp->getParentRegion() == region)
                    {
                        return true;
                    }

                    // ...or read out of a temporary made in `region` that the branch gives back at its end: an owned
                    // optional a call returns, or a nested conditional's owned result, turned into `| null` by
                    // another conditional (`c ? mo(i) : null`, `c ? (b ? [mk(i)]?.[0] : undefined) : null`)
                    if (readOut && definingOp->getParentRegion() == region && ::typescript::mayTakeOverReference(value))
                    {
                        return true;
                    }

                    if (auto ifOp = mlir::dyn_cast<mlir_ts::IfOp>(definingOp))
                    {
                        if (!region->isAncestor(ifOp->getParentRegion()) || !visited.insert({value, readOut ? 1u : 0u}).second)
                        {
                            return false;
                        }

                        auto index = mlir::cast<mlir::OpResult>(value).getResultNumber();
                        for (auto &inner : ifOp->getRegions())
                        {
                            auto innerResult = inner.empty() ? mlir_ts::ResultOp()
                                                             : mlir::dyn_cast_or_null<mlir_ts::ResultOp>(inner.back().getTerminator());
                            if (innerResult && innerResult->getNumOperands() > index &&
                                readsBranchLiteral(innerResult->getOperand(index), region, readOut))
                            {
                                return true;
                            }
                        }

                        return false;
                    }

                    auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(definingOp);
                    if (!loadOp)
                    {
                        return false;
                    }

                    // the place read: a field of a value held in place (a tuple element) is in the same place; a
                    // field of an object, an element of an array, is read out of that object or array
                    auto place = loadOp.getReference();
                    while (auto propertyRefOp = place.getDefiningOp<mlir_ts::PropertyRefOp>())
                    {
                        auto object = propertyRefOp.getObjectRef();
                        if (!mlir::isa<mlir_ts::RefType>(object.getType()))
                        {
                            return readsBranchLiteral(object, region, true);
                        }

                        place = object;
                    }

                    auto elementRefOp = place.getDefiningOp<mlir_ts::ElementRefOp>();
                    return elementRefOp && readsBranchLiteral(elementRefOp->getOperand(0), region, true);
                };
            auto borrowsBranchTemporary = [&](mlir_ts::ResultOp resultOp) {
                auto value = producerOf(resultOp->getOperand(0));
                visited.clear();
                if (compileOptions.isRefCounted() &&
                    readsBranchLiteral(resultOp->getOperand(0), resultOp->getParentRegion(), false))
                {
                    return true;
                }

                while (auto loadOp = value.getDefiningOp<mlir_ts::LoadOp>())
                {
                    auto propertyRefOp = loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>();
                    if (!propertyRefOp)
                    {
                        return false;
                    }

                    value = propertyRefOp.getObjectRef();
                    auto *definingOp = value.getDefiningOp();
                    if (definingOp && ::typescript::mayTakeOverReference(value) &&
                        definingOp->getParentRegion() == resultOp->getParentRegion())
                    {
                        return true;
                    }
                }

                return false;
            };

            // ...or, under -mm=own, a field read out of an object a `const` names (`c ? init : r.value`),
            // the merge kept by a receiver: a retain in the branch, right after the read, is what own
            // takes out of the field (spec 64). A merge only read stays a borrow, as it was. Under rc a
            // retain after the `ts.If` does the same, and is left so.
            auto mergeKept = !findReceiverRetains(result, /*claim=*/false).empty();
            auto readsNamedObjectField = [&](mlir_ts::ResultOp resultOp) {
                auto loadOp = producerOf(resultOp->getOperand(0)).getDefiningOp<mlir_ts::LoadOp>();
                auto propertyRefOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::PropertyRefOp>() : mlir_ts::PropertyRefOp();
                auto *definingOp = propertyRefOp ? propertyRefOp.getObjectRef().getDefiningOp() : nullptr;
                // (or an owning `let` holds, spec 65)
                auto slotRead = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(definingOp);
                auto *slot = slotRead ? slotRead.getReference().getDefiningOp() : nullptr;
                return compileOptions.memoryModel == MemoryModelOwn && mergeKept && definingOp &&
                       (definingOp->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME) ||
                        (mlir::isa_and_nonnull<mlir_ts::VariableOp>(slot) && slot->hasAttr(OWNED_LOCAL_ATTR_NAME)));
            };

            // ...or some branch yields a record literal it built whose field holds a temporary it made
            // (`c ? { start: this.pos, dir: new V(1) } : ...`): the branch gives the temporary back before
            // it ends, so the receiver's retain after the `ts.If` - which retains the fields - comes too
            // late, and the field is freed under it. The branch retains it first. A field may hold
            // another literal the branch built (`{ one: { h: new H() } }` of a @value type holding a
            // @value type), whose fields are looked into the same way. Not under -mm=own, which
            // settles a literal's fields itself.
            std::function<bool(mlir::Value, mlir::Region *)> isLiteralOfBranchTemporary = [&](mlir::Value value,
                                                                                               mlir::Region *region) {
                auto loadOp = producerOf(value).getDefiningOp<mlir_ts::LoadOp>();
                auto varOp = loadOp ? loadOp.getReference().getDefiningOp<mlir_ts::VariableOp>() : mlir_ts::VariableOp();
                if (!varOp || varOp->getParentRegion() != region)
                {
                    return false;
                }

                return llvm::any_of(varOp->getUsers(), [&](mlir::Operation *user) {
                    auto propertyRefOp = mlir::dyn_cast<mlir_ts::PropertyRefOp>(user);
                    return propertyRefOp && llvm::any_of(propertyRefOp->getUsers(), [&](mlir::Operation *fieldUser) {
                               auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(fieldUser);
                               if (!storeOp || storeOp.getReference() != propertyRefOp.getResult())
                               {
                                   return false;
                               }

                               auto stored = producerOf(storeOp.getValue());
                               return (stored.getDefiningOp() && ::typescript::mayTakeOverReference(stored) &&
                                       stored.getDefiningOp()->getParentRegion() == region) ||
                                      isLiteralOfBranchTemporary(storeOp.getValue(), region);
                           });
                });
            };
            auto yieldsLiteralOfBranchTemporary = [&](mlir_ts::ResultOp resultOp) {
                return compileOptions.memoryModel != MemoryModelOwn &&
                       isLiteralOfBranchTemporary(resultOp->getOperand(0), resultOp->getParentRegion());
            };

            if (llvm::none_of(yields, carriesOwned) && llvm::none_of(yields, borrowsBranchTemporary) &&
                llvm::none_of(yields, readsNamedObjectField) && llvm::none_of(yields, yieldsLiteralOfBranchTemporary))
            {
                return;
            }

            for (auto resultOp : yields)
            {
                if (carriesOwned(resultOp))
                {
                    // The `ts.If` result now stands for this reference, so the producer's +1 has
                    // an owner and must not also be released as a discarded temporary.
                    producerOf(resultOp->getOperand(0))
                        .getDefiningOp()
                        ->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                    continue;
                }

                mlir::OpBuilder builder(resultOp);
                builder.create<mlir_ts::RetainOp>(resultOp->getLoc(), resultOp->getOperand(0));
            }

            ifOp->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));

            auto retains = findReceiverRetains(result);
            if (!retains.empty())
            {
                ifOp->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                toErase.append(retains.begin(), retains.end());
            }
        });

        for (auto *op : toErase)
        {
            op->erase();
        }

        consumeValueOrDefaultResults(mth, module);
    }

    // -mm=own: a pattern's default stores what it yields into a slot of the scope (MLIRGenImpl::optionalValueOrDefault),
    // which owns it and gives it back when the scope ends; the pattern's value borrows the slot. Only a new value is the
    // slot's to own - a call's result, a new object, a fresh string, as settled here, once calls are marked: it moves in.
    // A default that borrows (`p = e`, a field, a literal's element) is not stored: what the region yields is that
    // value, as before, and the slot stays empty. (An array pattern's element default is the else branch of the bounds
    // test, a `ts.If`: MLIRGenImpl::processDeclarationArrayBindingPatternSubPath.)
    void settleOwnPatternDefaults(mlir::ModuleOp module)
    {
        if (compileOptions.memoryModel != MemoryModelOwn)
        {
            return;
        }

        llvm::SmallVector<mlir_ts::StoreOp> stores;
        module.walk([&](mlir_ts::StoreOp storeOp) {
            auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
            if (varOp && varOp->hasAttr(PATTERN_DEFAULT_SLOT_ATTR_NAME) &&
                mlir::isa_and_nonnull<mlir_ts::OptionalValueOrDefaultOp, mlir_ts::ParamOptionalOp, mlir_ts::IfOp>(
                    storeOp->getParentOp()))
            {
                stores.push_back(storeOp);
            }
        });

        // the new value a region yields, where it is made: at its own `ts.Result`, or in a branch of a conditional
        // there (`p = e ?? mk(1)`: only the branch that makes one stores it, the other yields its borrow)
        std::function<void(mlir::Operation *, unsigned, mlir::Value)> storeNewValue = [&](mlir::Operation *resultOp, unsigned index,
                                                                                         mlir::Value slot) {
            auto yielded = resultOp->getOperand(index);
            auto slotType = mlir::cast<mlir_ts::RefType>(slot.getType()).getElementType();
            auto produced = yielded;
            while (true)
            {
                // (an element of an array of `T | undefined`: the default wrapped as an optional, `ts.OptionalValue`)
                if (auto optionalValueOp = produced.getDefiningOp<mlir_ts::OptionalValueOp>())
                {
                    produced = optionalValueOp.getIn();
                    continue;
                }

                auto castOp = produced.getDefiningOp<mlir_ts::CastOp>();
                if (!castOp || !mlir::isa<mlir_ts::UnionType, mlir_ts::OptionalType>(castOp.getType()))
                {
                    break;
                }

                produced = castOp.getIn();
            }

            auto *producer = produced.getDefiningOp();
            if (producer && ::typescript::mayTakeOverReference(produced) && yielded.getType() == slotType)
            {
                producer->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                mlir::OpBuilder builder(resultOp);
                builder.create<mlir_ts::StoreOp>(resultOp->getLoc(), yielded, slot);
                auto loadOp = builder.create<mlir_ts::LoadOp>(resultOp->getLoc(), slotType, slot);
                resultOp->setOperand(index, loadOp.getResult());
                return;
            }

            // (the one element of a literal made there only to read it, `p = [mk(1)][0]`: that element is the value; the
            // literal is not made - it held the element and was never given back)
            if (auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(producer); loadOp && produced == yielded)
            {
                auto elementRefOp = loadOp.getReference().getDefiningOp<mlir_ts::ElementRefOp>();
                auto literal = elementRefOp ? elementRefOp->getOperand(0).getDefiningOp<mlir_ts::CreateArrayOp>()
                                            : mlir_ts::CreateArrayOp();
                if (literal && literal->getNumOperands() == 1 && literal->getParentRegion() == resultOp->getParentRegion() &&
                    literal->getResult(0).hasOneUse() && elementRefOp->getResult(0).hasOneUse() && loadOp->getResult(0).hasOneUse() &&
                    literal->getOperand(0).getType() == yielded.getType() && literal->getOperand(0).getDefiningOp() &&
                    literal->getOperand(0).getDefiningOp()->hasAttr(OWNED_RESULT_ATTR_NAME) &&
                    !literal->getOperand(0).getDefiningOp()->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME))
                {
                    auto element = literal->getOperand(0);
                    resultOp->setOperand(index, element);
                    loadOp->erase();
                    elementRefOp->erase();
                    literal->erase();
                    if (auto *elementProducer = element.getDefiningOp())
                    {
                        elementProducer->removeAttr(OWNED_RESULT_CONSUMED_ATTR_NAME);
                    }

                    storeNewValue(resultOp, index, slot);
                    return;
                }
            }

            if (auto ifOp = mlir::dyn_cast_or_null<mlir_ts::IfOp>(producer);
                ifOp && produced == yielded && resultOp->getParentRegion()->isAncestor(ifOp->getParentRegion()))
            {
                auto resultIndex = mlir::cast<mlir::OpResult>(produced).getResultNumber();
                for (auto &region : ifOp->getRegions())
                {
                    auto innerResult = region.empty() ? nullptr : region.back().getTerminator();
                    if (mlir::isa_and_nonnull<mlir_ts::ResultOp>(innerResult) && innerResult->getNumOperands() > resultIndex)
                    {
                        storeNewValue(innerResult, resultIndex, slot);
                    }
                }
            }
        };

        for (auto storeOp : stores)
        {
            // the store MLIRGen made at the region's end, and the read that follows it: put back to what the region
            // yields, then stored where a new value is made
            // (a parameter's default ends with `ts.ParamDefaultValue`, processOptionalParam)
            auto loadOp = mlir::dyn_cast_or_null<mlir_ts::LoadOp>(storeOp->getNextNode());
            auto *resultOp = loadOp ? loadOp->getNextNode() : nullptr;
            if (!loadOp || loadOp.getReference() != storeOp.getReference() ||
                !mlir::isa_and_nonnull<mlir_ts::ResultOp, mlir_ts::ParamDefaultValueOp>(resultOp) ||
                resultOp->getNumOperands() != 1 || resultOp->getOperand(0) != loadOp.getResult())
            {
                continue;
            }

            auto slot = storeOp.getReference();
            resultOp->setOperand(0, storeOp.getValue());
            loadOp->erase();
            storeOp->erase();

            // a slot no scope exit releases - made inside an expression (a destructuring assignment in `?:`), which
            // does not reach the scope's end (MLIRGenImpl::releaseScopeOwnedVars): taken out, the default is what it
            // was without one
            if (llvm::none_of(slot.getUsers(), [](mlir::Operation *user) { return mlir::isa<mlir_ts::ReleaseSlotOp>(user); }))
            {
                for (auto *user : llvm::make_early_inc_range(slot.getUsers()))
                {
                    auto emptyStore = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                    if (!emptyStore)
                    {
                        continue;
                    }

                    auto *emptyValue = emptyStore.getValue().getDefiningOp();
                    emptyStore->erase();
                    if (mlir::isa_and_nonnull<mlir_ts::DefaultOp>(emptyValue) && emptyValue->use_empty())
                    {
                        emptyValue->erase();
                    }
                }

                if (slot.use_empty())
                {
                    slot.getDefiningOp()->erase();
                }

                continue;
            }

            // A parameter that may be kept - stored, returned, given to a call or a new literal (`constructor(public
            // a = new C())`) - holds its default as it holds an argument: own keeps it, the default moving out with
            // it (own_param_default_kept). Only a parameter that is only read borrows the slot.
            if (auto paramOptionalOp = mlir::dyn_cast<mlir_ts::ParamOptionalOp>(resultOp->getParentOp()))
            {
                auto mayKeep = llvm::any_of(paramOptionalOp->getResult(0).getUsers(), [&](mlir::Operation *user) {
                    auto readOp = mlir::dyn_cast<mlir_ts::LoadOp>(user);
                    if (!readOp)
                    {
                        return !mlir::isa<mlir_ts::StoreOp>(user);
                    }

                    std::function<bool(mlir::Value)> handedOn = [&](mlir::Value value) {
                        return llvm::any_of(value.getUsers(), [&](mlir::Operation *reader) {
                            if (mlir::isa<mlir_ts::CastOp>(reader))
                            {
                                return handedOn(reader->getResult(0));
                            }

                            return !mlir::isa<mlir_ts::PropertyRefOp, mlir_ts::ElementRefOp, mlir_ts::LengthOfOp,
                                              mlir_ts::LogicalBinaryOp, mlir_ts::StringLengthOp>(reader);
                        });
                    };
                    return handedOn(readOp.getResult());
                });
                if (mayKeep)
                {
                    continue;
                }
            }

            storeNewValue(resultOp, 0, slot);
        }
    }

    // The same disagreement in a default of a destructuring pattern (`const { p = mk(1) } = { p: c ? mk(i) :
    // undefined }`): `ts.OptValueOrDefault` hands back the value the optional holds - a borrow of the field - or
    // what its default region makes - a +1 nobody took, so every evaluation that ran the default leaked it. Under rc
    // it is made to carry one reference whichever way it goes, as a conditional's result is: the default's +1 is
    // its own (or, when the default reads a temporary it makes and gives back - a record literal of a new object -
    // what it yields is retained in the region first), the optional's value is retained before the op (a retain of
    // an undefined optional does nothing), and the op is an owned result its receiver takes over, or that is given
    // back at the end of its block.
    void consumeValueOrDefaultResults(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        if (!compileOptions.isRefCounted())
        {
            return;
        }

        llvm::SmallVector<mlir::Operation *> toErase;
        module.walk([&](mlir_ts::OptionalValueOrDefaultOp op) {
            auto result = op.getResult();
            if (op->hasAttr(OWNED_RESULT_ATTR_NAME) || !mth.ownsHeapMemory(op.getLoc(), result.getType()) ||
                op.getDefaultValueRegion().empty())
            {
                return;
            }

            auto resultOp = mlir::dyn_cast_or_null<mlir_ts::ResultOp>(op.getDefaultValueRegion().back().getTerminator());
            if (!resultOp || resultOp->getNumOperands() != 1)
            {
                return;
            }

            // only when the default carries a +1 (a borrowed default agrees with the borrowed value already)
            auto produced = resultOp->getOperand(0);
            while (auto castOp = produced.getDefiningOp<mlir_ts::CastOp>())
            {
                if (!mlir::isa<mlir_ts::UnionType, mlir_ts::OptionalType>(castOp.getType()))
                {
                    break;
                }

                produced = castOp.getIn();
            }

            // ...or what it yields reads a temporary it makes (a record literal `{ p: { w = mk(1) } = { w: mk(2) } }`,
            // a field of one): the region gives the temporary back before it ends, so what it yields is retained
            // there first, as a conditional's branch does (consumeConditionalResults)
            auto *region = &op.getDefaultValueRegion();
            auto *producer = produced.getDefiningOp();
            // (an array literal it makes too: `p = [mk(1)][0]`, given back at the end of the region by
            // holdReadOnlyNewArrays once the region's yield is counted)
            auto makesTemporary = [&]() {
                auto found = false;
                region->walk([&](mlir::Operation *inner) {
                    if (inner->getNumResults() == 1 && inner->getParentRegion() == region &&
                        (::typescript::mayTakeOverReference(inner->getResult(0)) || isNewArray(inner)))
                    {
                        found = true;
                    }
                });
                return found;
            };

            if (producer && ::typescript::mayTakeOverReference(produced) && producer->getParentRegion() == region)
            {
                producer->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            }
            else if (makesTemporary())
            {
                mlir::OpBuilder regionBuilder(resultOp);
                regionBuilder.create<mlir_ts::RetainOp>(resultOp->getLoc(), resultOp->getOperand(0));
            }
            else
            {
                return;
            }

            mlir::OpBuilder builder(op);
            builder.create<mlir_ts::RetainOp>(op.getLoc(), op.getArgValue());

            op->setAttr(OWNED_RESULT_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
            auto retains = findReceiverRetains(result);
            if (!retains.empty())
            {
                op->setAttr(OWNED_RESULT_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(&getContext()));
                toErase.append(retains.begin(), retains.end());
            }
        });

        for (auto *op : toErase)
        {
            op->erase();
        }
    }

  private:
    // Gives back the +1 on a produced reference that no receiver ever took.
    //
    // Every function retains its result on the way out (§9.24), so a call hands back a reference
    // whether or not the caller does anything with it. Where a receiver takes it over the pair is
    // balanced (§9.25, §9.27); where nothing does, the reference stands with no owner and the
    // value is never freed. That is not a corner case: `raytrace.ts` is built out of
    // `Vector.plus(Vector.times(k, a), b)`, so nearly every allocation it makes is an
    // intermediate passed straight as an argument, and it reclaimed nothing at all before this.
    //
    // WHERE the release goes is the whole difficulty - "after the last use" is not something
    // MLIRGen can see while it is still building the expression. The answer here is the END OF
    // THE PRODUCER'S OWN BLOCK, which is a temporary's natural lifetime (the enclosing statement,
    // or one iteration of a loop body) and, more importantly, is unconditionally after every use
    // in that block. Placing it after the last *user* instead looks tighter and is wrong: the
    // receiver of `let x = <T>f()` retains the result of the CAST, not of the call, so the call's
    // last user is the cast and a release put there would run before that retain and free the
    // value out from under it. End-of-block cannot get that ordering wrong.
    //
    // Two things disqualify a value, and both simply leave it leaking as before:
    //
    //  - a user outside the producer's block, so the value outlives the block or is used on a
    //    path this cannot see - the body of a statement written in the block is not outside it
    //    (runsWithinBlock);
    //  - a user that is a terminator, since a value handed to a successor as a block argument is
    //    still live after the point this would release it - save a conditional expression's
    //    branch, whose `ts.If` result is judged in the value's place (allUsesReleasableInOwnBlock);
    //  - a user at or after a `ts.StateLabel` that follows the definition, which is a generator's
    //    resume point: the state machine re-enters the block THERE, so the end of the block is
    //    reachable on a path that never ran the op that produced the value. Nothing about that is
    //    visible while the generator is still one block - it only becomes a use before definition
    //    once the state machine is expanded, and it surfaces as a dominance failure in the affine
    //    lowering rather than as anything this pass could notice. So in a generator the release
    //    does not go at the end of the block but in front of the first yield, resume point or
    //    statement holding one (releaseRangeEnd), inside the stretch the definition dominates.
    //
    // That bias is the same one the rest of this arc takes: an unreleased reference is invisible,
    // a released one that was still owned is a use-after-free.
    void releaseDiscardedTemporaries(MLIRTypeHelper &mth, mlir::ModuleOp module)
    {
        // What a generator's body looks like here is not what runs: the state machine cuts its
        // blocks at every resume point afterwards, so a position that is plainly after a
        // definition now may not be on the path that reaches it later - and the failure is a
        // dominance error in the affine lowering, not something visible here. A temporary in a
        // generator is released only inside the stretch of its block that no resume point cuts
        // (releaseRangeEnd), and only when it does not go out with what the generator yields
        // (escapesThroughStore). Every other one leaks, which is the side of the line this arc
        // keeps everything uncertain on.
        llvm::DenseSet<mlir::Operation *> generators;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            auto isGenerator = false;
            funcOp.walk([&](mlir_ts::YieldReturnValOp) { isGenerator = true; });
            if (isGenerator)
            {
                generators.insert(funcOp.getOperation());
            }
        });

        llvm::SmallVector<mlir::Operation *> discarded;
        llvm::SmallVector<mlir::Operation *> unread;
        module.walk([&](mlir::Operation *op) {
            if (!op->hasAttr(OWNED_RESULT_ATTR_NAME) || op->hasAttr(OWNED_RESULT_CONSUMED_ATTR_NAME))
            {
                return;
            }

            // A read through an interface's field is owned only once it is lowered (InterfaceFieldAccess), so one
            // nothing uses - what an assignment `i.f = v` leaves of the place it wrote - goes, rather than calling
            // a getter to give its result back.
            if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(op))
            {
                if (loadOp->use_empty() && loadOp.getReference().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>())
                {
                    unread.push_back(op);
                    return;
                }
            }

            if (op->getNumResults() != 1 || !mth.ownsHeapMemory(op->getLoc(), op->getResult(0).getType()))
            {
                return;
            }

            if (auto funcOp = op->getParentOfType<mlir_ts::FuncOp>())
            {
                if (!compileOptions.isRefCounted() && generators.contains(funcOp.getOperation()) &&
                    escapesThroughStore(op->getResult(0)))
                {
                    return;
                }
            }

            discarded.push_back(op);
        });

        for (auto *op : unread)
        {
            op->erase();
        }

        mlir::OpBuilder builder(&getContext());
        std::set<std::tuple<mlir::Operation *, mlir::Operation *, std::string>> placedAfterFinally;

        // The ways out of a block from inside its ops' regions (see below), each with the op of the block it is
        // in, in the block's order: found once for a block, as they are the block's - not one temporary's. Looked
        // for again after each temporary, a function of many of them (1300 lines of `any` arithmetic,
        // binaryArithmeticControlFlowGraphNotTooLarge) walked the rest of its block once for each: 2.7 s here.
        llvm::DenseMap<mlir::Block *, llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>>> exitsOfBlock;
        auto exitsOf = [&](mlir::Block *block) -> llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>> & {
            auto found = exitsOfBlock.find(block);
            if (found != exitsOfBlock.end())
            {
                return found->second;
            }

            auto &exits = exitsOfBlock[block];
            for (auto &top : *block)
            {
                if (top.getNumRegions() == 0)
                {
                    continue;
                }

                top.walk([&](mlir::Operation *nested) {
                    // A nested function's `return` returns from that function, and its jumps are
                    // its own; nothing inside one is on a path out of this block.
                    if (mlir::isa<mlir_ts::FuncOp>(nested))
                    {
                        return mlir::WalkResult::skip();
                    }

                    // A `throw` leaves as a `return` does - `const s = "x" + i; if (c) throw i;` lost
                    // `s` on every throw - unless a `try` between it and this block takes it: the
                    // block may then go on and give the value back at its own end.
                    auto leavesBlock = mlir::isa<mlir_ts::ReturnValOp, mlir_ts::ReturnOp>(nested) ||
                                       (mlir::isa<mlir_ts::BreakOp, mlir_ts::ContinueOp>(nested) &&
                                        jumpLeavesBlock(nested, block)) ||
                                       (mlir::isa<mlir_ts::ThrowOp>(nested) && !throwLandsWithin(nested, block));
                    if (leavesBlock)
                    {
                        exits.push_back({&top, nested});
                    }

                    return mlir::WalkResult::advance();
                });
            }

            return exits;
        };

        BlockScan scan;
        // a temporary, the op its release goes in front of (null: the block's end), and the op it goes right
        // after instead (rc: its last use), when there is one
        llvm::SmallVector<std::tuple<mlir::Operation *, mlir::Operation *, mlir::Operation *>> endReleases;
        for (auto *op : discarded)
        {
            if (!allUsesReleasableInOwnBlock(op, scan) || storedIntoUnownedLocalOutside(op))
            {
                continue;
            }

            auto *block = op->getBlock();
            auto rangeEnd = scan.releaseRangeEnd(op);

            // Where the stretch would end (see below): in front of the first op that leaves the block, of the
            // end of the range, or of the block's terminator
            mlir::Operation *before = nullptr;
            auto *exiting = scan.firstExitingOpAfter(op);
            if (exiting && (rangeEnd == block->end() || exiting->isBeforeInBlock(&*rangeEnd)))
            {
                before = exiting;
            }
            else if (rangeEnd != block->end())
            {
                before = &*rangeEnd;
            }
            else if (auto *terminator = block->getTerminator())
            {
                before = terminator;
            }

            // -mm=rc: the temporary is given back right after the op of its block that uses it last
            // (lastUseInBlock), not where its stretch ends. Held to the end, each one was live across every call
            // after it in its block, and each call that may throw had to give all of them back on its unwind
            // path (UnwindReleasePass): a function of many - 1300 lines of `any` arithmetic,
            // binaryArithmeticControlFlowGraphNotTooLarge - got a cleanup pad of thousands of releases for each
            // call, and took half an hour to compile. Not under -mm=own, whose passes after this read where the
            // releases are; and not where a use can't be followed (lastUseInBlock gives none) or is not before
            // the stretch's end.
            // Only a temporary with no name: a `const` holding the value itself (OWNED_RESULT_NAMED_ATTR_NAME) is a
            // local, given back where its scope ends - an object let go of earlier is seen let go of
            // (`Shared.count`, a `WeakRef`'s deref, a destructor).
            // Nor a view (own_facts::isView: an interface made of an object, a union of a class): its release is the
            // release of the block it shows, which the value it is made of, and that value's other views, may go on
            // using - `use(<Something>o); use(<Something>o)` gave the first interface back after its call, and the
            // object literal's block, born with no count, was freed before the second.
            mlir::Operation *after = nullptr;
            if (compileOptions.isRefCounted() && !op->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME) && !own_facts::isView(op))
            {
                auto *last = lastUseInBlock(op);
                if (last && (!before || last->isBeforeInBlock(before)))
                {
                    after = last;
                }
            }

            // The end of the block is not the only way out of it. A `return` written inside an
            // `if` - which is where returns are usually written - leaves from a nested region and
            // never reaches the release placed below, so the temporary was simply lost on that
            // path. That is what `raytrace` was leaking: `addLightAt` builds a ray, tests it, and
            // returns early when the light is blocked, so the ray's reference came back only on
            // the path that falls through (section 9.60).
            //
            // Only exits that come after the definition in this block are covered, because only
            // those are dominated by it. A nested exit may read the value - a branch that reads a
            // `const` and returns - but it cannot return it: a `return` of the value takes it over
            // and marks it consumed, which keeps it out of this set. The release goes in front of
            // the exit, after everything the branch read.
            //
            // A `return` leaves for good wherever it is written. A `break` or `continue` only
            // sometimes does, and which it is decides between a leak and a double release:
            //
            //   - caught by a loop BELOW this block - the definition sits outside that loop - so
            //     control comes back and runs the end-of-block release as well. Releasing at the
            //     jump too would give the same reference back twice;
            //   - caught by a loop ABOVE this block, which is to say this block is that loop's
            //     body. The end-of-block release is then skipped for the iteration that jumps,
            //     and without one here the value is lost.
            //
            // jumpLeavesBlock answers it by walking from the jump up to this block and asking
            // whether anything on the way catches it.
            //
            // The exits of the ops between the definition and the end of its range (exitsOf) - with the release
            // right after the last use (`after`), the exits up to that op only: one after it leaves with the
            // value given back already.
            for (auto [top, nested] : exitsOf(block))
            {
                if (!op->isBeforeInBlock(top) || (rangeEnd != block->end() && !top->isBeforeInBlock(&*rangeEnd)) ||
                    (after && after->isBeforeInBlock(top)))
                {
                    continue;
                }

                if (!releaseAfterFinally(builder, op, nested, block, placedAfterFinally))
                {
                    builder.setInsertionPoint(nested);
                    builder.create<mlir_ts::ReleaseOp>(op->getLoc(), op->getResult(0));
                }
            }

            endReleases.push_back({op, before, after});
        }

        // The releases at the end of the temporaries' stretches go in once every place is found: one put
        // into a block left no room in its order between its neighbours, and the next temporary's
        // isBeforeInBlock numbered the whole block again - for each of them, in a block of many. In the
        // order found, each in front of its op (or at the block's end), as they went in one by one.
        for (auto [op, before, after] : endReleases)
        {
            if (after)
            {
                builder.setInsertionPointAfter(after);
            }
            else if (before)
            {
                builder.setInsertionPoint(before);
            }
            else
            {
                builder.setInsertionPointToEnd(op->getBlock());
            }

            builder.create<mlir_ts::ReleaseOp>(op->getLoc(), op->getResult(0));
        }
    }

    // A jump out of a `try` body or catch clause runs the `finally` first, and the finally may read
    // the temporary: `const v = mv(); try { if (c) continue; } finally { s += v.get(); }` gave `v`
    // back at the `continue`, before the finally read it. Or the finally may leave by a jump of its
    // own, which releases the temporary in front of it too: `try { continue; } finally { break; }`
    // gave it back twice. When a jump passes through a finally between it and the temporary's block,
    // the release goes at the end of the outermost such finally instead - the last one the jump
    // runs - inside a `ts.If` tagged with the jump, as MLIRGen does for the scopes the jump leaves
    // (mlirGenScopeExitAfterFinally), so that TryOpLowering keeps it in the copy of the finally that
    // jump runs and in no other. A finally that leaves on its own never reaches it. False when the
    // jump passes through no finally.
    //
    // Every jump with one key runs one copy, so the release goes in once for each (`placed`).
    static bool releaseAfterFinally(mlir::OpBuilder &builder, mlir::Operation *op, mlir::Operation *jump,
                                    mlir::Block *block,
                                    std::set<std::tuple<mlir::Operation *, mlir::Operation *, std::string>> &placed)
    {
        mlir_ts::TryOp lastTry;
        for (auto *child = jump; child->getBlock() != block; child = child->getParentOp())
        {
            auto tryOp = mlir::dyn_cast<mlir_ts::TryOp>(child->getParentOp());
            if (!tryOp || child->getParentRegion() == &tryOp.getFinally() ||
                child->getParentRegion() == &tryOp.getCleanup())
            {
                continue;
            }

            auto &finally = tryOp.getFinally();
            if (finally.empty() || &finally.front().front() == finally.front().getTerminator())
            {
                continue;
            }

            lastTry = tryOp;
        }

        if (!lastTry)
        {
            return false;
        }

        auto value = op->getResult(0);

        auto isReturn = mlir::isa<mlir_ts::ReturnValOp, mlir_ts::ReturnOp>(jump);
        std::string key;
        if (!isReturn)
        {
            auto isBreak = mlir::isa<mlir_ts::BreakOp>(jump);
            auto label = isBreak ? mlir::cast<mlir_ts::BreakOp>(jump).getLabelAttr()
                                 : mlir::cast<mlir_ts::ContinueOp>(jump).getLabelAttr();
            key = std::string(isBreak ? JUMP_ONLY_BREAK_KEY : JUMP_ONLY_CONTINUE_KEY) +
                  (label ? label.getValue().str() : std::string());
        }

        if (!placed.insert({op, lastTry.getOperation(), isReturn ? std::string(RETURN_ONLY_ATTR_NAME) : key}).second)
        {
            return true;
        }

        auto location = op->getLoc();
        auto &finallyEnd = lastTry.getFinally().back();
        builder.setInsertionPoint(finallyEnd.getTerminator());
        auto alwaysTrue = builder.create<mlir_ts::ConstantOp>(location, mlir_ts::BooleanType::get(builder.getContext()),
                                                              builder.getBoolAttr(true));
        auto ifOp = builder.create<mlir_ts::IfOp>(location, alwaysTrue, /*withElseRegion=*/false);
        if (isReturn)
        {
            ifOp->setAttr(RETURN_ONLY_ATTR_NAME, builder.getUnitAttr());
        }
        else
        {
            ifOp->setAttr(JUMP_ONLY_ATTR_NAME, builder.getStringAttr(key));
        }

        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        builder.create<mlir_ts::ReleaseOp>(location, value);
        return true;
    }

    // Does `op` catch this jump - is it the loop, switch or labelled statement the jump names?
    //
    // An unlabelled `break` is caught by the nearest enclosing loop or `switch`; an unlabelled
    // `continue` by the nearest enclosing loop. A labelled one is caught by whatever carries that
    // label, which may be several levels further out - `outer: while (..) { while (..) { break
    // outer; } }` is the case that makes position alone the wrong question.
    static bool catchesJump(mlir::Operation *op, bool isBreak, mlir::StringAttr label)
    {
        auto isLoop = mlir::isa<mlir_ts::WhileOp, mlir_ts::DoWhileOp, mlir_ts::ForOp>(op);
        auto isSwitch = mlir::isa<mlir_ts::SwitchOp>(op);

        if (label && !label.getValue().empty())
        {
            auto ownLabel = op->getAttrOfType<mlir::StringAttr>(LABEL_ATTR_NAME);
            return ownLabel && ownLabel.getValue() == label.getValue();
        }

        return isLoop || (isBreak && isSwitch);
    }

    // Does this `break` or `continue` leave `block` for good, or does control come back into it?
    //
    // Walks outwards from the jump to the op that sits in `block`, asking each level whether it
    // catches the jump. Nothing on the way - including that outermost op itself - means the jump
    // is caught further out, so `block` is inside the loop being left and everything after the
    // jump in it is skipped.
    //
    // Answers false when the walk runs out of ancestors without meeting `block`, which cannot
    // happen for a jump this pass reached but is the safe answer either way: a missing release
    // leaks, a surplus one frees memory twice.
    // Is there a `try` between a `throw` and `block` that takes the exception - one whose body the
    // throw is in, or that has a finally to run? Then the block may go on after it. A throw out of
    // a catch clause with no finally passes its `try` by.
    static bool throwLandsWithin(mlir::Operation *throwOp, mlir::Block *block)
    {
        auto finallyIsEmpty = [](mlir_ts::TryOp tryOp) {
            return llvm::all_of(tryOp.getFinally(), [](mlir::Block &finallyBlock) {
                return finallyBlock.empty() || &finallyBlock.front() == finallyBlock.getTerminator();
            });
        };

        for (auto *child = throwOp, *parent = throwOp->getParentOp(); parent && parent != block->getParentOp();
             child = parent, parent = parent->getParentOp())
        {
            if (auto tryOp = mlir::dyn_cast<mlir_ts::TryOp>(parent))
            {
                if (child->getParentRegion() != &tryOp.getCatches() || !finallyIsEmpty(tryOp))
                {
                    return true;
                }
            }
        }

        return false;
    }

    static bool jumpLeavesBlock(mlir::Operation *jump, mlir::Block *block)
    {
        auto isBreak = mlir::isa<mlir_ts::BreakOp>(jump);
        auto label = isBreak ? mlir::cast<mlir_ts::BreakOp>(jump).getLabelAttr()
                             : mlir::cast<mlir_ts::ContinueOp>(jump).getLabelAttr();

        for (auto *parent = jump->getParentOp(); parent != nullptr; parent = parent->getParentOp())
        {
            if (catchesJump(parent, isBreak, label))
            {
                return false;
            }

            if (parent->getBlock() == block)
            {
                return true;
            }
        }

        return false;
    }

    // Does `user` run before control reaches the op that follows its ancestor in `block`? True
    // for `block`'s own ops, and for ops nested in structured statements only. Under rc and own also in the default
    // of a value-or-default (`const { p = d ?? mk(1) } = ...`), which runs there at most once, as a branch does
    // (gc and none keep their IR as it was).
    bool runsWithinBlock(mlir::Operation *user, mlir::Block *block)
    {
        for (auto *op = user; op->getBlock() != block; op = op->getParentOp())
        {
            auto *parent = op->getParentOp();
            if (!parent || !(mlir::isa<mlir_ts::IfOp, mlir_ts::WhileOp, mlir_ts::DoWhileOp, mlir_ts::ForOp,
                                       mlir_ts::LabelOp, mlir_ts::SwitchOp, mlir_ts::TryOp>(parent) ||
                             (compileOptions.tracksOwnership() && mlir::isa<mlir_ts::OptionalValueOrDefaultOp>(parent))))
            {
                return false;
            }
        }

        return true;
    }

    // Is the value stored into a local that owns nothing, declared outside the value's block? Such
    // a local - one declared with no initializer (`let n: N;`), or in a catch or finally clause
    // (MLIRGenImpl::localTakesOwnership) - takes no count for what it is given, so the value's own
    // reference is all that keeps it, and the local holds it until its own scope ends: later than
    // the end of the value's block. `let n: N; if (c) { n = mk(); } use(n)` gave the new value back
    // at the end of the `if`, and `finally { let n = mk(); for (..) { use(n); n = mk(); } }` at the
    // end of the loop body; the local then read a freed block. Such a value is left with no one to
    // give it back - a leak - which is the side the whole arc keeps the uncertain case on.
    bool storedIntoUnownedLocalOutside(mlir::Operation *op)
    {
        llvm::SmallVector<mlir::Value> values{op->getResult(0)};
        while (!values.empty())
        {
            auto value = values.pop_back_val();
            for (auto *user : value.getUsers())
            {
                if (auto castOp = mlir::dyn_cast<mlir_ts::CastOp>(user))
                {
                    values.push_back(castOp.getRes());
                    continue;
                }

                // `n = b ? h : h2` stores what the branch hands on
                if (auto handedOn = handedOnByConditional(user, value); !handedOn.empty())
                {
                    values.append(handedOn.begin(), handedOn.end());
                    continue;
                }

                auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user);
                auto varOp = storeOp && storeOp.getValue() == value
                                 ? storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>()
                                 : mlir_ts::VariableOp();
                if (varOp && !varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) && varOp->getBlock() != op->getBlock())
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Can a release at the end of this value's own block give its reference back safely? See
    // releaseDiscardedTemporaries for what disqualifies a use and why.
    //
    // A user inside the body of an `if`, a loop, a `switch`, a label or a `try` written in this
    // block is not outside it: those regions run to completion before the op after them does, so
    // the end of the block still follows every use. A folded `const` read inside a branch -
    // `const a = new C(); if (x) { let b = a; }` - is exactly that, and leaked before this. Any
    // other region is: a nested function or an `async.execute` body runs later, from somewhere
    // else.
    //
    // A conditional expression's branch is the one terminator that is followed: `b ? h : h2`
    // yields `h` through `ts.Result`, and from there the value is the `ts.If`'s result, so it is
    // that result's users that decide (handedOnByConditional). A receiver of a result nobody owns
    // retains, and a branch that makes it owned retained `h` for it (consumeConditionalResults),
    // so `h`'s own reference is still given back at the end of its block either way.
    // The op of `op`'s block that holds the last use of its value (rc's release goes right after it), or none
    // when one can't be followed - the release then stays where the stretch ends. The value goes by other names
    // than its own, and their uses are its uses:
    // - what an op makes of it, unless that comes with a reference of its own (OWNED_RESULT_ATTR_NAME): a view
    //   (a cast, a union made of it), a place in it (`ts.PropertyRef`, then a load), a number read out of it;
    // - what a conditional hands on (handedOnByConditional);
    // - what is loaded from a local of the block it is stored into: every use of that local is a load or a
    //   store, or the local is one it can't follow.
    // A store anywhere else, or a terminator handing it on otherwise (a loop's), is a use it can't follow.
    mlir::Operation *lastUseInBlock(mlir::Operation *op)
    {
        auto *block = op->getBlock();
        mlir::Operation *last = op;
        auto later = [&](mlir::Operation *user) {
            auto *inBlock = block->findAncestorOpInBlock(*user);
            if (!inBlock)
            {
                return false;
            }

            if (last->isBeforeInBlock(inBlock))
            {
                last = inBlock;
            }

            return true;
        };

        llvm::SmallVector<mlir::Value> values{op->getResult(0)};
        llvm::DenseSet<mlir::Value> seen;
        while (!values.empty())
        {
            auto value = values.pop_back_val();
            if (!seen.insert(value).second)
            {
                continue;
            }

            for (auto *user : value.getUsers())
            {
                if (!later(user))
                {
                    return nullptr;
                }

                if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
                {
                    if (storeOp.getValue() != value)
                    {
                        continue;
                    }

                    auto varOp = storeOp.getReference().getDefiningOp<mlir_ts::VariableOp>();
                    if (!varOp)
                    {
                        return nullptr;
                    }

                    for (auto *varUser : varOp->getUsers())
                    {
                        if (!later(varUser))
                        {
                            return nullptr;
                        }

                        if (auto loadOp = mlir::dyn_cast<mlir_ts::LoadOp>(varUser))
                        {
                            values.push_back(loadOp.getResult());
                        }
                        else if (auto varStoreOp = mlir::dyn_cast<mlir_ts::StoreOp>(varUser);
                                 !varStoreOp || varStoreOp.getReference() != varOp.getResult())
                        {
                            return nullptr;
                        }
                    }

                    continue;
                }

                if (auto handedOn = handedOnByConditional(user, value); !handedOn.empty())
                {
                    values.append(handedOn.begin(), handedOn.end());
                    continue;
                }

                // handed on by a terminator other than a conditional's
                if (user->hasTrait<mlir::OpTrait::IsTerminator>())
                {
                    return nullptr;
                }

                // (a view is the value still, whatever it is marked: an interface made of an object literal's copy,
                // `ts.NewInterface`, is that copy - given back right after it, `use(<Something>o)` read freed memory)
                if (user->hasAttr(OWNED_RESULT_ATTR_NAME) && !own_facts::isView(user))
                {
                    continue;
                }

                for (auto result : user->getResults())
                {
                    values.push_back(result);
                }
            }
        }

        return last;
    }

    struct BlockScan;

    bool allUsesReleasableInOwnBlock(mlir::Operation *op, BlockScan &scan)
    {
        auto *block = op->getBlock();
        auto rangeEnd = scan.releaseRangeEnd(op);

        llvm::SmallVector<mlir::Value> values{op->getResult(0)};
        while (!values.empty())
        {
            auto value = values.pop_back_val();
            for (auto *user : value.getUsers())
            {
                auto handedOn = handedOnByConditional(user, value);
                if ((user->hasTrait<mlir::OpTrait::IsTerminator>() && handedOn.empty()) || !runsWithinBlock(user, block))
                {
                    return false;
                }

                // A use at or after the end of the range runs on a path the release has already
                // left: after the yield, or inside the statement that holds one (the resume point
                // there re-enters past the definition). Such a value leaks. MLIRGen keeps a
                // generator's locals in its state object, so a temporary used across a yield does
                // not occur.
                auto *inBlock = block->findAncestorOpInBlock(*user);
                if (rangeEnd != block->end() && !inBlock->isBeforeInBlock(&*rangeEnd))
                {
                    return false;
                }

                values.append(handedOn.begin(), handedOn.end());
            }
        }

        return true;
    }

    // What a branch of a conditional expression hands `value` on as: the result of the `ts.If` whose branch this
    // `ts.Result` ends, at each position the branch yields `value` in - a conditional may yield more than one value
    // (holdPastConditional adds the literal its branch holds). None for any other user.
    // (under rc and own also the default of a value-or-default - `const { p = e } = r` hands `e` on as the op's result;
    // gc and none keep their IR)
    llvm::SmallVector<mlir::Value, 1> handedOnByConditional(mlir::Operation *user, mlir::Value value)
    {
        llvm::SmallVector<mlir::Value, 1> handedOn;
        auto *conditional = mlir::isa<mlir_ts::ResultOp>(user) ? user->getParentOp() : nullptr;
        if (!conditional ||
            !(mlir::isa<mlir_ts::IfOp>(conditional) ||
              (compileOptions.tracksOwnership() && mlir::isa<mlir_ts::OptionalValueOrDefaultOp>(conditional))) ||
            conditional->getNumResults() != user->getNumOperands())
        {
            return handedOn;
        }

        for (auto &operand : user->getOpOperands())
        {
            if (operand.get() == value)
            {
                handedOn.push_back(conditional->getResult(operand.getOperandNumber()));
            }
        }

        return handedOn;
    }

    // Where the stretch of `op`'s block that a release of it may cover ends: at the first op after
    // it that is a yield or a resume point or has one inside it, or at the end of the block. The
    // yield ends the call (the affine lowering turns it into a store and a jump to the exit
    // block), and the next call re-enters at the resume point after it, past the definition - so
    // a release has to come in front of it, and nothing after is on the path that ran the
    // definition. A statement with a yield inside - `if (c) { yield 1; }` - is the same: resuming
    // in the middle of it falls through to the rest of the block. A plain function has neither and
    // gets the whole block.
    //
    // The ops of a block that end such a stretch, and the ones that end its execution (firstExitingOpAfter), are
    // the block's: BlockScan finds them once for a block, in its order, and answers each temporary from them.
    // Looked for again from each temporary on, a function of many of them (1300 lines of `any` arithmetic,
    // binaryArithmeticControlFlowGraphNotTooLarge) walked the rest of its block, regions and all, for each.
    // Releases put in afterwards neither end a stretch nor leave the block.
    struct BlockScan
    {
        llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Operation *>> resumesOf;
        llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Operation *>> exitingOf;

        llvm::SmallVector<mlir::Operation *> &resumes(mlir::Block *block)
        {
            auto found = resumesOf.find(block);
            if (found != resumesOf.end())
            {
                return found->second;
            }

            auto &ops = resumesOf[block];
            for (auto &it : *block)
            {
                if (mlir::isa<mlir_ts::YieldReturnValOp, mlir_ts::StateLabelOp>(it))
                {
                    ops.push_back(&it);
                    continue;
                }

                if (it.getNumRegions() == 0)
                {
                    continue;
                }

                auto resumes = it.walk([&](mlir::Operation *nested) {
                    // a nested function's yields are its own
                    if (mlir::isa<mlir_ts::FuncOp>(nested))
                    {
                        return mlir::WalkResult::skip();
                    }

                    return mlir::isa<mlir_ts::YieldReturnValOp, mlir_ts::StateLabelOp>(nested)
                               ? mlir::WalkResult::interrupt()
                               : mlir::WalkResult::advance();
                });

                if (resumes.wasInterrupted())
                {
                    ops.push_back(&it);
                }
            }

            return ops;
        }

        mlir::Block::iterator releaseRangeEnd(mlir::Operation *op)
        {
            auto *block = op->getBlock();
            for (auto *resume : resumes(block))
            {
                if (op->isBeforeInBlock(resume))
                {
                    return mlir::Block::iterator(resume);
                }
            }

            return block->end();
        }

        mlir::Operation *firstExitingOpAfter(mlir::Operation *op)
        {
            auto *block = op->getBlock();
            auto found = exitingOf.find(block);
            if (found == exitingOf.end())
            {
                auto &ops = exitingOf[block];
                for (auto &it : *block)
                {
                    if (exitsBlock(&it))
                    {
                        ops.push_back(&it);
                    }
                }

                found = exitingOf.find(block);
            }

            for (auto *exiting : found->second)
            {
                if (op->isBeforeInBlock(exiting))
                {
                    return exiting;
                }
            }

            return nullptr;
        }
    };

    // Does this generator temporary, or a view of it, get stored somewhere? Asked under every model but
    // rc (only own has releases left by the time it matters). Under own the `{ value, done }` record a `yield` hands back is filled by stores that move the
    // value in, so `yield "a" + i` keeps the concatenation alive only through the reference it
    // was born with, and a release before the yield would free what the caller reads. Any store
    // is treated that way, rather than telling the record apart from a place that does count.
    // Under rc the record takes a reference of its own (retainYieldedRecords).
    bool escapesThroughStore(mlir::Value value)
    {
        for (auto *user : value.getUsers())
        {
            if (auto storeOp = mlir::dyn_cast<mlir_ts::StoreOp>(user))
            {
                if (storeOp.getValue() == value)
                {
                    return true;
                }

                continue;
            }

            if (mlir::isa<mlir_ts::CastOp, mlir_ts::OptionalValueOp, mlir_ts::CreateUnionInstanceOp>(user) &&
                user->getNumResults() == 1 && escapesThroughStore(user->getResult(0)))
            {
                return true;
            }

            // `yield b ? h : h2` stores what the branch hands on
            if (llvm::any_of(handedOnByConditional(user, value), [&](mlir::Value handedOn) { return escapesThroughStore(handedOn); }))
            {
                return true;
            }
        }

        return false;
    }

    // The first op after `op` that ends this block's execution, if there is one.
    //
    // "End of the block" is not the end of what runs. `ts.ReturnVal` is not a terminator - the
    // scope-exit releases and `ts.Exit` follow it in the same block - but the affine lowering
    // turns it into a branch to the exit block, and everything the pass appended after it is
    // dropped as unreachable. A release placed there does not merely run late; it never runs.
    //
    // That is not a corner: any function whose block ends with a `return` has this shape, which
    // is most of them. Before this, 36 of `raytrace.ts`'s 47 releases were discarded on the way
    // to affine, so §9.30 was largely inert for exactly the code it was written for. The
    // scope-exit releases MLIRGen emits never had the problem, because it emits them before
    // building the return.
    //
    // Releasing before the return is right for a discarded temporary by definition: what is
    // being returned was consumed by the return's own retain (§9.24) and so is not in this set.
    // A generator's `yield` ends the call the same way (releaseRangeEnd).
    // (BlockScan::firstExitingOpAfter) - and a `throw`: what follows it in the block never runs, so a temporary of
    // the thrown expression - `throw "x" + i`'s number printed into a string - was never given back
    static bool exitsBlock(mlir::Operation *op)
    {
        return mlir::isa<mlir_ts::ReturnValOp, mlir_ts::ReturnOp, mlir_ts::ExitOp, mlir_ts::YieldReturnValOp,
                         mlir_ts::ThrowOp>(op);
    }

    static bool producesOwnedResult(mlir::Value value)
    {
        auto *definingOp = value.getDefiningOp();
        return definingOp && definingOp->hasAttr(OWNED_RESULT_ATTR_NAME);
    }

    // The symbol a call names, when it names one. An indirect call through a value - a callback,
    // a method off an interface, a function-typed field - answers empty and is left alone: there
    // is no one callee to inspect, so the caller keeps its retain and leaks rather than guessing.
    //
    // A method call is not that. `obj.m(x)` builds a bound function and then splits it apart
    // again - `GetMethod` for the code, `GetThis` for the receiver - so the callee is a step
    // further back than a plain function's. The shapes below are the ones the dialect's own
    // canonicalizer (SimplifyIndirectCallWithKnownCallee) already rewrites into direct calls,
    // which is the argument that they name one callee: it is the same judgement, made here
    // before canonicalization has run.
    // Does the callee say `@record` (OWN_DECLARED_RECORD_ATTR_NAME)? By name, through the global a
    // dynamically imported function is read from, or as a method - virtual or not: an override says
    // the same markers as what it overrides (OwnershipSignaturePass::checkDeclared).
    static bool calleeDeclaresRecord(mlir::ModuleOp module, mlir_ts::CallIndirectOp callOp)
    {
        if (callOp.getNumOperands() == 0)
        {
            return false;
        }

        auto callee = callOp.getOperand(0);
        mlir::StringRef name;
        if (auto symbolRefOp = callee.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            name = symbolRefOp.getIdentifier();
        }
        else if (auto loadOp = callee.getDefiningOp<mlir_ts::LoadOp>())
        {
            if (auto addressOfOp = loadOp.getReference().getDefiningOp<mlir_ts::AddressOfOp>())
            {
                name = addressOfOp.getGlobalName();
            }
        }
        else if (auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>())
        {
            auto *boundOp = getMethodOp.getBoundFunc().getDefiningOp();
            if (auto thisSymbolRefOp = mlir::dyn_cast_or_null<mlir_ts::ThisSymbolRefOp>(boundOp))
            {
                name = thisSymbolRefOp.getIdentifier();
            }
            else if (auto thisVirtualSymbolRefOp = mlir::dyn_cast_or_null<mlir_ts::ThisVirtualSymbolRefOp>(boundOp))
            {
                name = thisVirtualSymbolRefOp.getIdentifier();
            }
        }

        auto *symbol = name.empty() ? nullptr : mlir::SymbolTable::lookupSymbolIn(module, name);
        return symbol && symbol->hasAttr(OWN_DECLARED_RECORD_ATTR_NAME);
    }

    static mlir::StringRef calleeNameOf(mlir_ts::CallIndirectOp callOp, const own_facts::FieldFunctions *fieldFunctions)
    {
        if (callOp.getNumOperands() == 0)
        {
            return {};
        }

        auto callee = callOp.getOperand(0);

        if (auto symbolRefOp = callee.getDefiningOp<mlir_ts::SymbolRefOp>())
        {
            return symbolRefOp.getIdentifier();
        }

        // a non-virtual method, called without the bound-function detour
        if (auto thisSymbolRefOp = callee.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            return thisSymbolRefOp.getIdentifier();
        }

        // a function an object literal holds (`o.make(i)` of `const o = { make: (x) => new P(x) }`): it names it. Named
        // by nothing, its result was taken for one the call does not hand over, and never given back - a leak under
        // rc and -mm=own whenever the program links anything whose returns are not classified
        if (auto symbol = own_facts::constantFieldFunction(callee))
        {
            return symbol.getValue();
        }

        auto getMethodOp = callee.getDefiningOp<mlir_ts::GetMethodOp>();
        if (!getMethodOp)
        {
            return {};
        }

        auto boundFunc = getMethodOp.getBoundFunc();

        if (auto thisSymbolRefOp = boundFunc.getDefiningOp<mlir_ts::ThisSymbolRefOp>())
        {
            return thisSymbolRefOp.getIdentifier();
        }

        // A bound function built here from a known function - a trampoline - names it outright.
        if (auto createBoundFunctionOp = boundFunc.getDefiningOp<mlir_ts::CreateBoundFunctionOp>())
        {
            if (auto symbolRefOp = createBoundFunctionOp.getFunc().getDefiningOp<mlir_ts::SymbolRefOp>())
            {
                return symbolRefOp.getIdentifier();
            }

            return {};
        }

        // A generator's `next`, read out of its state object: a field every state object of that
        // type holds the same function in, and nothing writes (own_facts::FieldFunctions). Asked
        // under rc only, where a yield hands its record over retained (retainYieldedRecords).
        if (fieldFunctions)
        {
            if (auto symbol = fieldFunctions->of(boundFunc.getDefiningOp()))
            {
                return symbol.getValue();
            }
        }

        // `ts.ThisVirtualSymbolRef` is deliberately absent here. It carries an identifier, but
        // that names the declaration the call was written against, not what the runtime class put
        // in the slot - so reading it as *the* callee would consume a reference an override may
        // never have taken. `private` looks like it would settle this and does not: this compiler
        // accepts a subclass redeclaring a private method and dispatches to the override, where
        // TypeScript rejects the program outright. See §9.32.
        //
        // Asking about every method that could be in the slot answers it instead -
        // virtualCallReturnsOwned.
        return {};
    }

    // A virtual call has no single callee, but it does have a set of possible ones, and the
    // question this pass asks is answerable for a set: consume only if *every* candidate returns
    // owned. One that does not leaves the call retaining, exactly as an unclassified callee does.
    //
    // The candidate set used here is every method in the module whose name after the last dot
    // matches the call's - a superset of the overrides, since an override is `<Subclass>.<same
    // member>` by construction, and a superset is the safe direction. It costs precision only
    // where two unrelated classes share a method name and disagree about ownership, and the cost
    // there is a leak rather than a free.
    //
    // Without this, `raytrace.ts` reclaimed about a quarter of what it allocated: nearly every
    // call in it is a method call, so nearly every allocation it made kept the reference its
    // callee's return had added. See section 9.53.
    static bool virtualCallReturnsOwned(mlir_ts::CallIndirectOp callOp,
                                        const llvm::DenseSet<mlir::StringRef> &returnsOwned,
                                        const llvm::StringMap<llvm::SmallVector<std::pair<mlir::StringRef, mlir::Type>>> &methodsByMemberName)
    {
        if (callOp.getNumOperands() == 0)
        {
            return false;
        }

        auto getMethodOp = callOp.getOperand(0).getDefiningOp<mlir_ts::GetMethodOp>();
        if (!getMethodOp)
        {
            return false;
        }

        auto virtualRefOp = getMethodOp.getBoundFunc().getDefiningOp<mlir_ts::ThisVirtualSymbolRefOp>();
        if (!virtualRefOp)
        {
            return false;
        }

        return everyOverrideReturnsOwned(virtualRefOp.getIdentifier(), returnsOwned, methodsByMemberName,
                                         virtualRefOp.getThisVal().getType());
    }

    // Is `classType` the class `receiverType` names, or on its chain of bases or derived classes? A
    // method of another class shares the member name only: no object of the receiver's type has
    // it in that slot. A base is the first field of a class's storage; a class whose storage this
    // cannot see may be on the chain.
    static bool onReceiverChain(mlir::Type classType, mlir::Type receiverType)
    {
        auto storageOf = [](mlir::Type type) {
            if (auto refClass = mlir::dyn_cast<mlir_ts::ClassType>(type))
            {
                return mlir::dyn_cast<mlir_ts::ClassStorageType>(refClass.getStorageType());
            }

            return mlir::dyn_cast<mlir_ts::ClassStorageType>(type);
        };

        auto candidate = storageOf(classType);
        auto receiver = storageOf(receiverType);
        if (!candidate || !receiver)
        {
            return true;
        }

        auto extends = [](mlir_ts::ClassStorageType derived, mlir_ts::ClassStorageType base) {
            for (auto depth = 0; derived && depth < 64; ++depth)
            {
                if (derived.getName() == base.getName())
                {
                    return true;
                }

                auto fields = derived.getFields();
                derived = fields.empty() ? mlir_ts::ClassStorageType() : mlir::dyn_cast<mlir_ts::ClassStorageType>(fields.front().type);
            }

            return false;
        };

        return extends(candidate, receiver) || extends(receiver, candidate);
    }

    // Does every method that can be in the slot `identifier` was written against return owned?
    // With the receiver's type, only methods of classes on its chain can be there.
    static bool everyOverrideReturnsOwned(mlir::StringRef identifier,
                                          const llvm::DenseSet<mlir::StringRef> &returnsOwned,
                                          const llvm::StringMap<llvm::SmallVector<std::pair<mlir::StringRef, mlir::Type>>> &methodsByMemberName,
                                          mlir::Type receiverType = mlir::Type())
    {
        auto dot = identifier.rfind('.');
        if (dot == mlir::StringRef::npos || dot + 1 >= identifier.size())
        {
            return false;
        }

        auto candidates = methodsByMemberName.find(identifier.substr(dot + 1));
        if (candidates == methodsByMemberName.end() || candidates->second.empty())
        {
            return false;
        }

        // none on the chain: the method is not one this module has, unclassified
        auto any = false;
        for (auto [candidate, classType] : candidates->second)
        {
            if (receiverType && !onReceiverChain(classType, receiverType))
            {
                continue;
            }

            if (!returnsOwned.contains(candidate))
            {
                return false;
            }

            any = true;
        }

        return any;
    }

    // The same question for a call through an interface, where the member name is not enough to
    // name the candidates: an interface method can be implemented by an object literal, whose
    // function is named for where it was written (`Surfaces..feL166C18FH19436811`) rather than for
    // the member it fills. Matching on the member name would miss it entirely, and missing a
    // candidate is the direction that frees memory nobody retained.
    //
    // The vtables say it exactly. An interface value is built by `ts.NewInterface` over one of
    // them, and every vtable in the module is a global: a class implementing `Thing` gets
    // `Sphere.Thing..vtbl`, an object literal gets `Thing.<hash>..vtbl`, and the interface's own
    // name is a component of both. So the candidates for member `index` of interface `I` are its
    // slot in every vtable global naming `I`, and a slot that is absent - not initialised from a
    // symbol - makes the whole call unclassifiable rather than being skipped.
    static bool interfaceCallReturnsOwned(mlir_ts::CallIndirectOp callOp,
                                          const llvm::DenseSet<mlir::StringRef> &returnsOwned,
                                          const llvm::StringMap<llvm::DenseMap<int64_t, mlir::StringRef>> &vtableSlots)
    {
        if (callOp.getNumOperands() == 0)
        {
            return false;
        }

        auto getMethodOp = callOp.getOperand(0).getDefiningOp<mlir_ts::GetMethodOp>();
        if (!getMethodOp)
        {
            return false;
        }

        auto interfaceRefOp = getMethodOp.getBoundFunc().getDefiningOp<mlir_ts::InterfaceSymbolRefOp>();
        if (!interfaceRefOp)
        {
            return false;
        }

        return interfaceMemberReturnsOwned(interfaceRefOp, returnsOwned, vtableSlots);
    }

    // Does every implementation in the vtables of the interface `interfaceRefOp` reads return
    // owned?
    static bool interfaceMemberReturnsOwned(mlir_ts::InterfaceSymbolRefOp interfaceRefOp,
                                            const llvm::DenseSet<mlir::StringRef> &returnsOwned,
                                            const llvm::StringMap<llvm::DenseMap<int64_t, mlir::StringRef>> &vtableSlots)
    {
        auto interfaceType = dyn_cast<mlir_ts::InterfaceType>(interfaceRefOp.getInterfaceVal().getType());
        if (!interfaceType)
        {
            return false;
        }

        auto interfaceName = interfaceType.getName().getValue();
        // The op's index is the member's, and the vtable global has the `.instanceOf` header in
        // front of the members: read at the index itself, the call looked at its neighbour.
        auto index = (int64_t)interfaceRefOp.getIndex() + INTERFACE_VTABLE_HEADER_SLOTS;

        auto sawCandidate = false;
        for (auto &vtable : vtableSlots)
        {
            if (!namesInterface(vtable.getKey(), interfaceName))
            {
                continue;
            }

            auto slot = vtable.getValue().find(index);
            if (slot == vtable.getValue().end() || !returnsOwned.contains(slot->second))
            {
                return false;
            }

            sawCandidate = true;
        }

        return sawCandidate;
    }

    // `h.cc`, `h[i]` through a `get` accessor: the ops that call a getter and have its result.
    static bool isGetterRead(mlir::Operation *op)
    {
        return op->getNumResults() == 1 &&
               mlir::isa<mlir_ts::AccessorOp, mlir_ts::ThisAccessorOp, mlir_ts::ThisIndirectAccessorOp,
                         mlir_ts::ThisIndexAccessorOp, mlir_ts::ThisIndirectIndexAccessorOp,
                         mlir_ts::BoundIndirectAccessorOp, mlir_ts::BoundIndirectIndexAccessorOp>(op);
    }

    // Is anything going to read this value? A use that is itself dead - a `ts.Cast` of it that
    // nothing reads - is not one: the canonicalizer deletes the pair.
    static bool hasLiveUse(mlir::Value value)
    {
        for (auto *user : value.getUsers())
        {
            if (!mlir::isOpTriviallyDead(user))
            {
                return true;
            }
        }

        return false;
    }

    // The getter a getter read calls, by the same rules as calleeNameOf and the two
    // candidate-set questions: named outright, a virtual slot (every method of that member
    // name), or an interface slot (what the vtables put there). A getter given as anything
    // else - `ts.Undef` when there is only a setter - is unclassified.
    static bool getterReturnsOwned(mlir::Operation *op, const llvm::DenseSet<mlir::StringRef> &returnsOwned,
                                   const llvm::StringMap<llvm::SmallVector<std::pair<mlir::StringRef, mlir::Type>>> &methodsByMemberName,
                                   const llvm::StringMap<llvm::DenseMap<int64_t, mlir::StringRef>> &vtableSlots)
    {
        auto named = [&](mlir::FlatSymbolRefAttr getter) {
            return getter && returnsOwned.contains(getter.getValue());
        };

        auto throughValue = [&](mlir::Value getter) {
            auto *definingOp = getter.getDefiningOp();
            if (!definingOp)
            {
                return false;
            }

            if (auto symbolRefOp = mlir::dyn_cast<mlir_ts::SymbolRefOp>(definingOp))
            {
                return returnsOwned.contains(symbolRefOp.getIdentifier());
            }

            if (auto thisSymbolRefOp = mlir::dyn_cast<mlir_ts::ThisSymbolRefOp>(definingOp))
            {
                return returnsOwned.contains(thisSymbolRefOp.getIdentifier());
            }

            if (auto virtualSymbolRefOp = mlir::dyn_cast<mlir_ts::VirtualSymbolRefOp>(definingOp))
            {
                return everyOverrideReturnsOwned(virtualSymbolRefOp.getIdentifier(), returnsOwned,
                                                 methodsByMemberName);
            }

            if (auto thisVirtualSymbolRefOp = mlir::dyn_cast<mlir_ts::ThisVirtualSymbolRefOp>(definingOp))
            {
                return everyOverrideReturnsOwned(thisVirtualSymbolRefOp.getIdentifier(), returnsOwned,
                                                 methodsByMemberName, thisVirtualSymbolRefOp.getThisVal().getType());
            }

            if (auto interfaceRefOp = mlir::dyn_cast<mlir_ts::InterfaceSymbolRefOp>(definingOp))
            {
                return interfaceMemberReturnsOwned(interfaceRefOp, returnsOwned, vtableSlots);
            }

            return false;
        };

        return llvm::TypeSwitch<mlir::Operation *, bool>(op)
            .Case<mlir_ts::AccessorOp, mlir_ts::ThisAccessorOp, mlir_ts::ThisIndexAccessorOp>(
                [&](auto accessorOp) { return named(accessorOp.getGetAccessorAttr()); })
            .Case<mlir_ts::ThisIndirectAccessorOp, mlir_ts::ThisIndirectIndexAccessorOp,
                  mlir_ts::BoundIndirectAccessorOp, mlir_ts::BoundIndirectIndexAccessorOp>(
                [&](auto accessorOp) { return throughValue(accessorOp.getGetAccessor()); })
            .Default([](mlir::Operation *) { return false; });
    }

    // Is `vtableName` a vtable for `interfaceName`? Both shapes spell the interface as a whole
    // dot-separated component: `Sphere.Thing..vtbl` for a class that implements it,
    // `Thing.19585545..vtbl` for an object literal that satisfies it. A class's own vtable,
    // `Sphere..vtbl`, names no interface and is left out.
    static bool namesInterface(mlir::StringRef vtableName, mlir::StringRef interfaceName)
    {
        auto rest = vtableName;
        while (!rest.empty())
        {
            auto split = rest.split('.');
            if (split.first == interfaceName)
            {
                return true;
            }

            rest = split.second;
        }

        return false;
    }

    // Does every return of a heap-owning value in this function retain it first?
    //
    // Looked up rather than assumed, and answered "no" for anything unclear: a function with no
    // A function this module only holds a declaration of: the definition is in another module,
    // so there are no returns here to read and the body test above can never say yes. Refusing
    // them all is what item 5al was - and it is not a small refusal. One imported method with a
    // heap-returning signature also poisons the candidate set for its whole member name, so
    // local classes' calls stop being consumed too. Measured on a two-module program whose work
    // is an imported method returning a string, `rc` held 18.0 MB against `none`'s 18.0 and
    // `gc`'s 5.7: reference counting reclaimed nothing whatsoever across a module boundary.
    //
    // Two things have to be true to say yes here, and both now are.
    //
    // First, it must be a tslang function rather than a foreign one. A `declare`d C function
    // returning a `string` has no convention at all and must never be consumed; an imported
    // tslang declaration carries `export`, re-printed from the exporting module's source, and a
    // foreign one carries nothing. That is the whole test.
    //
    // Second, the defining module must return +1 (§9.24). If it was built `-mm=rc` it does:
    // swept over the corpus, every one of the 36 exported functions with a body classifies as
    // returning owned, and the only functions that fail the classification are `.next` methods
    // of generator state objects, which are anonymous and internal and cannot be exported. If
    // it was built `gc` or `none` it does not - but its blocks are now born
    // HEAP_BLOCK_IMMORTAL, so the release this consumption adds is a no-op and the object
    // leaks rather than being freed twice. That is section 4's agreed policy for a mixed link,
    // and moving it into the object is what lets this say yes without asking which model the
    // other module was built under - a question a static link cannot answer, because the import
    // is resolved by re-parsing source before any artifact of it exists.
    static bool importedFunctionReturnsOwned(MLIRTypeHelper &mth, mlir_ts::FuncOp funcOp, bool refCounted)
    {
        // ...or, under rc, a declaration that says it is one: `@owned`
        // (DECLARED_OWNED_RESULT_ATTR_NAME), which the default library's lib.d.ts says of what lib.ts
        // defines. Without it every library declaration was unclassified, so closed world never held
        // in a program that has the library - every call through a function value, and every direct
        // library call, kept its result's +1 for good. -mm=own goes by its own declared facts instead
        // (`@lends`: `JsonValue.get` lends what it returns, which counted as fresh here was released
        // under the borrow), and gc and none have no count to give back.
        auto declaredOwned = refCounted && funcOp->hasAttr(DECLARED_OWNED_RESULT_ATTR_NAME);
        if (!funcOp->hasAttr("export") && !declaredOwned)
        {
            return false;
        }

        for (auto resultType : funcOp.getFunctionType().getResults())
        {
            if (mth.ownsHeapMemory(funcOp.getLoc(), resultType))
            {
                return true;
            }
        }

        return false;
    }

    // body, a return whose retain is not in the same block, a return with no retain at all. A
    // false "no" costs a leak; a false "yes" frees a value the callee never retained.
    static bool functionReturnsOwned(MLIRTypeHelper &mth, mlir_ts::FuncOp funcOp, bool refCounted)
    {
        if (funcOp.isExternal() || funcOp.getBody().empty())
        {
            return importedFunctionReturnsOwned(mth, funcOp, refCounted);
        }

        // A generator's `next` - the function the yields are in - hands its caller a `{ value,
        // done }` record from every yield and every return. Under rc each of them retains it
        // (retainYieldedRecords for the yields), so it is classified like any other function. Under
        // the other models the yields do not, and it is left alone; its callers keep retaining.
        auto isGenerator = false;
        funcOp.walk([&](mlir_ts::YieldReturnValOp) { isGenerator = true; });
        if (isGenerator && !refCounted)
        {
            return false;
        }

        auto sawOwningReturn = false;
        auto everyReturnRetains = true;
        funcOp.walk([&](mlir::Operation *op) {
            if (!mlir::isa<mlir_ts::ReturnValOp, mlir_ts::YieldReturnValOp>(op))
            {
                return;
            }

            auto value = op->getOperand(0);
            if (!mth.ownsHeapMemory(op->getLoc(), value.getType()))
            {
                return;
            }

            sawOwningReturn = true;

            // Two ways a return can hand back a reference. It retains one of its own, which is
            // the ordinary case - or it forwards a reference it was already given, and then
            // there is no retain to find: `return new C()` consumes the instance's own +1
            // rather than adding a second (§9.25). Reading only the first shape as "returns
            // owned" left every `static times(...) { return new Vector(...) }` unclassified,
            // which is most of what expression-shaped code is built from.
            if (!retainPrecedes(op, value) && !producesOwnedResult(value))
            {
                everyReturnRetains = false;
            }
        });

        return sawOwningReturn && everyReturnRetains;
    }

    // Is there a `ts.Retain` of `value` ahead of `op` in its own block? The scope-exit releases
    // sit between the two in the ordinary case, so this is not an adjacency test - but it stays
    // inside one block deliberately. A retain somewhere else may not run on the path that
    // reaches this return, and reading "retained" off a path that never retained is the one
    // mistake here that frees live memory.
    static bool retainPrecedes(mlir::Operation *op, mlir::Value value)
    {
        for (auto it = mlir::Block::iterator(op); it != op->getBlock()->begin();)
        {
            --it;
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(*it))
            {
                if (retainOp.getReference() == value)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // Is this retain the one a `return` of `result` took? Scope exit may release other locals
    // between the two, but nothing else may use the value before it is returned.
    static bool retainFeedsReturn(mlir_ts::RetainOp retainOp, mlir::Value result)
    {
        for (auto *op = retainOp->getNextNode(); op; op = op->getNextNode())
        {
            if (mlir::isa<mlir_ts::ReturnValOp>(op))
            {
                return op->getNumOperands() > 0 && op->getOperand(0) == result;
            }

            if (llvm::is_contained(op->getOperands(), result))
            {
                return false;
            }
        }

        return false;
    }

    // The retain a receiver put on this call's result, if it took one. Two shapes, matching the
    // two ways §9.25's receivers acquire: a `ts.Retain` on the value itself (a field or element
    // store, a literal capturing it, a return passing it on), and a `ts.RetainSlot` on the
    // storage of a local declared from it.
    //
    // Returning none is the ordinary answer for a result nobody took - `f();` on its own, or
    // `f().n` - and it is left exactly as it is. Consuming a reference no receiver balances
    // would free the value while the expression is still using it.
    //
    // A folded `const` (OWNED_RESULT_NAMED_ATTR_NAME) is the exception: its name is this very
    // value, so the one receiver that may take its reference over is a `return`, after which
    // nothing reads the name. Any other receiver keeps its retain, and the call's +1 is given
    // back as a temporary like any other nobody took. Every `return` of it takes it over, since
    // each is on a path of its own - `if (x) return r; return r;` - and one left retaining would
    // hand its caller two references.
    // (`claim` false only asks: the declaration is not marked as the acquisition)
    static llvm::SmallVector<mlir::Operation *> findReceiverRetains(mlir::Value result, bool claim = true)
    {
        auto *definingOp = result.getDefiningOp();
        if (definingOp && definingOp->hasAttr(OWNED_RESULT_NAMED_ATTR_NAME))
        {
            llvm::SmallVector<mlir::Operation *> returned;
            for (auto *user : result.getUsers())
            {
                auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(user);
                if (retainOp && retainOp.getReference() == result && retainFeedsReturn(retainOp, result))
                {
                    returned.push_back(user);
                }
            }

            return returned;
        }

        for (auto *user : result.getUsers())
        {
            if (auto retainOp = mlir::dyn_cast<mlir_ts::RetainOp>(user))
            {
                if (retainOp.getReference() == result)
                {
                    return {retainOp.getOperation()};
                }
            }
        }

        for (auto *user : result.getUsers())
        {
            auto varOp = mlir::dyn_cast<mlir_ts::VariableOp>(user);
            if (!varOp || !varOp->hasAttr(OWNED_LOCAL_ATTR_NAME) ||
                varOp->hasAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME))
            {
                continue;
            }

            for (auto *slotUser : varOp.getResult().getUsers())
            {
                if (auto retainSlotOp = mlir::dyn_cast<mlir_ts::RetainSlotOp>(slotUser))
                {
                    // the declaration becomes the acquisition, which is what keeps the
                    // ownership verifier able to pair the release still to come
                    if (claim)
                    {
                        varOp->setAttr(OWNED_LOCAL_CONSUMED_ATTR_NAME, mlir::UnitAttr::get(varOp.getContext()));
                    }

                    return {retainSlotOp.getOperation()};
                }
            }
        }

        return {};
    }
};

} // end anonymous namespace

#undef DEBUG_TYPE

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createOwnedReturnConsumptionPass(CompileOptions &compileOptions)
{
    return std::make_unique<OwnedReturnConsumptionPass>(compileOptions);
}
