#include "TypeScript/Pass/WasmExceptionPass.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/IR/Module.h"

using namespace llvm;

#define DEBUG_TYPE "pass"

// wasm exception handling (the exception-handling feature, -wasm-enable-eh) is built on funclets,
// the catchswitch/catchpad/cleanuppad form MSVC's EH uses; the backend has nothing for a
// `landingpad`. tslang emits the Itanium form on wasm, as it does on Linux, so this pass turns each
// landing pad into a catch that is left at once:
//
//     invoke ... unwind label %lpad              invoke ... unwind label %wasm.dispatch
//                                                wasm.dispatch:
//                                                  %cs = catchswitch within none [label %wasm.catch] unwind to caller
//                                                wasm.catch:
//                                                  %cp = catchpad within %cs [ptr null]
//                                                  store (llvm.wasm.get.exception(%cp)), %wasm.exn
//                                                  catchret from %cp to label %lpad
//     lpad:                                      lpad:
//       %lp = landingpad {ptr, i32} ...            %lp = { load %wasm.exn, selector }
//
// The code after an Itanium landing pad already runs as ordinary code - the handler, the cleanup,
// and the `resume` or rethrow that ends it - so the funclet holds nothing but the catch. A catch of
// `ptr null` catches every C++ exception (the __cpp_exception tag) and needs no personality
// routine, no LSDA and no C++ runtime: the host (docs/how/wasm/test/host.mjs) keeps what a C++
// runtime keeps - each exception's type_info, the exceptions being handled - and throws a thrown
// object as a WebAssembly.Exception of that tag. A JS exception (exit(), an assert) is not one, so
// no catch sees it.
//
// What the personality routine would decide is decided after the catch instead:
// - the selector a typed catch clause compares (llvm.eh.typeid.for) is the type_info's address,
//   and the pad's selector that of the first clause that takes the exception - the same type, or
//   a base class's pointer for a class's (__tslang_wasm_catches) - or 0;
// - a pad with only typed clauses, which the personality would skip for any other type, rethrows
//   an exception none of them matches;
// - `resume` rethrows (llvm.wasm.throw).
//
// The C++ runtime's type_infos for the fundamental types (_ZTIi, _ZTIPc, ...) and the vtables of
// its type_info classes are not there to link against, and wasm-ld with --allow-undefined puts an
// undefined data symbol at address 0, so every one of them would be the same type: each gets a
// definition here.

namespace
{

constexpr const char *EXCEPTION_TYPE_FUNC_NAME = "__tslang_wasm_exception_type";
constexpr const char *PERSONALITY_FUNC_NAME = "__gxx_wasm_personality_v0";

struct WasmExceptionPassCode
{
    Module &M;
    LLVMContext &C;
    PointerType *ptrTy;
    IntegerType *i32Ty;
    IntegerType *intPtrTy;

    WasmExceptionPassCode(Module &M)
        : M(M), C(M.getContext()), ptrTy(PointerType::get(M.getContext(), 0)), i32Ty(Type::getInt32Ty(M.getContext())),
          intPtrTy(M.getDataLayout().getIntPtrType(M.getContext()))
    {
    }

    bool defineRuntimeTypeInfos()
    {
        SmallVector<GlobalVariable *> declarations;
        for (auto &G : M.globals())
        {
            auto name = G.getName();
            if (G.isDeclaration() && (name.starts_with("_ZTI") || name.starts_with("_ZTS") || name.starts_with("_ZTV")))
            {
                declarations.push_back(&G);
            }
        }

        for (auto *G : declarations)
        {
            // room for what is read through one: a vtable is referenced at an offset into it
            auto *type = ArrayType::get(ptrTy, 4);
            auto *definition = new GlobalVariable(M, type, /*isConstant=*/true, GlobalValue::LinkOnceODRLinkage,
                                                  Constant::getNullValue(type), "", G);
            definition->takeName(G);
            G->replaceAllUsesWith(definition);
            G->eraseFromParent();
        }

        return !declarations.empty();
    }

    // the selector of a type_info: its address, as an i32 like llvm.eh.typeid.for's
    Value *selectorOf(IRBuilder<> &builder, Value *typeInfo)
    {
        return builder.CreateTrunc(builder.CreatePtrToInt(typeInfo, intPtrTy), i32Ty);
    }

    // i1 __tslang_wasm_catches(ptr thrown, ptr clause): whether a catch clause of type_info `clause`
    // takes an exception thrown with type_info `thrown`, as the C++ personality decides it for what
    // tslang throws: the same type, or a pointer to a class and a pointer to one of its bases (a
    // class is thrown as a pointer to it; TypeScript classes have one base, an
    // __si_class_type_info's).
    Function *catchesFunction = nullptr;
    Function *getCatchesFunction()
    {
        if (catchesFunction)
        {
            return catchesFunction;
        }

        auto *i1Ty = Type::getInt1Ty(C);
        catchesFunction = Function::Create(FunctionType::get(i1Ty, {ptrTy, ptrTy}, false), GlobalValue::InternalLinkage,
                                           "__tslang_wasm_catches", M);
        auto *thrown = catchesFunction->getArg(0);
        auto *clause = catchesFunction->getArg(1);

        auto pointerSize = M.getDataLayout().getPointerSize();
        // a type_info's vtable pointer points two pointers into its class's vtable
        auto vtableOf = [&](StringRef name) -> Constant * {
            auto *vtable = M.getOrInsertGlobal(name, ptrTy);
            return ConstantExpr::getGetElementPtr(Type::getInt8Ty(C), vtable,
                                                  ConstantInt::get(intPtrTy, 2 * pointerSize));
        };
        auto *pointerTypeInfoVtable = vtableOf("_ZTVN10__cxxabiv119__pointer_type_infoE");
        auto *singleBaseTypeInfoVtable = vtableOf("_ZTVN10__cxxabiv120__si_class_type_infoE");
        // __pointer_type_info { vtable, name, i32 flags, pointee }; __si_class_type_info { vtable, name, base }
        auto *pointerTypeInfoType = StructType::get(C, {ptrTy, ptrTy, i32Ty, ptrTy});
        auto pointeeOffset = M.getDataLayout().getStructLayout(pointerTypeInfoType)->getElementOffset(3);

        auto *entry = BasicBlock::Create(C, "entry", catchesFunction);
        auto *pointers = BasicBlock::Create(C, "pointers", catchesFunction);
        auto *walk = BasicBlock::Create(C, "walk", catchesFunction);
        auto *next = BasicBlock::Create(C, "next", catchesFunction);
        auto *base = BasicBlock::Create(C, "base", catchesFunction);
        auto *yes = BasicBlock::Create(C, "yes", catchesFunction);
        auto *no = BasicBlock::Create(C, "no", catchesFunction);

        IRBuilder<> b(entry);
        auto *nullPtr = ConstantPointerNull::get(ptrTy);
        auto *eitherNull = b.CreateOr(b.CreateICmpEQ(thrown, nullPtr), b.CreateICmpEQ(clause, nullPtr));
        auto *same = BasicBlock::Create(C, "same", catchesFunction, pointers);
        b.CreateCondBr(eitherNull, no, same);

        b.SetInsertPoint(same);
        auto *bothPointersCheck = BasicBlock::Create(C, "kinds", catchesFunction, pointers);
        b.CreateCondBr(b.CreateICmpEQ(thrown, clause), yes, bothPointersCheck);

        b.SetInsertPoint(bothPointersCheck);
        auto *thrownIsPointer = b.CreateICmpEQ(b.CreateLoad(ptrTy, thrown), pointerTypeInfoVtable);
        auto *clauseIsPointer = b.CreateICmpEQ(b.CreateLoad(ptrTy, clause), pointerTypeInfoVtable);
        b.CreateCondBr(b.CreateAnd(thrownIsPointer, clauseIsPointer), pointers, no);

        b.SetInsertPoint(pointers);
        auto pointeeOf = [&](Value *typeInfo) {
            return b.CreateLoad(ptrTy, b.CreateConstGEP1_64(Type::getInt8Ty(C), typeInfo, pointeeOffset));
        };
        auto *thrownClass = pointeeOf(thrown);
        auto *clauseClass = pointeeOf(clause);
        b.CreateBr(walk);

        b.SetInsertPoint(walk);
        auto *current = b.CreatePHI(ptrTy, 2, "class");
        current->addIncoming(thrownClass, pointers);
        b.CreateCondBr(b.CreateICmpEQ(current, clauseClass), yes, next);

        b.SetInsertPoint(next);
        auto *hasBase = b.CreateAnd(b.CreateICmpNE(current, nullPtr),
                                    b.CreateICmpEQ(b.CreateLoad(ptrTy, current), singleBaseTypeInfoVtable));
        b.CreateCondBr(hasBase, base, no);

        b.SetInsertPoint(base);
        auto *baseClass = b.CreateLoad(ptrTy, b.CreateConstGEP1_64(Type::getInt8Ty(C), current, 2 * pointerSize));
        current->addIncoming(baseClass, base);
        b.CreateBr(walk);

        b.SetInsertPoint(yes);
        b.CreateRet(b.getTrue());
        b.SetInsertPoint(no);
        b.CreateRet(b.getFalse());

        return catchesFunction;
    }

    // A clause of a pointer type (a string is a char *, a class instance is thrown as a pointer to
    // it) that matches binds the pointer itself: the C++ runtime's __cxa_begin_catch returns the
    // thrown pointer then, not the address of the thrown object, which it returns for any other
    // clause and for a catch-all. The host's returns the object's address, always - it does not know
    // which clause matched - so where a pointer clause did, the pointer is read from there.
    void adjustPointerCatches(IRBuilder<> &builder, Instruction *padValue, Value *selector, ArrayRef<Value *> typedClauses)
    {
        SmallVector<Value *> pointerClauses;
        for (auto *clause : typedClauses)
        {
            if (auto *global = dyn_cast<GlobalValue>(clause->stripPointerCasts()); global && global->getName().starts_with("_ZTIP"))
            {
                pointerClauses.push_back(clause);
            }
        }

        if (pointerClauses.empty())
        {
            return;
        }

        SmallVector<CallInst *> beginCatches;
        for (auto *user : padValue->users())
        {
            auto *extract = dyn_cast<ExtractValueInst>(user);
            if (!extract || extract->getNumIndices() != 1 || extract->getIndices()[0] != 0)
            {
                continue;
            }

            for (auto *extractUser : extract->users())
            {
                if (auto *call = dyn_cast<CallInst>(extractUser);
                    call && call->getCalledFunction() && call->getCalledFunction()->getName() == "__cxa_begin_catch")
                {
                    beginCatches.push_back(call);
                }
            }
        }

        for (auto *beginCatch : beginCatches)
        {
            builder.SetInsertPoint(beginCatch->getNextNode());
            Value *pointerMatched = builder.getFalse();
            for (auto *clause : pointerClauses)
            {
                pointerMatched = builder.CreateOr(pointerMatched, builder.CreateICmpEQ(selector, selectorOf(builder, clause)));
            }

            auto *thrownPointer = builder.CreateLoad(ptrTy, beginCatch, "wasm.exn.pointer");
            auto *adjusted = builder.CreateSelect(pointerMatched, thrownPointer, beginCatch, "wasm.exn.adjusted");
            beginCatch->replaceUsesWithIf(adjusted, [&](Use &use) {
                return use.getUser() != thrownPointer && use.getUser() != adjusted;
            });
        }
    }

    bool runOnFunction(Function &F)
    {
        SmallVector<LandingPadInst *> pads;
        SmallVector<ResumeInst *> resumes;
        SmallVector<IntrinsicInst *> typeIds;
        for (auto &I : instructions(F))
        {
            if (auto *landingPad = dyn_cast<LandingPadInst>(&I))
            {
                pads.push_back(landingPad);
            }
            else if (auto *resume = dyn_cast<ResumeInst>(&I))
            {
                resumes.push_back(resume);
            }
            else if (auto *intrinsic = dyn_cast<IntrinsicInst>(&I); intrinsic && intrinsic->getIntrinsicID() == Intrinsic::eh_typeid_for)
            {
                typeIds.push_back(intrinsic);
            }
        }

        if (pads.empty() && resumes.empty() && typeIds.empty())
        {
            return false;
        }

        IRBuilder<> builder(C);
        auto *getException = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_get_exception);
        auto *wasmThrow = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::wasm_throw);
        auto cppExceptionTag = builder.getInt32(0);

        auto rethrow = [&](Value *exception) {
            builder.CreateCall(wasmThrow, {cppExceptionTag, exception});
            builder.CreateUnreachable();
        };

        if (!pads.empty())
        {
            F.setPersonalityFn(cast<Constant>(
                M.getOrInsertFunction(PERSONALITY_FUNC_NAME, FunctionType::get(i32Ty, /*isVarArg=*/true)).getCallee()));
        }

        // the exception, from the catch to the code after it (the catch is a funclet of its own)
        AllocaInst *exceptionSlot = nullptr;
        if (!pads.empty())
        {
            auto &entry = F.getEntryBlock();
            exceptionSlot = new AllocaInst(ptrTy, M.getDataLayout().getAllocaAddrSpace(), "wasm.exn", entry.getFirstInsertionPt());
        }

        for (auto *landingPad : pads)
        {
            auto *padBlock = landingPad->getParent();
            auto *dispatchBlock = BasicBlock::Create(C, "wasm.dispatch", &F, padBlock);
            auto *catchBlock = BasicBlock::Create(C, "wasm.catch", &F, padBlock);

            for (auto *pred : SmallVector<BasicBlock *>(predecessors(padBlock)))
            {
                auto *invoke = dyn_cast<InvokeInst>(pred->getTerminator());
                if (!invoke || invoke->getUnwindDest() != padBlock)
                {
                    report_fatal_error(Twine("wasm exceptions: a landing pad in '") + F.getName() +
                                       "' is reached other than by an invoke's unwind edge");
                }

                invoke->setUnwindDest(dispatchBlock);
            }

            builder.SetInsertPoint(dispatchBlock);
            auto *catchSwitch = builder.CreateCatchSwitch(ConstantTokenNone::get(C), nullptr, 1, "wasm.cs");
            catchSwitch->addHandler(catchBlock);

            // the pad's phis merge the invokes' values, which now come to the catchswitch
            for (auto &phi : make_early_inc_range(padBlock->phis()))
            {
                phi.moveBefore(catchSwitch->getIterator());
            }

            builder.SetInsertPoint(catchBlock);
            auto *catchPad = builder.CreateCatchPad(catchSwitch, {ConstantPointerNull::get(ptrTy)}, "wasm.cp");
            auto *exception = builder.CreateCall(getException, {catchPad}, "wasm.exn.value");
            builder.CreateStore(exception, exceptionSlot);
            builder.CreateCatchRet(catchPad, padBlock);

            auto catchesAll = landingPad->isCleanup();
            SmallVector<Value *> typedClauses;
            for (unsigned i = 0, n = landingPad->getNumClauses(); i < n; i++)
            {
                auto *clause = landingPad->getClause(i);
                if (landingPad->isFilter(i) || isa<ConstantPointerNull>(clause->stripPointerCasts()))
                {
                    catchesAll = true;
                }
                else
                {
                    typedClauses.push_back(clause);
                }
            }

            builder.SetInsertPoint(landingPad);
            auto *caught = builder.CreateLoad(ptrTy, exceptionSlot, "wasm.exn.caught");
            Value *selector = builder.getInt32(0);
            if (!typedClauses.empty())
            {
                // the selector of the first clause that takes the exception, as the personality
                // picks it; 0 for none (a catch-all clause's, or none at all - see below)
                auto exceptionTypeFunc = M.getOrInsertFunction(EXCEPTION_TYPE_FUNC_NAME, FunctionType::get(ptrTy, {ptrTy}, false));
                auto *thrownType = builder.CreateCall(exceptionTypeFunc, {caught}, "wasm.exn.type");
                for (auto *clause : llvm::reverse(typedClauses))
                {
                    auto *takes = builder.CreateCall(getCatchesFunction(), {thrownType, clause});
                    selector = builder.CreateSelect(takes, selectorOf(builder, clause), selector);
                }
            }

            Value *value = PoisonValue::get(landingPad->getType());
            value = builder.CreateInsertValue(value, caught, 0);
            auto *lastInsert = cast<Instruction>(builder.CreateInsertValue(value, selector, 1));
            landingPad->replaceAllUsesWith(lastInsert);
            landingPad->eraseFromParent();

            adjustPointerCatches(builder, lastInsert, selector, typedClauses);

            if (!catchesAll && !typedClauses.empty())
            {
                // what the personality would have skipped goes on to the caller
                builder.SetInsertPoint(lastInsert->getNextNode());
                auto *matchesInst = cast<Instruction>(builder.CreateICmpNE(selector, builder.getInt32(0), "wasm.matched"));
                auto *matches = matchesInst;
                auto *continueBlock = padBlock->splitBasicBlock(matchesInst->getNextNode(), "wasm.matched");
                auto *rethrowBlock = BasicBlock::Create(C, "wasm.unmatched", &F, continueBlock);
                padBlock->getTerminator()->eraseFromParent();
                builder.SetInsertPoint(padBlock);
                builder.CreateCondBr(matches, continueBlock, rethrowBlock);
                builder.SetInsertPoint(rethrowBlock);
                rethrow(caught);
            }
        }

        for (auto *resume : resumes)
        {
            builder.SetInsertPoint(resume);
            auto *exception = builder.CreateExtractValue(resume->getValue(), 0);
            rethrow(exception);
            resume->eraseFromParent();
        }

        for (auto *typeId : typeIds)
        {
            builder.SetInsertPoint(typeId);
            typeId->replaceAllUsesWith(selectorOf(builder, typeId->getArgOperand(0)));
            typeId->eraseFromParent();
        }

        return true;
    }

    bool run()
    {
        auto changed = false;
        for (auto &F : make_early_inc_range(M))
        {
            if (!F.isDeclaration() && &F != catchesFunction)
            {
                changed |= runOnFunction(F);
            }
        }

        // last: __tslang_wasm_catches refers to the type_info vtables
        changed |= defineRuntimeTypeInfos();
        return changed;
    }
};

} // namespace

namespace ts
{
llvm::PreservedAnalyses WasmExceptionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM)
{
    WasmExceptionPassCode code(M);
    if (!code.run())
    {
        return llvm::PreservedAnalyses::all();
    }

    return llvm::PreservedAnalyses::none();
}
} // namespace ts
