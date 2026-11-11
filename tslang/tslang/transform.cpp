#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRGen.h"
#include "TypeScript/Passes.h"
#include "TypeScript/DiagnosticHelper.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialectTranslation.h"
#include "TypeScript/TypeScriptGC.h"
#ifdef ENABLE_ASYNC
#include "TypeScript/AsyncDialectTranslation.h"
#endif
#ifdef ENABLE_EXCEPTIONS
#include "TypeScript/Pass/LandingPadFixPass.h"
#include "TypeScript/Pass/Win32ExceptionPass.h"
#include "TypeScript/Pass/CxxThrowCallingConvPass.h"
#endif
#include "TypeScript/Pass/ExportFixPass.h"
#ifdef ENABLE_DEBUGINFO_PATCH_INFO
#include "TypeScript/Pass/DebugInfoPatchPass.h"
#endif
#include "TypeScript/Pass/MemAllocFixPass.h"
#include "TypeScript/Pass/AliasPass.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#ifdef ENABLE_ASYNC
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#endif

#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Operator.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/JumpThreading.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FormatVariadic.h"

#ifdef GC_ENABLE
#include "llvm/IR/GCStrategy.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#endif

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/DataStructs.h"

#define ENABLE_CUSTOM_PASSES 1
#define ENABLE_OPT_PASSES 1
// #define SAVE_VIA_PASS 1
//  TODO: if you uncomment it you will have exception in test 00try_finally.ts error: empty block: expect at least a terminator
// #define AFFINE_MODULE_PASS 1

#define DEBUG_TYPE "tslang"

namespace cl = llvm::cl;

extern cl::opt<std::string> inputFilename;
extern cl::opt<enum Action> emitAction;
extern cl::opt<bool> enableOpt;
extern cl::opt<int> optLevel;
extern cl::opt<int> sizeLevel;
extern cl::opt<bool> disableWarnings;
extern cl::opt<bool> verifyOwnership;
extern cl::opt<bool> ownSkipInference;
extern cl::opt<std::string> mainFuncName;
extern cl::opt<bool> noStackPromotion;
extern cl::opt<bool> noBorrowFactsExport;
extern cl::opt<unsigned> stackPromotionMaxObject;
extern cl::opt<unsigned> stackPromotionMaxFrame;
extern cl::opt<bool> stackPromotionReport;

int runMLIRPasses(mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr, mlir::OwningOpRef<mlir::ModuleOp> &module, CompileOptions &compileOptions)
{
    mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> postponedMessages;
    mlir::ScopedDiagnosticHandler diagHandler(&context, [&](mlir::Diagnostic &diag)
                                              { postponedMessages.emplace_back(new mlir::Diagnostic(std::move(diag))); });

    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

    // Before anything is lowered, and in every memory model: the ops it removes erase on the
    // way to LLVM under `gc` and `none` anyway, so doing this here keeps the IR the same shape
    // in all three and keeps the ownership verifier checking one thing rather than two.
    pm.addPass(mlir::typescript::createOwnedReturnConsumptionPass(compileOptions));

    if (isLoweringToAffine)
    {
        pm.addPass(mlir::createCanonicalizerPass());

#ifdef ENABLE_ASYNC
        pm.addPass(mlir::createAsyncToAsyncRuntimePass());
#endif

#ifndef AFFINE_MODULE_PASS
        mlir::OpPassManager &optPM = pm.nest<mlir::typescript::FuncOp>();

        // Partially lower the TypeScript dialect with a few cleanups afterwards.
        optPM.addPass(mlir::typescript::createLowerToAffineTSFuncPass(compileOptions));
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::typescript::createRelocateConstantPass());

        mlir::OpPassManager &optPM2 = pm.nest<mlir::func::FuncOp>();

        // Partially lower the TypeScript dialect with a few cleanups afterwards.
        optPM2.addPass(mlir::typescript::createLowerToAffineFuncPass(compileOptions));
        optPM2.addPass(mlir::createCanonicalizerPass());

        pm.addPass(mlir::typescript::createLowerToAffineModulePass(compileOptions));
        pm.addPass(mlir::createCanonicalizerPass());
#else
        pm.addPass(mlir::typescript::createLowerToAffineModulePass(compileOptions));
        pm.addPass(mlir::createCanonicalizerPass());

        mlir::OpPassManager &optPM = pm.nest<mlir::typescript::FuncOp>();
        optPM.addPass(mlir::typescript::createRelocateConstantPass());
#endif

        // Ahead of the optimisation passes, so it checks what MLIRGen and the affine lowering
        // actually produced rather than what the inliner left of it, and after the lowering,
        // because that is what turns unwind paths into ordinary CFG edges.
        if (verifyOwnership)
        {
            pm.nest<mlir::typescript::FuncOp>().addPass(mlir::typescript::createOwnershipVerifierPass());
        }

        // Under own, after the verifier: the verifier pairs rc's acquisitions with their
        // releases, and inference erases the acquisitions it proves are the one owner's. Inference
        // either erases a retain or reports it, and a program it reports stops here.
        // Named after the module's own file, the main buffer, not inputFilename, so a symbol is
        // never another module's.
        llvm::StringRef moduleFile = inputFilename;
        if (sourceMgr.getNumBuffers() > 0 && inputFilename != "-")
        {
            moduleFile = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBufferIdentifier();
        }

        // Nothing links the object of a JIT-run program, nor those the JIT cache compiles beside
        // it: they are only loaded into the JIT.
        auto linkedByOthers = !compileOptions.isJit && !compileOptions.jitCache;

        if (compileOptions.memoryModel == MemoryModelOwn && !ownSkipInference)
        {
            // a program's exports are seen by no other module (own spec 31): an executable, or the
            // module the JIT enters (a module the JIT cache compiles beside it has isJit off). What
            // its object tells a module linking it is written whatever it exports: a static
            // library is compiled with --export=none.
            pm.addPass(mlir::typescript::createOwnershipSignaturePass(
                compileOptions.isExecutable || compileOptions.isJit,
                linkedByOthers ? SHARED_LIB_OWN_OBJECT_FACTS + typescript::moduleSymbolSuffix(moduleFile) : ""));
            pm.nest<mlir::typescript::FuncOp>().addPass(mlir::typescript::createOwnershipInferencePass());
            // an `await`'s continuation and a `for await` body, outlined by the async lowering
            pm.nest<mlir::func::FuncOp>().addPass(mlir::typescript::createOwnershipInferencePass());
        }

        // What this module promises importers about its parameters, whether or not it promotes
        // anything itself. Nothing imports a JIT-run program, nor the modules the JIT cache
        // compiles beside it (isJit off, jitCache on): their objects are only loaded into the
        // JIT, never linked, and the program run there does not read facts from them
        // (StackPromotionPass).
        if (linkedByOthers && !noBorrowFactsExport)
        {
            pm.addPass(mlir::typescript::createBorrowFactsExportPass(
                SHARED_LIB_BORROW_FACTS + typescript::moduleSymbolSuffix(moduleFile)));
        }

        // Under every model, after own's inference: what it decides about a stack object's retains
        // is already settled, and the optimiser has not yet inlined anything
        if (!noStackPromotion)
        {
            mlir::typescript::StackPromotionLimits limits;
            limits.maxObjectBytes = stackPromotionMaxObject;
            limits.maxFrameBytes = stackPromotionMaxFrame;
            limits.report = stackPromotionReport;
            pm.addPass(mlir::typescript::createStackPromotionPass(compileOptions, limits));
        }

#ifdef ENABLE_OPT_PASSES
        if (enableOpt)
        {
            // optPM runs ahead of own's inference. Not under own: CSE merges two equal allocations
            // (`"a" + b` twice) into one value, which before inference has two owners and after it
            // two releases. Under rc their retains and releases stay paired.
            if (compileOptions.memoryModel != MemoryModelOwn)
            {
                optPM.addPass(mlir::createCSEPass());
            }
            // No MLIR loop-invariant code motion: it moves code out of ops with
            // LoopLikeOpInterface, and the lowering above has already turned every While,
            // DoWhile and For into branches. LLVM's LICM does the work.
            // Not under --di: stripping left every function a subprogram and every call no
            // location, which the LLVM dialect's verifier rejects for a call it could inline
            if (!compileOptions.generateDebugInfo)
            {
                pm.addPass(mlir::createStripDebugInfoPass());
            }
            pm.addPass(mlir::createInlinerPass());
            pm.addPass(mlir::createSCCPPass());
            pm.addPass(mlir::createSymbolDCEPass());
        }
#endif

#ifdef ENABLE_ASYNC
        // pm.addPass(mlir::createAsyncToAsyncRuntimePass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createAsyncRuntimeRefCountingPass());
        if (enableOpt)
        {
            pm.addPass(mlir::createAsyncRuntimeRefCountingOptPass());
        }
#endif
    }

    if (isLoweringToLLVM)
    {
#ifdef ENABLE_ASYNC
        pm.addPass(mlir::createConvertAsyncToLLVMPass());
        // Before GCPass, which rewrites aligned_alloc to GC_malloc_uncollectable and so keeps the corrected
        // signature; before LowerToLLVM, whose memref alloc lowering (AlignedAlloc) looks up
        // aligned_alloc with a pointer-width index and would reject the (i64, i64) declaration as a
        // redefinition.
        if (compileOptions.sizeBits() < 64)
        {
            pm.addPass(mlir::typescript::createAsyncTargetWidthPass(compileOptions.sizeBits()));
        }
#endif
        pm.addPass(mlir::typescript::createLowerToLLVMPass(compileOptions));
#ifdef ENABLE_ASYNC
        // After LowerToLLVM: the index casts it folds are half made by ConvertAsyncToLLVM and half
        // by LowerToLLVM, so this is the first point at which the whole chain exists.
        if (compileOptions.sizeBits() < 64)
        {
            pm.addPass(mlir::typescript::createAsyncIndexCastPass());
        }
#endif
        if (compileOptions.generateDebugInfo)
        {
            if (enableOpt)
            {
                pm.addPass(mlir::typescript::createInlinedLocationPass());
            }

            pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
        }

        if (compileOptions.needsGCRuntime())
        {
            pm.addPass(mlir::typescript::createGCPass(compileOptions));
        }
        else if (compileOptions.isWindows && !compileOptions.isWasm)
        {
            // one allocator for every module on Windows - see TypeScript/ProcessHeap.h
            pm.addPass(mlir::typescript::createProcessHeapPass());
        }
    }

    auto result = 0;
    if (mlir::failed(pm.run(*module)))
    {
        result = 4;
    }

    // A pass can report an error and still succeed - a lowering pattern that cannot cast says so
    // and carries on with a value that is not one - so the run's result alone is not the answer.
    // Without this the error was printed and the module compiled, linked and run anyway.
    if (llvm::any_of(postponedMessages, [](auto &diag) { return diag->getSeverity() == mlir::DiagnosticSeverity::Error; }))
    {
        result = 4;
    }

    SourceMgrDiagnosticHandlerEx sourceMgrHandler(sourceMgr, &context);
    printDiagnostics(sourceMgrHandler, postponedMessages, disableWarnings);
    return result;
}

int registerMLIRDialects(mlir::ModuleOp module)
{
    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());
    mlir::typescript::registerTypeScriptDialectTranslation(*module->getContext());

#ifdef TSGC_ENABLE
    mlir::typescript::registerTypeScriptGC();
#endif

#ifdef ENABLE_ASYNC
    mlir::typescript::registerAsyncDialectTranslation(*module->getContext());
#endif

    return 0;
}

static std::optional<llvm::OptimizationLevel> mapToLevel(unsigned optLevel, unsigned sizeLevel)
{
    switch (optLevel)
    {
    case 0:
        return llvm::OptimizationLevel::O0;

    case 1:
    case 2:
    case 3:
        // A size level is a pipeline of its own, as clang's -Os and -Oz are: it replaces the
        // speed level rather than being read only beside --opt_level=2. The default level is 3,
        // so `--opt --size_level=2` used to be a plain -O3.
        switch (sizeLevel)
        {
        case 0:
            return optLevel == 1   ? llvm::OptimizationLevel::O1
                   : optLevel == 2 ? llvm::OptimizationLevel::O2
                                   : llvm::OptimizationLevel::O3;

        case 1:
            return llvm::OptimizationLevel::Os;

        case 2:
            return llvm::OptimizationLevel::Oz;
        }
        break;
    }
    return std::nullopt;
}

// A C math function has two results: its value, and `errno` when the argument is out of its
// domain. TypeScript has no `errno` (`Math.sqrt(-1)` is NaN and nothing else), so a call of one,
// whether the default library's or a program's own `declare function sqrt(v: number): number`,
// is the LLVM intrinsic, as clang's -fno-math-errno and rustc make it. As a libm call that may
// write `errno` it kept loads from being hoisted over it, counted as a call against inlining the
// function around it, and `sqrt` kept a branch to the library for a negative argument.
static void applyNoMathErrno(llvm::Module &m)
{
    llvm::TargetLibraryInfoImpl tlii(llvm::Triple(m.getTargetTriple()));
    llvm::TargetLibraryInfo tli(tlii);

    for (auto &func : m)
    {
        llvm::LibFunc libFunc;
        if (!func.isDeclaration() || func.isIntrinsic() || !tli.getLibFunc(func, libFunc))
        {
            continue;
        }

        auto intrinsic = llvm::Intrinsic::not_intrinsic;
        switch (libFunc)
        {
        case llvm::LibFunc_sqrt: case llvm::LibFunc_sqrtf: intrinsic = llvm::Intrinsic::sqrt; break;
        case llvm::LibFunc_pow: case llvm::LibFunc_powf: intrinsic = llvm::Intrinsic::pow; break;
        case llvm::LibFunc_exp: case llvm::LibFunc_expf: intrinsic = llvm::Intrinsic::exp; break;
        case llvm::LibFunc_exp2: case llvm::LibFunc_exp2f: intrinsic = llvm::Intrinsic::exp2; break;
        case llvm::LibFunc_log: case llvm::LibFunc_logf: intrinsic = llvm::Intrinsic::log; break;
        case llvm::LibFunc_log2: case llvm::LibFunc_log2f: intrinsic = llvm::Intrinsic::log2; break;
        case llvm::LibFunc_log10: case llvm::LibFunc_log10f: intrinsic = llvm::Intrinsic::log10; break;
        case llvm::LibFunc_sin: case llvm::LibFunc_sinf: intrinsic = llvm::Intrinsic::sin; break;
        case llvm::LibFunc_cos: case llvm::LibFunc_cosf: intrinsic = llvm::Intrinsic::cos; break;
        case llvm::LibFunc_floor: case llvm::LibFunc_floorf: intrinsic = llvm::Intrinsic::floor; break;
        case llvm::LibFunc_ceil: case llvm::LibFunc_ceilf: intrinsic = llvm::Intrinsic::ceil; break;
        case llvm::LibFunc_trunc: case llvm::LibFunc_truncf: intrinsic = llvm::Intrinsic::trunc; break;
        case llvm::LibFunc_round: case llvm::LibFunc_roundf: intrinsic = llvm::Intrinsic::round; break;
        case llvm::LibFunc_rint: case llvm::LibFunc_rintf: intrinsic = llvm::Intrinsic::rint; break;
        case llvm::LibFunc_nearbyint: case llvm::LibFunc_nearbyintf: intrinsic = llvm::Intrinsic::nearbyint; break;
        case llvm::LibFunc_fabs: case llvm::LibFunc_fabsf: intrinsic = llvm::Intrinsic::fabs; break;
        default: continue;
        }

        // a use that is not a direct call (the function's address is taken) keeps the function
        llvm::SmallVector<llvm::CallInst *, 8> calls;
        for (auto *user : func.users())
        {
            auto *call = llvm::dyn_cast<llvm::CallInst>(user);
            if (call && call->getCalledOperand() == &func && call->getFunctionType() == func.getFunctionType()
                && !call->isMustTailCall())
            {
                calls.push_back(call);
            }
        }

        if (calls.empty())
        {
            continue;
        }

        auto *intrinsicFunc = llvm::Intrinsic::getOrInsertDeclaration(&m, intrinsic, {func.getReturnType()});
        for (auto *call : calls)
        {
            llvm::SmallVector<llvm::Value *, 2> args(call->args());
            auto *newCall = llvm::CallInst::Create(intrinsicFunc, args, "", call->getIterator());
            newCall->setDebugLoc(call->getDebugLoc());
            newCall->setTailCallKind(call->getTailCallKind());
            newCall->copyFastMathFlags(call);
            newCall->takeName(call);
            call->replaceAllUsesWith(newCall);
            call->eraseFromParent();
        }
    }
}

// An executable built with --opt: a function no other module can name (not `export`ed, not the
// entry point) is internal, as a `static` C function and a Rust function that is not `pub` are.
// The inliner then knows every caller: a function with one call left is inlined whatever its
// size and its body dropped, which is how the ray tracer's `shade` and `getNaturalColor` end up
// inside `traceRay`. A library's object, a shared library and JIT modules are loaded beside other
// modules that find functions by name, and stay as they were.
static void internalizeModulePrivateFunctions(llvm::Module &m)
{
    for (auto &func : m)
    {
        if (func.isDeclaration() || !func.hasExternalLinkage() || func.hasDLLExportStorageClass()
            || func.getName() == MAIN_ENTRY_NAME || func.getName() == mainFuncName.getValue()
            || func.hasFnAttribute("export") || func.hasFnAttribute(DLL_EXPORT) || func.hasFnAttribute(DLL_NAME)
            || func.hasSection())
        {
            continue;
        }

        func.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
}

// A select whose condition compares a phi of the same block, when the comparison is decided for
// one of the phi's constant values: made a branch again, which jump threading (run next) threads
// for the predecessors that bring those constants. SimplifyCFG turns `if (d < closest) { closest =
// d; index = i; }` into selects before the function computing `d` is inlined; once it is, a path
// on which `d` is Infinity still runs them, and jump threading unfolds a select only over integer
// constants compared with a constant.
struct UnfoldSelectOverPhiPass : llvm::PassInfoMixin<UnfoldSelectOverPhiPass>
{
    static llvm::CmpInst *decidedCompare(llvm::SelectInst *select, const llvm::DataLayout &dl)
    {
        auto *block = select->getParent();
        auto *cmp = llvm::dyn_cast<llvm::CmpInst>(select->getCondition());
        if (!cmp || cmp->getParent() != block || !select->getType()->isSingleValueType())
        {
            return nullptr;
        }

        auto *phi = llvm::dyn_cast<llvm::PHINode>(cmp->getOperand(0));
        auto phiIsLhs = true;
        if (!phi || phi->getParent() != block)
        {
            phi = llvm::dyn_cast<llvm::PHINode>(cmp->getOperand(1));
            phiIsLhs = false;
        }

        if (!phi || phi->getParent() != block)
        {
            return nullptr;
        }

        for (unsigned i = 0, e = phi->getNumIncomingValues(); i != e; ++i)
        {
            auto *constant = llvm::dyn_cast<llvm::Constant>(phi->getIncomingValue(i));
            if (!constant)
            {
                continue;
            }

            auto *other = cmp->getOperand(phiIsLhs ? 1 : 0)->DoPHITranslation(block, phi->getIncomingBlock(i));
            auto *lhs = phiIsLhs ? static_cast<llvm::Value *>(constant) : other;
            auto *rhs = phiIsLhs ? other : static_cast<llvm::Value *>(constant);
            if (auto *folded = llvm::simplifyCmpInst(cmp->getPredicate(), lhs, rhs, {dl}))
            {
                if (llvm::isa<llvm::ConstantInt>(folded))
                {
                    return cmp;
                }
            }
        }

        return nullptr;
    }

    llvm::PreservedAnalyses run(llvm::Function &func, llvm::FunctionAnalysisManager &)
    {
        auto &dl = func.getDataLayout();
        auto changed = false;
        llvm::SmallVector<llvm::SelectInst *, 8> candidates;
        for (auto &block : func)
        {
            for (auto &inst : block)
            {
                auto *select = llvm::dyn_cast<llvm::SelectInst>(&inst);
                if (select && decidedCompare(select, dl))
                {
                    candidates.push_back(select);
                    break;
                }
            }
        }

        for (auto *first : candidates)
        {
            auto *block = first->getParent();
            auto *cmp = llvm::cast<llvm::CmpInst>(first->getCondition());

            // every select on that comparison whose operands are computed before the first one
            llvm::SmallVector<llvm::SelectInst *, 4> selects;
            for (auto it = first->getIterator(), end = block->end(); it != end; ++it)
            {
                auto *select = llvm::dyn_cast<llvm::SelectInst>(&*it);
                if (!select || select->getCondition() != cmp || !select->getType()->isSingleValueType())
                {
                    continue;
                }

                auto before = [&](llvm::Value *value) {
                    auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
                    return !inst || inst->getParent() != block || inst->comesBefore(first);
                };
                if (before(select->getTrueValue()) && before(select->getFalseValue()))
                {
                    selects.push_back(select);
                }
            }

            auto *thenTerminator = llvm::SplitBlockAndInsertIfThen(cmp, first->getIterator(), /*Unreachable=*/false);
            auto *thenBlock = thenTerminator->getParent();
            auto *tail = first->getParent();
            for (auto *select : selects)
            {
                auto *phi = llvm::PHINode::Create(select->getType(), 2, "", tail->begin());
                phi->addIncoming(select->getTrueValue(), thenBlock);
                phi->addIncoming(select->getFalseValue(), block);
                phi->takeName(select);
                select->replaceAllUsesWith(phi);
                select->eraseFromParent();
            }

            changed = true;
        }

        return changed ? llvm::PreservedAnalyses::none() : llvm::PreservedAnalyses::all();
    }
};

// --size_level: the pipeline's level alone decides little - the inliner's thresholds, the
// vectorizer, the unroller and the backend all ask the function whether it is optimized for
// size. clang marks every function `optsize` under -Os and `optsize minsize` under -Oz.
static void applySizeAttributes(llvm::Module &m, llvm::OptimizationLevel level)
{
    auto minSize = level == llvm::OptimizationLevel::Oz;
    if (!minSize && level != llvm::OptimizationLevel::Os)
    {
        return;
    }

    for (auto &func : m)
    {
        // `optnone` and the size attributes exclude each other (the verifier rejects the pair)
        if (func.isDeclaration() || func.hasFnAttribute(llvm::Attribute::OptimizeNone))
        {
            continue;
        }

        func.addFnAttr(llvm::Attribute::OptimizeForSize);
        if (minSize)
        {
            func.addFnAttr(llvm::Attribute::MinSize);
        }
    }
}

// --fast-math: relax IEEE-754 strictness the same way clang's
// -funsafe-math-optimizations does, but deliberately WITHOUT nnan/ninf
// (finite-math-only). In TypeScript NaN and Infinity are first-class values
// produced in ordinary control flow (Number('abc'), parseFloat failures,
// overflow), and the default lib's isNaN/isFinite are `x !== x`-style checks
// that nnan would constant-fold away. reassoc+nsz is what actually licenses
// FP-reduction vectorization; arcp/contract/afn add reciprocal, FMA
// contraction, and approximate libm calls.
static void applyFastMathFlags(llvm::Module &m)
{
    for (auto &func : m)
    {
        if (func.isDeclaration())
        {
            continue;
        }

        // backend (SelectionDAG/GlobalISel) counterparts of the IR flags
        func.addFnAttr("unsafe-fp-math", "true");
        func.addFnAttr("no-signed-zeros-fp-math", "true");
        func.addFnAttr("approx-func-fp-math", "true");

        for (auto &inst : llvm::instructions(func))
        {
            if (llvm::isa<llvm::FPMathOperator>(&inst))
            {
                auto fmf = inst.getFastMathFlags();
                fmf.setAllowReassoc();
                fmf.setNoSignedZeros();
                fmf.setAllowReciprocal();
                fmf.setAllowContract();
                fmf.setApproxFunc();
                inst.setFastMathFlags(fmf);
            }
        }
    }
}

std::function<llvm::Error(llvm::Module *)> makeCustomPassesWithOptimizingTransformer(
    std::optional<unsigned> mbOptLevel, std::optional<unsigned> mbSizeLevel, llvm::TargetMachine *targetMachine, CompileOptions &compileOptions)
{
    return [mbOptLevel, mbSizeLevel, targetMachine, compileOptions](llvm::Module *m) -> llvm::Error
    {
        std::optional<llvm::OptimizationLevel> ol = mapToLevel(mbOptLevel.value(), mbSizeLevel.value());
        if (!ol)
        {
            return llvm::make_error<llvm::StringError>(
                llvm::formatv("invalid optimization/size level {0}/{1}", mbOptLevel.value(), mbSizeLevel.value()).str(),
                llvm::inconvertibleErrorCode());
        }

        if (compileOptions.enableFastMath)
        {
            applyFastMathFlags(*m);
        }

        applyNoMathErrno(*m);

        if (compileOptions.isExecutable && *ol != llvm::OptimizationLevel::O0)
        {
            internalizeModulePrivateFunctions(*m);
        }

        applySizeAttributes(*m, *ol);

        // Without a TargetMachine the -O2/-O3 cost model has no target information: no vector
        // registers are known, so LoopVectorize/SLP never consider vectorization profitable.
        // A caller that generates the code hands in the machine it generates it with, so the
        // cost model and the code generator agree on the CPU - the JIT's is the host's, which
        // -mcpu/-mattr (empty unless given) do not name. Only a caller with no machine of its
        // own (--emit=llvm, --emit=bc) gets one built here from the module triple and
        // -mcpu/-mattr.
        std::unique_ptr<llvm::TargetMachine> ownedTargetMachine;
        auto *effectiveTargetMachine = targetMachine;
        if (!effectiveTargetMachine)
        {
            std::string lookupError;
            if (auto *target = llvm::TargetRegistry::lookupTarget(m->getTargetTriple(), lookupError))
            {
                ownedTargetMachine.reset(target->createTargetMachine(
                    m->getTargetTriple(), llvm::codegen::getCPUStr(), llvm::codegen::getFeaturesStr(),
                    llvm::TargetOptions(), std::nullopt));
                effectiveTargetMachine = ownedTargetMachine.get();
            }
        }

        llvm::LoopAnalysisManager lam;
        llvm::FunctionAnalysisManager fam;
        llvm::CGSCCAnalysisManager cgam;
        llvm::ModuleAnalysisManager mam;

        // The inliner's threshold at the highest speed level is the one LLVM keeps for functions
        // with an inline hint (325, what clang gives every C++ function defined in its class), not
        // the plain 250: a TypeScript program is small methods and functions over small values.
        // The ray tracer's `intersections` (cost 285; its Rust twin 245 against rustc's 275) was
        // left as a call in each ray's loop, with every live number spilled around it.
        llvm::PipelineTuningOptions pto;
        if (*ol == llvm::OptimizationLevel::O3)
        {
            pto.InlinerThreshold = 325;
        }

        llvm::PassBuilder pb(effectiveTargetMachine, pto);
        // Jump threading before the pipeline's first SimplifyCFG. `let d = 0; if (..) d = ..;
        // return d === 0 ? Infinity : d` reaches that pass as a phi of constants and a test of it;
        // SimplifyCFG makes the test a select, and jump threading later unfolds a select only over
        // integer constants, so a path that knew its answer still ran the select and what the
        // caller does with the result. Threaded first, such a path goes straight to its result
        // (the ray tracer's misses to the loop's next thing). SROA first: the values are in
        // allocas until then.
        pb.registerPipelineStartEPCallback([](llvm::ModulePassManager &pm, llvm::OptimizationLevel level) {
            if (level != llvm::OptimizationLevel::O2 && level != llvm::OptimizationLevel::O3)
            {
                return;
            }

            llvm::FunctionPassManager fpm;
            fpm.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
            fpm.addPass(llvm::JumpThreadingPass());
            pm.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(fpm)));
        });


        // after a function's callees are inlined into it
        pb.registerScalarOptimizerLateEPCallback([](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
            if (level != llvm::OptimizationLevel::O2 && level != llvm::OptimizationLevel::O3)
            {
                return;
            }

            fpm.addPass(UnfoldSelectOverPhiPass());
            fpm.addPass(llvm::JumpThreadingPass());
        });
        pb.registerModuleAnalyses(mam);
        pb.registerCGSCCAnalyses(cgam);
        pb.registerFunctionAnalyses(fam);
        pb.registerLoopAnalyses(lam);
        pb.crossRegisterProxies(lam, fam, cgam, mam);

        llvm::ModulePassManager mpm;

        // pb.parsePassPipeline(mpm, "module(function(landing-pad-fix))");

#ifdef ENABLE_DEBUGINFO_PATCH_INFO
        // debug info patch
        mpm.addPass(llvm::createModuleToFunctionPassAdaptor(ts::DebugInfoPatchPass()));
#endif

        // add custom passes
        mpm.addPass(llvm::createModuleToFunctionPassAdaptor(ts::LandingPadFixPass()));
        if (compileOptions.isWindows)
        {
            mpm.addPass(llvm::createModuleToFunctionPassAdaptor(ts::Win32ExceptionPass()));
        }

        // After Win32ExceptionPass, which adds throw calls and turns calls into invokes, and
        // before the optimization pipeline, which treats a calling-convention mismatch as UB.
        if (compileOptions.targetInfo.stdcallDecoratesCxxThrow)
        {
            mpm.addPass(ts::CxxThrowCallingConvPass());
        }

        llvm::Triple triple(m->getTargetTriple());
        mpm.addPass(ts::ExportFixPass(triple.isWindowsMSVCEnvironment()));

        if (compileOptions.isWasm)
        {
            mpm.addPass(ts::MemAllocFixPass(compileOptions.sizeBits()));
            mpm.addPass(ts::AliasPass(true, compileOptions.sizeBits()));
        }

        if (*ol == llvm::OptimizationLevel::O0)
            mpm.addPass(pb.buildO0DefaultPipeline(*ol));
        else
            mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));

#ifdef SAVE_VIA_PASS
        std::unique_ptr<llvm::ToolOutputFile> FDOut;
        if (emitAction == Action::DumpLLVMIR)
        {
            FDOut = GetOutputStream(emitAction);
            mpm.addPass(llvm::PrintModulePass(FDOut ? FDOut->os() : llvm::errs()));
        }

        if (emitAction == Action::DumpByteCode)
        {
            FDOut = GetOutputStream(emitAction);
            mpm.addPass(llvm::BitcodeWriterPass(FDOut ? FDOut->os() : llvm::errs()));
        }
#endif

        mpm.run(*m, mam);

#ifdef SAVE_VIA_PASS
        if (FDOut)
        {
            FDOut->keep();
        }
#endif

        return llvm::Error::success();
    };
}

std::function<llvm::Error(llvm::Module *)> getTransformer(bool enableOpt, int optLevel, int sizeLevel, CompileOptions &compileOptions,
                                                          llvm::TargetMachine *targetMachine)
{
#ifdef ENABLE_CUSTOM_PASSES
    auto optPipeline = makeCustomPassesWithOptimizingTransformer(
        /*optLevel=*/enableOpt ? optLevel : 0,
        /*sizeLevel=*/enableOpt ? sizeLevel : 0,
        targetMachine,
        compileOptions);
#else
    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? optLevel : 0,
        /*sizeLevel=*/enableOpt ? sizeLevel : 0,
        targetMachine);
#endif

    return optPipeline;
}
