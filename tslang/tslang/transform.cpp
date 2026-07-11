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
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Operator.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
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
extern cl::opt<bool> disableGC;
extern cl::opt<bool> disableWarnings;

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

#ifdef ENABLE_OPT_PASSES
        if (enableOpt)
        {
            optPM.addPass(mlir::createCSEPass());
            pm.addPass(mlir::createLoopInvariantCodeMotionPass());
            pm.addPass(mlir::createStripDebugInfoPass());
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
#endif
        pm.addPass(mlir::typescript::createLowerToLLVMPass(compileOptions));
        if (compileOptions.generateDebugInfo)
        {
            pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
        }

        if (!disableGC)
        {
            pm.addPass(mlir::typescript::createGCPass(compileOptions));
        }
    }

    auto result = 0;
    if (mlir::failed(pm.run(*module)))
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
        return llvm::OptimizationLevel::O1;

    case 2:
        switch (sizeLevel)
        {
        case 0:
            return llvm::OptimizationLevel::O2;

        case 1:
            return llvm::OptimizationLevel::Os;

        case 2:
            return llvm::OptimizationLevel::Oz;
        }
        break;
    case 3:
        return llvm::OptimizationLevel::O3;
    }
    return std::nullopt;
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

        // Every caller hands in a null TargetMachine, which leaves the -O2/-O3
        // cost model without target information: no vector registers are known,
        // so LoopVectorize/SLP never consider vectorization profitable. Build a
        // TargetMachine from the module triple (honoring -mcpu/-mattr) so the
        // optimization pipeline sees the real target.
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

        llvm::PassBuilder pb(effectiveTargetMachine);

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

        llvm::Triple triple(m->getTargetTriple());
        mpm.addPass(ts::ExportFixPass(triple.isWindowsMSVCEnvironment()));

        if (compileOptions.isWasm)
        {
            mpm.addPass(ts::MemAllocFixPass(compileOptions.sizeBits));
            mpm.addPass(ts::AliasPass(true, compileOptions.sizeBits));
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

std::function<llvm::Error(llvm::Module *)> getTransformer(bool enableOpt, int optLevel, int sizeLevel, CompileOptions &compileOptions)
{
#ifdef ENABLE_CUSTOM_PASSES
    auto optPipeline = makeCustomPassesWithOptimizingTransformer(
        /*optLevel=*/enableOpt ? optLevel : 0,
        /*sizeLevel=*/enableOpt ? sizeLevel : 0,
        /*targetMachine=*/nullptr,
        compileOptions);
#else
    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? optLevel : 0,
        /*sizeLevel=*/enableOpt ? sizeLevel : 0,
        /*targetMachine=*/nullptr);
#endif

    return optPipeline;
}
