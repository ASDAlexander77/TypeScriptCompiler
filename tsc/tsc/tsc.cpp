#define DEBUG_TYPE "tsc"

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRGen.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptToLLVMIRTranslation.h"
#include "TypeScript/TypeScriptGC.h"

#include "TypeScript/rt.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#ifdef GC_ENABLE
#include "llvm/IR/GCStrategy.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#endif

using namespace typescript;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input TypeScript file>"), cl::init("-"), cl::value_desc("filename"));

namespace
{
enum InputType
{
    TypeScript,
    MLIR
};
} // namespace

static cl::opt<enum InputType> inputType("x", cl::init(TypeScript), cl::desc("Decided the kind of output desired"),
                                         cl::values(clEnumValN(TypeScript, "TypeScript", "load the input file as a TypeScript source.")),
                                         cl::values(clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

namespace
{
enum Action
{
    None,
    DumpAST,
    DumpMLIR,
    DumpMLIRAffine,
    DumpMLIRLLVM,
    DumpLLVMIR,
    RunJIT
};
} // namespace

static cl::opt<enum Action> emitAction("emit", cl::desc("Select the kind of output desired"),
                                       cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
                                       cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
                                       cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine", "output the MLIR dump after affine lowering")),
                                       cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm", "output the MLIR dump after llvm lowering")),
                                       cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
                                       cl::values(clEnumValN(RunJIT, "jit", "JIT the code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

// dump obj
cl::OptionCategory clOptionsCategory{"linking options"};
cl::list<std::string> clSharedLibs{"shared-libs", cl::desc("Libraries to link dynamically"), cl::ZeroOrMore, cl::MiscFlags::CommaSeparated,
                                   cl::cat(clOptionsCategory)};

static cl::opt<std::string> mainFuncName{"e", cl::desc("The function to be called"), cl::value_desc("<function name>"), cl::init("main")};

static cl::opt<bool> dumpObjectFile{"dump-object-file", cl::desc("Dump JITted-compiled object to file specified with "
                                                                 "-object-filename (<input file>.o by default).")};

static cl::opt<std::string> objectFilename{"object-filename", cl::desc("Dump JITted-compiled object to file <input file>.o")};

// static cl::opt<std::string> targetTriple("mtriple", cl::desc("Override target triple for module"));

cl::OptionCategory clTsCompilingOptionsCategory{"TypeScript compiling options"};
static cl::opt<bool> disableGC("nogc", cl::desc("Disable Garbage collection"), cl::cat(clTsCompilingOptionsCategory));

int loadMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module)
{
    auto fileName = llvm::StringRef(inputFilename);

    // Handle '.ts' input to the compiler.
    if (inputType != InputType::MLIR && !fileName.endswith(".mlir"))
    {
        auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
        if (std::error_code ec = fileOrErr.getError())
        {
            llvm::errs() << "Could not open input file: " << ec.message() << "\n";
            return -1;
        }

        CompileOptions compileOptions;
        compileOptions.disableGC = disableGC;
        module = mlirGenFromSource(context, fileName, fileOrErr.get()->getBuffer(), compileOptions);
        return !module ? 1 : 0;
    }

    // Otherwise, the input is '.mlir'.
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code EC = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: " << EC.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile(sourceMgr, &context);
    if (!module)
    {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }

    return 0;
}

void publishDiagnostic(mlir::Diagnostic &diag)
{
    auto printMsg = [](llvm::raw_fd_ostream &os, mlir::Diagnostic &diag, const char *msg) {
        if (!diag.getLocation().isa<mlir::UnknownLoc>())
            os << diag.getLocation() << ": ";
        os << msg;

        // The default behavior for errors is to emit them to stderr.
        os << diag << '\n';
        os.flush();
    };

    switch (diag.getSeverity())
    {
    case mlir::DiagnosticSeverity::Note:
        printMsg(llvm::outs(), diag, "note: ");
        for (auto &note : diag.getNotes())
        {
            printMsg(llvm::outs(), note, "note: ");
        }

        break;
    case mlir::DiagnosticSeverity::Warning:
        printMsg(llvm::outs(), diag, "warning: ");
        break;
    case mlir::DiagnosticSeverity::Error:
        printMsg(llvm::errs(), diag, "error: ");
        break;
    case mlir::DiagnosticSeverity::Remark:
        printMsg(llvm::outs(), diag, "information: ");
        break;
    }
}

int loadAndProcessMLIR(mlir::MLIRContext &context, mlir::OwningModuleRef &module)
{
    if (int error = loadMLIR(context, module))
    {
        return error;
    }

    mlir::ScopedDiagnosticHandler diagHandler(&context, [&](mlir::Diagnostic &diag) { publishDiagnostic(diag); });

    mlir::PassManager pm(&context);
    // Apply any generic pass manager command line options and run the pipeline.
    applyPassManagerCLOptions(pm);

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

    if (enableOpt || isLoweringToAffine)
    {
        // Inline all functions into main and then delete them.
        pm.addPass(mlir::createInlinerPass());
        // TODO: experiment
        pm.addPass(mlir::createCanonicalizerPass());

        // mlir::OpPassManager &optPM = pm.nest<mlir::typescript::FuncOp>();
        // disabled as it was experiment
        // optPM.addPass(mlir::typescript::createLoadBoundPropertiesPass());
        // optPM.addPass(mlir::createCanonicalizerPass());

        // TODO: this failing test about accessors
        // optPM.addPass(mlir::createCSEPass());
    }

    if (isLoweringToAffine)
    {
        mlir::OpPassManager &optPM = pm.nest<mlir::typescript::FuncOp>();

        // Partially lower the TypeScript dialect with a few cleanups afterwards.
        optPM.addPass(mlir::typescript::createLowerToAffinePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::typescript::createRelocateConstantPass());
        optPM.addPass(mlir::createCSEPass());

        pm.addPass(mlir::createSymbolDCEPass());

        // Add optimizations if enabled.
        if (enableOpt)
        {
            // TODO: do I need this?
            // optPM.addPass(mlir::createLoopFusionPass());
            // TODO: do I need this?
            // optPM.addPass(mlir::createMemRefDataFlowOptPass());
        }
    }

    if (isLoweringToLLVM)
    {
        // Finish lowering the TypeScript IR to the LLVM dialect.
        pm.addPass(mlir::typescript::createLowerToLLVMPass());
        if (!disableGC)
        {
            pm.addPass(mlir::typescript::createGCPass());
        }
    }

    if (mlir::failed(pm.run(*module)))
    {
        return 4;
    }

    return 0;
}

int dumpAST()
{
    if (inputType == InputType::MLIR && !llvm::StringRef(inputFilename).endswith(".mlir"))
    {
        llvm::errs() << "Can't dump a TypeScript AST when the input is MLIR\n";
        return 5;
    }

    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError())
    {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return 0;
    }

    llvm::outs() << dumpFromSource(inputFilename, fileOrErr.get()->getBuffer());

    return 0;
}

int dumpLLVMIR(mlir::ModuleOp module)
{
    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerLLVMDialectTranslation(*module->getContext());
    mlir::typescript::registerTypeScriptDialectTranslation(*module->getContext());

#ifdef TSGC_ENABLE
    mlir::typescript::registerTypeScriptGC();
#endif

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule)
    {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }

    llvm::errs() << *llvmModule << "\n";
    return 0;
}

int runJit(mlir::ModuleOp module)
{
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    mlir::registerLLVMDialectTranslation(*module->getContext());
    mlir::typescript::registerTypeScriptDialectTranslation(*module->getContext());

    // force linking
    // mlir::typescript::registerTypeScriptGC();

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    // If shared library implements custom mlir-runner library init and destroy
    // functions, we'll use them to register the library with the execution
    // engine. Otherwise we'll pass library directly to the execution engine.
    mlir::SmallVector<mlir::SmallString<256>, 4> libPaths;

    // Use absolute library path so that gdb can find the symbol table.
    transform(clSharedLibs, std::back_inserter(libPaths), [](std::string libPath) {
        mlir::SmallString<256> absPath(libPath.begin(), libPath.end());
        cantFail(llvm::errorCodeToError(llvm::sys::fs::make_absolute(absPath)));
        return absPath;
    });

    // Libraries that we'll pass to the ExecutionEngine for loading.
    mlir::SmallVector<mlir::StringRef, 4> executionEngineLibs;

    using MlirRunnerInitFn = void (*)(llvm::StringMap<void *> &);
    using MlirRunnerDestroyFn = void (*)();

    llvm::StringMap<void *> exportSymbols;
    mlir::SmallVector<MlirRunnerDestroyFn> destroyFns;

    // Handle libraries that do support mlir-runner init/destroy callbacks.
    for (auto &libPath : libPaths)
    {
        LLVM_DEBUG(llvm::dbgs() << "checking path: " << libPath.c_str() << "\n";);

        auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(libPath.c_str());
        void *initSym = lib.getAddressOfSymbol("__mlir_runner_init");
        void *destroySim = lib.getAddressOfSymbol("__mlir_runner_destroy");

        // Library does not support mlir runner, load it with ExecutionEngine.
        if (!initSym || !destroySim)
        {
            executionEngineLibs.push_back(libPath);
            continue;
        }

        LLVM_DEBUG(llvm::dbgs() << "added path: " << libPath.c_str() << "\n";);

        auto initFn = reinterpret_cast<MlirRunnerInitFn>(initSym);
        initFn(exportSymbols);

        auto destroyFn = reinterpret_cast<MlirRunnerDestroyFn>(destroySim);
        destroyFns.push_back(destroyFn);
    }

    auto noGC = false;

    // Build a runtime symbol map from the config and exported symbols.
    auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
        auto symbolMap = llvm::orc::SymbolMap();
        for (auto &exportSymbol : exportSymbols)
        {
            LLVM_DEBUG(llvm::dbgs() << "loading symbol: " << exportSymbol.getKey() << "\n";);
            symbolMap[interner(exportSymbol.getKey())] = llvm::JITEvaluatedSymbol::fromPointer(exportSymbol.getValue());
        }

        // adding my ref to __enable_execute_stack
        symbolMap[interner("__enable_execute_stack")] = llvm::JITEvaluatedSymbol::fromPointer(_mlir__enable_execute_stack);

        if (!disableGC && symbolMap.count(interner("GC_init")) == 0)
        {
            noGC = true;
        }

        return symbolMap;
    };

    if (noGC)
    {
        llvm::errs() << "JIT initialization failed. Missing GC library. Did you forget to provide it via '--shared-libs'?\n";
        return -1;
    }

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    auto maybeEngine = mlir::ExecutionEngine::create(module, /*llvmModuleBuilder=*/nullptr, optPipeline);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    engine->registerSymbols(runtimeSymbolMap);

    if (dumpObjectFile)
    {
        auto expectedFPtr = engine->lookup(mainFuncName);
        if (!expectedFPtr)
        {
            llvm::errs() << expectedFPtr.takeError();
            return -1;
        }

        engine->dumpToObjectFile(objectFilename.empty() ? inputFilename + ".o" : objectFilename);
        return 0;
    }

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked(mainFuncName);

    // Run all dynamic library destroy callbacks to prepare for the shutdown.
    llvm::for_each(destroyFns, [](MlirRunnerDestroyFn destroy) { destroy(); });

    if (invocationResult)
    {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "TypeScript compiler\n");

    if (emitAction == Action::DumpAST)
    {
        return dumpAST();
    }

    // If we aren't dumping the AST, then we are compiling with/to MLIR.

    mlir::MLIRContext context;
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::typescript::TypeScriptDialect>();
    context.getOrLoadDialect<mlir::StandardOpsDialect>();
    context.getOrLoadDialect<mlir::math::MathDialect>();
    context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

    mlir::OwningModuleRef module;
    if (int error = loadAndProcessMLIR(context, module))
    {
        return error;
    }

    // If we aren't exporting to non-mlir, then we are done.
    bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
    if (isOutputingMLIR)
    {
        module->dump();
        return 0;
    }

    // Check to see if we are compiling to LLVM IR.
    if (emitAction == Action::DumpLLVMIR)
    {
        return dumpLLVMIR(*module);
    }

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
    {
        return runJit(*module);
    }

    llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    return -1;
}
