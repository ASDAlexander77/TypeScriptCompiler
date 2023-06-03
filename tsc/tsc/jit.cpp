#include "TypeScript/Version.h"
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
#include "TypeScript/LandingPadFixPass.h"
#ifdef WIN_EXCEPTION
#include "TypeScript/Win32ExceptionPass.h"
#endif
#endif

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/PassInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#define DEBUG_TYPE "tsc"

using namespace typescript;
namespace cl = llvm::cl;

extern cl::opt<bool> enableOpt;
extern cl::opt<int> optLevel;
extern cl::opt<int> sizeLevel;
extern cl::list<std::string> clSharedLibs;
extern cl::opt<bool> dumpObjectFile;
extern cl::opt<std::string> objectFilename;
extern cl::opt<bool> disableGC;
extern cl::opt<std::string> mainFuncName;
extern cl::opt<std::string> inputFilename;

int registerMLIRDialects(mlir::ModuleOp);
std::function<llvm::Error(llvm::Module *)> getTransformer(bool, int, int);

int runJit(int argc, char **argv, mlir::ModuleOp module)
{
    // Print a stack trace if we signal out.
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);
    llvm::PrettyStackTraceProgram X(argc, argv);
    llvm::setBugReportMsg("PLEASE submit a bug report to https://github.com/ASDAlexander77/TypeScriptCompiler/issues and include the crash backtrace.");

    llvm::llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

    registerMLIRDialects(module);

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel);

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
        LLVM_DEBUG(llvm::dbgs() << "loading lib at path: " << libPath.c_str() << "\n";);

        std::string errMsg;
        auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(libPath.c_str(), &errMsg);

        if (errMsg.size() > 0)
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << "Loading error lib: " << errMsg << "\n";
            return -1;
        }

        LLVM_DEBUG(llvm::dbgs() << "loaded path: " << libPath.c_str() << "\n";);

        void *initSym = lib.getAddressOfSymbol("__mlir_runner_init");
        if (!initSym)
        {
            LLVM_DEBUG(llvm::dbgs() << "missing __mlir_runner_init";);
        }

        void *destroySim = lib.getAddressOfSymbol("__mlir_runner_destroy");
        if (!destroySim)
        {
            LLVM_DEBUG(llvm::dbgs() << "missing __mlir_runner_destroy";);
        }

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

#ifdef ENABLE_STACK_EXEC
        // adding my ref to __enable_execute_stack
        symbolMap[interner("__enable_execute_stack")] = llvm::JITEvaluatedSymbol::fromPointer(_mlir__enable_execute_stack);
#endif        

        if (!disableGC && symbolMap.count(interner("GC_init")) == 0)
        {
            noGC = true;
        }

        return symbolMap;
    };

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    engineOptions.enableObjectDump = dumpObjectFile;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    engine->registerSymbols(runtimeSymbolMap);
    if (noGC)
    {
#ifdef WIN32
#define LIB_NAME ""
#define LIB_EXT "dll"
#else
#define LIB_NAME "lib"
#define LIB_EXT "so"
#endif
        llvm::WithColor::error(llvm::errs(), "tsc") << "JIT initialization failed. Missing GC library. Did you forget to provide it via "
                        "'--shared-libs=" LIB_NAME "TypeScriptRuntime." LIB_EXT "'? or you can switch it off by using '-nogc'\n";
        return -1;
    }

    if (dumpObjectFile)
    {
        auto expectedFPtr = engine->lookup(mainFuncName);
        if (!expectedFPtr)
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << expectedFPtr.takeError();
            return -1;
        }

        llvm::Triple theTriple;
        theTriple.setTriple(llvm::sys::getDefaultTargetTriple());

        engine->dumpToObjectFile(
            objectFilename.empty() 
                ? inputFilename + (
                    (theTriple.getOS() == llvm::Triple::Win32) 
                        ? ".obj" 
                        : ".o") 
                : objectFilename);

        return 0;
    }

    if (module.lookupSymbol("__mlir_gctors"))
    {
        auto gctorsResult = engine->invokePacked("__mlir_gctors");
        if (gctorsResult)
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << "JIT calling global constructors failed, error: " << gctorsResult << "\n";
            return -1;
        }
    }

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked(mainFuncName);

    // Run all dynamic library destroy callbacks to prepare for the shutdown.
    llvm::for_each(destroyFns, [](MlirRunnerDestroyFn destroy) { destroy(); });

    if (invocationResult)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "JIT invocation failed, error: " << invocationResult << "\n";
        return -1;
    }

    return 0;
}
