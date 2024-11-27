#include "TypeScript/DataStructs.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/TargetParser/Host.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/ManagedStatic.h"

#include "TypeScript/Defines.h"

#define DEBUG_TYPE "tsc"

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

// obj
extern cl::opt<std::string> TargetTriple;

std::string getDefaultLibPath();
std::string mergeWithDefaultLibPath(std::string, std::string);

int registerMLIRDialects(mlir::ModuleOp);
std::function<llvm::Error(llvm::Module *)> getTransformer(bool, int, int, CompileOptions&);

using MlirRunnerInitFn = void (*)(llvm::StringMap<void *> &);
using MlirRunnerDestroyFn = void (*)();

int loadLibrary(mlir::SmallString<256> &libPath, llvm::StringMap<void *> &exportSymbols, mlir::SmallVector<MlirRunnerDestroyFn> &destroyFns)
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

    void *initSym = lib.getAddressOfSymbol("__mlir_execution_engine_init");
    if (!initSym)
    {
        LLVM_DEBUG(llvm::dbgs() << "missing __mlir_execution_engine_init";);
    }

    void *destroySim = lib.getAddressOfSymbol("__mlir_execution_engine_destroy");
    if (!destroySim)
    {
        LLVM_DEBUG(llvm::dbgs() << "missing __mlir_execution_engine_destroy";);
    }

    // Library does not support mlir runner, load it with ExecutionEngine.
    if (!initSym || !destroySim)
    {
        return 0;
    }

    LLVM_DEBUG(llvm::dbgs() << "added path: " << libPath.c_str() << "\n";);

    auto initFn = reinterpret_cast<MlirRunnerInitFn>(initSym);
    initFn(exportSymbols);

    auto destroyFn = reinterpret_cast<MlirRunnerDestroyFn>(destroySim);
    destroyFns.push_back(destroyFn);

    return 0;
}

bool addLibFileIfExists(mlir::SmallVector<mlir::StringRef>& sharedLibs, mlir::StringRef tsLibPath)
{
    mlir::SmallString<256> absPathTypeScriptLib(tsLibPath.begin(), tsLibPath.end());
    cantFail(llvm::errorCodeToError(llvm::sys::fs::make_absolute(absPathTypeScriptLib)));        

    if (llvm::sys::fs::exists(absPathTypeScriptLib))
    {
        return true;
    }

    return false;
}

int runJit(int argc, char **argv, mlir::ModuleOp module, CompileOptions &compileOptions)
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

    auto optPipeline = getTransformer(enableOpt, optLevel, sizeLevel, compileOptions);

    // If shared library implements custom mlir-runner library init and destroy
    // functions, we'll use them to register the library with the execution
    // engine. Otherwise we'll pass library directly to the execution engine.
    if (!compileOptions.noDefaultLib)
    {
        clSharedLibs.push_back(mergeWithDefaultLibPath(getDefaultLibPath(), 
#ifdef WIN32        
            DEFAULT_LIB_DIR "/dll/" DEFAULT_LIB_NAME ".dll"
#else
            DEFAULT_LIB_DIR "/dll/lib" DEFAULT_LIB_NAME ".so"
#endif        
        ));
    }      

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::SmallVector<mlir::StringRef> sharedLibPaths;
    sharedLibPaths.append(begin(clSharedLibs), end(clSharedLibs));

    // add default libs in case they are not part of options
#ifdef WIN32
#define LIB_NAME ""
#define LIB_EXT "dll"
#else
#define LIB_NAME "lib"
#define LIB_EXT "so"
#endif
    if (!disableGC.getValue())
        if (!addLibFileIfExists(sharedLibPaths, "../lib/" LIB_NAME "TypeScriptRuntime." LIB_EXT))
            if (!addLibFileIfExists(sharedLibPaths, LIB_NAME "TypeScriptRuntime." LIB_EXT))
            {
                /*
                llvm::WithColor::error(llvm::errs(), "tsc") << "JIT initialization failed. Missing GC library. Did you forget to provide it via "
                                "'--shared-libs=" LIB_NAME "TypeScriptRuntime." LIB_EXT "'? or you can switch it off by using '-nogc'\n";
                return -1;            
                */
            }        

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    engineOptions.enableObjectDump = dumpObjectFile;
    engineOptions.enableGDBNotificationListener = !enableOpt;
    engineOptions.sharedLibPaths = sharedLibPaths;
    if (enableOpt.getValue())
    {
        engineOptions.jitCodeGenOptLevel = (llvm::CodeGenOptLevel) optLevel.getValue();
    }

    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    if (dumpObjectFile)
    {
        auto expectedFPtr = engine->lookup(mainFuncName);
        if (!expectedFPtr)
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << expectedFPtr.takeError();
            return -1;
        }

        llvm::Triple TheTriple;
        std::string targetTriple = llvm::sys::getDefaultTargetTriple();
        if (!TargetTriple.empty())
        {
            targetTriple = llvm::Triple::normalize(TargetTriple);
        }
        
        TheTriple = llvm::Triple(targetTriple);

        engine->dumpToObjectFile(
            objectFilename.empty() 
                ? inputFilename + ((TheTriple.getOS() == llvm::Triple::Win32) ? ".obj" : ".o") 
                : objectFilename);

        return 0;
    }

    if (module.lookupSymbol(MLIR_GCTORS))
    {
        if (auto gctorsResult = engine->invokePacked(MLIR_GCTORS))
        {
            llvm::WithColor::error(llvm::errs(), "tsc") << "JIT calling global constructors failed, error: " << gctorsResult << "\n";
            return -1;
        }
    }

    // Invoke the JIT-compiled function.
    if (auto invocationResult = engine->invokePacked(mainFuncName))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "JIT invocation failed, error: " << invocationResult << "\n";
        return -1;
    }

    return 0;
}
