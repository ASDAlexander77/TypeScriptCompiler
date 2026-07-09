#include "TypeScript/DataStructs.h"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"

#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#endif

#include "llvm/TargetParser/Host.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/ManagedStatic.h"

#include "TypeScript/Defines.h"

#define DEBUG_TYPE "tslang"

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
std::string getTslangLibPath();
std::string mergeWithDefaultLibPath(std::string, std::string);
std::string makeAbsolutePath(std::string);

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
        llvm::WithColor::error(llvm::errs(), "tslang") << "Loading error lib: " << errMsg << "\n";
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

#ifdef _WIN64
// Image base of the JIT'd module (tslang JITs a single module per run), needed by
// the _CxxThrowException shim below. Filled in by JitSectionMemoryManager.
static uint64_t jitImageBase = 0;
#endif

// A SectionMemoryManager that makes JIT'd code behave like AOT'd code:
//
// 1. GC roots. JIT'd globals live in RTDyld-allocated data sections (plain
//    VirtualAlloc/mmap memory), which the Boehm GC does not scan: for AOT binaries
//    the loader-mapped data segment is registered as a root set automatically, but
//    in the JIT an object reachable only from a global (e.g. a static class member)
//    is collected on the first GC cycle and its memory recycled. Register every RW
//    data section via GC_add_roots, resolved dynamically from the already-loaded
//    TypeScriptRuntime library so this stays inert under --nogc.
//
// 2. Win64 unwind info. LLVM's RTDyld never registers .pdata with the OS
//    (RTDyldMemoryManager::registerEHFramesInProcess only speaks the Itanium
//    __register_frame protocol), so a C++ exception thrown from JIT'd code aborts:
//    RtlUnwindEx finds no RUNTIME_FUNCTION for JIT PCs and never reaches the catch
//    funclet. Register every .pdata section RTDyld hands us with
//    RtlAddFunctionTable, using the same "image base" RTDyld resolves
//    IMAGE_REL_AMD64_ADDR32NB relocations against: the lowest section load address
//    (see RuntimeDyldCOFFX86_64::getImageBase).
//
// 3. Executable read-only data (Win64). For external ADDR32NB targets — e.g. the
//    __CxxFrameHandler3 personality referenced from .xdata — RTDyld materializes a
//    jump thunk in the *referencing* section's stub area, and the OS exception
//    dispatcher calls through it. Read-only sections therefore must be mapped
//    executable; route them through the code allocator.
class JitSectionMemoryManager : public llvm::SectionMemoryManager
{
    using GCRootsFn = void (*)(void *, void *);

  public:
    uint8_t *allocateCodeSection(uintptr_t size, unsigned alignment, unsigned sectionID,
                                 llvm::StringRef sectionName) override
    {
        auto *addr = llvm::SectionMemoryManager::allocateCodeSection(size, alignment, sectionID, sectionName);
        noteSectionAddress(addr);
        return addr;
    }

    uint8_t *allocateDataSection(uintptr_t size, unsigned alignment, unsigned sectionID,
                                 llvm::StringRef sectionName, bool isReadOnly) override
    {
#ifdef _WIN64
        if (isReadOnly)
        {
            // see (3): unwind personality thunks live in read-only sections
            return allocateCodeSection(size, alignment, sectionID, sectionName);
        }
#endif

        auto *addr = llvm::SectionMemoryManager::allocateDataSection(size, alignment, sectionID, sectionName, isReadOnly);
        noteSectionAddress(addr);
        if (!isReadOnly && addr != nullptr)
        {
            if (auto addRoots = reinterpret_cast<GCRootsFn>(
                    llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("GC_add_roots")))
            {
                addRoots(addr, addr + size);
                gcRootSections.push_back({addr, size});
            }
        }

        return addr;
    }

#ifdef _WIN64
    void registerEHFrames(uint8_t *addr, uint64_t loadAddr, size_t size) override
    {
        auto entryCount = size / sizeof(RUNTIME_FUNCTION);
        if (entryCount == 0 || imageBase == 0)
        {
            return;
        }

        if (RtlAddFunctionTable(reinterpret_cast<PRUNTIME_FUNCTION>(addr), static_cast<DWORD>(entryCount), imageBase))
        {
            functionTables.push_back(reinterpret_cast<PRUNTIME_FUNCTION>(addr));
        }
    }

    void deregisterEHFrames() override
    {
        for (auto *table : functionTables)
        {
            RtlDeleteFunctionTable(table);
        }

        functionTables.clear();
    }
#endif

    ~JitSectionMemoryManager() override
    {
        if (auto removeRoots = reinterpret_cast<GCRootsFn>(
                llvm::sys::DynamicLibrary::SearchForAddressOfSymbol("GC_remove_roots")))
        {
            for (auto &section : gcRootSections)
            {
                removeRoots(section.first, section.first + section.second);
            }
        }
    }

  private:
    void noteSectionAddress(uint8_t *addr)
    {
        if (addr == nullptr)
        {
            return;
        }

        if (imageBase == 0 || reinterpret_cast<uint64_t>(addr) < imageBase)
        {
            imageBase = reinterpret_cast<uint64_t>(addr);
#ifdef _WIN64
            jitImageBase = imageBase;
#endif
        }
    }

    uint64_t imageBase = 0;
    mlir::SmallVector<std::pair<uint8_t *, uintptr_t>> gcRootSections;
#ifdef _WIN64
    mlir::SmallVector<PRUNTIME_FUNCTION> functionTables;
#endif
};

#ifdef _WIN64
// MSVC x64 C++ EH encodes throw-site type information as image-relative offsets.
// vcruntime's _CxxThrowException recovers the base with RtlPcToFileHeader on the
// ThrowInfo pointer, which fails for JIT'd memory (it is not a loader-mapped
// image), so __CxxFrameHandler3 would resolve the throw-side RVAs against a null
// base and never match a catch clause. Raise the exception ourselves, substituting
// the image base RTDyld resolved those RVAs against.
static void jitCxxThrowException(void *exceptionObject, void *throwInfo)
{
    constexpr DWORD cxxExceptionCode = 0xE06D7363;   // 'msc' | 0xE0000000
    constexpr ULONG_PTR cxxMagicNumber = 0x19930520; // EH_MAGIC_NUMBER1

    // ThrowInfo from an AOT module (e.g. a runtime DLL) still resolves normally
    void *moduleBase = nullptr;
    if (throwInfo != nullptr)
    {
        RtlPcToFileHeader(throwInfo, &moduleBase);
    }

    ULONG_PTR args[] = {cxxMagicNumber, reinterpret_cast<ULONG_PTR>(exceptionObject),
                        reinterpret_cast<ULONG_PTR>(throwInfo),
                        moduleBase != nullptr ? reinterpret_cast<ULONG_PTR>(moduleBase)
                                              : static_cast<ULONG_PTR>(jitImageBase)};
    RaiseException(cxxExceptionCode, EXCEPTION_NONCONTINUABLE, 4, args);
}

// from the statically linked vcruntime; bound into the JIT'd module as its
// exception personality so throw and catch sides use the same CRT
extern "C" EXCEPTION_DISPOSITION __CxxFrameHandler3(struct _EXCEPTION_RECORD *, void *, struct _CONTEXT *,
                                                    struct _DISPATCHER_CONTEXT *);
#endif

int runJit(int argc, char **argv, mlir::ModuleOp module, CompileOptions &compileOptions)
{
    // to avoid false positive memory leak reports in release builds
    // Print a stack trace if we signal out.
    llvm::sys::PrintStackTraceOnErrorSignal(argv[0]);

    llvm::PrettyStackTraceProgram X(argc, argv);
    llvm::setBugReportMsg("PLEASE submit a bug report to https://github.com/ASDAlexander77/TypeScriptCompiler/issues and include the crash backtrace.");

    llvm::llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

    // No leaks until here

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
        // per-build subfolder (debug/release) must match the JIT compilation mode.
        // Keyed on --di (generate debug info): with debug info use the debug lib.
        auto defaultLibBuildDir = compileOptions.generateDebugInfo ? DEFAULT_LIB_BUILD_DIR_DEBUG : DEFAULT_LIB_BUILD_DIR_RELEASE;
        clSharedLibs.push_back(mergeWithDefaultLibPath(getDefaultLibPath(),
#ifdef WIN32
            std::string(DEFAULT_LIB_DIR "/dll/") + defaultLibBuildDir + "/" DEFAULT_LIB_NAME ".dll"
#else
            std::string(DEFAULT_LIB_DIR "/dll/") + defaultLibBuildDir + "/lib" DEFAULT_LIB_NAME ".so"
#endif
        ));
    }      

    // add default libs in case they are not part of options
#ifdef WIN32
#define LIB_NAME ""
#define LIB_EXT "dll"
#else
#define LIB_NAME "lib"
#define LIB_EXT "so"
#endif
    // Only locate a GC runtime when the user has not already supplied one via
    // --shared-libs: loading a second TypeScriptRuntime copy from a different path
    // (e.g. TSLANG_LIB_PATH) creates two independent GC instances, so roots
    // registered in one are invisible to collections running in the other and
    // GC-heap pointers cross between the two heaps.
    auto hasTypeScriptRuntime = llvm::any_of(clSharedLibs, [](const std::string &libPath) {
        return llvm::sys::path::stem(libPath).contains_insensitive("TypeScriptRuntime");
    });

    std::string pathTypeScriptLib("../lib/" LIB_NAME "TypeScriptRuntime." LIB_EXT);
    if (!disableGC.getValue() && !hasTypeScriptRuntime)
    {
        auto absPath3 = makeAbsolutePath(mergeWithDefaultLibPath(getTslangLibPath(), LIB_NAME "TypeScriptRuntime." LIB_EXT));
        if (absPath3.empty())
        {
            auto absPath = makeAbsolutePath(pathTypeScriptLib);
            if (absPath.empty())
            {            
                pathTypeScriptLib = LIB_NAME "TypeScriptRuntime." LIB_EXT;
                auto absPath2 = makeAbsolutePath(pathTypeScriptLib);
                if (absPath2.empty())
                {
                    /*
                    llvm::WithColor::error(llvm::errs(), "tslang") << "JIT initialization failed. Missing GC library. Did you forget to provide it via "
                                    "'--shared-libs=" LIB_NAME "TypeScriptRuntime." LIB_EXT "'? or you can switch it off by using '-nogc'\n";
                    return -1;            
                    */
                }        
                else
                {
                    clSharedLibs.push_back(absPath2);
                }
            }
            else
            {
                clSharedLibs.push_back(absPath);
            }
        }     
        else
        {
            clSharedLibs.push_back(absPath3);
        }
    }

#ifdef _WIN32
    // tslang.exe links the CRT statically (/MT[d]), so its libc symbols are not in
    // any DLL export table. The JIT's process-symbol resolver would otherwise bind
    // libc calls (puts/printf/malloc/...) to a *different* CRT instance loaded as a
    // DLL (ucrtbase.dll), giving JIT'd code a separate stdout buffer and heap from
    // tslang.exe. That mismatch loses output and corrupts state on teardown.
    //
    // Add our own CRT entry points to the process symbol table *before* creating
    // the engine: the JIT's GetForCurrentProcess generator consults this table
    // (SearchForAddressOfSymbol checks explicitly-added symbols first) only for
    // unresolved symbols, so it overrides ucrtbase without an ORC JITDylib
    // "duplicate definition" conflict, and it is in place before create() eagerly
    // materializes any global constructors that reference these symbols.
    {
        auto addSym = [](const char *name, void *addr) {
            llvm::sys::DynamicLibrary::AddSymbol(name, addr);
        };
        addSym("puts", (void*)&puts);
        addSym("printf", (void*)&printf);
        addSym("malloc", (void*)&malloc);
        addSym("free", (void*)&free);
        addSym("realloc", (void*)&realloc);
        addSym("calloc", (void*)&calloc);
        addSym("memset", (void*)&memset);
        addSym("memcpy", (void*)&memcpy);
#ifdef _WIN64
        // C++ EH: bind the JIT'd module's personality to our static CRT and route
        // throws through the shim that fixes up the throw-site image base (see
        // jitCxxThrowException above).
        addSym("__CxxFrameHandler3", (void *)&__CxxFrameHandler3);
        addSym("_CxxThrowException", (void *)&jitCxxThrowException);
#endif
    }
#endif

    if (dumpObjectFile)
    {
        // Compile-and-dump only, no execution: the stock MLIR engine is enough.
        mlir::SmallVector<mlir::StringRef> sharedLibPaths;
        sharedLibPaths.append(begin(clSharedLibs), end(clSharedLibs));

        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = optPipeline;
        engineOptions.enableObjectDump = true;
        engineOptions.enableGDBNotificationListener = !enableOpt;
        engineOptions.sharedLibPaths = sharedLibPaths;
        if (enableOpt.getValue())
        {
            engineOptions.jitCodeGenOptLevel = (llvm::CodeGenOptLevel) optLevel.getValue();
        }

        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        if (!maybeEngine)
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "failed to construct an execution engine, error: " << maybeEngine.takeError() << "\n";
            return -1;
        }
        auto &engine = maybeEngine.get();

        auto expectedFPtr = engine->lookup(mainFuncName);
        if (!expectedFPtr)
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << expectedFPtr.takeError();
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

    // Run path: build our own LLJIT instead of mlir::ExecutionEngine — the stock
    // engine hard-codes a plain SectionMemoryManager, which neither registers GC
    // roots for JIT'd globals nor Win64 unwind info (see JitSectionMemoryManager).

    // Load the shared libraries into the process; the JIT resolves external
    // symbols from their export tables via the current-process generator below.
    for (auto &libPathStr : clSharedLibs)
    {
        std::string errMsg;
        llvm::sys::DynamicLibrary::getPermanentLibrary(libPathStr.c_str(), &errMsg);
        if (!errMsg.empty())
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "Loading error lib: " << libPathStr << " error: " << errMsg << "\n";
            return -1;
        }
    }

    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmContext);
    if (!llvmModule)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "failed to emit LLVM IR\n";
        return -1;
    }

    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "failed to create a JITTargetMachineBuilder for the host, error: " << tmBuilderOrError.takeError() << "\n";
        return -1;
    }

    if (enableOpt.getValue())
    {
        tmBuilderOrError->setCodeGenOptLevel((llvm::CodeGenOptLevel) optLevel.getValue());
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "failed to create a TargetMachine for the host, error: " << tmOrError.takeError() << "\n";
        return -1;
    }

    llvmModule->setDataLayout((*tmOrError)->createDataLayout());
    llvmModule->setTargetTriple((*tmOrError)->getTargetTriple());

    if (auto err = optPipeline(llvmModule.get()))
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "failed to optimize LLVM IR, error: " << std::move(err) << "\n";
        return -1;
    }

    auto maybeJit =
        llvm::orc::LLJITBuilder()
            .setJITTargetMachineBuilder(std::move(*tmBuilderOrError))
            .setDataLayout(llvmModule->getDataLayout())
            .setObjectLinkingLayerCreator(
                [targetTriple = llvmModule->getTargetTriple()](llvm::orc::ExecutionSession &session)
                    -> llvm::Expected<std::unique_ptr<llvm::orc::ObjectLayer>> {
                    auto objectLayer = std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(
                        session, [](const llvm::MemoryBuffer &) { return std::make_unique<JitSectionMemoryManager>(); });

                    if (!enableOpt.getValue())
                    {
                        if (auto *gdbListener = llvm::JITEventListener::createGDBRegistrationListener())
                        {
                            objectLayer->registerJITEventListener(*gdbListener);
                        }
                    }

                    // COFF format binaries (Windows) need special handling to deal
                    // with exported symbol visibility (cf mlir::ExecutionEngine)
                    if (targetTriple.isOSBinFormatCOFF())
                    {
                        objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
                        objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
                    }

                    return objectLayer;
                })
            .create();
    if (!maybeJit)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "failed to construct the JIT engine, error: " << maybeJit.takeError() << "\n";
        return -1;
    }

    auto &jit = maybeJit.get();

    // Resolve symbols from the current process, including the loaded shared
    // libraries and the AddSymbol overrides above.
    auto generator = llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit->getDataLayout().getGlobalPrefix());
    if (!generator)
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "failed to create a process symbol generator, error: " << generator.takeError() << "\n";
        return -1;
    }

    jit->getMainJITDylib().addGenerator(std::move(*generator));

#ifdef _WIN32
    // Bind CRT entry points to tslang.exe's static CRT (/MT[d]) explicitly: the
    // process generator above resolves from DLL export tables only, so without
    // these definitions the JIT'd code would bind to ucrtbase.dll — a different
    // CRT instance with its own stdout buffers and heap. JITDylib definitions
    // take precedence over generators, making the binding deterministic.
    {
        llvm::orc::MangleAndInterner interner(jit->getExecutionSession(), jit->getDataLayout());
        llvm::orc::SymbolMap crtOverrides;
        auto addOverride = [&](const char *name, void *addr) {
            crtOverrides[interner(name)] = {llvm::orc::ExecutorAddr::fromPtr(addr), llvm::JITSymbolFlags::Exported};
        };
        addOverride("puts", (void *)&puts);
        addOverride("printf", (void *)&printf);
        addOverride("malloc", (void *)&malloc);
        addOverride("free", (void *)&free);
        addOverride("realloc", (void *)&realloc);
        addOverride("calloc", (void *)&calloc);
        addOverride("memset", (void *)&memset);
        addOverride("memcpy", (void *)&memcpy);
#ifdef _WIN64
        // C++ EH: same-CRT personality, and throws routed through the shim that
        // fixes up the throw-site image base (see jitCxxThrowException above)
        addOverride("__CxxFrameHandler3", (void *)&__CxxFrameHandler3);
        addOverride("_CxxThrowException", (void *)&jitCxxThrowException);
#endif
        if (auto err = jit->getMainJITDylib().define(llvm::orc::absoluteSymbols(std::move(crtOverrides))))
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "failed to define CRT overrides, error: " << std::move(err) << "\n";
            return -1;
        }
    }
#endif

    if (auto err = jit->addIRModule(llvm::orc::ThreadSafeModule(std::move(llvmModule), std::move(llvmContext))))
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "failed to add the module to the JIT engine, error: " << std::move(err) << "\n";
        return -1;
    }

    // run platform initializers (llvm.global_ctors etc.)
    if (auto err = jit->initialize(jit->getMainJITDylib()))
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "JIT initialization failed, error: " << std::move(err) << "\n";
        return -1;
    }

    auto invoke = [&](llvm::StringRef name) {
        auto sym = jit->lookup(name);
        if (!sym)
        {
            llvm::WithColor::error(llvm::errs(), "tslang") << "JIT invocation failed, error: " << sym.takeError() << "\n";
            return -1;
        }

        sym->toPtr<void (*)()>()();
        return 0;
    };

    if (module.lookupSymbol(MLIR_GCTORS) && invoke(MLIR_GCTORS) != 0)
    {
        return -1;
    }

    if (invoke(mainFuncName.getValue()) != 0)
    {
        return -1;
    }

    // The JIT program has finished. On Windows/COFF the MLIR/ORC LLJIT platform
    // registers process-level atexit glue that faults during teardown (exception
    // 0x80000003), so the normal CRT exit path crashes after a successful run.
    // ExitProcess is not safe either: it runs DLL_PROCESS_DETACH, and
    // TypeScriptRuntime.dll's static AsyncRuntime destructor then waits on its
    // thread pool whose worker threads ExitProcess has already terminated,
    // deadlocking forever when async work is still pending. TerminateProcess
    // skips all teardown, so nothing can crash or deadlock — but it also skips
    // the detach-time stdio flush, so flush every CRT the JIT'd code may have
    // bound to (ucrtbase buffers its streams separately from our static CRT).
    fflush(stdout);
    fflush(stderr);
#ifdef _WIN32
    #if _DEBUG
    for (auto crtName : {"ucrtbase.dll", "ucrtbased.dll"})
    {
        if (HMODULE crt = GetModuleHandleA(crtName))
        {
            if (auto crtFflush = (int(__cdecl *)(FILE *))GetProcAddress(crt, "fflush"))
            {
                crtFflush(nullptr); // fflush(NULL) flushes all of that CRT's output streams
            }
        }
    }
    #endif

    // Must stay unconditional: without it release builds crash (0xC0000005) or
    // deadlock in teardown after async tests — the AsyncRuntime static destructor
    // waits on thread-pool workers during DLL_PROCESS_DETACH (see above).
    TerminateProcess(GetCurrentProcess(), 0);
#endif
    return 0;
}
