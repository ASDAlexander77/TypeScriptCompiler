#include "clang/Driver/Driver.h"
#include "TypeScript/TypeScriptLang/TextDiagnosticPrinter.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "clang/Options/Options.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/WithColor.h"
#include "llvm/CodeGen/CommandFlags.h"

#include "TypeScript/DataStructs.h"
#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/Defines.h"

namespace cl = llvm::cl;

extern cl::opt<enum Action> emitAction;
extern cl::opt<std::string> outputFilename;
extern cl::opt<std::string> TargetTriple;
extern cl::opt<std::string> defaultlibpath;
extern cl::opt<std::string> gclibpath;
extern cl::opt<std::string> gcsharedlibpath;

// From TypeScript/ObjDumper.h, declared here as jit.cpp does, where that header's
// llvm/BinaryFormat/COFF.h collides with <windows.h> macros. This file does not include
// <windows.h>, so it can take the COFF machine constants from llvm/BinaryFormat/COFF.h itself.
namespace Dump
{
    bool containsGarbageCollector(llvm::StringRef);
    uint16_t coffMachine(llvm::StringRef);
}
extern cl::opt<std::string> llvmlibpath;
extern cl::opt<std::string> tslanglibpath;
extern cl::opt<std::string> emsdksysrootpath;
extern cl::opt<bool> enableOpt;
extern cl::list<std::string> libs;
extern cl::list<std::string> objs;
extern cl::opt<bool> verbose;

std::string getDefaultOutputFileName(enum Action);
std::string mergeWithDefaultLibPath(std::string, std::string);

using llvm::StringRef;

std::string getExecutablePath(const char *argv0)
{
    // This just needs to be some symbol in the binary
    void *p = (void *)(intptr_t)getExecutablePath;
    return llvm::sys::fs::getMainExecutable(argv0, p);
}

// This lets us create the DiagnosticsEngine with a properly-filled-out
// DiagnosticOptions instance
static clang::DiagnosticOptions *
createAndPopulateDiagOpts(llvm::ArrayRef<const char *> argv)
{
    auto *diagOpts = new clang::DiagnosticOptions;

    // Ignore missingArgCount and the return value of ParseDiagnosticArgs.
    // Any errors that would be diagnosed here will also be diagnosed later,
    // when the DiagnosticsEngine actually exists.
    unsigned missingArgIndex, missingArgCount;
    llvm::opt::InputArgList args = clang::getDriverOptTable().ParseArgs(
        argv.slice(1), missingArgIndex, missingArgCount,
        /*FlagsToInclude=*/clang::options::FlangOption);

    // parse args

    return diagOpts;
}

static void ExpandResponseFiles(llvm::StringSaver &saver,
                                llvm::SmallVectorImpl<const char *> &args)
{
    // We're defaulting to the GNU syntax, since we don't have a CL mode.
    llvm::cl::TokenizerCallback tokenizer = &llvm::cl::TokenizeGNUCommandLine;
    llvm::cl::ExpansionContext ExpCtx(saver.getAllocator(), tokenizer);
    if (llvm::Error Err = ExpCtx.expandResponseFiles(args))
    {
        llvm::errs() << toString(std::move(Err)) << '\n';
    }
}

std::string getDefaultLibPath()
{
    if (!defaultlibpath.empty())
    {
        return defaultlibpath;
    }

    if (auto gcDefaultLibEnvValue = llvm::sys::Process::GetEnv("DEFAULT_LIB_PATH")) 
    {
        return gcDefaultLibEnvValue.value();
    }    

    return "";    
}

bool checkFileExistsAtPath(std::string path, std::string fileName)
{
    llvm::SmallVector<char> destPath(0);
    destPath.reserve(256);
    destPath.append(path.begin(), path.end());

    llvm::sys::path::append(destPath, fileName);

    if (!llvm::sys::fs::exists(destPath))
    {
        llvm::WithColor::error(llvm::errs(), "tslang") << "path: '" << path << "' is not pointing to file '" << fileName << "'\n";        
        return false;
    }    

    return true;
}

// Windows x86 takes its libraries from an `x86` subdirectory of each configured lib path, so the
// flat path has none of them by design; buildExe checks the subdirectory instead (see
// resolveWindowsLibPath). Every other target keeps the flat layout.
static bool isWindowsX86Target()
{
    llvm::Triple triple(TargetTriple.empty() ? llvm::sys::getDefaultTargetTriple() : llvm::Triple::normalize(TargetTriple));
    return triple.getOS() == llvm::Triple::Win32 && triple.getArch() == llvm::Triple::x86;
}

// The directory a lib path's libraries are in: its `x86` subdirectory for Windows x86, else itself.
static std::string getTargetLibDir(llvm::StringRef path, bool windowsX86)
{
    llvm::SmallString<256> dir(path);
    if (windowsX86)
    {
        llvm::sys::path::append(dir, "x86");
    }

    return dir.str().str();
}

void checkGCLibPath(std::string path)
{
    if (isWindowsX86Target())
    {
        return;
    }

#ifdef WIN32
    const auto libName = "gc.lib";
#else    
    const auto libName = "libgc.a";
#endif    
    checkFileExistsAtPath(path, libName);
}

void checkTslangLibPath(std::string path) 
{
    if (isWindowsX86Target())
    {
        return;
    }

#ifdef WIN32
    const auto libName = "TypeScriptAsyncRuntime.lib";
#else    
    const auto libName = "libTypeScriptAsyncRuntime.a";
#endif    
    checkFileExistsAtPath(path, libName);
}

std::string getGCLibPath()
{
    if (!gclibpath.empty())
    {
        checkGCLibPath(gclibpath);
        return gclibpath;
    }

    if (auto gcLibEnvValue = llvm::sys::Process::GetEnv("GC_LIB_PATH")) 
    {
        checkGCLibPath(gcLibEnvValue.value());
        return gcLibEnvValue.value();
    }    

    return "";    
}

// Boehm built as a DLL: a directory holding its import library `gc.lib`, with `gc.dll` beside it
// (the release package's gcdll folder) or in `../bin` (a CMake install).
static std::string findGCDll(llvm::StringRef libDir)
{
    llvm::SmallString<256> beside(libDir);
    llvm::sys::path::append(beside, "gc.dll");
    if (llvm::sys::fs::exists(beside))
    {
        return beside.str().str();
    }

    llvm::SmallString<256> bin(libDir);
    llvm::sys::path::append(bin, "..", "bin", "gc.dll");
    if (llvm::sys::fs::exists(bin))
    {
        llvm::sys::path::remove_dots(bin, /*remove_dot_dot=*/true);
        return bin.str().str();
    }

    return "";
}

static bool isSharedGCLibDir(llvm::StringRef libDir)
{
    auto targetLibDir = getTargetLibDir(libDir, isWindowsX86Target());
    llvm::SmallString<256> lib(targetLibDir);
    llvm::sys::path::append(lib, "gc.lib");
    return llvm::sys::fs::exists(lib) && !findGCDll(targetLibDir).empty();
}

// Where the shared collector is, or "" when there is none to be found. Not a fallback to the
// static gc.lib: that is exactly the build this exists to avoid, and it links fine and then frees
// live objects, which is far harder to find than a link that stops here.
std::string getGCSharedLibPath()
{
    if (!gcsharedlibpath.empty())
    {
        checkGCLibPath(gcsharedlibpath);
        return gcsharedlibpath;
    }

    if (auto gcSharedLibEnvValue = llvm::sys::Process::GetEnv("GC_SHARED_LIB_PATH"))
    {
        if (!gcSharedLibEnvValue->empty())
        {
            checkGCLibPath(gcSharedLibEnvValue.value());
            return gcSharedLibEnvValue.value();
        }
    }

    std::string staticPath = gclibpath;
    if (staticPath.empty())
    {
        staticPath = llvm::sys::Process::GetEnv("GC_LIB_PATH").value_or("");
    }

    if (!staticPath.empty())
    {
        // the release package: the shared build in a gcdll folder beside the static gc.lib
        llvm::SmallString<256> gcdll(staticPath);
        llvm::sys::path::append(gcdll, "gcdll");
        if (isSharedGCLibDir(gcdll))
        {
            return gcdll.str().str();
        }

        // --gc-lib-path already names a shared build
        if (isSharedGCLibDir(staticPath))
        {
            return staticPath;
        }
    }

    return "";
}

std::string getLLVMLibPath()
{
    if (!llvmlibpath.empty())
    {
        return llvmlibpath;
    }

    if (auto llvmLibEnvValue = llvm::sys::Process::GetEnv("LLVM_LIB_PATH")) 
    {
        return llvmLibEnvValue.value();
    }    

    return "";    
}

std::string getTslangLibPath()
{
    if (!tslanglibpath.empty())
    {
        checkTslangLibPath(tslanglibpath);
        return tslanglibpath;
    }

    if (auto tslangLibEnvValue = llvm::sys::Process::GetEnv("TSLANG_LIB_PATH")) 
    {
        checkTslangLibPath(tslangLibEnvValue.value());
        return tslangLibEnvValue.value();
    }   

    return "";    
}

std::string getEMSDKSysRootPath()
{
    if (!emsdksysrootpath.empty())
    {
        return emsdksysrootpath;
    }

    if (auto emsdksysrootpathEnvValue = llvm::sys::Process::GetEnv("EMSDK_SYSROOT_PATH")) 
    {
        return emsdksysrootpathEnvValue.value();
    }   

    return "";    
}

std::string concatIfNotEmpty(const char *prefix, std::string path)
{
    return path.empty() ? path : prefix + path;
}

std::string getLibsPathOpt(std::string path)
{
    return concatIfNotEmpty("-L", path);
}

std::string getLibOpt(std::string path)
{
    return concatIfNotEmpty("-l", path);
}

static std::string getCOFFMachineName(uint16_t machine)
{
    switch (machine)
    {
    case llvm::COFF::IMAGE_FILE_MACHINE_I386:
        return "x86";
    case llvm::COFF::IMAGE_FILE_MACHINE_AMD64:
        return "x64";
    case llvm::COFF::IMAGE_FILE_MACHINE_ARM64:
        return "arm64";
    default:
        return "0x" + llvm::utohexstr(machine);
    }
}

// Windows x86 and x64: turns `path` into the directory the link takes `libName` from - its `x86`
// subdirectory for x86, the path itself for x64 - and refuses a library built for another
// machine. The linker only warns about one (LNK4272) and then fails on every symbol it was to
// supply, which names neither the library nor the fix. Not a fallback to the flat path for x86:
// the libraries there are x64.
static bool resolveWindowsLibPath(const llvm::Triple &triple, std::string &path, llvm::StringRef libName,
                                  llvm::StringRef component, llvm::StringRef flag, llvm::StringRef buildScript)
{
    if (path.empty())
    {
        return true;
    }

    auto x86 = triple.getArch() == llvm::Triple::x86;
    auto libDir = getTargetLibDir(path, x86);
    llvm::SmallString<256> lib(libDir);
    llvm::sys::path::append(lib, libName);

    if (x86)
    {
        if (!llvm::sys::fs::exists(lib))
        {
            llvm::WithColor::error(llvm::errs(), "tslang")
                << "no x86 build of " << component << " in " << libDir << ". Build it with scripts\\" << buildScript
                << ".bat, or point " << flag << " at a directory holding an x86 subdirectory.\n";
            return false;
        }

        path = libDir;
    }

    auto expected = x86 ? llvm::COFF::IMAGE_FILE_MACHINE_I386 : llvm::COFF::IMAGE_FILE_MACHINE_AMD64;
    auto machine = Dump::coffMachine(lib);
    if (machine != 0 && machine != expected)
    {
        llvm::WithColor::error(llvm::errs(), "tslang")
            << lib << " is built for " << getCOFFMachineName(machine) << ", but this program targets "
            << getCOFFMachineName(expected) << ".\n";
        return false;
    }

    return true;
}

void addCommandArgs(clang::driver::Compilation *c, llvm::ArrayRef<const char*> cmdParts)
{
    for (auto &job : c->getJobs())
    {
        llvm::opt::ArgStringList newArgs;
        for (auto arg : job.getArguments())
        {
            newArgs.push_back(arg);
        }

        for (auto newCmdArg : cmdParts)
        {
            newArgs.push_back(newCmdArg);
        }

        job.replaceArguments(newArgs);
        break;
    }
}

void removeCommandArgs(clang::driver::Compilation *c, llvm::ArrayRef<const char*> cmdParts)
{
    for (auto &job : c->getJobs())
    {
        auto replace = false;
        llvm::opt::ArgStringList newArgs;
        for (auto arg : job.getArguments())
        {
            StringRef argStr(arg);
            if (llvm::any_of(cmdParts, [argStr](auto &cmdPart) { return argStr.contains(cmdPart); }))
            {
                replace = true;
                continue;
            }
            newArgs.push_back(arg);
        }

        if (replace)
        {
            job.replaceArguments(newArgs);
        }
    }
}

int buildExe(int argc, char **argv, std::string objFileName, std::string additionalObjFileName, CompileOptions &compileOptions)
{
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);
    //llvm::SmallVector<const char *, 256> args(argv, argv + argc);
    llvm::SmallVector<const char *, 256> args(argv, argv + 1);    

    clang::driver::ParsedClangName targetAndMode("tslang", "--driver-mode=tslang");
    std::string driverPath = getExecutablePath(args[0]);

    llvm::BumpPtrAllocator a;
    llvm::StringSaver saver(a);
    ExpandResponseFiles(saver, args);

    // Check if tslang-new is in the frontend mode
    auto firstArg = std::find_if(args.begin() + 1, args.end(),
                                 [](const char *a)
                                 { return a != nullptr; });
    if (firstArg != args.end())
    {
        if (llvm::StringRef(args[1]).starts_with("-cc1"))
        {
            llvm::errs() << "error: unknown integrated tool '" << args[1] << "'. "
                         << "Valid tools include '-tslang'.\n";
            return 1;
        }

        // Call tslang
        // ...
    }

    llvm::Triple TheTriple;
    std::string targetTriple = llvm::sys::getDefaultTargetTriple();
    if (!TargetTriple.empty())
    {
        targetTriple = llvm::Triple::normalize(TargetTriple);
    }

    TheTriple = llvm::Triple(targetTriple);

    // Specify Visual Studio C runtime library. “static” and “static_dbg” correspond to the cl flags /MT and /MTd which use the multithread, 
    // static version. “dll” and “dll_dbg” correspond to the cl flags /MD and /MDd which use the multithread, dll version. <arg> must be ‘static’, ‘static_dbg’, ‘dll’ or ‘dll_dbg’.    
    //args.insert(args.begin() + 1, "-nodefaultlibs");
    //args.insert(args.begin() + 1, "-fms-omit-default-lib=dll");
    //args.insert(args.begin() + 1, "-fms-runtime-lib=static_dbg");

    std::string gcLibPathOpt;
    std::string tslangLibPathOpt;
    std::string emsdkSysRootPathOpt;
    std::string defaultLibPathOpt;
    std::string defaultLibFileOpt;

    auto isTslangLibNeeded = true;

    auto os = TheTriple.getOS();
    auto arch = TheTriple.getArch();
    auto win = os == llvm::Triple::Win32;
    auto wasm = arch == llvm::Triple::wasm32 || arch == llvm::Triple::wasm64;
    auto emscripten = os == llvm::Triple::Emscripten;
    auto shared = emitAction == BuildDll;
    // Windows x86 and x64 link libraries whose machine tslang checks; see resolveWindowsLibPath.
    auto winX86OrX64 = win && (arch == llvm::Triple::x86 || arch == llvm::Triple::x86_64);
    // the same debug/release choice as the CRT below
    std::string libConfig = enableOpt ? "release" : "debug";
    
    std::optional<llvm::Reloc::Model> RM = llvm::codegen::getExplicitRelocModel();

    if (wasm)
    {
        isTslangLibNeeded = false;        
    }

    args.push_back(objFileName.c_str());
    llvm::SmallVector<std::string> objOpts;
    for (auto obj : objs)
    {
        objOpts.push_back(obj);
    }

    for (auto &obj : objOpts)
    {
        args.push_back(obj.c_str());
    }

    if (!additionalObjFileName.empty())
    {
        args.push_back(additionalObjFileName.c_str());
    }    

    if (outputFilename.empty())
    {
        outputFilename = getDefaultOutputFileName(emitAction);
    }

    std::string resultFile = "-o" + outputFilename;
    args.push_back(resultFile.c_str());
    if (shared)
    {
        args.push_back("-shared");
        if (!win)
        {
            // added search path
            args.push_back("-Wl,-rpath=.");
        }
    }

    // add extra libs
    llvm::SmallVector<std::string> libOpts;
    for (auto lib : libs)
    {
        auto libPathOpt = getLibOpt(lib);
        if (!libPathOpt.empty())
        {
            libOpts.push_back(libPathOpt);
        }
    }

    for (auto &lib : libOpts)
    {
        args.push_back(lib.c_str());
    }

    if (!compileOptions.noDefaultLib)
    {
        // default lib file
        args.push_back("-l" DEFAULT_LIB_NAME);    

        // default lib path (per-build subfolder: debug/release must match how
        // this program is being compiled so the CRT and default-lib binaries agree).
        // Keyed on --di (generate debug info): with debug info use the debug lib.
        // ...and per memory model: the default lib allocates the way the model it was built for
        // allocates, so a `gc` build linked into an `-mm=rc` program would drag Boehm in and hand
        // back objects this program's ownership rules do not describe. See getDefaultLibSubDir.
        auto defaultLibSubDir = getDefaultLibSubDir(shared, compileOptions.generateDebugInfo,
                                                    memoryModelName(compileOptions.memoryModel));
        auto defaultLibDir = mergeWithDefaultLibPath(getDefaultLibPath(), defaultLibSubDir);

        // Checked here rather than left to the linker. mergeWithDefaultLibPath only joins the
        // path, so a model that has not been built reaches lld as a `-L` to nowhere and comes
        // back as "cannot open input file 'TypeScriptDefaultLib.lib'", which says nothing about
        // which model is missing or how to get it. Deliberately not a fallback to another
        // model's build either: the wrong one links and then misbehaves at run time, which is
        // far harder to diagnose than a directory that is not there.
        if (!defaultLibDir.empty() && !llvm::sys::fs::is_directory(defaultLibDir))
        {
            llvm::errs() << "error: no default library built for -mm="
                         << memoryModelName(compileOptions.memoryModel) << ": " << defaultLibDir
                         << " does not exist. Build it (see the default-lib build scripts), "
                         << "or compile with --no-default-lib.\n";
            return 1;
        }

        // A shared library links the default library's DLL and shares a process with it. If that
        // DLL predates linking gc.dll it brings a second collector, which frees what the library
        // and its host hold - the case step 4's import check cannot see, since the library does
        // not import it by name. See docs/single-gc-collector-design.md.
        if (win && shared && compileOptions.needsGCRuntime() && !defaultLibDir.empty())
        {
            llvm::SmallString<256> defaultLibDll(defaultLibDir);
            llvm::sys::path::append(defaultLibDll, DEFAULT_LIB_NAME ".dll");
            if (llvm::sys::fs::exists(defaultLibDll) && Dump::containsGarbageCollector(defaultLibDll))
            {
                llvm::WithColor::error(llvm::errs(), "tslang")
                    << defaultLibDll << " links its own garbage collector (the static gc.lib), so a shared library "
                    << "linked with it would run two collectors in one process. Rebuild the default library: its "
                    << "DLL has to link gc.dll.\n";
                return 1;
            }
        }

        defaultLibPathOpt = getLibsPathOpt(defaultLibDir);
        if (!defaultLibPathOpt.empty())
        {
            args.push_back(defaultLibPathOpt.c_str());
        }
    }    

    // Which Boehm. A shared library, and a program that loads one, share a process with other gc
    // code, so they take the collector from gc.dll: linked statically, each binary brings a
    // collector of its own, and one frees objects only the other's memory references. A program
    // that is alone keeps the static gc.lib and ships as one file. Windows only for now - Linux has
    // not been measured. See docs/single-gc-collector-design.md.
    auto useSharedGC = win && compileOptions.needsGCRuntime() && (shared || compileOptions.importsSharedLibrary);
    std::string gcSharedLibPath;
    std::string gcDllPath;
    if (useSharedGC)
    {
        gcSharedLibPath = getGCSharedLibPath();
        if (gcSharedLibPath.empty())
        {
            llvm::WithColor::error(llvm::errs(), "tslang")
                << (shared ? "a shared library" : "a program that imports a shared library")
                << " built with -mm=gc links the garbage collector from gc.dll, so that the process has only one collector"
                   " - linked statically, it would free objects another module still holds. Point --gc-shared-lib-path"
                   " (or GC_SHARED_LIB_PATH) at the directory with gc.dll's import library 'gc.lib'; the release package"
                   " ships it as 'gcdll'.\n";
            return 1;
        }

        if (winX86OrX64
            && !resolveWindowsLibPath(TheTriple, gcSharedLibPath, "gc.lib", "gc.dll's import library gc.lib",
                                      "--gc-shared-lib-path (or GC_SHARED_LIB_PATH)",
                                      "build_gc_" + libConfig + "_shared_vs_x86"))
        {
            return 1;
        }

        gcDllPath = findGCDll(gcSharedLibPath);
    }

    if (compileOptions.needsGCRuntime())
    {
        auto gcLibPath = useSharedGC ? gcSharedLibPath : getGCLibPath();
        if (!useSharedGC && winX86OrX64
            && !resolveWindowsLibPath(TheTriple, gcLibPath, "gc.lib", "gc.lib", "--gc-lib-path (or GC_LIB_PATH)",
                                      "build_gc_" + libConfig + "_vs_x86"))
        {
            return 1;
        }

        gcLibPathOpt = getLibsPathOpt(gcLibPath);
        if (!gcLibPathOpt.empty())
        {
            args.push_back(gcLibPathOpt.c_str());    
        }
    }
    
    if (isTslangLibNeeded)
    {
        auto tslangLibPath = getTslangLibPath();
        if (winX86OrX64
            && !resolveWindowsLibPath(TheTriple, tslangLibPath, "TypeScriptAsyncRuntime.lib", "TypeScriptAsyncRuntime.lib",
                                      "--tslang-lib-path (or TSLANG_LIB_PATH)",
                                      "build_tslang_runtime_" + libConfig + "_x86"))
        {
            return 1;
        }

        tslangLibPathOpt = getLibsPathOpt(tslangLibPath);
        if (!tslangLibPathOpt.empty())
        {
            args.push_back(tslangLibPathOpt.c_str());    
        }
    }

    // system
    if (win)
    {
        args.push_back("-luser32");

        // Link the static CRT (/MT[d]) to match the prebuilt static LLVM/MLIR libs,
        // the TypeScript runtime libs and gc.lib. Mixing the static and dynamic CRT
        // gives the generated program a separate heap/stdout buffer from those libs
        // and crashes at startup/teardown (0xC0000005). The release config uses the
        // non-debug import-name suffix; debug uses the 'd' suffix.
        if (enableOpt)
        {
            args.push_back("-llibucrt");
            args.push_back("-llibcmt");
            args.push_back("-llibvcruntime");
            args.push_back("-Wl,-nodefaultlib:libcmtd");
        }
        else
        {
            args.push_back("-llibucrtd");
            args.push_back("-llibcmtd");
            args.push_back("-llibvcruntimed");
            args.push_back("-Wl,-nodefaultlib:libcmt");
        }
    }

    // tslang libs
    // ELF: a program that loads a tslang shared object exports its whole collector, so the shared
    // object's calls to GC_* bind to it at load time instead of to the static copy linked into the
    // shared object, and the process runs one collector. Measured: without the export, strings the
    // shared object built were freed while the program held them. The JIT needs nothing here, as
    // libTypeScriptRuntime.so already exports GC_*. See docs/single-gc-collector-design.md.
    auto exportGC = !win && !wasm && !shared && compileOptions.needsGCRuntime() && compileOptions.importsSharedLibrary;
    if (compileOptions.needsGCRuntime())
    {
        if (exportGC)
        {
            // whole archive: the shared object may call GC_* functions this program never does
            args.push_back("-Wl,--whole-archive");
            args.push_back("-lgc");
            args.push_back("-Wl,--no-whole-archive");
            args.push_back("-Wl,--export-dynamic-symbol=GC_*");
        }
        else
        {
            args.push_back("-lgc");
        }
    }

    if (isTslangLibNeeded)
    {
        args.push_back("-lTypeScriptAsyncRuntime");
    }

    if (!win && !wasm)
    {
        if (RM && *RM == llvm::Reloc::PIC_)
        {
            args.push_back("-fPIC");
            if (!shared)
            {
                args.push_back("-Wl,-pie");
            }
        }
        else
        {
            if (!shared)
            {
                args.push_back("-Wl,-no-pie");
            }
        }

        if (shared)
        {
            // something missing in .so compiling        
            args.push_back("-Wl,--build-id");
        }

        // TODO: review some options
        args.push_back("-frtti");
        args.push_back("-fexceptions");
        args.push_back("-lstdc++");
        args.push_back("-lm");
        args.push_back("-lpthread");
        args.push_back("-ldl");
        args.push_back("-lrt");

        // The default library's HTTP wrapper (http_linux.cpp) is implemented on
        // top of libcurl (the Linux counterpart of WinHTTP on Windows, which is
        // linked via a #pragma in http.cpp). Only needed when the default library
        // is actually linked, so this is skipped under --no-default-lib. Warn
        // early with the exact install command if the development library is not
        // present, then still pass -lcurl so the linker error is emitted too if
        // it really is missing.
        if (!compileOptions.noDefaultLib)
        {
            static const char *curlSharedLibs[] = {
                "/usr/lib/x86_64-linux-gnu/libcurl.so",
                "/usr/lib/aarch64-linux-gnu/libcurl.so",
                "/usr/lib/libcurl.so",
                "/usr/local/lib/libcurl.so",
                "/usr/lib64/libcurl.so",
            };

            bool curlFound = false;
            for (auto *curlLib : curlSharedLibs)
            {
                if (llvm::sys::fs::exists(curlLib))
                {
                    curlFound = true;
                    break;
                }
            }

            if (!curlFound)
            {
                llvm::WithColor::warning(llvm::errs(), "tslang")
                    << "the curl development library was not found; the HTTP support in the default "
                       "library needs it. Install it with: sudo apt install libcurl4-openssl-dev\n";
            }

            args.push_back("-lcurl");
        }
        //args.push_back("-rdynamic"); // do we need it?
    }

    if (wasm && emscripten)
    {
        //args.push_back("--sysroot=C:/utils/emsdk/upstream/emscripten/cache/sysroot");
        emsdkSysRootPathOpt = concatIfNotEmpty("--sysroot=", getEMSDKSysRootPath());
        if (!emsdkSysRootPathOpt.empty())
        {
            args.push_back(emsdkSysRootPathOpt.c_str());
        }

        args.push_back("-lcompiler_rt");
    }

    if (compileOptions.generateDebugInfo)
    {
        args.push_back("-g");
    }

    // Create DiagnosticsEngine for the compiler driver
    auto diagOpts = createAndPopulateDiagOpts(args);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(new clang::DiagnosticIDs());
    auto *diagClient = new typescript::tslang::TextDiagnosticPrinter(llvm::errs(), &*diagOpts);

    diagClient->setPrefix(
        std::string(llvm::sys::path::stem(driverPath)));

    clang::DiagnosticsEngine diags(diagID, *diagOpts, diagClient);

    // Prepare the driver
    clang::driver::Driver theDriver(driverPath,
                                    targetTriple, diags,
                                    "tslang LLVM compiler");

    theDriver.setTargetAndMode(targetAndMode);
    std::unique_ptr<clang::driver::Compilation> c(theDriver.BuildCompilation(args));

    if (wasm)
    {
        if (emscripten)
        {
            removeCommandArgs(c.get(), {"clang_rt.builtins"});
            addCommandArgs(c.get(), {"-ldlmalloc", "-lstandalonewasm"});
        }
        else
        {
            removeCommandArgs(c.get(), {"crt1.o", "-lc", "clang_rt.builtins"});
            addCommandArgs(c.get(), {"--no-entry", "--export-all", "--allow-undefined"});
        }
    }

    llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4> failingCommands;

    if (verbose.getValue())
        c->getJobs().Print(llvm::errs(), "\n", /*Quote=*/false);

    // Run the driver
    int res = 1;
    bool isCrash = false;
    res = theDriver.ExecuteCompilation(*c, failingCommands);

    for (const auto &p : failingCommands)
    {
        int commandRes = p.first;
        const clang::driver::Command *failingCommand = p.second;
        if (!res)
            res = commandRes;

        // If result status is < 0 (e.g. when sys::ExecuteAndWait returns -1),
        // then the driver command signalled an error. On Windows, abort will
        // return an exit code of 3. In these cases, generate additional diagnostic
        // information if possible.
        isCrash = commandRes < 0;
#ifdef _WIN32
        isCrash |= commandRes == 3;
#endif
        if (isCrash)
        {
            theDriver.generateCompilationDiagnostics(*c, *failingCommand);
            break;
        }
    }

    diags.getClient()->finish();

    // A binary linked against gc.dll does not start without it, so put it beside the output.
    if (res == 0 && useSharedGC)
    {
        auto outputDir = llvm::sys::path::parent_path(outputFilename.getValue());
        llvm::SmallString<256> destination(outputDir.empty() ? "." : outputDir);
        llvm::sys::path::append(destination, "gc.dll");

        bool sameFile = false;
        if (gcDllPath.empty())
        {
            llvm::WithColor::warning(llvm::errs(), "tslang")
                << "linked against gc.dll, but no gc.dll was found next to '" << gcSharedLibPath
                << "' or in its '../bin'; ship gc.dll beside '" << outputFilename.getValue() << "'\n";
        }
        else if (llvm::sys::fs::equivalent(gcDllPath, destination, sameFile) || !sameFile)
        {
            if (auto error = llvm::sys::fs::copy_file(gcDllPath, destination))
            {
                llvm::WithColor::warning(llvm::errs(), "tslang")
                    << "could not copy '" << gcDllPath << "' to '" << destination << "': " << error.message()
                    << "; ship gc.dll beside '" << outputFilename.getValue() << "'\n";
            }
        }
    }

    // If we have multiple failing commands, we return the result of the first
    // failing command.
    return res;
}
