#include "clang/Driver/Driver.h"
#include "TypeScript/TypeScriptLang/TextDiagnosticPrinter.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/CodeGen/CommandFlags.h"

#include "TypeScript/DataStructs.h"
#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/Defines.h"

namespace cl = llvm::cl;

extern cl::opt<enum Action> emitAction;
extern cl::opt<std::string> outputFilename;
extern cl::opt<bool> disableGC;
extern cl::opt<std::string> TargetTriple;
extern cl::opt<std::string> defaultlibpath;
extern cl::opt<std::string> gclibpath;
extern cl::opt<std::string> llvmlibpath;
extern cl::opt<std::string> tsclibpath;
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
    llvm::opt::InputArgList args = clang::driver::getDriverOptTable().ParseArgs(
        argv.slice(1), missingArgIndex, missingArgCount,
        /*FlagsToInclude=*/clang::driver::options::FlangOption);

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

std::string getGCLibPath()
{
    if (!gclibpath.empty())
    {
        return gclibpath;
    }

    if (auto gcLibEnvValue = llvm::sys::Process::GetEnv("GC_LIB_PATH")) 
    {
        return gcLibEnvValue.value();
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

std::string getTscLibPath()
{
    if (!tsclibpath.empty())
    {
        return tsclibpath;
    }

    if (auto tscLibEnvValue = llvm::sys::Process::GetEnv("TSC_LIB_PATH")) 
    {
        return tscLibEnvValue.value();
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

int buildExe(int argc, char **argv, std::string objFileName, CompileOptions &compileOptions)
{
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);
    //llvm::SmallVector<const char *, 256> args(argv, argv + argc);
    llvm::SmallVector<const char *, 256> args(argv, argv + 1);    

    clang::driver::ParsedClangName targetAndMode("tsc", "--driver-mode=tsc");
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
        if (llvm::StringRef(args[1]).startswith("-cc1"))
        {
            llvm::errs() << "error: unknown integrated tool '" << args[1] << "'. "
                         << "Valid tools include '-tsc'.\n";
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
    std::string tscLibPathOpt;
    std::string llvmLibPathOpt;
    std::string emsdkSysRootPathOpt;
    std::string defaultLibPathOpt;
    std::string defaultLibFileOpt;

    auto isLLVMLibNeeded = true;
    auto isTscLibNeeded = true;

    auto os = TheTriple.getOS();
    auto arch = TheTriple.getArch();
    auto win = os == llvm::Triple::Win32;
    auto wasm = arch == llvm::Triple::wasm32 || arch == llvm::Triple::wasm64;
    auto emscripten = os == llvm::Triple::Emscripten;
    auto shared = emitAction == BuildDll;
    
    std::optional<llvm::Reloc::Model> RM = llvm::codegen::getExplicitRelocModel();

    if (wasm)
    {
        isLLVMLibNeeded = false;
        isTscLibNeeded = false;        
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

    if (win && shared)
    {
        //args.push_back("-Wl,-nodefaultlib:libcmt");
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

        // default lib path
        defaultLibPathOpt = getLibsPathOpt(mergeWithDefaultLibPath(getDefaultLibPath(), shared ? DEFAULT_LIB_DIR "/dll" : DEFAULT_LIB_DIR "/lib"));
        if (!defaultLibPathOpt.empty())
        {
            args.push_back(defaultLibPathOpt.c_str());    
        }        
    }    

    if (!disableGC)
    {
        gcLibPathOpt = getLibsPathOpt(getGCLibPath());
        if (!gcLibPathOpt.empty())
        {
            args.push_back(gcLibPathOpt.c_str());    
        }
    }
    
    // add logic to detect if libs are used and needed
    if (isLLVMLibNeeded)
    {
        llvmLibPathOpt = getLibsPathOpt(getLLVMLibPath());
        if (!llvmLibPathOpt.empty())
        {
            args.push_back(llvmLibPathOpt.c_str());    
        }
    }

    if (isTscLibNeeded)
    {
        tscLibPathOpt = getLibsPathOpt(getTscLibPath());
        if (!tscLibPathOpt.empty())
        {
            args.push_back(tscLibPathOpt.c_str());    
        }
    }

    // system
    if (win)
    {
        args.push_back("-luser32");    
        if (shared || !disableGC)
        {
            // needed to resolve DLL ref
            if (enableOpt)
            {
                args.push_back("-lmsvcrt");
            }
            else
            {
                args.push_back("-lmsvcrtd");
            }
        }
    }

    // tsc libs
    if (!disableGC)
    {    
        args.push_back("-lgcmt-lib");
    }

    if (isTscLibNeeded)
    {
        args.push_back("-lTypeScriptAsyncRuntime");
    }

    if (isLLVMLibNeeded)
    {
        args.push_back("-lLLVMSupport");
        if (!win)
        {
            args.push_back("-lLLVMDemangle");
        }
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
        args.push_back("-ltinfo");
        args.push_back("-ldl");
        args.push_back("-lrt");
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

    clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagClient);

    // Prepare the driver
    clang::driver::Driver theDriver(driverPath,
                                    targetTriple, diags,
                                    "tsc LLVM compiler");

    theDriver.setTargetAndMode(targetAndMode);
    std::unique_ptr<clang::driver::Compilation> c(theDriver.BuildCompilation(args));

    if (win && (shared || !disableGC))
    {
        args.push_back("-Wl,-nodefaultlib:libcmt");
        removeCommandArgs(c.get(), {"defaultlib:libcmt"});
    }

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

    // If we have multiple failing commands, we return the result of the first
    // failing command.
    return res;
}
