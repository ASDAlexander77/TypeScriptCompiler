#include "clang/Driver/Driver.h"
#include "TypeScript/TypeScriptLang/TextDiagnosticPrinter.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticIDs.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/Compilation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/VirtualFileSystem.h"

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

int main(int argc, char **argv)
{
    // Initialize variables to call the driver
    llvm::InitLLVM x(argc, argv);
    llvm::SmallVector<const char *, 256> args(argv, argv + argc);

    clang::driver::ParsedClangName targetandMode("tslang", "--driver-mode=tslang");
    std::string driverPath = getExecutablePath(args[0]);

    llvm::BumpPtrAllocator a;
    llvm::StringSaver saver(a);
    ExpandResponseFiles(saver, args);

    // Check if flang-new is in the frontend mode
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

    // Specify Visual Studio C runtime library. “static” and “static_dbg” correspond to the cl flags /MT and /MTd which use the multithread, 
    // static version. “dll” and “dll_dbg” correspond to the cl flags /MD and /MDd which use the multithread, dll version. <arg> must be ‘static’, ‘static_dbg’, ‘dll’ or ‘dll_dbg’.    
    //args.insert(args.begin() + 1, "-nodefaultlibs");
    //args.insert(args.begin() + 1, "-fms-omit-default-lib=dll");
    //args.insert(args.begin() + 1, "-fms-runtime-lib=static_dbg");

    auto win = true;
    auto shared = false;
    
    if (win)
    {
        args.insert(args.begin() + 1, "-Wl,-nodefaultlib:libcmt");
    }

    if (shared)
    {
        args.insert(args.begin() + 1, "-shared");
        args.insert(args.begin() + 1, "-oliba1.so");
        if (!win)
        {
            // added search path
            args.insert(args.begin() + 1, "-Wl,-rpath=.");
        }
    }
    else
    {
        args.insert(args.begin() + 1, "-oa1.exe");
    }

    args.insert(args.begin() + 1, "-LC:/dev/TypeScriptCompiler/3rdParty/gc/x64/debug");    
    args.insert(args.begin() + 1, "-LC:/dev/TypeScriptCompiler/3rdParty/llvm/x64/debug/lib");    
    args.insert(args.begin() + 1, "-LC:/dev/TypeScriptCompiler/__build/tsc/windows-msbuild-debug/lib");    

    // system
    if (win)
    {
        args.insert(args.begin() + 1, "-luser32");    
        if (shared)
        {
            // needed to resolve DLL ref
            args.insert(args.begin() + 1, "-lmsvcrt");
        }
    }

    // tsc libs
    args.insert(args.begin() + 1, "-lgcmt-lib");
    args.insert(args.begin() + 1, "-lTypeScriptAsyncRuntime");
    args.insert(args.begin() + 1, "-lLLVMSupport");

    // Create DiagnosticsEngine for the compiler driver
    auto diagOpts = createAndPopulateDiagOpts(args);
    llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> diagID(new clang::DiagnosticIDs());
    auto *diagClient = new typescript::tslang::TextDiagnosticPrinter(llvm::errs(), &*diagOpts);

    diagClient->setPrefix(
        std::string(llvm::sys::path::stem(getExecutablePath(args[0]))));

    clang::DiagnosticsEngine diags(diagID, &*diagOpts, diagClient);

    // Prepare the driver
    clang::driver::Driver theDriver(driverPath,
                                    llvm::sys::getDefaultTargetTriple(), diags,
                                    "tslang LLVM compiler");

    theDriver.setTargetAndMode(targetandMode);
    std::unique_ptr<clang::driver::Compilation> c(theDriver.BuildCompilation(args));
    llvm::SmallVector<std::pair<int, const clang::driver::Command *>, 4> failingCommands;

#ifndef _NDEBUG
    c->getJobs().Print(llvm::errs(), "\n", /*Quote=*/false);
#endif    

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
