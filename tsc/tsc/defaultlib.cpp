#include "TypeScript/MLIRGen.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/IR/LLVMContext.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/FormatVariadic.h"

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/VSCodeTemplate/Files.h"

#include <regex>

#define DEBUG_TYPE "tsc"

using namespace typescript;
using namespace llvm;
namespace cl = llvm::cl;
namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

int clone();
int build();
#ifdef WIN32
int buildWin32(const SmallVectorImpl<char>&);
#else
int buildLinux(const SmallVectorImpl<char>&);
#endif
std::string getExecutablePath(const char *);
std::string getGCLibPath();
std::string getLLVMLibPath();
std::string getTscLibPath();
std::string getDefaultLibPath();
std::string getpath(std::string, const SmallVectorImpl<char>&);

int installDefaultLib(int argc, char **argv)
{
    SmallVector<char> tempFolder;
    if (auto error_code = fs::createUniqueDirectory("defaultLibBuild", tempFolder))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't create Temp folder/directory '" << tempFolder << "' : " << error_code.message() << "\n";
        return -1;
    }

    if (auto error_code = fs::set_current_path(tempFolder))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open folder/directory '" << tempFolder << "' : " << error_code.message() << "\n";
        return -1;
    }
    
    if (auto cloneResult = clone())
    {
        return -1;
    }

    // open TypeScriptCompilerDefaultLib
    if (auto error_code = fs::set_current_path("TypeScriptCompilerDefaultLib"))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open folder/directory '" << "TypeScriptCompilerDefaultLib " << "' : " << error_code.message() << "\n";
        return -1;
    }    

    // run build
    llvm::SmallVector<const char *, 256> args(argv, argv + 1);    
    auto driverPath = getExecutablePath(args[0]);

    llvm::SmallVector<char> appPath{};
    appPath.append(driverPath.begin(), driverPath.end());
    path::remove_filename(appPath);

    if (auto buildResult = 
#ifdef WIN32    
        buildWin32(appPath)
#else
        buildLinux(appPath)
#endif        
    )
    {
        return -1;
    }

    return 0;
}

int clone()
{
    auto fromPath = llvm::sys::findProgramByName("git");
    if (!fromPath)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "'git' not found on PATH" << "\n";
        return -1;        
    }

    auto gitPath = fromPath.get();

    std::optional<StringRef> redirects[] = {
        std::nullopt, // Stdin
        std::nullopt, // Stdout
        std::nullopt  // Stderr
    };

    // 1, run git to get data

    SmallVector<StringRef, 4> args{gitPath, "clone", "https://github.com/ASDAlexander77/TypeScriptCompilerDefaultLib.git"};

    std::string errMsg;
    auto returnCode = sys::ExecuteAndWait(
        gitPath, args, /*envp*/ std::nullopt, redirects, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

    if (returnCode < 0)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Error running git clone. " << errMsg << "\n";
        return -1;         
    }

    return 0;
}

int buildWin32(const SmallVectorImpl<char>& appPath)
{
    auto fromPath = llvm::sys::findProgramByName("cmd");
    if (!fromPath)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "'cmd' not found on PATH" << "\n";
        return -1;        
    }

    auto cmdPath = fromPath.get();

    fromPath = llvm::sys::findProgramByName("vswhere");
    if (!fromPath)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "'vswhere' not found on PATH" << "\n";
        return -1;        
    }

    auto vswherePath = fromPath.get();    

    std::optional<StringRef> redirects[] = {
        std::nullopt, // Stdin
        std::nullopt, // Stdout
        std::nullopt  // Stderr
    };

    // 1, run git to get data

    SmallVector<StringRef, 4> args{cmdPath, "/S /C", "build.bat"};

    SmallVector<StringRef, 4> envp{};

    auto gcLibPath = getpath(getGCLibPath(), appPath);
    auto llvmLibPath = getpath(getLLVMLibPath(), appPath);
    auto tscLibPath = getpath(getTscLibPath(), appPath);
    auto defaultLibPath = getpath(getDefaultLibPath(), appPath);

    std::string appPathVar = llvm::formatv("{0}={1}", "TOOL_PATH", appPath);
    std::string gcLibPathVar = llvm::formatv("{0}={1}", "GC_LIB_PATH", gcLibPath);
    std::string llvmLibPathVar = llvm::formatv("{0}={1}", "LLVM_LIB_PATH", llvmLibPath);
    std::string tscLibPathVar = llvm::formatv("{0}={1}", "TSC_LIB_PATH", tscLibPath);
    std::string defaultLibPathVar = llvm::formatv("{0}={1}", "DEFAULT_LIB_PATH", defaultLibPath);    
    std::string vswherePathVar = llvm::formatv("{0}={1}", "VSWHERE_PATH", vswherePath);    

    envp.push_back(StringRef(appPathVar));
    envp.push_back(StringRef(gcLibPathVar));
    envp.push_back(StringRef(llvmLibPathVar));
    envp.push_back(StringRef(tscLibPathVar));
    envp.push_back(StringRef(defaultLibPathVar));
    envp.push_back(StringRef(vswherePathVar));

    std::string errMsg;
    auto returnCode = sys::ExecuteAndWait(
        cmdPath, args, envp, redirects, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

    if (returnCode < 0)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Error running build command. " << errMsg << "\n";
        return -1;         
    }

    return 0;
}

std::string getpath(std::string path, const SmallVectorImpl<char>& defaultPath)
{
    if (path.empty())
    {
        return std::string(defaultPath.begin(), defaultPath.end());
    }

    return path;    
}