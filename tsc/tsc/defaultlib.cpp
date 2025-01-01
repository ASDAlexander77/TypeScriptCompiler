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

#include <cstdlib>
#include <regex>
#include <filesystem>

#define DEBUG_TYPE "tsc"

#ifdef WIN32
#define PUTENV _putenv
#else
#define PUTENV putenv
#endif

using namespace typescript;
using namespace llvm;
namespace cl = llvm::cl;
namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;
namespace fs_ = std::filesystem;

int clone();
int build();
#ifdef WIN32
int buildWin32(const SmallVectorImpl<char>&, SmallVectorImpl<char>&);
#else
int buildLinux(const SmallVectorImpl<char>&, SmallVectorImpl<char>&);
#endif
std::string getExecutablePath(const char *);
std::string getGCLibPath();
std::string getLLVMLibPath();
std::string getTscLibPath();
std::string getDefaultLibPath();
std::string getpath(std::string, const SmallVectorImpl<char>&);
std::error_code copy_from_to(const SmallVectorImpl<char>&, const SmallVectorImpl<char>&);

bool checkFileExistsAtPath(const SmallVectorImpl<char>& path, std::string sub1, std::string sub2, std::string fileName)
{
    llvm::SmallVector<char> destPath(0);
    destPath.reserve(256);
    destPath.append(path);

    llvm::sys::path::append(destPath, sub1, sub2, fileName);
    if (!llvm::sys::fs::exists(destPath))
    {
        return false;
    }    

    return true;
}

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

    llvm::SmallVector<char> appPath(0);
    appPath.reserve(256);
    appPath.append(driverPath.begin(), driverPath.end());
    path::remove_filename(appPath);

    llvm::SmallVector<char> builtPath(0);
    builtPath.reserve(256);
    if (auto buildResult = 
#ifdef WIN32    
        buildWin32(appPath, builtPath)
#else
        buildLinux(appPath, builtPath)
#endif        
    )
    {
        return -1;
    }

    // get destination path
    llvm::SmallVector<char> destPath(0);
    destPath.reserve(256);
    auto defaultLibPath = getDefaultLibPath();
    if (!defaultLibPath.empty())
    {
        path::append(destPath, defaultLibPath);
    }

    if (destPath.empty())
    {
        path::append(destPath, appPath);
    }

    if (destPath.empty())
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "installation destination is not provided. use option --default-lib-path to set it or set environment variable DEFAULT_LIB_PATH\n";
        return -1;
    }

    auto result = checkFileExistsAtPath(
        builtPath, 
        "defaultlib", 
        "lib", 
#ifdef WIN32        
        "TypeScriptDefaultLib.lib"
#else
        "libTypeScriptDefaultLib.a"
#endif        
    );

    if (!result)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "It seems the compilation process has failed, try to check settings for --gc-lib-path, --llvm-lib-path and --tsc-lib-path or their environment variables GC_LIB_PATH, LLVM_LIB_PATH, TSC_LIB_PATH\n";
        return -1;
    }

    // copy
    if (auto error_code = copy_from_to(builtPath, destPath))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't copy built library into destination folder/directory : " << error_code.message() << "\n";
        return -1;
    }

    // set environment variable
    std::string defaultLibPathVar = llvm::formatv("{0}={1}", "DEFAULT_LIB_PATH", destPath);    
    PUTENV(defaultLibPathVar.data());

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

#ifdef WIN32
int buildWin32(const SmallVectorImpl<char>& appPath, SmallVectorImpl<char>& builtPath)
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

    PUTENV(appPathVar.data());
    PUTENV(gcLibPathVar.data());
    PUTENV(llvmLibPathVar.data());
    PUTENV(tscLibPathVar.data());
    PUTENV(defaultLibPathVar.data());
    PUTENV(vswherePathVar.data());

    std::string errMsg;
    auto returnCode = sys::ExecuteAndWait(
        cmdPath, args, std::nullopt, redirects, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

    if (returnCode < 0)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Error running build command. " << errMsg << "\n";
        return -1;         
    }

    if (auto error_code = fs::current_path(builtPath))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open get info about current folder/directory : " << error_code.message() << "\n";
        return -1;        
    }

    path::append(builtPath, "__build", "release");

    return 0;
}
#else
int buildLinux(const SmallVectorImpl<char>& appPath, SmallVectorImpl<char>& builtPath)
{
    std::optional<StringRef> redirects[] = {
        std::nullopt, // Stdin
        std::nullopt, // Stdout
        std::nullopt  // Stderr
    };

    // 1, run git to get data

    SmallVector<StringRef, 4> args{};

    auto gcLibPath = getpath(getGCLibPath(), appPath);
    auto llvmLibPath = getpath(getLLVMLibPath(), appPath);
    auto tscLibPath = getpath(getTscLibPath(), appPath);
    auto defaultLibPath = getpath(getDefaultLibPath(), appPath);

    std::string appPathVar = llvm::formatv("{0}={1}", "TOOL_PATH", appPath);
    std::string gcLibPathVar = llvm::formatv("{0}={1}", "GC_LIB_PATH", gcLibPath);
    std::string llvmLibPathVar = llvm::formatv("{0}={1}", "LLVM_LIB_PATH", llvmLibPath);
    std::string tscLibPathVar = llvm::formatv("{0}={1}", "TSC_LIB_PATH", tscLibPath);
    std::string defaultLibPathVar = llvm::formatv("{0}={1}", "DEFAULT_LIB_PATH", defaultLibPath);    

    PUTENV(appPathVar.data());
    PUTENV(gcLibPathVar.data());
    PUTENV(llvmLibPathVar.data());
    PUTENV(tscLibPathVar.data());
    PUTENV(defaultLibPathVar.data());

    std::string errMsg;
    auto returnCode = sys::ExecuteAndWait(
        "build.sh", args, std::nullopt, redirects, /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

    if (returnCode < 0)
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Error running build command. " << errMsg << "\n";
        return -1;         
    }

    if (auto error_code = fs::current_path(builtPath))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open get info about current folder/directory : " << error_code.message() << "\n";
        return -1;        
    }

    path::append(builtPath, "__build", "release");

    return 0;    
}
#endif

std::error_code copy_from_to(const SmallVectorImpl<char>& source, const SmallVectorImpl<char>& dest)
{
    if (!llvm::sys::fs::is_directory(source))
    {
        return std::make_error_code(std::errc::no_such_file_or_directory);
    }

    std::string srcStr;
    srcStr.reserve(source.size());
    for (auto c : source) srcStr += c;

    std::string dstStr;
    dstStr.reserve(dest.size());
    for (auto c : dest) dstStr += c;

    fs_::copy(srcStr, dstStr, fs_::copy_options::recursive);

    return {};
}

std::string getpath(std::string path, const SmallVectorImpl<char>& defaultPath)
{
    if (path.empty())
    {
        return std::string(defaultPath.begin(), defaultPath.end());
    }

    return path;    
}