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

#include "TypeScript/TypeScriptCompiler/Defines.h"
#include "TypeScript/VSCodeTemplate/Files.h"

#include <regex>

#define DEBUG_TYPE "tsc"

using namespace typescript;
using namespace llvm;
namespace cl = llvm::cl;
namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

extern cl::opt<std::string> inputFilename;

int create_file_base(StringRef filepath, StringRef data);
int substitute(StringRef data, StringMap<StringRef> &values, SmallString<128> &result);

std::string getExecutablePath(const char *);
std::string getGCLibPath();
std::string getLLVMLibPath();
std::string getTscLibPath();
std::string getDefaultLibPath();
std::string fixpath(std::string, const SmallVectorImpl<char>&);

int createVSCodeFolder(int argc, char **argv)
{
    auto projectName = llvm::StringRef(inputFilename);
    if (projectName == "-") {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Name is not provided. (use file name without file extension)\n";
        return -1;
    }

    if (auto error_code = fs::create_directory(projectName))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not create project: " << error_code.message() << "\n";
        return -1;            
    }

    if (auto error_code = fs::set_current_path(projectName))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open folder/directory: " << error_code.message() << "\n";
        return -1;
    }

    SmallString<256> fullFilePath(projectName);
    path::replace_extension(fullFilePath, ".ts");   
    if (auto error_code = create_file_base(fullFilePath.str(), R"(print("Hello World!");)"))
    {
        return -1;
    }

    if (auto error_code = create_file_base("tsnc.natvis", TSNC_NATVIS))
    {
        return -1;
    }    

    StringMap<StringRef> vals;
    vals["PROJECT"] = projectName;

    StringRef tsconfig(TSCONFIG_JSON_DATA);
    SmallString<128> result;
    substitute(tsconfig, vals, result);

    if (auto error_code = create_file_base("tsconfig.json", result.str()))
    {
        return -1;
    }

    SmallString<128> projectPath;
    if (auto error_code = fs::current_path(projectPath)) 
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't get info about current folder: " << error_code.message() << "\n";
        return -1;
    }

    // node_modules
    if (auto error_code = fs::create_directories(NODE_MODULE_TSNC_PATH))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not create folder/directory '" << NODE_MODULE_TSNC_PATH << "' : " << error_code.message() << "\n";
        return -1;            
    }    

    if (auto error_code = fs::set_current_path(NODE_MODULE_TSNC_PATH))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open folder/directory '" << NODE_MODULE_TSNC_PATH << "' : " << error_code.message() << "\n";
        return -1;
    }

    if (auto error_code = create_file_base("index.d.ts", TSNC_INDEX_D_TS))
    {
        return -1;
    }

    // need to create .vscode
    if (auto error_code = fs::set_current_path(projectPath))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open folder/directory '" << projectPath << "' : " << error_code.message() << "\n";
        return -1;
    }    

    if (auto error_code = fs::create_directory(DOT_VSCODE_PATH))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Could not create folder/directory '" << DOT_VSCODE_PATH << "' : " << error_code.message() << "\n";
        return -1;            
    }    

    if (auto error_code = fs::set_current_path(DOT_VSCODE_PATH))
    {
        llvm::WithColor::error(llvm::errs(), "tsc") << "Can't open folder/directory '" << DOT_VSCODE_PATH << "' : " << error_code.message() << "\n";
        return -1;
    }     

    if (auto error_code = create_file_base("settings.json", R"({})"))
    {
        return -1;
    }    

    // set params

    llvm::SmallVector<const char *, 256> args(argv, argv + 1);    
    auto driverPath = getExecutablePath(args[0]);

    llvm::SmallVector<char> appPath{};
    appPath.append(driverPath.begin(), driverPath.end());
    path::remove_filename(appPath);

    auto tscCmd = fixpath(driverPath, appPath);
    auto gcLibPath = fixpath(getGCLibPath(), appPath);
    auto llvmLibPath = fixpath(getLLVMLibPath(), appPath);
    auto tscLibPath = fixpath(getTscLibPath(), appPath);
    auto defaultLibPath = fixpath(getDefaultLibPath(), appPath);

    vals["TSC_CMD"] = tscCmd;
    vals["GC_LIB_PATH"] = gcLibPath;
    vals["LLVM_LIB_PATH"] = llvmLibPath;
    vals["TSC_LIB_PATH"] = tscLibPath;
    vals["DEFAULT_LIB_PATH"] = defaultLibPath;

    StringRef tasks(TASKS_JSON_DATA);
    SmallString<128> resultTasks;
    substitute(tasks, vals, resultTasks);

    if (auto error_code = create_file_base("tasks.json", resultTasks.str()))
    {
        return -1;
    }    

    StringRef launch(
#if WIN32        
        LAUNCH_JSON_DATA_WIN32
#else
        LAUNCH_JSON_DATA_LINUX
#endif
    );
    SmallString<128> resultLaunch;
    substitute(launch, vals, resultLaunch);

    if (auto error_code = create_file_base("launch.json", resultLaunch.str()))
    {
        return -1;
    }    

    return 0;
}

int create_file_base(StringRef filepath, StringRef data)
{
    std::error_code ec;
    llvm::ToolOutputFile out(filepath, ec, 
#ifdef WIN32    
    fs::OpenFlags::OF_TextWithCRLF
#else
    fs::OpenFlags::OF_Text
#endif    
    );

    // ... print into out
    out.os() << data;

    out.os().flush();
    out.keep();
    out.os().close();

    if (out.os().has_error())
    {
        llvm::report_fatal_error(llvm::Twine("Error emitting data to file '") + filepath);
        return -1;
    }

    return 0;
}

std::regex paramsRegEx = std::regex(R"(<<(.*?)>>)", std::regex_constants::ECMAScript); 

int substitute(StringRef data, StringMap<StringRef> &values, SmallString<128> &result)
{
    auto str = data.str();
    auto begin = std::sregex_iterator(str.begin(), str.end(), paramsRegEx);
    auto end = std::sregex_iterator();     

    std::string suffix;
    for (auto it = begin; it != end; it++) 
    {
        auto match = *it;
        result.append(match.prefix().str());
        result.append(values[match[1].str()]);
        suffix = match.suffix().str();
    }

    result.append(suffix);

    return 0;
}

std::string fixpath(std::string path, const SmallVectorImpl<char>& defaultPath)
{
    if (path.empty())
    {
        path = std::string(defaultPath.begin(), defaultPath.end());
    }

#ifdef WIN32    
    std::string output;
    output.reserve(path.size());
    for (const auto c: path) {
        switch (c) {
            case '\\':
            case '/':  
                output += "\\\\";        
                break;
            default:    
                output += c;            
                break;
        }
    }

    return output;
#else
    return path;    
#endif    
}