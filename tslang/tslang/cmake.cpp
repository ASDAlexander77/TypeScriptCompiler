#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/WithColor.h"

#include "TypeScript/VSCodeTemplate/Files.h"

#define DEBUG_TYPE "tslang"

using namespace llvm;
using namespace std;
namespace cl = llvm::cl;
namespace fs = llvm::sys::fs;
namespace path = llvm::sys::path;

extern cl::opt<string> inputFilename;

int create_file_base(StringRef filepath, StringRef data);
int substitute(StringRef data, StringMap<StringRef> &values, SmallString<128> &result);

string getExecutablePath(const char *);
string fixpath(string, const SmallVectorImpl<char>&);

int createCMakeFolder(int argc, char **argv)
{
    auto projectName = StringRef(inputFilename);
    if (projectName == "-") {
        WithColor::error(errs(), "tslang") << "Name is not provided. (use file name without file extension)\n";
        return -1;
    }

    if (auto error_code = fs::create_directory(projectName))
    {
        WithColor::error(errs(), "tslang") << "Could not create project: " << error_code.message() << "\n";
        return -1;
    }

    if (auto error_code = fs::set_current_path(projectName))
    {
        WithColor::error(errs(), "tslang") << "Can't open folder/directory: " << error_code.message() << "\n";
        return -1;
    }

    SmallString<256> projectPath;
    if (auto error_code = fs::current_path(projectPath))
    {
        WithColor::error(errs(), "tslang") << "Can't get info about current folder: " << error_code.message() << "\n";
        return -1;
    }

    StringMap<StringRef> vals;
    vals["PROJECT"] = projectName;

    StringRef cmakeLists(CMAKE_LISTS_TXT_DATA);
    SmallString<128> resultCMakeLists;
    substitute(cmakeLists, vals, resultCMakeLists);

    if (auto error_code = create_file_base("CMakeLists.txt", resultCMakeLists.str()))
    {
        return -1;
    }

    if (auto error_code = create_file_base("CMakePresets.json", CMAKE_PRESETS_JSON_DATA))
    {
        return -1;
    }

    if (auto error_code = create_file_base("main.cpp", CMAKE_MAIN_CPP_DATA))
    {
        return -1;
    }

    if (auto error_code = create_file_base("mycode.ts", CMAKE_MYCODE_TS_DATA))
    {
        return -1;
    }

    if (auto error_code = create_file_base("adder.ts", CMAKE_ADDER_TS_DATA))
    {
        return -1;
    }

    if (auto error_code = create_file_base("README.md", CMAKE_README_MD_DATA))
    {
        return -1;
    }

    // cmake folder
    if (auto error_code = fs::create_directory(CMAKE_FOLDER_PATH))
    {
        WithColor::error(errs(), "tslang") << "Could not create folder/directory '" << CMAKE_FOLDER_PATH << "' : " << error_code.message() << "\n";
        return -1;
    }

    if (auto error_code = fs::set_current_path(CMAKE_FOLDER_PATH))
    {
        WithColor::error(errs(), "tslang") << "Can't open folder/directory '" << CMAKE_FOLDER_PATH << "' : " << error_code.message() << "\n";
        return -1;
    }

    // hint for finding tslang app (same logic as in createVSCodeFolder)
    SmallVector<const char *, 256> args(argv, argv + 1);
    auto driverPath = getExecutablePath(args[0]);

    SmallVector<char> appPath{};
    appPath.append(driverPath.begin(), driverPath.end());
    path::remove_filename(appPath);

    auto tslangAppPath = fixpath(string(appPath.begin(), appPath.end()), appPath);
    vals["TSLANG_APP_PATH"] = tslangAppPath;

    StringRef determineCompiler(CMAKE_DETERMINE_TSLANG_COMPILER_DATA);
    SmallString<128> resultDetermineCompiler;
    substitute(determineCompiler, vals, resultDetermineCompiler);

    if (auto error_code = create_file_base("CMakeDetermineTSLANGCompiler.cmake", resultDetermineCompiler.str()))
    {
        return -1;
    }

    if (auto error_code = create_file_base("CMakeTestTSLANGCompiler.cmake", CMAKE_TEST_TSLANG_COMPILER_DATA))
    {
        return -1;
    }

    if (auto error_code = create_file_base("CMakeTSLANGCompiler.cmake.in", CMAKE_TSLANG_COMPILER_IN_DATA))
    {
        return -1;
    }

    if (auto error_code = create_file_base("CMakeTSLANGInformation.cmake", CMAKE_TSLANG_INFORMATION_DATA))
    {
        return -1;
    }

    return 0;
}
