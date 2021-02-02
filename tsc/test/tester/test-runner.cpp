#include "helper.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <iostream>
#include <fstream>

#if __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#if WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

#ifndef TEST_LIBPATH
#define TEST_LIBPATH "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/VC/lib"
#endif

#ifndef TEST_EXEPATH
#define TEST_EXEPATH "C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin"
#endif

#ifndef TEST_TSC_EXEPATH
#define TEST_TSC_EXEPATH "C:/dev/TypeScriptCompiler/__build/tsc/bin"
#endif

#ifndef TEST_FILE
#define TEST_FILE "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/01arguments.ts"
#endif

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length())
    {
        return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
    }
    else
    {
        return false;
    }
}

std::string exec(std::string cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&PCLOSE)> pipe(POPEN(cmd.c_str(), "rt"), PCLOSE);
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }

    return result;
}

inline bool exists(std::string name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

int runFolder(const char *folder)
{
    std::string path = folder;
    for (const auto &entry : fs::directory_iterator(path))
    {
        if (!hasEnding(entry.path().extension().string(), ".ts"))
        {
            std::cout << "skipping: " << entry.path() << std::endl;
            continue;
        }

        std::cout << "Testing: " << entry.path() << std::endl;
    }

    return 0;
}

void createBatchFile()
{
    std::ofstream batFile("compile.bat");
    batFile << "set FILENAME=test_run" << std::endl;
    batFile << "set LIBPATH=" << TEST_LIBPATH << std::endl;
    batFile << "set EXEPATH=" << TEST_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=mlir-llvm %1 2> %FILENAME%.mlir" << std::endl;
    batFile << "%EXEPATH%\\mlir-translate.exe --mlir-to-llvmir -o=%FILENAME%.il %FILENAME%.mlir" << std::endl;
    batFile << "%EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\lld.exe -flavor link %FILENAME%.o \"%LIBPATH%\\libcmt.lib\" \"%LIBPATH%\\libvcruntime.lib\" \"%LIBPATH%\\kernel32.lib\" \"%LIBPATH%\\libucrt.lib\" \"%LIBPATH%\\uuid.lib\"" << std::endl;
    batFile.close();
}

void testFile(const char *file)
{
    // compile
    std::stringstream ss;
    ss << "compile.bat " << file;
    auto compileResult = exec(ss.str());

    std::cout << "Compiling: " << std::endl;
    std::cout << compileResult << std::endl;

    ASSERT_THROW_MSG(exists("test_run.exe"), "compile error");

    // run
    auto result = exec("test_run.exe");

    std::cout << "Test result: " << std::endl;
    std::cout << result << std::endl;
}

int main(int argc, char **argv)
{
    createBatchFile();
    try
    {
        testFile(TEST_FILE);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
        throw;
    }

    return 0;
}