#include "helper.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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

#ifndef TEST_SDKPATH
#define TEST_SDKPATH "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/SDK/lib"
#endif

#ifndef TEST_EXEPATH
#define TEST_EXEPATH "C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin"
#endif

#ifndef TEST_TSC_EXEPATH
#define TEST_TSC_EXEPATH "C:/dev/TypeScriptCompiler/__build/tsc/bin"
#endif

#ifndef TEST_CLANGLIBPATH
#define TEST_CLANGLIBPATH "C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/lib/clang/13.0.0/lib/windows"
#endif

#ifndef TEST_FILE
#define TEST_FILE "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00funcs_capture.ts"
#endif

bool isJit = true;
bool enableBuiltins = false;

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

    FILE *pipe = POPEN(cmd.c_str(), "rt");
    if (!pipe)
    {
        throw std::runtime_error("popen() failed!");
    }

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
    {
        result += buffer.data();
    }

    if (feof(pipe))
    {
        auto code = PCLOSE(pipe);
        if (code)
        {
            std::cerr << "Error: return code is not 0" << std::endl;
        }
    }
    else
    {
        std::cerr << "Error: Failed to read the pipe to the end" << std::endl;
    }

    return result;
}

inline bool exists(std::string name)
{
    return fs::exists(name);
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

void createCompileBatchFile()
{
    if (exists("compile.bat"))
    {
        return;
    }

    std::ofstream batFile("compile.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=" << TEST_LIBPATH << std::endl;
    batFile << "set SDKPATH=" << TEST_SDKPATH << std::endl;
    batFile << "set EXEPATH=" << TEST_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm %2 2> %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\lld.exe -flavor link %FILENAME%.o \"%LIBPATH%\\libcmt.lib\" \"%LIBPATH%\\libvcruntime.lib\" "
               "\"%SDKPATH%\\kernel32.lib\" \"%SDKPATH%\\libucrt.lib\" \"%SDKPATH%\\uuid.lib\""
            << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createCompileBatchFileWithRT()
{
    if (exists("compile_rt.bat"))
    {
        return;
    }

    std::ofstream batFile("compile_rt.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=" << TEST_LIBPATH << std::endl;
    batFile << "set SDKPATH=" << TEST_SDKPATH << std::endl;
    batFile << "set EXEPATH=" << TEST_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set CLANGLIBPATH=" << TEST_CLANGLIBPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm %2 2> %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile
        << "%EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:\"%LIBPATH%\" \"%LIBPATH%\\libcmt.lib\" \"%LIBPATH%\\libvcruntime.lib\" "
           "\"%SDKPATH%\\kernel32.lib\" \"%SDKPATH%\\libucrt.lib\" \"%SDKPATH%\\uuid.lib\" \"%CLANGLIBPATH%\\clang_rt.builtins-x86_64.lib\""
        << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile.close();
}

void createJitBatchFile()
{
    if (exists("jit.bat"))
    {
        return;
    }

    std::ofstream batFile("jit.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=jit %2 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile.close();
}

void testFile(const char *file)
{
    std::chrono::milliseconds ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    auto fileName = fs::path(file).filename();
    auto stem = fs::path(file).stem();

    std::stringstream sfn;
    sfn << stem << ms.count() << ".exe";
    auto exeFile = sfn.str();

    std::stringstream tfn;
    tfn << stem << ms.count() << ".txt";
    auto txtFile = tfn.str();

    std::stringstream efn;
    efn << stem << ms.count() << ".err";
    auto errFile = efn.str();

    std::cout << "Test file: " << fileName << " path: " << file << std::endl;

    auto cleanup = [&]() {
        std::stringstream mask;
        mask << "del " << stem << ms.count() << ".*";
        auto delCmd = mask.str();

        // read output result
        std::ifstream infileO;
        infileO.open(txtFile, std::fstream::in);
        std::string lineO;
        auto anyDoneMsg = false;
        while (std::getline(infileO, lineO))
        {
            if (lineO.find("done.") != std::string::npos)
            {
                anyDoneMsg = true;
            }
        }

        infileO.close();

        // read test result
        std::ifstream infile;
        infile.open(errFile, std::fstream::in);
        std::string line;
        std::stringstream errors;
        auto anyError = false;
        while (std::getline(infile, line))
        {
            errors << line << std::endl;
            anyError = true;
        }

        infile.close();

        exec(delCmd);

        if (anyError)
        {
            auto errStr = errors.str();
            return errStr;
        }

        if (!anyDoneMsg)
        {
            return std::string("no 'done.' msg.");
        }

        return std::string();
    };

    // compile
    std::stringstream ss;
    if (isJit)
    {
        ss << "jit.bat " << stem << ms.count() << " " << file;
    }
    else if (enableBuiltins)
    {
        ss << "compile_rt.bat " << stem << ms.count() << " " << file;
    }
    else
    {
        ss << "compile.bat " << stem << ms.count() << " " << file;
    }

    try
    {
        auto compileResult = exec(ss.str());

        // std::cout << std::endl << "Compiling: " << std::endl;
        // std::cout << compileResult << std::endl;

        auto index = compileResult.find("error:");
        if (index != std::string::npos)
        {
            throw "compile error";
        }

        index = compileResult.find("failed");
        if (index != std::string::npos)
        {
            throw "run error";
        }
    }
    catch (const std::exception &)
    {
    }

    auto res = cleanup();
    if (!res.empty())
    {
        throw std::exception(res.c_str());
    }
}

int main(int argc, char **argv)
{
    try
    {
        char *filePath = nullptr;
        auto index = 1;
        for (; index < argc; index++)
        {
            if (std::string(argv[index]) == "-jit")
            {
                isJit = true;
            }
            else if (std::string(argv[index]) == "-builtins")
            {
                enableBuiltins = true;
            }
            else
            {
                filePath = argv[index];
            }
        }

        if (isJit)
        {
            createJitBatchFile();
        }
        else
        {
            createCompileBatchFileWithRT();
        }

        if (index > 1)
        {
            testFile(filePath);
        }
        else
        {
            testFile(TEST_FILE);
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}