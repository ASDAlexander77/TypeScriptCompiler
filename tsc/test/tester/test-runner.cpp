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

#ifdef WIN32
#include <windows.h>
#elif _POSIX_C_SOURCE >= 199309L
#include <time.h> // for nanosleep
#else
#include <unistd.h> // for usleep
#endif

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

//#define NEW_BAT 1

//#define SEARCH_LIB 1
//#define SEARCH_SDK 1
#define SEARCH_LIBPATH "\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\""
#define SEARCH_SDKPATH "\"C:\\Program Files (x86)\\Windows Kits\\10\\Lib\""
#define FILTER_LIB "\"lib\\x64\""
#define FILTER_SDK "\"ucrt\\x64\""

#ifndef TEST_LIBPATH
//#define TEST_LIBPATH "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/VC/lib"
#error TEST_LIBPATH must be provided
#endif

#ifndef TEST_SDKPATH
//#define TEST_SDKPATH "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/SDK/lib"
#error TEST_SDKPATH must be provided
#endif

#ifndef TEST_EXEPATH
//#define TEST_EXEPATH "C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin"
#error TEST_EXEPATH must be provided
#endif

#ifndef TEST_TSC_EXEPATH
//#define TEST_TSC_EXEPATH "C:/dev/TypeScriptCompiler/__build/tsc/bin"
#error TEST_TSC_EXEPATH must be provided
#endif

#ifndef TEST_CLANGLIBPATH
//#define TEST_CLANGLIBPATH "C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/lib/clang/13.0.0/lib/windows"
#error TEST_CLANGLIBPATH must be provided
#endif

#ifndef TEST_FILE
#define TEST_FILE "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00funcs_capture.ts"
#endif

bool isJit = true;
bool isJitCompile = true;
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

void sleep_ms(int milliseconds)
{ // cross-platform sleep function
#ifdef WIN32
    Sleep(milliseconds);
#elif _POSIX_C_SOURCE >= 199309L
    struct timespec ts;
    ts.tv_sec = milliseconds / 1000;
    ts.tv_nsec = (milliseconds % 1000) * 1000000;
    nanosleep(&ts, NULL);
#else
    if (milliseconds >= 1000)
        sleep(milliseconds / 1000);
    usleep((milliseconds % 1000) * 1000);
#endif
}

std::string exec(std::string cmd)
{
    auto retry = 3;

    do
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
                if (retry <= 1)
                {
                    std::cerr << "Error: return code is not 0, code: " << code << " cmd: " << cmd << " output: " << result << std::endl;
                }
                else
                {
                    std::cerr << "retrying..." << std::endl;
                    sleep_ms(1000);
                    continue;
                }
            }
        }
        else
        {
            std::cerr << "Error: Failed to read the pipe to the end" << std::endl;
        }

        return result;
    } while (--retry > 0);

    return "";
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
#ifndef NEW_BAT
    if (exists("compile.bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
#ifdef SEARCH_LIB
    batFile << "FOR /F \"tokens=* USEBACKQ\" %%F IN (`where.exe /R " SEARCH_LIBPATH " libvcruntime.lib ^| find " FILTER_LIB
               "`) DO ( SET libname=%%F )"
            << std::endl;
    batFile << "FOR %%A in (\"%libname%\") do ( Set LIBPATH1=\"%%~dpA\" )" << std::endl;
    batFile << "Set LIBPATH=%LIBPATH1:~0,-3%\"" << std::endl;
#else
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
#endif
#ifdef SEARCH_SDK
    batFile << "FOR /F \"tokens=* USEBACKQ\" %%F IN (`where.exe /R " SEARCH_SDKPATH " libucrt.lib ^| find " FILTER_LIB
               "`) DO ( SET libname=%%F )"
            << std::endl;
    batFile << "FOR %%A in (\"%libname%\") do ( Set SDKPATH1=\"%%~dpA\" )" << std::endl;
    batFile << "Set SDKPATH=%SDKPATH1:~0,-3%\"" << std::endl;
#else
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
#endif
    batFile << "set EXEPATH=" << TEST_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm %2 2> %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% libcmt.lib libvcruntime.lib kernel32.lib"
            << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createCompileBatchFileWithRT()
{
#ifndef NEW_BAT
    if (exists("compile_rt.bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_rt.bat");
    // batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
#ifdef SEARCH_LIB
    batFile << "FOR /F \"tokens=* USEBACKQ\" %%F IN (`where.exe /R " SEARCH_LIBPATH " libvcruntime.lib ^| find " FILTER_LIB
               "`) DO ( SET libname=%%F )"
            << std::endl;
    batFile << "FOR %%A in (\"%libname%\") do ( Set LIBPATH1=\"%%~dpA\" )" << std::endl;
    batFile << "Set LIBPATH=%LIBPATH1:~0,-3%\"" << std::endl;
#else
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
#endif
#ifdef SEARCH_SDK
    batFile << "FOR /F \"tokens=* USEBACKQ\" %%F IN (`where.exe /R " SEARCH_SDKPATH " libucrt.lib ^| find " FILTER_LIB
               "`) DO ( SET libname=%%F )"
            << std::endl;
    batFile << "FOR %%A in (\"%libname%\") do ( Set SDKPATH1=\"%%~dpA\" )" << std::endl;
    batFile << "Set SDKPATH=%SDKPATH1:~0,-3%\"" << std::endl;
#else
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
#endif
    batFile << "set EXEPATH=" << TEST_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set CLANGLIBPATH=" << TEST_CLANGLIBPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm %2 2> %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%CLANGLIBPATH% libcmt.lib "
               "libvcruntime.lib kernel32.lib clang_rt.builtins-x86_64.lib"
            << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile.close();
}

void createJitCompileBatchFile()
{
#ifndef NEW_BAT
    if (exists("compile_jit.bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_jit.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
#ifdef SEARCH_LIB
    batFile << "FOR /F \"tokens=* USEBACKQ\" %%F IN (`where.exe /R " SEARCH_LIBPATH " libvcruntime.lib ^| find " FILTER_LIB
               "`) DO ( SET libname=%%F )"
            << std::endl;
    batFile << "FOR %%A in (\"%libname%\") do ( Set LIBPATH1=\"%%~dpA\" )" << std::endl;
    batFile << "Set LIBPATH=%LIBPATH1:~0,-3%\"" << std::endl;
#else
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
#endif
#ifdef SEARCH_SDK
    batFile << "FOR /F \"tokens=* USEBACKQ\" %%F IN (`where.exe /R " SEARCH_SDKPATH " libucrt.lib ^| find " FILTER_LIB
               "`) DO ( SET libname=%%F )"
            << std::endl;
    batFile << "FOR %%A in (\"%libname%\") do ( Set SDKPATH1=\"%%~dpA\" )" << std::endl;
    batFile << "Set SDKPATH=%SDKPATH1:~0,-3%\"" << std::endl;
#else
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
#endif
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=jit -dump-object-file -object-filename=%FILENAME%.o %2" << std::endl;
    batFile << "%EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% libcmt.lib libvcruntime.lib kernel32.lib"
            << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createJitBatchFile()
{
#ifndef NEW_BAT
    if (exists("jit.bat"))
    {
        return;
    }
#endif

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
    sfn << stem.generic_string() << ms.count() << ".exe";
    auto exeFile = sfn.str();

    std::stringstream tfn;
    tfn << stem.generic_string() << ms.count() << ".txt";
    auto txtFile = tfn.str();

    std::stringstream efn;
    efn << stem.generic_string() << ms.count() << ".err";
    auto errFile = efn.str();

    std::cout << "Test file: " << fileName.generic_string() << " path: " << file << std::endl;

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
        ss << "jit.bat " << stem.generic_string() << ms.count() << " " << file;
    }
    else if (isJitCompile)
    {
        ss << "compile_jit.bat " << stem.generic_string() << ms.count() << " " << file;
    }
    else if (enableBuiltins)
    {
        ss << "compile_rt.bat " << stem.generic_string() << ms.count() << " " << file;
    }
    else
    {
        ss << "compile.bat " << stem.generic_string() << ms.count() << " " << file;
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
        throw std::runtime_error(res.c_str());
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
            else if (std::string(argv[index]) == "-llc")
            {
                isJit = false;
                isJitCompile = false;
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
        else if (isJitCompile)
        {
            createJitCompileBatchFile();
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