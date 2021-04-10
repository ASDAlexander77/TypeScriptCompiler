#include "helper.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

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

#ifndef TEST_VCPATH
#define TEST_VCPATH "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/VC/lib"
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

#ifndef TEST_FILE
#define TEST_FILE "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/02numbers.ts"
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
            std::cerr << "Error: return code is not 0";
        }
    }
    else
    {
        std::cerr << "Error: Failed to read the pipe to the end";
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
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set VCPATH=" << TEST_VCPATH << std::endl;
    batFile << "set SDKPATH=" << TEST_SDKPATH << std::endl;
    batFile << "set EXEPATH=" << TEST_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm %2 2> %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%EXEPATH%\\lld.exe -flavor link %FILENAME%.o \"%VCPATH%\\libcmt.lib\" \"%VCPATH%\\libvcruntime.lib\" \"%SDKPATH%\\kernel32.lib\" \"%SDKPATH%\\libucrt.lib\" \"%SDKPATH%\\uuid.lib\"" << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile.close();
}

void testFile(const char *file)
{
    std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
        std::chrono::system_clock::now().time_since_epoch()
    );

    auto fileName = fs::path(file).filename();
    auto stem = fs::path(file).stem();
    
    std::stringstream sfn;
    sfn << stem << ms.count() << ".exe";
    auto exeFile = sfn.str();

    std::stringstream efn;
    efn << stem << ms.count() << ".err";
    auto errFile = efn.str();    

    std::cout << "Test file: " << fileName << " path: " << file << std::endl;

    auto cleanup = [&]() {
        std::stringstream mask;
        mask << "del " << stem << ms.count() << "*.*";
        auto delCmd = mask.str();

        // read test result
        std::ifstream infile(errFile);
        std::string line;
        std::stringstream errors;
        auto anyError = false;
        while (std::getline(infile, line))
        {
            errors << line << std::endl;
            anyError = true;
        }        

        exec(delCmd);       

        if (anyError)
        {
            auto errStr = errors.str();
            std::cerr << errStr << std::endl;
            return errStr;
        } 

        return std::string();
    };

    // compile
    std::stringstream ss;
    ss << "compile.bat " << stem << ms.count() << " " << file;
    try
    {
        auto compileResult = exec(ss.str());

        std::cout << "Compiling: " << std::endl;
        std::cout << compileResult << std::endl;

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
        createBatchFile();

        if (argc > 1)
        {
            testFile(argv[1]);
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