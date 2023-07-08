#include "helper.h"

#if WIN32
#define GC_LIB "gcmt-lib.lib "
#else
#define GC_LIB "-lgcmt-lib "
#endif
// for Ubuntu 20.04 add -ldl and optionally -rdynamic 
#define LIBS "-frtti -fexceptions -lstdc++ -lrt -ldl -lpthread -lm -lz -ltinfo"
#ifdef WIN32
#define TYPESCRIPT_LIBS "TypeScriptAsyncRuntime.lib LLVMSupport.lib "
#else
#define TYPESCRIPT_LIBS "-lTypeScriptAsyncRuntime -lLLVMSupport -lLLVMDemangle "
#endif

#ifdef WIN32
#ifndef CMAKE_C_STANDARD_LIBRARIES
// #define CMAKE_C_STANDARD_LIBRARIES kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib
#error CMAKE_C_STANDARD_LIBRARIES must be provided
#endif

#ifndef TEST_LIBPATH
//#define TEST_LIBPATH "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30037\lib\x64"
#error TEST_LIBPATH must be provided
#endif

#ifndef TEST_SDKPATH
//#define TEST_SDKPATH "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\um\x64"
#error TEST_SDKPATH must be provided
#endif

#ifndef TEST_UCRTPATH
//#define TEST_UCRTPATH "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.18362.0\ucrt\x64"
#error TEST_UCRTPATH must be provided
#endif
#endif

#ifndef TEST_LLVM_EXEPATH
#error TEST_LLVM_EXEPATH must be provided
#endif

#ifndef TEST_LLVM_LIBPATH
#error TEST_LLVM_LIBPATH must be provided
#endif

#ifndef TEST_TSC_EXEPATH
#error TEST_TSC_EXEPATH must be provided
#endif

#ifndef TEST_TSC_LIBPATH
#error TEST_TSC_LIBPATH must be provided
#endif

#ifndef TEST_GCPATH
#error TEST_GCPATH must be provided
#endif

#ifndef WIN32
#error TEST_COMPILER must be provided
#endif

#ifndef TEST_FILE
#define TEST_FILE "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00funcs_capture.ts"
#endif

bool jitRun = false;
bool sharedLibCompiler = false;
bool opt = true;

void createCompileBatchFile()
{
    if (exists("compile.bat"))
    {
        return;
    }

    std::ofstream batFile("compile.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set FILEPATH=%2" << std::endl;
    batFile << "set TSC_OPTS=%3" << std::endl;
    batFile << "set LINKER_OPTS=%4" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVMEXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVMLIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "set GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=obj %TSC_OPTS% %FILEPATH% -o=%FILENAME%.obj" << std::endl;
    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link %FILENAME%.obj %LINKER_OPTS% " << GC_LIB << TYPESCRIPT_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GCLIBPATH% /libpath:%LLVMLIBPATH% /libpath:%TSCLIBPATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;
    batFile << "del %FILENAME%.obj" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
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
#if WIN32
        mask << "del " << stem << ms.count() << ".*";
#else
        mask << "rm " << stem << ms.count() << ".*";
#endif
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

        if (anyDoneMsg)
        {
            return std::string();
        }

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
#if WIN32
#define RUN_CMD ""
#define BAT_NAME ".bat "
#else
#define RUN_CMD "/bin/sh -f ./"
#define BAT_NAME ".sh "
#endif

    ss << RUN_CMD << "compile" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
    ss << opt ? "--opt" : "--opt_level=0";
    if (sharedLibCompiler)
    {
#if WIN32
        ss << " /DLL";
#else
        ss << " -shared";
#endif        
    }

    try
    {
        auto compileResult = exec(ss.str());

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
                jitRun = true;
            }
            else if (std::string(argv[index]) == "-shared")
            {
                sharedLibCompiler = true;
            }
            else if (std::string(argv[index]) == "-noopt")
            {
                opt = false;
            }
            else
            {
                filePath = argv[index];
            }
        }

        createCompileBatchFile();
        testFile(filePath);
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}