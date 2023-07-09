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

#if WIN32
#define RUN_CMD ""
#define BAT_NAME ".bat "
#else
#define RUN_CMD "/bin/sh -f ./"
#define BAT_NAME ".sh "
#endif

bool jitRun = false;
bool sharedLibCompiler = false;
bool opt = true;

void createJitBatchFile()
{
    if (exists("jit.bat"))
    {
        return;
    }

    std::ofstream batFile("jit_gc.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set FILEPATH=%2" << std::endl;
    batFile << "set TSC_OPTS=%3" << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=jit %TSC_OPTS% --shared-libs=%TSCEXEPATH%/TypeScriptRuntime.dll %FILEPATH% 1> %FILENAME%.txt 2> %FILENAME%.err"
            << std::endl;
    batFile.close();
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

void buildJitExecCommand(std::stringstream &ss, std::string fileNameNoExt, const char *file)
{
    ss << RUN_CMD << "jit" << BAT_NAME << fileNameNoExt << " " << file;
    ss << opt ? "--opt" : "--opt_level=0";
}

void buildCompileExecCommand(std::stringstream &ss, std::string fileNameNoExt, const char *file)
{
    ss << RUN_CMD << "compile" _D_ << BAT_NAME << fileNameNoExt << " " << file;
    ss << opt ? "--opt" : "--opt_level=0";
    if (sharedLibCompiler)
    {
#if WIN32
        ss << " /DLL";
#else
        ss << " -shared";
#endif        
    }    
}

std::string buildExecCommand(std::string tempOutputFileNameNoExt, const char *file)
{
    std::stringstream ss;
    if (jitRun)
    {
        buildJitExecCommand(ss, tempOutputFileNameNoExt, file);
    }
    else
    {
        buildCompileExecCommand(ss, tempOutputFileNameNoExt, file);
    }

    return ss.str();
}

std::string readOutput(std::string fileName)
{
    std::stringstream output;

    std::ifstream fileInputStream;
    fileInputStream.open(fileName, std::fstream::in);

    std::string line;
    while (std::getline(fileInputStream, line))
    {
        output << line << std::endl;
    }

    fileInputStream.close();    

    return output.str();
}

void deleteFiles(std::string tempOutputFileNameNoExt)
{
    std::stringstream mask;
#if WIN32
    mask << "del " << tempOutputFileNameNoExt << ".*";
#else
    mask << "rm " << tempOutputFileNameNoExt << ".*";
#endif

    auto delCmd = mask.str();
    exec(delCmd);
}

std::string checkOutputAndCleanup(std::string tempOutputFileNameNoExt)
{
    auto txtFile = tempOutputFileNameNoExt + ".txt";
    auto errFile = tempOutputFileNameNoExt + ".err";

    auto output = readOutput(txtFile);
    auto errors = readOutput(errFile);

    deleteFiles(tempOutputFileNameNoExt);    

    if (output.find("done.") != std::string::npos)
    {
        return std::string();
    }

    if (!errors.empty())
    {
        return errors;
    }

    return "no 'done.' msg.";
}

std::string getTempOutputFileNameNoExt(const char *file)
{
    std::chrono::milliseconds ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    auto fileName = fs::path(file).filename();
    auto stem = fs::path(file).stem();

    std::stringstream fn;
    fn << stem.generic_string() << ms.count();
    auto fileNameNoExt = fn.str();

    std::cout << "Test file: " << fileName.generic_string() << " path: " << file << std::endl;

    return fileNameNoExt;
}

void checkExecOutput(std::string compileResult)
{
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

void checkedExecCommand(std::string batchFileCmd)
{
    try
    {
        checkExecOutput(exec(batchFileCmd));
    }
    catch (const std::exception &)
    {
    }
}

void testFile(const char *file)
{
    auto tempOutputFileNameNoExt = getTempOutputFileNameNoExt(file);

    checkedExecCommand(buildExecCommand(tempOutputFileNameNoExt, file));

    auto res = checkOutputAndCleanup(tempOutputFileNameNoExt);
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