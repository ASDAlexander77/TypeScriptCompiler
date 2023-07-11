#include "helper.h"

#if WIN32
#define GC_LIB "gcmt-lib.lib "
#else
#define GC_LIB "-lgcmt-lib "
#endif
#ifdef WIN32
#define TYPESCRIPT_LIB "TypeScriptAsyncRuntime.lib "
#define LLVM_LIBS "LLVMSupport.lib "
#define LIBS "msvcrt.lib ucrt"_D_".lib "
#else
// for Ubuntu 20.04 add -ldl and optionally -rdynamic 
#define LIBS "-frtti -fexceptions -lstdc++ -lrt -ldl -lpthread -lm -ltinfo"
#define TYPESCRIPT_LIB "-lTypeScriptAsyncRuntime "
#define LLVM_LIBS "-lLLVMSupport -lLLVMDemangle "
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
#ifndef TEST_COMPILER
#error TEST_COMPILER must be provided
#endif
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

#if WIN32
#define SHARED_LIB_OPT "/DLL"
#else
#define SHARED_LIB_OPT "-shared"
#endif        

bool jitRun = false;
bool sharedLibCompiler = false;
bool opt = true;

void createJitBatchFile()
{
#ifdef WIN32    
    if (exists("jit.bat"))
    {
        return;
    }

    std::ofstream batFile("jit.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set FILEPATH=%2" << std::endl;
    batFile << "set TSC_OPTS=%3" << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=jit %TSC_OPTS% --shared-libs=%TSCEXEPATH%/TypeScriptRuntime.dll %FILEPATH% 1> %FILENAME%.txt 2> %FILENAME%.err"
            << std::endl;
    batFile.close();
#else
    if (exists("jit.sh"))
    {
        return;
    }

    std::ofstream batFile("jit.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "FILEPATH=$2" << std::endl;
    batFile << "TSC_OPTS=$3" << std::endl;
    batFile << "LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=jit $TSC_OPTS --shared-libs=../../lib/libTypeScriptRuntime.so $FILEPATH 1> $FILENAME.txt 2> $FILENAME.err"
            << std::endl;
    batFile.close();    
#endif    
}

void createCompileBatchFile()
{
#ifdef WIN32     
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
    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link %FILENAME%.obj %LINKER_OPTS% " 
            << LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GCLIBPATH% /libpath:%LLVMLIBPATH% /libpath:%TSCLIBPATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;
    batFile << "del %FILENAME%.obj" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
#else
    if (exists("compile.sh"))
    {
        return;
    }

    std::ofstream batFile("compile.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "FILEPATH=$2" << std::endl;
    batFile << "TSC_OPTS=$3" << std::endl;
    batFile << "LINKER_OPTS=$4" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=obj $TSC_OPTS $FILEPATH -relocation-model=pic -o=$FILENAME.o" << std::endl;
    batFile << TEST_COMPILER << " -o $FILENAME $LINKER_OPTS -L$LLVM_LIBPATH -L$GCLIBPATH -L$TSCLIBPATH $FILENAME.o " 
            << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();    
#endif    
}

void createBatchFile()
{
    if (jitRun)
    {
        createJitBatchFile();
    }
    else
    {
        createCompileBatchFile();
    }
}

void buildJitExecCommand(std::stringstream &ss, std::string fileNameNoExt, std::string file)
{
    ss << RUN_CMD << "jit" << BAT_NAME << fileNameNoExt << " " << file;
    ss << " " << (opt ? "--opt" : "--opt_level=0");
}

void buildCompileExecCommand(std::stringstream &ss, std::string fileNameNoExt, std::string file)
{
    ss << RUN_CMD << "compile" << BAT_NAME << fileNameNoExt << " " << file;
    ss << " " << (opt ? "--opt" : "--opt_level=0");
    if (sharedLibCompiler)
    {
        ss << SHARED_LIB_OPT;
    }    
}

std::string buildExecCommand(std::string tempOutputFileNameNoExt, std::string file)
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

std::string getTempOutputFileNameNoExt(std::string file)
{
    std::chrono::milliseconds ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    std::string fileNameNoExt = fs::path(file).stem().string();
    auto fileNameNoExtWithMs = fileNameNoExt + std::to_string(ms.count());

    std::cout << "Test file: " << fileNameNoExtWithMs << " path: " << file << std::endl;

    return fileNameNoExtWithMs;
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

void testFile(std::string file)
{
    auto tempOutputFileNameNoExt = getTempOutputFileNameNoExt(file);

    checkedExecCommand(buildExecCommand(tempOutputFileNameNoExt, file));

    auto res = checkOutputAndCleanup(tempOutputFileNameNoExt);
    if (!res.empty())
    {
        throw std::runtime_error(res.c_str());
    }
}

void createMultiCompileBatchFile(std::string tempOutputFileNameNoExt, std::vector<std::string> &files)
{
    auto tsc_opt = opt ? "--opt" : "--opt_level=0";

#ifdef WIN32
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVMEXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVMLIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "set GCLIBPATH=" << TEST_GCPATH << std::endl;

    std::stringstream objs;
    for (auto &file : files)
    {
        auto fileNameWithoutExt = fs::path(file).stem().string();
        objs << fileNameWithoutExt << ".obj ";
        batFile << "%TSCEXEPATH%\\tsc.exe --emit=obj " << tsc_opt << " " << file << " -o=" << fileNameWithoutExt << ".obj" << std::endl;
    }

    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link /out:%FILENAME%.exe " << objs.str() << " "
            << LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GCLIBPATH% /libpath:%LLVMLIBPATH% /libpath:%TSCLIBPATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;

    batFile << "del " << objs.str() << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
#else
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;

    std::stringstream objs;
    for (auto &file : files)
    {
        auto fileNameWithoutExt = fs::path(file).stem().string();
        objs << <fileNameWithoutExt << ".o ";
        batFile << "$TSCEXEPATH/tsc --emit=obj " << tsc_opt << " $FILEPATH -relocation-model=pic -o=" << fileNameWithoutExt << ".o" << std::endl;
    }

    batFile << TEST_COMPILER << " -o $FILENAME " << objs.str() 
            << "-L$LLVM_LIBPATH -L$GCLIBPATH -L$TSCLIBPATH "
            << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    
    batFile << "rm " << objs << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();    
#endif    
}

void createSharedMultiCompileBatchFile(std::string tempOutputFileNameNoExt, std::vector<std::string> &files)
{
    auto tsc_opt = opt ? "--opt" : "--opt_level=0";
    auto linker_opt = SHARED_LIB_OPT;

#ifdef WIN32
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVMEXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVMLIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "set GCLIBPATH=" << TEST_GCPATH << std::endl;

    auto first = true;
    std::stringstream shared_objs;
    std::stringstream exec_objs;
    std::string shared_filenameNoExt;
    std::stringstream execBat;
    std::stringstream sharedBat;
    for (auto &file : files)
    {
        auto fileNameWithoutExt = fs::path(file).stem().string();
        if (first)
        {
            exec_objs << fileNameWithoutExt << ".obj ";
        }
        else
        {
            shared_objs << fileNameWithoutExt << ".obj ";
            if (shared_filenameNoExt.empty())
            {
                shared_filenameNoExt = fileNameWithoutExt;
            }
        }

        (first ? execBat : sharedBat) << "%TSCEXEPATH%\\tsc.exe --emit=obj " << tsc_opt << " " << file << " -o=" << fileNameWithoutExt << ".obj" << std::endl;

        first = false;
    }

    batFile << sharedBat.str();
    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link /out:" << shared_filenameNoExt << ".dll " << linker_opt << " " << shared_objs.str() << " "
            <<  LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GCLIBPATH% /libpath:%LLVMLIBPATH% /libpath:%TSCLIBPATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;

    batFile << execBat.str();
    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link /out:%FILENAME%.exe " << exec_objs.str() << " "
            << LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GCLIBPATH% /libpath:%LLVMLIBPATH% /libpath:%TSCLIBPATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;

    batFile << "del " << shared_objs.str() << std::endl;
    batFile << "del " << exec_objs.str() << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
#else
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "SHARED=" << tempOutputFileNameNoExt << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;

    auto first = true;
    std::stringstream shared_objs;
    std::stringstream exec_objs;
    std::string shared_filenameNoExt;
    std::stringstream execBat;
    std::stringstream sharedBat;    
    for (auto &file : files)
    {
        auto fileNameWithoutExt = fs::path(file).stem().string();
        if (first)
        {
            exec_objs << fileNameWithoutExt << ".o ";
        }
        else
        {
            shared_objs << fileNameWithoutExt << ".o ";
            if (shared_filenameNoExt.empty())
            {
                shared_filenameNoExt = fileNameWithoutExt;
            }
        }

        (first ? execBat : sharedBat) << "$TSCEXEPATH/tsc --emit=obj " << tsc_opt << " $FILEPATH -relocation-model=pic -o=" << fileNameWithoutExt << ".o" << std::endl;

        first = false;
    }

    batFile << sharedBat.str();
    batFile << TEST_COMPILER << " -o " << shared_filenameNoExt << " " << linker_opt << " " << shared_objs.str() 
            << "-L$LLVM_LIBPATH -L$GCLIBPATH -L$TSCLIBPATH "
            << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;

    batFile << execBat.str();
    batFile << TEST_COMPILER << " -o $FILENAME " << exec_objs.str() 
            << "-L$LLVM_LIBPATH -L$GCLIBPATH -L$TSCLIBPATH "
            << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;

    batFile << "rm " << shared_objs << std::endl;
    batFile << "rm " << exec_objs << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();    
#endif    
}

std::string buildMultiCompileExecCommand(std::string fileNameNoExt)
{
    std::stringstream ss;
    ss << RUN_CMD << fileNameNoExt << BAT_NAME;
    return ss.str();
}

void testMutliFiles(std::vector<std::string> &files)
{    
    auto tempOutputFileNameNoExt = getTempOutputFileNameNoExt(*files.begin());
    if (sharedLibCompiler)
    {
        createSharedMultiCompileBatchFile(tempOutputFileNameNoExt, files);
    }
    else
    {
        createMultiCompileBatchFile(tempOutputFileNameNoExt, files);
    }

    checkedExecCommand(buildMultiCompileExecCommand(tempOutputFileNameNoExt));

    auto res = checkOutputAndCleanup(tempOutputFileNameNoExt);
    if (!res.empty())
    {
        throw std::runtime_error(res.c_str());
    }    
}

void readParams(int argc, char **argv, std::vector<std::string> &files)
{
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
            files.push_back(argv[index]);
        }
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::vector<std::string> files;
        readParams(argc, argv, files);
        if (files.size() == 1)
        {
            createBatchFile();
            testFile(*files.begin());
        }
        else if (files.size() > 1)
        {
            if (jitRun)
            {
                throw "jit supports 1 file only";    
            }

            testMutliFiles(files);
        }
        else
        {
            throw "no file provided";
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }

    return 0;
}