#include "helper.h"

//#define NEW_BAT 1

#if WIN32
#define GC_LIB "gcmt-lib"
#else
#define GC_LIB "-lgcmt-lib"
#endif
// for Ubuntu 20.04 add -ldl and optionally -rdynamic 
#define LIBS "-frtti -fexceptions -lstdc++ -lrt -ldl -lpthread -lm -lz -ltinfo"
//#define LIBS "-frtti -fexceptions -lstdc++ -lrt -ldl -lpthread -lm -lz -ltinfo -lxml2"
#define TYPESCRIPT_ASYNC_LIB "-lTypeScriptAsyncRuntime -lLLVMSupport -lLLVMDemangle"

//#define LINUX_ASYNC_ENABLED 1

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

#ifndef TEST_LLVM_EXEPATH
//#define TEST_LLVM_EXEPATH "C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin"
#error TEST_LLVM_EXEPATH must be provided
#endif

#ifndef TEST_LLVM_LIBPATH
//#define TEST_LLVM_LIBPATH "C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/lib"
#error TEST_LLVM_LIBPATH must be provided
#endif

#ifndef TEST_TSC_EXEPATH
//#define TEST_TSC_EXEPATH "C:/dev/TypeScriptCompiler/__build/tsc/bin"
#error TEST_TSC_EXEPATH must be provided
#endif

#ifndef TEST_TSC_LIBPATH
//#define TEST_TSC_LIBPATH "C:/dev/TypeScriptCompiler/__build/tsc/lib"
#error TEST_TSC_LIBPATH must be provided
#endif

#ifndef TEST_GCPATH
//#define TEST_GCPATH "C:\dev\TypeScriptCompiler\3rdParty\gc\Release"
#error TEST_GCPATH must be provided
#endif

#ifndef TEST_FILE
#define TEST_FILE "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00funcs_capture.ts"
#endif

#define _OPT_ "--opt "

bool isJit = false;
bool isJitCompile = false;
bool isLlc = false;
bool enableBuiltins = false;
bool noGC = false;
bool asyncRuntime = false;
bool llvmGenerate = false;

#ifdef WIN32
void createCompileBatchFile()
{
#ifndef NEW_BAT
    if (exists("compile" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=obj " _OPT_ "%2 -o=%FILENAME%.o" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/defaultlib:libcmt" _D_ ".lib libvcruntime" _D_ ".lib"
            << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createCompileBatchFileWithAsyncRT()
{
#ifndef NEW_BAT
    if (exists("compile_async" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_async" _D_ ".bat");
    // batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=obj " _OPT_ "%2 -o=%FILENAME%.o" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/libpath:%LLVM_LIBPATH% /libpath:%TSCLIBPATH% /defaultlib:libcmt" _D_ ".lib libvcruntime" _D_
               ".lib TypeScriptAsyncRuntime.lib LLVMSupport.lib"
            << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile.close();
}

void createCompileBatchFileGC()
{
#ifndef NEW_BAT
    if (exists("compile_gc" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=obj " _OPT_ "%2 -o=%FILENAME%.o" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/libpath:%GCLIBPATH% msvcrt" _D_ ".lib ucrt" _D_ ".lib kernel32.lib user32.lib "
            << GC_LIB << ".lib" << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createCompileBatchFileGCWithAsyncRT()
{
#ifndef NEW_BAT
    if (exists("compile_gc_async" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc_async" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "set GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=obj " _OPT_ "%2 -o=%FILENAME%.o" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/libpath:%GCLIBPATH% /libpath:%LLVM_LIBPATH% /libpath:%TSCLIBPATH% "
               "msvcrt" _D_ ".lib ucrt" _D_ ".lib kernel32.lib user32.lib "
            << GC_LIB << ".lib TypeScriptAsyncRuntime.lib LLVMSupport.lib" << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createCompileBatchFile_LLC()
{
#ifndef NEW_BAT
    if (exists("compile_llc" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_llc" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm " _OPT_ "%2 -o=%FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/defaultlib:libcmt" _D_ ".lib libvcruntime" _D_ ".lib"
            << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createCompileBatchFileWithAsyncRT_LLC()
{
#ifndef NEW_BAT
    if (exists("compile_async_llc" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_async_llc" _D_ ".bat");
    // batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm " _OPT_ "%2 -o=%FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/libpath:%LLVM_LIBPATH% /libpath:%TSCLIBPATH% /defaultlib:libcmt" _D_ ".lib libvcruntime" _D_
               ".lib TypeScriptAsyncRuntime.lib LLVMSupport.lib"
            << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile.close();
}

void createCompileBatchFileGC_LLC()
{
#ifndef NEW_BAT
    if (exists("compile_gc_llc" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc_llc" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm " _OPT_ "%2 -o=%FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/libpath:%GCLIBPATH% msvcrt" _D_ ".lib ucrt" _D_ ".lib kernel32.lib user32.lib "
            << GC_LIB << ".lib" << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createCompileBatchFileGCWithAsyncRT_LLC()
{
#ifndef NEW_BAT
    if (exists("compile_gc_async_llc" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc_async_llc" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "set TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "set GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=llvm " _OPT_ "%2 -o=%FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\llc.exe --filetype=obj -o=%FILENAME%.o %FILENAME%.il" << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/libpath:%GCLIBPATH% /libpath:%LLVM_LIBPATH% /libpath:%TSCLIBPATH% "
               "msvcrt" _D_ ".lib ucrt" _D_ ".lib kernel32.lib user32.lib "
            << GC_LIB << ".lib TypeScriptAsyncRuntime.lib LLVMSupport.lib" << std::endl;
    batFile << "del %FILENAME%.il" << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createJitCompileBatchFile()
{
#ifndef NEW_BAT
    if (exists("compile_jit" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_jit" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=jit -nogc --shared-libs=%TSCEXEPATH%/TypeScriptRuntime.dll -dump-object-file "
               "-object-filename=%FILENAME%.o %2"
            << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/defaultlib:libcmt" _D_ ".lib libvcruntime" _D_ ".lib"
            << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createJitCompileBatchFileGC()
{
#ifndef NEW_BAT
    if (exists("compile_jit_gc" _D_ ".bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_jit_gc" _D_ ".bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=jit --shared-libs=%TSCEXEPATH%/TypeScriptRuntime.dll -dump-object-file "
               "-object-filename=%FILENAME%.o %2"
            << std::endl;
    batFile << "%LLVM_EXEPATH%\\lld.exe -flavor link %FILENAME%.o /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH% "
               "/defaultlib:libcmt" _D_ ".lib libvcruntime" _D_ ".lib"
            << std::endl;
    batFile << "del %FILENAME%.o" << std::endl;
    batFile << "call %FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "echo on" << std::endl;
    batFile.close();
}

void createLLVMBatchFile()
{
#ifndef NEW_BAT
    if (exists("llvm.bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("llvm.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "echo on" << std::endl;
    batFile
        << "%TSCEXEPATH%\\tsc.exe --emit=llvm -opt -o=- %2 1> %FILENAME%.txt 2> %FILENAME%.err"
        << std::endl;
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
    batFile << "set LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "echo on" << std::endl;
    batFile
        << "%TSCEXEPATH%\\tsc.exe --emit=jit -nogc --shared-libs=%LLVMPATH%/TypeScriptRuntime.dll %2 1> %FILENAME%.txt 2> %FILENAME%.err"
        << std::endl;
    batFile.close();
}

void createJitBatchFileGC()
{
#ifndef NEW_BAT
    if (exists("jit_gc.bat"))
    {
        return;
    }
#endif

    std::ofstream batFile("jit_gc.bat");
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "%TSCEXEPATH%\\tsc.exe --emit=jit --shared-libs=%TSCEXEPATH%/TypeScriptRuntime.dll %2 1> %FILENAME%.txt 2> %FILENAME%.err"
            << std::endl;
    batFile.close();
}
#else
void createCompileBatchFile()
{
#ifndef NEW_BAT
    if (exists("compile.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=obj " _OPT_ "-nogc $2 -relocation-model=pic -o=$FILENAME.il" << std::endl;
    batFile << "gcc -o $FILENAME $FILENAME.o -lm -frtti -fexceptions -lstdc++" << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createCompileBatchFileWithAsyncRT()
{
#ifndef NEW_BAT
    if (exists("compile_async.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_async.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=obj " _OPT_ "-nogc $2 -relocation-model=pic -o=$FILENAME.o" << std::endl;
    batFile << "gcc -o $FILENAME $FILENAME.o -L$LLVM_LIBPATH -L$TSCLIBPATH " << TYPESCRIPT_ASYNC_LIB << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createCompileBatchFileGC()
{
#ifndef NEW_BAT
    if (exists("compile_gc.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=obj " _OPT_ "$2  -relocation-model=pic -o=$FILENAME.o" << std::endl;
    batFile << "gcc -o $FILENAME -L$GCLIBPATH $FILENAME.o " << GC_LIB << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createCompileBatchFileGCWithAsyncRT()
{
#ifndef NEW_BAT
    if (exists("compile_gc_async.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc_async.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=obj " _OPT_ "$2 -relocation-model=pic -o=$FILENAME.o" << std::endl;
    batFile << "gcc -o $FILENAME -L$LLVM_LIBPATH -L$GCLIBPATH -L$TSCLIBPATH $FILENAME.o " << GC_LIB
            << " " TYPESCRIPT_ASYNC_LIB << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createCompileBatchFile_LLC()
{
#ifndef NEW_BAT
    if (exists("compile_llc.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_llc.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=llvm " _OPT_ "-nogc $2 -o=$FILENAME.il" << std::endl;
    batFile << "$LLVM_EXEPATH/llc -relocation-model=pic --filetype=obj -o=$FILENAME.o $FILENAME.il" << std::endl;
    batFile << "gcc -o $FILENAME $FILENAME.o -lm -frtti -fexceptions -lstdc++" << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createCompileBatchFile_LLC_WithAsyncRT()
{
#ifndef NEW_BAT
    if (exists("compile_async_llc.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_async_llc.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=llvm " _OPT_ "-nogc $2 -o=$FILENAME.il" << std::endl;
    batFile << "$LLVM_EXEPATH/llc -relocation-model=pic --filetype=obj -o=$FILENAME.o $FILENAME.il" << std::endl;
    batFile << "gcc -o $FILENAME $FILENAME.o -L$LLVM_LIBPATH -L$TSCLIBPATH " << TYPESCRIPT_ASYNC_LIB << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createCompileBatchFile_LLC_GC()
{
#ifndef NEW_BAT
    if (exists("compile_gc_llc.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc_llc.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=llvm " _OPT_ "$2 -o=$FILENAME.il" << std::endl;
    batFile << "$LLVM_EXEPATH/llc -relocation-model=pic --filetype=obj -o=$FILENAME.o $FILENAME.il" << std::endl;
    batFile << "gcc -o $FILENAME -L$GCLIBPATH $FILENAME.o " << GC_LIB << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createCompileBatchFile_LLC_GCWithAsyncRT()
{
#ifndef NEW_BAT
    if (exists("compile_gc_async_llc.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_gc_async_llc.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "TSCLIBPATH=" << TEST_TSC_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=llvm " _OPT_ "$2 -o=$FILENAME.il" << std::endl;
    batFile << "$LLVM_EXEPATH/llc -relocation-model=pic --filetype=obj -o=$FILENAME.o $FILENAME.il" << std::endl;
    batFile << "gcc -o $FILENAME -L$LLVM_LIBPATH -L$GCLIBPATH -L$TSCLIBPATH $FILENAME.o " << GC_LIB
            << " " TYPESCRIPT_ASYNC_LIB << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createJitCompileBatchFile()
{
#ifndef NEW_BAT
    if (exists("compile_jit.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_jit.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=jit -nogc --shared-libs=../../lib/libTypeScriptRuntime.so -dump-object-file "
               "-object-filename=$FILENAME.o $2"
            << std::endl;
    batFile << "gcc -o $FILENAME $FILENAME.o"
            << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createJitCompileBatchFileGC()
{
#ifndef NEW_BAT
    if (exists("compile_jit_gc.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("compile_jit_gc.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "GCLIBPATH=" << TEST_GCPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=jit " _OPT_
               " --shared-libs=../../lib/libTypeScriptRuntime.so -dump-object-file -object-filename=$FILENAME.o $2"
            << std::endl;
    batFile << "gcc -o $FILENAME -L$GCLIBPATH $FILENAME.o " << GC_LIB << " " << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm $FILENAME.o" << std::endl;
    batFile << "rm $FILENAME" << std::endl;
    batFile.close();
}

void createLLVMBatchFile()
{
#ifndef NEW_BAT
    if (exists("llvm.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("llvm.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=llvm -opt -o=- " _OPT_ " $2 1> $FILENAME.txt 2> $FILENAME.err"
            << std::endl;
    batFile.close();
}

void createJitBatchFile()
{
#ifndef NEW_BAT
    if (exists("jit.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("jit.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=jit "
               " -nogc --shared-libs=../../lib/libTypeScriptRuntime.so $2 1> $FILENAME.txt 2> $FILENAME.err"
            << std::endl;
    batFile.close();
}

void createJitBatchFileGC()
{
#ifndef NEW_BAT
    if (exists("jit_gc.sh"))
    {
        return;
    }
#endif

    std::ofstream batFile("jit_gc.sh");
    batFile << "FILENAME=$1" << std::endl;
    batFile << "LLVMPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "TSCEXEPATH=" << TEST_TSC_EXEPATH << std::endl;
    batFile << "$TSCEXEPATH/tsc --emit=jit --shared-libs=../../lib/libTypeScriptRuntime.so $2 1> $FILENAME.txt 2> $FILENAME.err"
            << std::endl;
    batFile.close();
}
#endif

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

        if (anyError)
        {
            auto errStr = errors.str();
            return errStr;
        }

        if (!anyDoneMsg && !llvmGenerate)
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

    if (llvmGenerate)
    {
        ss << RUN_CMD << "llvm" << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
    }
    else if (isJit)
    {
        if (noGC)
        {
            ss << RUN_CMD << "jit" << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
        else
        {
            ss << RUN_CMD << "jit_gc" << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
    }
    else if (isJitCompile)
    {
        if (noGC)
        {
            ss << RUN_CMD << "compile_jit" << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
        else
        {
            ss << RUN_CMD << "compile_jit_gc" << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
    }
    else if (isLlc)
    {
        if (asyncRuntime)
        {
            if (noGC)
            {
                ss << RUN_CMD << "compile_async_llc" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
            }
            else
            {
                ss << RUN_CMD << "compile_gc_async_llc" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
            }
        }
        else
        {
            if (noGC)
            {
                ss << RUN_CMD << "compile_llc" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
            }
            else
            {
                ss << RUN_CMD << "compile_gc_llc" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
            }
        }
    }
    else if (asyncRuntime)
    {
        if (noGC)
        {
            ss << RUN_CMD << "compile_async" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
        else
        {
            ss << RUN_CMD << "compile_gc_async" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
    }
    else
    {
        if (noGC)
        {
            ss << RUN_CMD << "compile" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
        else
        {
            ss << RUN_CMD << "compile_gc" _D_ << BAT_NAME << stem.generic_string() << ms.count() << " " << file;
        }
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
                isLlc = true;
            }
            else if (std::string(argv[index]) == "-builtins")
            {
                enableBuiltins = true;
            }
            else if (std::string(argv[index]) == "-nogc")
            {
                noGC = true;
            }
            else if (std::string(argv[index]) == "-async")
            {
                asyncRuntime = true;
            }
            else if (std::string(argv[index]) == "-llvm")
            {
                llvmGenerate = true;
            }
            else
            {
                filePath = argv[index];
            }
        }

        if (llvmGenerate)
        {
            createLLVMBatchFile();
        }
        else if (isJit)
        {
            if (noGC)
            {
                createJitBatchFile();
            }
            else
            {
                createJitBatchFileGC();
            }
        }
        else if (isJitCompile)
        {
            if (noGC)
            {
                createJitCompileBatchFile();
            }
            else
            {
                createJitCompileBatchFileGC();
            }
        }
        else if (isLlc)
        {
            if (asyncRuntime)
            {
                if (noGC)
                {
                    createCompileBatchFileWithAsyncRT_LLC();
                }
                else
                {
                    createCompileBatchFileGCWithAsyncRT_LLC();
                }
            }
            else
            {
                if (noGC)
                {
                    createCompileBatchFile_LLC();
                }
                else
                {
                    createCompileBatchFileGC_LLC();
                }
            }            
        }
        else if (asyncRuntime)
        {
            if (noGC)
            {
                createCompileBatchFileWithAsyncRT();
            }
            else
            {
                createCompileBatchFileGCWithAsyncRT();
            }
        }
        else
        {
            if (noGC)
            {
                createCompileBatchFile();
            }
            else
            {
                createCompileBatchFileGC();
            }
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