#include "helper.h"

#if WIN32
#define GC_LIB "gc.lib "
#else
#define GC_LIB "-lgc "
#endif
#ifdef WIN32
#define TYPESCRIPT_LIB "TypeScriptAsyncRuntime.lib "
#define LLVM_LIBS "LLVMSupport.lib "
//#define LIBS "msvcrt" _D_ ".lib ucrt" _D_ ".lib "
// static CRT (/MT[d]) to match LLVM/TypeScript runtime libs and gc.lib; mixing static+dynamic CRT crashes at startup
#define LIBS "libcmt" _D_ ".lib libvcruntime" _D_ ".lib libucrt" _D_ ".lib ntdll.lib "
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

#ifndef TEST_TSLANG_EXEPATH
#error TEST_TSLANG_EXEPATH must be provided
#endif

#ifndef TEST_TSLANG_LIBPATH
#error TEST_TSLANG_LIBPATH must be provided
#endif

#ifndef TEST_GCPATH
#error TEST_GCPATH must be provided
#endif

#ifndef WIN32
#ifndef TEST_COMPILER
#error TEST_COMPILER must be provided
#endif
#endif

#if WIN32
#define RUN_CMD ".\\"
#define BAT_NAME ".bat"
#else
#define RUN_CMD "/bin/sh -f ./"
#define BAT_NAME ".sh"
#endif

#if WIN32
#define SHARED_LIB_OPT "/DLL"
#else
#define SHARED_LIB_OPT "-shared"
#endif        

auto jitRun = false;
auto sharedLib = false;
auto sharedLibCompileTime = false;
auto gctorsAsMethod = false;
#ifndef COMPILE_DEBUG
auto opt = true;
auto tslang_opt = "--opt --opt_level=3";
#define JIT_NAME "jit"
#define COMPILE_NAME "compile"
#else
auto opt = false;
auto tslang_opt = "--di --opt_level=0";
#define JIT_NAME "jitd"
#define COMPILE_NAME "compiled"
#endif

auto tslang_opt_ext = "";

void createJitBatchFile()
{
#ifdef WIN32    
    if (exists(JIT_NAME BAT_NAME))
    {
        return;
    }

    std::ofstream batFile(JIT_NAME BAT_NAME);
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set FILEPATH=%2" << std::endl;
    batFile << "set TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "echo on" << std::endl;
    batFile << "%TSLANGEXEPATH%\\tslang.exe --emit=jit " << tslang_opt << " " << tslang_opt_ext << " --shared-libs=%TSLANGEXEPATH%/TypeScriptRuntime.dll %FILEPATH% 1> %FILENAME%.txt 2> %FILENAME%.err"
            << std::endl;
    batFile.close();
#else
    if (exists(JIT_NAME BAT_NAME))
    {
        return;
    }

    std::ofstream batFile(JIT_NAME BAT_NAME);
    batFile << "FILENAME=$1" << std::endl;
    batFile << "FILEPATH=$2" << std::endl;
    batFile << "TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "$TSLANGEXEPATH/tslang --emit=jit " << tslang_opt << " " << tslang_opt_ext << " --shared-libs=../../lib/libTypeScriptRuntime.so $FILEPATH 1> $FILENAME.txt 2> $FILENAME.err"
            << std::endl;
    batFile.close();    
#endif    
}

void createCompileBatchFile()
{
#ifdef WIN32     
    if (exists(COMPILE_NAME BAT_NAME))
    {
        return;
    }

    std::ofstream batFile(COMPILE_NAME BAT_NAME);
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=%1" << std::endl;
    batFile << "set FILEPATH=%2" << std::endl;
    batFile << "set LINKER_OPTS=%3" << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVMEXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVM_LIB_PATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "set TSLANG_LIB_PATH=" << TEST_TSLANG_LIBPATH << std::endl;
    batFile << "set GC_LIB_PATH=" << TEST_GCPATH << std::endl;
    batFile << "%TSLANGEXEPATH%\\tslang.exe --emit=obj " << tslang_opt << " " << tslang_opt_ext << " %FILEPATH% -o=%FILENAME%.obj" << std::endl;
    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link %FILENAME%.obj %LINKER_OPTS% " 
            << LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GC_LIB_PATH% /libpath:%LLVM_LIB_PATH% /libpath:%TSLANG_LIB_PATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;
    batFile << "del %FILENAME%.obj" << std::endl;
    batFile << "call " RUN_CMD "%FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "del %FILENAME%.exe" << std::endl;
    batFile << "if exist %FILENAME%.lib (del %FILENAME%.lib)" << std::endl;
    batFile << "if exist %FILENAME%.dll (del %FILENAME%.dll)" << std::endl;        
    batFile << "echo on" << std::endl;
    batFile.close();
#else
    if (exists(COMPILE_NAME BAT_NAME))
    {
        return;
    }

    std::ofstream batFile(COMPILE_NAME BAT_NAME);
    batFile << "FILENAME=$1" << std::endl;
    batFile << "FILEPATH=$2" << std::endl;
    batFile << "LINKER_OPTS=$3" << std::endl;
    batFile << "TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "TSLANG_LIB_PATH=" << TEST_TSLANG_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GC_LIB_PATH=" << TEST_GCPATH << std::endl;
    batFile << "$TSLANGEXEPATH/tslang --emit=obj " << tslang_opt << " " << tslang_opt_ext << " $FILEPATH -relocation-model=pic -o=$FILENAME.o" << std::endl;
    batFile << TEST_COMPILER << " -o $FILENAME $LINKER_OPTS -L$LLVM_LIBPATH -L$GC_LIB_PATH -L$TSLANG_LIB_PATH $FILENAME.o " 
            << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    batFile << "rm -f $FILENAME.o" << std::endl;
    batFile << "rm -f $FILENAME" << std::endl;
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
    ss << RUN_CMD << JIT_NAME BAT_NAME << " " << fileNameNoExt << " " << file;
}

void buildCompileExecCommand(std::stringstream &ss, std::string fileNameNoExt, std::string file)
{
    ss << RUN_CMD << COMPILE_NAME BAT_NAME << " " << fileNameNoExt << " " << file;
    if (sharedLib)
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
    mask << "del " << tempOutputFileNameNoExt << ".bat " << tempOutputFileNameNoExt << ".txt " << tempOutputFileNameNoExt << ".err " << tempOutputFileNameNoExt << ".exe " << tempOutputFileNameNoExt << ".obj";
#else
    mask << "rm -f " << tempOutputFileNameNoExt << ".sh " << tempOutputFileNameNoExt << ".txt " << tempOutputFileNameNoExt << ".err " << tempOutputFileNameNoExt << " " << tempOutputFileNameNoExt << ".o";
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
    // include the process id: rand() is never seeded, so every test-runner process yields the same first value and
    // uniqueness would otherwise rely solely on the millisecond timestamp - same-stem tests (e.g. compile + jit
    // variants) launched within the same millisecond under parallel ctest would collide on temp file names.
#ifdef WIN32
    auto pid = static_cast<long long>(GetCurrentProcessId());
#else
    auto pid = static_cast<long long>(getpid());
#endif
    auto fileNameNoExtWithMs = fileNameNoExt + "-" + std::to_string(ms.count()) + "-" + std::to_string(pid) + "-" + std::to_string(rand());

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
    if (gctorsAsMethod)
    {
        tslang_opt_ext = "--gctors-as-method";
    }

#ifdef WIN32
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVMEXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVM_LIB_PATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "set TSLANG_LIB_PATH=" << TEST_TSLANG_LIBPATH << std::endl;
    batFile << "set GC_LIB_PATH=" << TEST_GCPATH << std::endl;

    std::stringstream objs;
    auto isFirst = true;
    for (auto &file : files)
    {
        auto fileNameWithoutExt = fs::path(file).stem().string();
        objs << fileNameWithoutExt << ".obj ";
        batFile << "%TSLANGEXEPATH%\\tslang.exe --emit=obj " << tslang_opt << " " << (isFirst ? "" : tslang_opt_ext) << " " << file << " -o=" << fileNameWithoutExt << ".obj" << std::endl;
        isFirst = false;
    }

    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link /out:%FILENAME%.exe " << objs.str() << " "
            << LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GC_LIB_PATH% /libpath:%LLVM_LIB_PATH% /libpath:%TSLANG_LIB_PATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;

    batFile << "del " << objs.str() << std::endl;
    batFile << "call " RUN_CMD "%FILENAME%.exe 1> %FILENAME%.txt 2> %FILENAME%.err" << std::endl;
    batFile << "del %FILENAME%.exe" << std::endl;
    batFile << "if exist %FILENAME%.lib (del %FILENAME%.lib)" << std::endl;
    batFile << "if exist %FILENAME%.dll (del %FILENAME%.dll)" << std::endl;    
    batFile << "echo on" << std::endl;
    batFile.close();
#else
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "TSLANG_LIB_PATH=" << TEST_TSLANG_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GC_LIB_PATH=" << TEST_GCPATH << std::endl;

    std::stringstream objs;
    auto isFirst = true;
    for (auto &file : files)
    {
        // prefix with the unique temp name so parallel tests reusing the same source files don't stomp each other's object files
        auto fileNameWithoutExt = tempOutputFileNameNoExt + "_" + fs::path(file).stem().string();
        objs << fileNameWithoutExt << ".o ";
        batFile << "$TSLANGEXEPATH/tslang --emit=obj " << tslang_opt << " " << (isFirst ? "" : tslang_opt_ext) << " " << file << " -relocation-model=pic -o=" << fileNameWithoutExt << ".o" << std::endl;
        isFirst = false;
    }

    batFile << TEST_COMPILER << " -o $FILENAME " << objs.str() 
            << "-L$LLVM_LIBPATH -L$GC_LIB_PATH -L$TSLANG_LIB_PATH "
            << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;
    batFile << "./$FILENAME 1> $FILENAME.txt 2> $FILENAME.err" << std::endl;
    
    batFile << "rm -f " << objs.str() << std::endl;
    batFile << "rm -f $FILENAME" << std::endl;
    batFile << "rm -f lib$FILENAME.so" << std::endl;    
    batFile.close();    
#endif    
}

void createSharedMultiBatchFile(std::string tempOutputFileNameNoExt, std::vector<std::string> &files)
{
    if (gctorsAsMethod)
    {
        tslang_opt_ext = "--gctors-as-method";
    }

    auto linker_opt = SHARED_LIB_OPT;

#ifdef WIN32
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "echo off" << std::endl;
    batFile << "set FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "set LIBPATH=\"" << TEST_LIBPATH << "\"" << std::endl;
    batFile << "set SDKPATH=\"" << TEST_SDKPATH << "\"" << std::endl;
    batFile << "set UCRTPATH=\"" << TEST_UCRTPATH << "\"" << std::endl;
    batFile << "set LLVMEXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "set LLVM_LIB_PATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "set TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "set TSLANG_LIB_PATH=" << TEST_TSLANG_LIBPATH << std::endl;
    batFile << "set GC_LIB_PATH=" << TEST_GCPATH << std::endl;

    // run everything inside a unique per-test working directory: the shared lib must keep its
    // real name (<stem>.dll) for `import './<stem>'` to resolve, but that name is not unique
    // across tests reusing the same source file (e.g. decl_class.ts feeds the compile/compile-time/jit
    // variants) - isolating the cwd avoids parallel collisions under `ctest -j`. The .txt/.err output
    // is written to the parent dir where the runner reads it from.
    batFile << "set WORKDIR=" << tempOutputFileNameNoExt << "_wd" << std::endl;
    batFile << "if exist %WORKDIR% rmdir /s /q %WORKDIR%" << std::endl;
    batFile << "mkdir %WORKDIR%" << std::endl;
    batFile << "cd %WORKDIR%" << std::endl;

    auto first = true;
    std::stringstream shared_objs;
    std::stringstream shared_libs;
    std::stringstream shared_dlls;
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

        (first ? execBat : sharedBat) << "%TSLANGEXEPATH%\\tslang.exe --emit=obj " << tslang_opt << " " << (first ? "" : tslang_opt_ext) << " " << file << " -o=" << fileNameWithoutExt << ".obj" << std::endl;

        first = false;
    }

    batFile << sharedBat.str();
    batFile << "%LLVMEXEPATH%\\lld.exe -flavor link /out:" << shared_filenameNoExt << ".dll " << linker_opt << " " << shared_objs.str() << " "
            <<  LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
            << " /libpath:%GC_LIB_PATH% /libpath:%LLVM_LIB_PATH% /libpath:%TSLANG_LIB_PATH%" 
            << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
            << std::endl;

    shared_libs << shared_filenameNoExt << ".lib ";
    shared_dlls << shared_filenameNoExt << ".dll ";

    batFile << "del " << shared_objs.str() << std::endl;
    batFile << "echo on" << std::endl;

    if (jitRun)
    {
        batFile << "%TSLANGEXEPATH%\\tslang.exe --emit=jit " << tslang_opt << " --shared-libs=%TSLANGEXEPATH%/TypeScriptRuntime.dll " << *files.begin() << " 1> ..\\%FILENAME%.txt 2> ..\\%FILENAME%.err"
                << std::endl;

        batFile << "del " << shared_libs.str() << std::endl;
        batFile << "del " << shared_dlls.str() << std::endl;
    }
    else
    {
        batFile << execBat.str();
        batFile << "%LLVMEXEPATH%\\lld.exe -flavor link /out:%FILENAME%.exe " << exec_objs.str() << " ";
        if (sharedLibCompileTime)
        {
            batFile << shared_filenameNoExt << ".lib ";
        }

        batFile << LIBS << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << CMAKE_C_STANDARD_LIBRARIES
                << " /libpath:%GC_LIB_PATH% /libpath:%LLVM_LIB_PATH% /libpath:%TSLANG_LIB_PATH%" 
                << " /libpath:%LIBPATH% /libpath:%SDKPATH% /libpath:%UCRTPATH%"
                << std::endl;

        batFile << "del " << exec_objs.str() << std::endl;

        batFile << "call " RUN_CMD "%FILENAME%.exe 1> ..\\%FILENAME%.txt 2> ..\\%FILENAME%.err" << std::endl;

        batFile << "echo off" << std::endl;
        batFile << "del " << shared_libs.str() << std::endl;
        batFile << "del " << shared_dlls.str() << std::endl;

        batFile << "del %FILENAME%.exe" << std::endl;
        batFile << "if exist %FILENAME%.lib (del %FILENAME%.lib)" << std::endl;
        batFile << "if exist %FILENAME%.dll (del %FILENAME%.dll)" << std::endl;
        batFile << "echo on" << std::endl;
    }

    // leave and remove the per-test working directory (with the shared lib, exe, objs, etc.)
    batFile << "echo off" << std::endl;
    batFile << "cd .." << std::endl;
    batFile << "rmdir /s /q %WORKDIR%" << std::endl;
    batFile << "echo on" << std::endl;

    batFile.close();
#else
    std::ofstream batFile(tempOutputFileNameNoExt + BAT_NAME);
    batFile << "FILENAME=" << tempOutputFileNameNoExt << std::endl;
    batFile << "TSLANGEXEPATH=" << TEST_TSLANG_EXEPATH << std::endl;
    batFile << "TSLANG_LIB_PATH=" << TEST_TSLANG_LIBPATH << std::endl;
    batFile << "LLVM_EXEPATH=" << TEST_LLVM_EXEPATH << std::endl;
    batFile << "LLVM_LIBPATH=" << TEST_LLVM_LIBPATH << std::endl;
    batFile << "GC_LIB_PATH=" << TEST_GCPATH << std::endl;

    auto first = true;
    std::stringstream shared_objs;
    std::stringstream exec_objs;
    std::string shared_filenameNoExt;
    std::stringstream execBat;
    std::stringstream sharedBat;    
    for (auto &file : files)
    {
        // prefix with the unique temp name so parallel tests reusing the same source files don't stomp each other's object/shared-lib files
        auto fileNameWithoutExt = tempOutputFileNameNoExt + "_" + fs::path(file).stem().string();
        if (first)
        {
            exec_objs << fileNameWithoutExt << ".o ";
        }
        else
        {
            shared_objs << fileNameWithoutExt << ".o ";
            if (shared_filenameNoExt.empty())
            {
                // the shared lib must keep its real (unprefixed) stem so that `import './<stem>'`
                // resolves to lib<stem>.so instead of falling back to recompiling the source
                shared_filenameNoExt = fs::path(file).stem().string();
            }
        }

        (first ? execBat : sharedBat) << "$TSLANGEXEPATH/tslang --emit=obj " << tslang_opt << " " << (first ? "" : tslang_opt_ext) << " " << file << " -relocation-model=pic -o=" << fileNameWithoutExt << ".o" << std::endl;

        first = false;
    }

    // run everything inside a unique per-test working directory: the shared lib must keep its
    // real name (lib<stem>.so) for `import './<stem>'` to resolve, but that name is not unique
    // across tests reusing the same source file - isolating the cwd avoids parallel collisions.
    // Output (.txt/.err) is written to the parent dir where the runner reads it from.
    batFile << "WORKDIR=" << tempOutputFileNameNoExt << "_wd" << std::endl;
    batFile << "rm -rf $WORKDIR && mkdir -p $WORKDIR && cd $WORKDIR" << std::endl;

    batFile << sharedBat.str();
    batFile << TEST_COMPILER << " " << linker_opt << " -o lib" << shared_filenameNoExt << ".so " << shared_objs.str()
            << "-L$LLVM_LIBPATH -L$GC_LIB_PATH -L$TSLANG_LIB_PATH "
            << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;
    batFile << "rm -f " << shared_objs.str() << std::endl;

    if (jitRun)
    {
        // one extra "../" because we run from the per-test working directory
        batFile << "$TSLANGEXEPATH/tslang --emit=jit " << tslang_opt << " --shared-libs=../../../lib/libTypeScriptRuntime.so --shared-libs=./lib" << shared_filenameNoExt << ".so " << *files.begin() << " 1> ../$FILENAME.txt 2> ../$FILENAME.err"
                << std::endl;
    }
    else
    {
        batFile << execBat.str();
        batFile << TEST_COMPILER << " -o $FILENAME " << exec_objs.str() << " ";
        batFile << "-L$LLVM_LIBPATH -L$GC_LIB_PATH -L$TSLANG_LIB_PATH ";
        if (sharedLib)
        {
            // dynamics and compile-time shared modes both link the produced shared lib;
            // we need "-Wl,-rpath=" to embed path for compiled shared lib path
            batFile << "-L`pwd` -Wl,-rpath=`pwd` -l" << shared_filenameNoExt << " ";
        }

        batFile << TYPESCRIPT_LIB << GC_LIB << LLVM_LIBS << LIBS << std::endl;

        batFile << "rm -f " << exec_objs.str() << std::endl;

        batFile << "./$FILENAME 1> ../$FILENAME.txt 2> ../$FILENAME.err" << std::endl;
    }

    // leave and remove the per-test working directory (with the shared lib, exe, etc.)
    batFile << "cd .. && rm -rf $WORKDIR" << std::endl;

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
    if (sharedLib)
    {
        createSharedMultiBatchFile(tempOutputFileNameNoExt, files);
    }
    else
    {
        if (jitRun)
        {
            throw "not supported";
        }

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
            sharedLib = true;
        }
        else if (std::string(argv[index]) == "-compile-time")
        {
            sharedLibCompileTime = true;
        }
        else if (std::string(argv[index]) == "-gctors-as-method")
        {
            gctorsAsMethod = true;
        }
        else if (std::string(argv[index]) == "-noopt")
        {
            opt = false;
        }
        else if (exists(argv[index]))
        {
            files.push_back(argv[index]);
        }
        else
        {
            std::string msg = "unknown param or file does not exist: ";
            msg.append(argv[index]);
            throw msg.c_str();
        }
    }

    if (sharedLibCompileTime && !sharedLib)
    {
        throw "-compile-time can be used with -shared";
    }

    if (sharedLibCompileTime && jitRun)
    {
        throw "-compile-time can't be used with -jit";
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