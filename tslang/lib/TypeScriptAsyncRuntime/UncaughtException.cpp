// An exception nothing in a program catches: said on stderr, as the JIT says it (tslang/jit.cpp), and the
// program ends with exit code 1. Left to the C++ runtime, the program ended in abort() - on Windows with
// exit code 3 (or 0xC0000409, a fast fail) and not a word, on Linux with libstdc++'s "terminate called after
// throwing an instance of 'char const*'" and SIGABRT - and what the program had printed, still in stdout's
// buffer, was lost. Under it, the frames the exception was thrown from (TypeScript/UncaughtStackTrace.h).
//
// An executable tslang links takes this file with `-include:__tslang_report_uncaught` (Windows) or
// `-u __tslang_report_uncaught` (Linux), from tslang/exe.cpp; its initializer sets the handler.

#include "TypeScript/UncaughtStackTrace.h"

#ifdef TSLANG_DESCRIBES_UNCAUGHT

#include <cstdio>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef _WIN32
// _CxxThrowException, which the compiler declares, is the throw as the program calls it: from
// vcruntime140.dll, or a copy of its own (a static C runtime).
//
// The program's main, by another name: C++ may not name main itself. Only an executable links this file.
extern "C" void tslangProgramMain();
#ifdef _M_IX86
#pragma comment(linker, "/alternatename:_tslangProgramMain=_main")
#else
#pragma comment(linker, "/alternatename:tslangProgramMain=main")
#endif
#endif

namespace
{

void sayUncaught(const char *description, const uintptr_t *pcs, int count)
{
    fflush(stdout);
    if (description[0])
    {
        fprintf(stderr, "error: uncaught exception: %s\n", description);
    }
    else
    {
        fprintf(stderr, "error: uncaught exception\n");
    }

    typescript::UncaughtStackOptions options;
#ifdef _WIN32
    options.throwFunction = reinterpret_cast<const void *>(&_CxxThrowException);
    options.mainFunction = reinterpret_cast<const void *>(&tslangProgramMain);
#endif
    typescript::sayUncaughtStack(stderr, pcs, count, options);
    fflush(stderr);
}

#ifdef _WIN32

using typescript::uncaught_stack::cxxExceptionCode;

LPTOP_LEVEL_EXCEPTION_FILTER runtimeFilter;

// in front of the C runtime's own filter, which is set before any initializer runs
LONG WINAPI reportUncaught(EXCEPTION_POINTERS *pointers)
{
    if (pointers->ExceptionRecord->ExceptionCode != cxxExceptionCode)
    {
        // an access violation and the like: as before
        return runtimeFilter ? runtimeFilter(pointers) : EXCEPTION_CONTINUE_SEARCH;
    }

    // the first throw of it, when this is a rethrow from a `finally`
    auto *thrown = typescript::uncaught_stack::thrownFrom(pointers);
    char description[512];
    typescript::describeUncaught(thrown, description, sizeof(description));
    uintptr_t pcs[typescript::uncaughtStackMaxFrames];
    auto count = typescript::uncaught_stack::walk(thrown->ContextRecord, pcs, typescript::uncaughtStackMaxFrames);
    sayUncaught(description, pcs, count);

    // as the JIT does: nothing of the program's teardown runs on a thread an exception has left half done
    TerminateProcess(GetCurrentProcess(), 1);
    return EXCEPTION_EXECUTE_HANDLER;
}

int installReportUncaught()
{
    runtimeFilter = SetUnhandledExceptionFilter(reportUncaught);
    typescript::uncaught_stack::noteThrows();
    return 1;
}

#else

std::terminate_handler runtimeHandler;

// std::terminate() with no exception (a std::thread destroyed while joinable, ...) goes on as before
[[noreturn]] void reportUncaught()
{
    char description[512];
    if (!typescript::describeCurrentException(description, sizeof(description)))
    {
        if (runtimeHandler)
        {
            runtimeHandler();
        }

        abort();
    }

    // where the throw found no catch: still on the stack, under this handler and std::terminate()
    uintptr_t pcs[typescript::uncaughtStackMaxFrames];
    auto count = typescript::uncaught_stack::walk(pcs, typescript::uncaughtStackMaxFrames);
    sayUncaught(description, pcs, count);

    // nothing of the program's teardown runs on a thread an exception has left half done
    _exit(1);
}

int installReportUncaught()
{
    runtimeHandler = std::set_terminate(reportUncaught);
    return 1;
}

#endif

} // namespace

// what the executable's link names to take this file; set once its initializer has run
extern "C" int __tslang_report_uncaught = installReportUncaught();

#endif
