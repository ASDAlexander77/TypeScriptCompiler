// An exception nothing in a program catches: said on stderr, as the JIT says it (tslang/jit.cpp), and the
// program ends with exit code 1. Left to the C++ runtime, the program ended in abort() - on Windows with
// exit code 3 (or 0xC0000409, a fast fail) and not a word, on Linux with libstdc++'s "terminate called after
// throwing an instance of 'char const*'" and SIGABRT - and what the program had printed, still in stdout's
// buffer, was lost.
//
// An executable tslang links takes this file with `-include:__tslang_report_uncaught` (Windows) or
// `-u __tslang_report_uncaught` (Linux), from tslang/exe.cpp; its initializer sets the handler.

#include "TypeScript/UncaughtException.h"

#ifdef TSLANG_DESCRIBES_UNCAUGHT

#include <cstdio>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace
{

void sayUncaught(const char *description)
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

    fflush(stderr);
}

#ifdef _WIN32

constexpr DWORD cxxExceptionCode = 0xE06D7363; // 'msc' | 0xE0000000, a throw

LPTOP_LEVEL_EXCEPTION_FILTER runtimeFilter;

// in front of the C runtime's own filter, which is set before any initializer runs
LONG WINAPI reportUncaught(EXCEPTION_POINTERS *pointers)
{
    if (pointers->ExceptionRecord->ExceptionCode != cxxExceptionCode)
    {
        // an access violation and the like: as before
        return runtimeFilter ? runtimeFilter(pointers) : EXCEPTION_CONTINUE_SEARCH;
    }

    char description[512];
    typescript::describeUncaught(pointers, description, sizeof(description));
    sayUncaught(description);

    // as the JIT does: nothing of the program's teardown runs on a thread an exception has left half done
    TerminateProcess(GetCurrentProcess(), 1);
    return EXCEPTION_EXECUTE_HANDLER;
}

int installReportUncaught()
{
    runtimeFilter = SetUnhandledExceptionFilter(reportUncaught);
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

    sayUncaught(description);

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
