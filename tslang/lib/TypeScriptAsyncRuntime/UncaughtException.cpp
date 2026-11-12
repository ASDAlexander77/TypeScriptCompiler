// An exception nothing in a program catches: said on stderr, as the JIT says it (tslang/jit.cpp), and the
// program ends with exit code 1. Left to the C runtime, terminate() called abort(), which exited with 3 (or
// 0xC0000409, a fast fail) and said nothing - not even what the program had printed, still in stdout's buffer.
//
// An executable tslang links on Windows takes this file with `-include:__tslang_report_uncaught` (tslang/exe.cpp);
// its initializer puts the filter in front of the C runtime's own, which is set before any initializer runs.

#ifdef _WIN32

#include "TypeScript/UncaughtException.h"

#include <cstdio>

namespace
{

constexpr DWORD cxxExceptionCode = 0xE06D7363; // 'msc' | 0xE0000000, a throw

LPTOP_LEVEL_EXCEPTION_FILTER runtimeFilter;

LONG WINAPI reportUncaught(EXCEPTION_POINTERS *pointers)
{
    if (pointers->ExceptionRecord->ExceptionCode != cxxExceptionCode)
    {
        // an access violation and the like: as before
        return runtimeFilter ? runtimeFilter(pointers) : EXCEPTION_CONTINUE_SEARCH;
    }

    char description[512];
    typescript::describeUncaught(pointers, description, sizeof(description));

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

    // as the JIT does: nothing of the program's teardown runs on a thread an exception has left half done
    TerminateProcess(GetCurrentProcess(), 1);
    return EXCEPTION_EXECUTE_HANDLER;
}

int installReportUncaught()
{
    runtimeFilter = SetUnhandledExceptionFilter(reportUncaught);
    return 1;
}

} // namespace

// what the executable's link names to take this file; set once its initializer has run
extern "C" int __tslang_report_uncaught = installReportUncaught();

#endif
