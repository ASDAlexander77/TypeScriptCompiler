#include <cstdint>

#ifndef _WIN32
#include <sys/mman.h>
#endif

// #include "config.h"
// FIXME: CMake - include when cmake system is ready.
// Remove #define HAVE_SYSCONF 1 line.
#define HAVE_SYSCONF 1

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#ifndef __APPLE__
#include <unistd.h>
#endif // __APPLE__
#endif // _WIN32

#if __LP64__
#define TRAMPOLINE_SIZE 48
#else
#define TRAMPOLINE_SIZE 40
#endif

// A helper function that will be called from the JIT
void _mlir__enable_execute_stack(void *addr)
{
#if _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if (!VirtualQuery(addr, &mbi, sizeof(mbi)))
        return; // We should probably assert here because there is no return value
    VirtualProtect(mbi.BaseAddress, mbi.RegionSize, PAGE_EXECUTE_READWRITE, &mbi.Protect);
#else
#if __APPLE__
    // On Darwin, pagesize is always 4096 bytes
    const uintptr_t pageSize = 4096;
#elif !defined(HAVE_SYSCONF)
#error "HAVE_SYSCONF not defined! See enable_execute_stack.c"
#else
    const uintptr_t pageSize = sysconf(_SC_PAGESIZE);
#endif // __APPLE__

    const uintptr_t pageAlignMask = ~(pageSize - 1);
    uintptr_t p = (uintptr_t)addr;
    unsigned char *startPage = (unsigned char *)(p & pageAlignMask);
    unsigned char *endPage = (unsigned char *)((p + TRAMPOLINE_SIZE + pageSize) & pageAlignMask);
    size_t length = endPage - startPage;
    (void)mprotect((void *)startPage, length, PROT_READ | PROT_WRITE | PROT_EXEC);
#endif
}
