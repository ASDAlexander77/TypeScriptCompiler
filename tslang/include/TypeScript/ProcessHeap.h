#ifndef TYPESCRIPT_PROCESS_HEAP_H_
#define TYPESCRIPT_PROCESS_HEAP_H_

// The allocator every module tslang builds on Windows uses, under every memory model but gc
// (ProcessHeapPass renames the module's malloc, calloc, realloc, free and aligned_alloc to these).
//
// A block made by one module is freed by another - an object of a library's class under -mm=rc or
// -mm=own is made in the library and destroyed by the program, a coroutine frame comes from the
// runtime and goes back through the program - so they have to share one allocator, and the C
// runtime's is not one: each module links its own copy of the static CRT, and a debug CRT keeps a
// list of its blocks per copy, while tslang.exe's own `malloc` is rpmalloc when LLVM is the
// prebuilt package. The process heap is the same heap in every module and every build.
//
// Header-only, so each place that has to provide them - the async runtime library a program links,
// TypeScriptRuntime.dll and the JIT - compiles the same code.

#ifdef _WIN32

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstddef>
#include <cstdint>

namespace typescript
{
namespace process_heap
{

// What HeapAlloc guarantees: 16 bytes on x64, 8 on x86.
constexpr size_t alignment = MEMORY_ALLOCATION_ALIGNMENT;

inline void *allocate(size_t size)
{
    // malloc(0) hands back a block of its own too
    return HeapAlloc(GetProcessHeap(), 0, size ? size : 1);
}

inline void *allocateZeroed(size_t count, size_t size)
{
    if (size != 0 && count > SIZE_MAX / size)
    {
        return nullptr;
    }

    auto bytes = count * size;
    return HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, bytes ? bytes : 1);
}

inline void release(void *ptr)
{
    if (ptr)
    {
        HeapFree(GetProcessHeap(), 0, ptr);
    }
}

inline void *reallocate(void *ptr, size_t size)
{
    if (!ptr)
    {
        return allocate(size);
    }

    // as the C runtime's realloc: a block resized to nothing is freed, and there is none
    if (size == 0)
    {
        release(ptr);
        return nullptr;
    }

    return HeapReAlloc(GetProcessHeap(), 0, ptr, size);
}

// Every request anything makes fits what HeapAlloc aligns to (the coroutine frame asks for 8). A
// stricter one cannot be served and still go back through release(); handing back under-aligned
// memory silently would be the worse failure, so it fails.
inline void *allocateAligned(size_t align, size_t size)
{
    if (align > alignment)
    {
        return nullptr;
    }

    return allocate(size);
}

} // namespace process_heap
} // namespace typescript

#endif // _WIN32

#endif // TYPESCRIPT_PROCESS_HEAP_H_
