// The heap entry points under the names ProcessHeapPass calls (PROCESS_HEAP_* in
// TypeScript/Defines.h). Included by the async runtime library, which a program and a library link,
// and by TypeScriptRuntime.dll, which the JIT loads; see TypeScript/ProcessHeap.h for why there is
// one heap per process.
//
// The program owns the heap. In the program's own module these are the process heap; an executable
// that links TypeScriptMimalloc gets that library's definitions instead (mimalloc), and tslang.exe
// gives a program it runs its own (jit.cpp). Those two export theirs, so a library - a module other
// than the process's executable - looks them up there at its first allocation and calls the
// program's: the library's blocks and the program's come from one heap, whichever it is. A program
// that exports none (one on the process heap, a host tslang did not build) leaves a library on the
// process heap, which is what such a program allocates from too.
//
// These are not exported. Only the program's are ever looked up, and these would add nothing there:
// they are the process heap, which a library falls back to anyway. Exported from a library they did
// harm: its import library defines them, and a DLL linking it - the default library's, say - took
// the five from there, then pulled this file in for `__tslang_heap_shared` and had each twice
// (LNK2005).
//
// `__tslang_heap_shared` says a module does this, so that a program on mimalloc can refuse a library
// that would keep a heap of its own (ProcessHeap.h, librarySharesHeap).

#ifdef _WIN32

#include <atomic>

#include "TypeScript/Defines.h"
#include "TypeScript/ProcessHeap.h"

extern "C" __declspec(dllexport) const int __tslang_heap_shared = 1;

namespace
{

using HeapMallocFn = void *(*)(size_t);
using HeapCallocFn = void *(*)(size_t, size_t);
using HeapReallocFn = void *(*)(void *, size_t);
using HeapFreeFn = void (*)(void *);
using HeapAlignedAllocFn = void *(*)(size_t, size_t);

void *processHeapMalloc(size_t size)
{
    return typescript::process_heap::allocate(size);
}

void *processHeapCalloc(size_t count, size_t size)
{
    return typescript::process_heap::allocateZeroed(count, size);
}

void *processHeapRealloc(void *ptr, size_t size)
{
    return typescript::process_heap::reallocate(ptr, size);
}

void processHeapFree(void *ptr)
{
    typescript::process_heap::release(ptr);
}

void *processHeapAlignedAlloc(size_t alignment, size_t size)
{
    return typescript::process_heap::allocateAligned(alignment, size);
}

// Filled once, by whichever thread allocates first; every thread that races to it computes the
// same five functions. The library's own global constructors may allocate before anything else
// in it runs, so this is not left to a constructor of this file.
std::atomic<HeapMallocFn> heapMalloc{nullptr};
std::atomic<HeapCallocFn> heapCalloc{nullptr};
std::atomic<HeapReallocFn> heapRealloc{nullptr};
std::atomic<HeapFreeFn> heapFree{nullptr};
std::atomic<HeapAlignedAllocFn> heapAlignedAlloc{nullptr};

void resolveHeap()
{
    HeapMallocFn mallocFn = processHeapMalloc;
    HeapCallocFn callocFn = processHeapCalloc;
    HeapReallocFn reallocFn = processHeapRealloc;
    HeapFreeFn freeFn = processHeapFree;
    HeapAlignedAllocFn alignedAllocFn = processHeapAlignedAlloc;

    // the program's heap, unless this is the program
    HMODULE self = nullptr;
    auto program = GetModuleHandleW(nullptr);
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&resolveHeap), &self)
        && program != nullptr && program != self)
    {
        auto programMalloc = reinterpret_cast<HeapMallocFn>(GetProcAddress(program, PROCESS_HEAP_MALLOC));
        auto programCalloc = reinterpret_cast<HeapCallocFn>(GetProcAddress(program, PROCESS_HEAP_CALLOC));
        auto programRealloc = reinterpret_cast<HeapReallocFn>(GetProcAddress(program, PROCESS_HEAP_REALLOC));
        auto programFree = reinterpret_cast<HeapFreeFn>(GetProcAddress(program, PROCESS_HEAP_FREE));
        auto programAlignedAlloc =
            reinterpret_cast<HeapAlignedAllocFn>(GetProcAddress(program, PROCESS_HEAP_ALIGNED_ALLOC));
        // all five or none: half of one heap and half of another frees blocks on the wrong one
        if (programMalloc && programCalloc && programRealloc && programFree && programAlignedAlloc)
        {
            mallocFn = programMalloc;
            callocFn = programCalloc;
            reallocFn = programRealloc;
            freeFn = programFree;
            alignedAllocFn = programAlignedAlloc;
        }
    }

    heapCalloc.store(callocFn, std::memory_order_relaxed);
    heapRealloc.store(reallocFn, std::memory_order_relaxed);
    heapFree.store(freeFn, std::memory_order_relaxed);
    heapAlignedAlloc.store(alignedAllocFn, std::memory_order_relaxed);
    // last, and released: a thread that sees malloc sees the other four
    heapMalloc.store(mallocFn, std::memory_order_release);
}

void ensureHeap()
{
    if (!heapMalloc.load(std::memory_order_acquire))
    {
        resolveHeap();
    }
}

} // namespace

extern "C" void *__tslang_heap_malloc(size_t size)
{
    ensureHeap();
    return heapMalloc.load(std::memory_order_relaxed)(size);
}

extern "C" void *__tslang_heap_calloc(size_t count, size_t size)
{
    ensureHeap();
    return heapCalloc.load(std::memory_order_relaxed)(count, size);
}

extern "C" void *__tslang_heap_realloc(void *ptr, size_t size)
{
    ensureHeap();
    return heapRealloc.load(std::memory_order_relaxed)(ptr, size);
}

extern "C" void __tslang_heap_free(void *ptr)
{
    ensureHeap();
    heapFree.load(std::memory_order_relaxed)(ptr);
}

extern "C" void *__tslang_heap_aligned_alloc(size_t alignment, size_t size)
{
    ensureHeap();
    return heapAlignedAlloc.load(std::memory_order_relaxed)(alignment, size);
}

#endif // _WIN32
