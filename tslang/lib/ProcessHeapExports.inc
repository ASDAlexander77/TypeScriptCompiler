// The process-heap allocator under the names ProcessHeapPass calls (PROCESS_HEAP_* in
// TypeScript/Defines.h). Included by the async runtime library, which a program and a library link,
// and by TypeScriptRuntime.dll, which the JIT loads; see TypeScript/ProcessHeap.h for why.

#ifdef _WIN32

#include "TypeScript/ProcessHeap.h"

extern "C" void *__tslang_heap_malloc(size_t size)
{
    return typescript::process_heap::allocate(size);
}

extern "C" void *__tslang_heap_calloc(size_t count, size_t size)
{
    return typescript::process_heap::allocateZeroed(count, size);
}

extern "C" void *__tslang_heap_realloc(void *ptr, size_t size)
{
    return typescript::process_heap::reallocate(ptr, size);
}

extern "C" void __tslang_heap_free(void *ptr)
{
    typescript::process_heap::release(ptr);
}

extern "C" void *__tslang_heap_aligned_alloc(size_t alignment, size_t size)
{
    return typescript::process_heap::allocateAligned(alignment, size);
}

#endif // _WIN32
