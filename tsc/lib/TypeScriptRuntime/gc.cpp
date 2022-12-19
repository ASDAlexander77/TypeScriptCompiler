#include <cstddef>
#include <cstdint>

#ifndef NDEBUG
#define GC_DEBUG
#endif

//#ifndef GC_THREADS
//#define GC_THREADS
//#endif

#define GC_INSIDE_DLL
#define GC_NAMESPACE

#if defined _WIN32 || defined _WIN64 || defined PLATFORM_ANDROID || defined __ANDROID__
#define GC_NOT_DLL
#endif

#include "gc.h"
#include "gc_typed.h"

void _mlir__GC_init()
{
    GC_INIT();
}

void *_mlir__GC_malloc(size_t size)
{
    return GC_MALLOC(size);
}

void *_mlir__GC_memalign(size_t align, size_t size)
{
    return GC_memalign(align, size);
}

void *_mlir__GC_realloc(void *ptr, size_t size)
{
    return GC_REALLOC(ptr, size);
}

void _mlir__GC_free(void *ptr)
{
    GC_FREE(ptr);
}

size_t _mlir__GC_get_heap_size()
{
    return GC_get_heap_size();
}

void _mlir__GC_win32_free_heap()
{
#ifdef WIN32
    GC_win32_free_heap();
#endif
}

void *_mlir__GC_malloc_explicitly_typed(size_t size, int64_t descr)
{
    return GC_MALLOC_EXPLICITLY_TYPED(size, descr);
}

int64_t _mlir__GC_make_descriptor(const int64_t *descr, size_t size)
{
    return GC_make_descriptor(reinterpret_cast<const GC_word *>(descr), size);
}