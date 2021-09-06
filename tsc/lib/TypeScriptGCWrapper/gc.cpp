#include <cstddef>

#ifndef NDEBUG
#define GC_DEBUG
#endif

#ifndef GC_THREADS
#define GC_THREADS
#endif

#define GC_INSIDE_DLL

#if defined _WIN32 || defined _WIN64 || defined PLATFORM_ANDROID || defined __ANDROID__
#define GC_NOT_DLL
#endif

#include "gc.h"

void _mlir__GC_init()
{
#ifdef GC_THREADS
    GC_use_threads_discovery();
#endif
    GC_INIT();
}

void *_mlir__GC_malloc(size_t size)
{
    return GC_MALLOC(size);
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