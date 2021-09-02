#include <cstddef>

#define GC_NOT_DLL 1
#include "gc.h"

void _mlir__GC_init()
{
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
    GC_win32_free_heap();
}