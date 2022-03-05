#include <cstddef>
#include <cstdint>

void _mlir__GC_init();

void *_mlir__GC_malloc(size_t size);

void *_mlir__GC_realloc(void *ptr, size_t size);

void _mlir__GC_free(void *ptr);

size_t _mlir__GC_get_heap_size();

void _mlir__GC_win32_free_heap();

void *_mlir__GC_malloc_explicitly_typed(size_t size, int64_t descr);

int64_t _mlir__GC_make_descriptor(const int64_t *descr, size_t size);
