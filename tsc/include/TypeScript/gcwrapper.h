void _mlir__GC_init();

void *_mlir__GC_malloc(size_t size);

void *_mlir__GC_realloc(void *ptr, size_t size);

void _mlir__GC_free(void *ptr);

size_t _mlir__GC_get_heap_size();