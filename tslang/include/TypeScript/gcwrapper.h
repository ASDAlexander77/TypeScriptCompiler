#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void _mlir__GC_init();

void *_mlir__GC_malloc(size_t size);

void *_mlir__GC_malloc_atomic(size_t size);

void *_mlir__GC_memalign(size_t align, size_t size);

void *_mlir__GC_realloc(void *ptr, size_t size);

void _mlir__GC_free(void *ptr);

size_t _mlir__GC_get_heap_size();

void _mlir__GC_add_roots(void *low, void *highPlus1);

void _mlir__GC_remove_roots(void *low, void *highPlus1);

void _mlir__GC_win32_free_heap();

void *_mlir__GC_malloc_explicitly_typed(size_t size, int64_t descr);

int64_t _mlir__GC_make_descriptor(const int64_t *descr, size_t size);

int _mlir__GC_general_register_disappearing_link(void **link, const void *obj);

int _mlir__GC_unregister_disappearing_link(void **link);

void _mlir__GC_gcollect();

#ifdef __cplusplus
}
#endif
