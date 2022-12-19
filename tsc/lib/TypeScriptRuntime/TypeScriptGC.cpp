#include "TypeScript/gcwrapper.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ThreadPool.h"

// to support shared_libs
void init_gcruntime(llvm::StringMap<void *> &exportSymbols)
{
    auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
        assert(exportSymbols.count(name) == 0 && "symbol already exists");
        exportSymbols[name] = reinterpret_cast<void *>(ptr);
    };

    exportSymbol("GC_init", &_mlir__GC_init);
    exportSymbol("GC_malloc", &_mlir__GC_malloc);
    exportSymbol("GC_memalign", &_mlir__GC_memalign);
    exportSymbol("GC_realloc", &_mlir__GC_realloc);
    exportSymbol("GC_free", &_mlir__GC_free);
    exportSymbol("GC_get_heap_size", &_mlir__GC_get_heap_size);
    exportSymbol("GC_malloc_explicitly_typed", &_mlir__GC_malloc_explicitly_typed);
    exportSymbol("GC_make_descriptor", &_mlir__GC_make_descriptor);
}

void destroy_gcruntime()
{
#ifdef WIN32
    _mlir__GC_win32_free_heap();
#endif
}
