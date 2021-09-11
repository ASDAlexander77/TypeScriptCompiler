#include "TypeScript/gcwrapper.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ThreadPool.h"

void __mlir_runner_init_gcruntime(llvm::StringMap<void *> &exportSymbols);
void __mlir_runner_destroy_gcruntime();

namespace mlir {
    namespace runtime {
        void __mlir_runner_init_asyncruntime(llvm::StringMap<void *> &exportSymbols);
        void __mlir_runner_destroy_asyncruntime();
    }
}

// to support shared_libs
void __mlir_runner_init_gcruntime(llvm::StringMap<void *> &exportSymbols)
{
    auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
        assert(exportSymbols.count(name) == 0 && "symbol already exists");
        exportSymbols[name] = reinterpret_cast<void *>(ptr);
    };

    exportSymbol("GC_init", &_mlir__GC_init);
    exportSymbol("GC_malloc", &_mlir__GC_malloc);
    exportSymbol("GC_realloc", &_mlir__GC_realloc);
    exportSymbol("GC_free", &_mlir__GC_free);
    exportSymbol("GC_get_heap_size", &_mlir__GC_get_heap_size);
}

void __mlir_runner_destroy_gcruntime()
{
#ifdef WIN32
    _mlir__GC_win32_free_heap();
#endif
}

// Export symbols for the MLIR runner integration. All other symbols are hidden.
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

extern "C" API void __mlir_runner_init(llvm::StringMap<void *> &exportSymbols);

// to support shared_libs
void __mlir_runner_init(llvm::StringMap<void *> &exportSymbols)
{
    __mlir_runner_init_gcruntime(exportSymbols);
    mlir::runtime::__mlir_runner_init_asyncruntime(exportSymbols);
}

extern "C" API void __mlir_runner_destroy()
{
    __mlir_runner_destroy_gcruntime();
    mlir::runtime::__mlir_runner_destroy_asyncruntime();
}
