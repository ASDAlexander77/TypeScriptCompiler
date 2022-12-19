#include "llvm/ADT/StringMap.h"

#include <typeinfo>

#ifdef _WIN32
#pragma comment(linker, "/EXPORT:??_7type_info@@6B@")
#endif

void init_gcruntime(llvm::StringMap<void *> &exportSymbols);
void destroy_gcruntime();

void init_memruntime(llvm::StringMap<void *> &exportSymbols);
//void destroy_memruntime();

void init_asyncruntime(llvm::StringMap<void *> &exportSymbols);
void destroy_asyncruntime();

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
    init_gcruntime(exportSymbols);
    init_memruntime(exportSymbols);
    init_asyncruntime(exportSymbols);
}

extern "C" API void __mlir_runner_destroy()
{
    destroy_gcruntime();
    //destory_memruntime();
    destroy_asyncruntime();
}