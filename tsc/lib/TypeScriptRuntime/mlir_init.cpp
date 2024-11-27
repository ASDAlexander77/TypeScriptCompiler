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

void init_dynamicruntime(llvm::StringMap<void *> &exportSymbols);
void destroy_dynamicruntime();

// Export symbols for the MLIR runner integration. All other symbols are hidden.
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

extern "C" API void __mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols);

// to support shared_libs
void __mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols)
{
    init_gcruntime(exportSymbols);
    init_memruntime(exportSymbols);
    init_asyncruntime(exportSymbols);
    init_dynamicruntime(exportSymbols);
}

extern "C" API void __mlir_execution_engine_destroy()
{
    destroy_gcruntime();
    //destory_memruntime();
    destroy_asyncruntime();
    destroy_dynamicruntime();
}