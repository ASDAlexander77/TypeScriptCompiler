#include "llvm/Support/DynamicLibrary.h"
#include "llvm/ADT/StringMap.h"

//===----------------------------------------------------------------------===//
// Dynamic runtime API.
//===----------------------------------------------------------------------===//

namespace mlir
{
namespace runtime
{

// Named as the generated code calls them: Linux has no .def to rename an export, so the shared
// object's own name for a function is the one the JIT finds.
extern "C" int tslang_load_library_permanently(const char* fileName) { 
    return llvm::sys::DynamicLibrary::LoadLibraryPermanently(fileName); 
}

extern "C" void *tslang_search_for_address_of_symbol(const char* symbolName) {
    return llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(symbolName);
}

} // namespace runtime
} // namespace mlir


//===----------------------------------------------------------------------===//
// MLIR Runner (JitRunner) dynamic library integration.
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void init_dynamicruntime(llvm::StringMap<void *> &exportSymbols)
{
    auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
        assert(exportSymbols.count(name) == 0 && "symbol already exists");
        exportSymbols[name] = reinterpret_cast<void *>(ptr);
    };

    exportSymbol("tslang_load_library_permanently", &mlir::runtime::tslang_load_library_permanently);
    exportSymbol("tslang_search_for_address_of_symbol", &mlir::runtime::tslang_search_for_address_of_symbol);
}

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void destroy_dynamicruntime()
{
}
