#include "llvm/Support/DynamicLibrary.h"
#include "llvm/ADT/StringMap.h"

//===----------------------------------------------------------------------===//
// Dynamic runtime API.
//===----------------------------------------------------------------------===//

namespace mlir
{
namespace runtime
{

extern "C" int LoadLibraryPermanently(const char* fileName) { 
    return llvm::sys::DynamicLibrary::LoadLibraryPermanently(fileName); 
}

extern "C" void *SearchForAddressOfSymbol(const char* symbolName) {
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

    exportSymbol("LLVMLoadLibraryPermanently", &mlir::runtime::LoadLibraryPermanently);
    exportSymbol("LLVMSearchForAddressOfSymbol", &mlir::runtime::SearchForAddressOfSymbol);
}

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void destroy_dynamicruntime()
{
}
