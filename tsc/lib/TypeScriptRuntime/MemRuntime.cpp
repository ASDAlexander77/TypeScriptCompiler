#ifndef _WIN32
#if defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
#include <cstdlib>
#else
#include <alloca.h>
#endif
#include <sys/time.h>
#else
#include "malloc.h"
#endif // _WIN32

#include <cinttypes>
#include <cstdlib>
#include <cassert>
#include "llvm/ADT/StringMap.h"

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

namespace mlir
{
namespace runtime
{

extern "C" void *Alloc(uint64_t size) { return malloc(size); }

extern "C" void *AlignedAlloc(uint64_t alignment, uint64_t size) {
#ifdef _WIN32
  // On Windows, alignment is the second argument
  return _aligned_malloc(size, alignment);
#else
  // Check alignment requirements for posix_memalign
  assert((alignment & (alignment - 1)) == 0 && "alignment must be a power of 2");
  assert(alignment >= sizeof(void *) && "alignment must be at least sizeof(void *)");

  void *result = nullptr;
  int ret = ::posix_memalign(&result, alignment, size);
  if (ret != 0) {
    return nullptr; // Allocation failed.
  }
  return result;
#endif
}

extern "C" void Free(void *ptr) { free(ptr); }

extern "C" void AlignedFree(void *ptr) {
#ifdef _WIN32
  // Use Windows-specific aligned free
  _aligned_free(ptr);
#else
  // Use regular free for posix aligned_alloc
  free(ptr);
#endif
}

} // namespace runtime
} // namespace mlir

//===----------------------------------------------------------------------===//
// MLIR Runner (JitRunner) dynamic library integration.
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void init_memruntime(llvm::StringMap<void *> &exportSymbols)
{
    auto exportSymbol = [&](llvm::StringRef name, auto ptr) {
        assert(exportSymbols.count(name) == 0 && "symbol already exists");
        exportSymbols[name] = reinterpret_cast<void *>(ptr);
    };

    exportSymbol("_mlir_alloc", &mlir::runtime::Alloc);
    exportSymbol("_mlir_aligned_alloc", &mlir::runtime::AlignedAlloc);
    exportSymbol("_mlir_free", &mlir::runtime::Free);
    exportSymbol("_mlir_aligned_free", &mlir::runtime::AlignedFree);

    exportSymbol("aligned_alloc", &mlir::runtime::AlignedAlloc);
    exportSymbol("aligned_free", &mlir::runtime::AlignedFree);
}

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void destroy_memruntime()
{
}
