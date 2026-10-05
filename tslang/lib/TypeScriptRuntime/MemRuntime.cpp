// The coroutine lowering asks for a frame with `aligned_alloc` and gives it back with plain
// `free`, which is the C11 pairing and is fine everywhere `aligned_alloc` exists. MSVC has no
// `aligned_alloc`, and `_aligned_malloc` - the obvious stand-in - hands back memory that only
// `_aligned_free` may release, so that pairing corrupts the CRT heap. Windows' own `malloc` is
// already aligned enough for everything that asks, so the request is served from the ordinary
// heap and `free` stays honest. (Under `-mm=gc` this never showed, because GCPass rewrites the
// whole pair to GC_malloc_uncollectable/GC_free.)

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
#include <cstdio>
#include <cstdlib>

#include "llvm/ADT/StringMap.h"

// the process-heap allocator the JIT'd program calls (ProcessHeapPass), and heapMalloc/heapFree
#include "../ProcessHeapExports.inc"

//===----------------------------------------------------------------------===//
// Async runtime API.
//===----------------------------------------------------------------------===//

namespace mlir
{
namespace runtime
{

// The heap these hand out is the process heap, which is what every module tslang builds allocates
// from and frees to (TypeScript/ProcessHeap.h): a coroutine frame this allocates, the program frees.
// This DLL's own `malloc` need not be that heap - a debug CRT keeps its blocks per copy, and against
// the prebuilt LLVM, LLVMSupport makes it rpmalloc.
#ifdef _WIN32
static void *heapMalloc(size_t size) { return typescript::process_heap::allocate(size); }

static void heapFree(void *ptr) { typescript::process_heap::release(ptr); }
#else
static void *heapMalloc(size_t size) { return malloc(size); }

static void heapFree(void *ptr) { free(ptr); }
#endif

extern "C" void *Alloc(uint64_t size) { return heapMalloc(size); }

// What MSVC's `malloc` guarantees: enough for any fundamental type, 16 bytes on x64.
static constexpr uint64_t kMallocAlignment = 2 * sizeof(void *);

extern "C" void *AlignedAlloc(uint64_t alignment, uint64_t size) {
#ifdef _WIN32
  // Everything here comes from `heapMalloc`, so the block can go back through either `free` or
  // AlignedFree and both are right. `malloc`'s own guarantee covers every request anything
  // makes - the coroutine frame asks for 8. A stricter request cannot be served and stay
  // `free`-compatible at the same time, and silently handing back under-aligned memory is the
  // worse of the two failures, so say so.
  if (alignment > kMallocAlignment)
  {
    fprintf(stderr, "tslang runtime: alignment of %" PRIu64 " requested, only %" PRIu64 " is available\n",
            alignment, kMallocAlignment);
  }

  return heapMalloc(size);
#else
  void *result = nullptr;
  (void)::posix_memalign(&result, alignment, size);
  return result;
#endif
}

extern "C" void Free(void *ptr) { heapFree(ptr); }

extern "C" void AlignedFree(void *ptr) { heapFree(ptr); }

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
