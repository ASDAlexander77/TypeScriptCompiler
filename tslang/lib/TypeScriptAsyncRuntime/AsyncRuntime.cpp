// Needed this file to resolve error in linux which not allowing to load 2 shared dlls
#define MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS

//===- AsyncRuntime.cpp - Async runtime reference implementation ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Built into the static TypeScriptAsyncRuntime library, linked directly into
// AOT-compiled executables. The implementation itself lives in
// AsyncRuntimeCommon.inc, shared with TypeScriptRuntime (the shared, JIT-loaded
// build of the same API); this file adds the MSVC `aligned_alloc` shim on top
// of it.
//
//===----------------------------------------------------------------------===//

#define MLIR_ASYNC_RUNTIME_EXPORT
#include "mlir/ExecutionEngine/AsyncRuntime.h"

#ifdef MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS

#include "../AsyncRuntimeCommon.inc"

#ifdef _WIN32
//===----------------------------------------------------------------------===//
// `aligned_alloc` for a platform that does not have one.
//===----------------------------------------------------------------------===//

// The coroutine lowering allocates an async function's frame by calling `aligned_alloc` by name
// and releases it with plain `free`. MSVC has no `aligned_alloc`, so an executable that awaits
// anything does not link at all unless the model rewrites the call (which only `-mm=gc` does,
// to GC_memalign). `_aligned_malloc` is not a stand-in: its memory may only go back through
// `_aligned_free`, and pairing it with `free` corrupts the CRT heap. Windows' `malloc` is
// already aligned enough for every request anything makes - the frame asks for 8 - so it serves
// the request and keeps the pairing honest.
extern "C" void *aligned_alloc(size_t alignment, size_t size)
{
    // what MSVC's `malloc` guarantees: enough for any fundamental type, 16 bytes on x64
    constexpr size_t mallocAlignment = 2 * sizeof(void *);
    if (alignment > mallocAlignment)
    {
        std::cerr << "tslang runtime: alignment of " << alignment << " requested, only " << mallocAlignment
                  << " is available" << std::endl;
    }

    return malloc(size);
}
#endif // _WIN32

#endif // MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
