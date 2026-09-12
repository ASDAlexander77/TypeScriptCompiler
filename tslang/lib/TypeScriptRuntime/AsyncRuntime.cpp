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
// Built into the shared TypeScriptRuntime library, loaded by the JIT via
// `--shared-libs`. The implementation itself lives in AsyncRuntimeCommon.inc,
// shared with TypeScriptAsyncRuntime (the static, AOT-linked build of the same
// API); this file adds the JIT's dynamic export table on top of it.
//
//===----------------------------------------------------------------------===//

#define MLIR_ASYNC_RUNTIME_EXPORT
#include "mlir/ExecutionEngine/AsyncRuntime.h"

#ifdef MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS

#include "../AsyncRuntimeCommon.inc"

//===----------------------------------------------------------------------===//
// MLIR Runner (JitRunner) dynamic library integration.
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void init_asyncruntime(llvm::StringMap<void *> &exportSymbols)
{
    auto exportSymbol = [&](llvm::StringRef name, auto ptr)
    {
        assert(exportSymbols.count(name) == 0 && "symbol already exists");
        exportSymbols[name] = reinterpret_cast<void *>(ptr);
    };

    exportSymbol("mlirAsyncRuntimeAddRef", &mlir::runtime::mlirAsyncRuntimeAddRef);
    exportSymbol("mlirAsyncRuntimeDropRef", &mlir::runtime::mlirAsyncRuntimeDropRef);
    exportSymbol("mlirAsyncRuntimeExecute", &mlir::runtime::mlirAsyncRuntimeExecute);
    exportSymbol("mlirAsyncRuntimeGetValueStorage", &mlir::runtime::mlirAsyncRuntimeGetValueStorage);
    exportSymbol("mlirAsyncRuntimeCreateToken", &mlir::runtime::mlirAsyncRuntimeCreateToken);
    exportSymbol("mlirAsyncRuntimeCreateValue", &mlir::runtime::mlirAsyncRuntimeCreateValue);
    exportSymbol("mlirAsyncRuntimeEmplaceToken", &mlir::runtime::mlirAsyncRuntimeEmplaceToken);
    exportSymbol("mlirAsyncRuntimeEmplaceValue", &mlir::runtime::mlirAsyncRuntimeEmplaceValue);
    exportSymbol("mlirAsyncRuntimeSetTokenError", &mlir::runtime::mlirAsyncRuntimeSetTokenError);
    exportSymbol("mlirAsyncRuntimeSetValueError", &mlir::runtime::mlirAsyncRuntimeSetValueError);
    exportSymbol("mlirAsyncRuntimeIsTokenError", &mlir::runtime::mlirAsyncRuntimeIsTokenError);
    exportSymbol("mlirAsyncRuntimeIsValueError", &mlir::runtime::mlirAsyncRuntimeIsValueError);
    exportSymbol("mlirAsyncRuntimeIsGroupError", &mlir::runtime::mlirAsyncRuntimeIsGroupError);
    exportSymbol("mlirAsyncRuntimeAwaitToken", &mlir::runtime::mlirAsyncRuntimeAwaitToken);
    exportSymbol("mlirAsyncRuntimeAwaitValue", &mlir::runtime::mlirAsyncRuntimeAwaitValue);
    exportSymbol("mlirAsyncRuntimeAwaitTokenAndExecute", &mlir::runtime::mlirAsyncRuntimeAwaitTokenAndExecute);
    exportSymbol("mlirAsyncRuntimeAwaitValueAndExecute", &mlir::runtime::mlirAsyncRuntimeAwaitValueAndExecute);
    exportSymbol("mlirAsyncRuntimeCreateGroup", &mlir::runtime::mlirAsyncRuntimeCreateGroup);
    exportSymbol("mlirAsyncRuntimeAddTokenToGroup", &mlir::runtime::mlirAsyncRuntimeAddTokenToGroup);
    exportSymbol("mlirAsyncRuntimeAwaitAllInGroup", &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroup);
    exportSymbol("mlirAsyncRuntimeAwaitAllInGroupAndExecute",
                 &mlir::runtime::mlirAsyncRuntimeAwaitAllInGroupAndExecute);
    exportSymbol("mlirAsyncRuntimGetNumWorkerThreads", &mlir::runtime::mlirAsyncRuntimGetNumWorkerThreads);
    exportSymbol("mlirAsyncRuntimePrintCurrentThreadId", &mlir::runtime::mlirAsyncRuntimePrintCurrentThreadId);
}

// NOLINTNEXTLINE(*-identifier-naming): externally called.
void destroy_asyncruntime()
{
    resetDefaultAsyncRuntime();
}

#endif // MLIR_ASYNCRUNTIME_DEFINE_FUNCTIONS
