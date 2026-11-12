#ifndef WASMEXCEPTIONPASS__H
#define WASMEXCEPTIONPASS__H

#include "llvm/Pass.h"
#include "llvm/IR/PassManager.h"

namespace ts
{
    // wasm32/wasm64 (not Emscripten): turns the Itanium landing pads tslang emits into the funclets
    // wasm exception handling is built on. See WasmExceptionPass.cpp.
    class WasmExceptionPass : public llvm::PassInfoMixin<WasmExceptionPass>
    {
    public:
        llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);

        static bool isRequired()
        {
            return true;
        }
    };
}

#endif // WASMEXCEPTIONPASS__H
