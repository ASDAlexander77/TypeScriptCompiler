#ifndef MLIR_TYPESCRIPT_MLIRGEN_H_
#define MLIR_TYPESCRIPT_MLIRGEN_H_

#include <memory>

namespace mlir
{
    class MLIRContext;
    class OwningModuleRef;
} // namespace mlir

namespace typescript
{
    class ModuleAST;

    /// Emit IR for the given TypeScript moduleAST, returns a newly created MLIR module
    /// or nullptr on failure.
    mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} // namespace typescript

#endif // MLIR_TYPESCRIPT_MLIRGEN_H_
