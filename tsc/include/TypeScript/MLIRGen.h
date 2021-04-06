#ifndef MLIR_TYPESCRIPT_MLIRGEN_H_
#define MLIR_TYPESCRIPT_MLIRGEN_H_

#include <memory>

namespace mlir
{
    class MLIRContext;
    class OwningModuleRef;
} // namespace mlir

namespace llvm
{
    class StringRef;
} // namespace mlir

namespace typescript
{
    llvm::StringRef dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source);
    mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName, const llvm::StringRef &source);
} // namespace typescript

#endif // MLIR_TYPESCRIPT_MLIRGEN_H_
