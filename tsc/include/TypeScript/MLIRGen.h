#ifndef MLIR_TYPESCRIPT_MLIRGEN_H_
#define MLIR_TYPESCRIPT_MLIRGEN_H_

#include <memory>
#include <string>

#include "TypeScript/DataStructs.h"

namespace mlir
{
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace llvm
{
class StringRef;
} // namespace llvm

namespace typescript
{
::std::string dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source);
mlir::OwningModuleRef mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName, const llvm::StringRef &source,
                                        CompileOptions compileOptions);
} // namespace typescript

#endif // MLIR_TYPESCRIPT_MLIRGEN_H_
