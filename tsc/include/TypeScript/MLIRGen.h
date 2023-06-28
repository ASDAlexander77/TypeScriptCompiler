#ifndef MLIR_TYPESCRIPT_MLIRGEN_H_
#define MLIR_TYPESCRIPT_MLIRGEN_H_

#include <memory>
#include <string>

#include "TypeScript/DataStructs.h"

#include "llvm/Support/SourceMgr.h"

namespace mlir
{
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace llvm
{
class StringRef;
} // namespace llvm

namespace typescript
{
::std::string dumpFromSource(const llvm::StringRef &fileName, const llvm::StringRef &source);
mlir::OwningOpRef<mlir::ModuleOp> mlirGenFromSource(const mlir::MLIRContext &context, const llvm::StringRef &fileName, const llvm::SourceMgr &sourceMgr,
                                        CompileOptions compileOptions);
} // namespace typescript

#endif // MLIR_TYPESCRIPT_MLIRGEN_H_
