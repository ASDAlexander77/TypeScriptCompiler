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

// `<stem>_<hash of the canonical path>`: the suffix of a module's exported marker symbols
// (`__tsmm_`, `__tsown_`, `__tsborrow_`), unique per source file.
::std::string moduleSymbolSuffix(llvm::StringRef filePath);

::std::string dumpFromSource(const llvm::StringRef &, const llvm::StringRef &);

mlir::OwningOpRef<mlir::ModuleOp> mlirGenFromMainSource(
    const mlir::MLIRContext &, const llvm::StringRef &, const llvm::SourceMgr &, CompileOptions &);

mlir::OwningOpRef<mlir::ModuleOp> mlirGenFromSource(
    const mlir::MLIRContext &, llvm::SMLoc &, const llvm::StringRef &, const llvm::SourceMgr &, CompileOptions &);

} // namespace typescript

#endif // MLIR_TYPESCRIPT_MLIRGEN_H_
