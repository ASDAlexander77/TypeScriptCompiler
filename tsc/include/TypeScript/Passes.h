#ifndef MLIR_TYPESCRIPT_PASSES_H
#define MLIR_TYPESCRIPT_PASSES_H

#include "TypeScript/DataStructs.h"

#include <memory>

namespace mlir
{
class Pass;

namespace typescript
{
/// Create a pass for lowering to operations in the `Affine` and `Std` dialects,
/// for a subset of the TypeScript IR (e.g. WhileOp etc).
std::unique_ptr<mlir::Pass> createLowerToAffineTSFuncPass(bool);
std::unique_ptr<mlir::Pass> createLowerToAffineFuncPass();
std::unique_ptr<mlir::Pass> createLowerToAffineModulePass();

/// Create a pass for lowering operations the remaining `TypeScript` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

// to move constant to root of function to avoid "dominating" issue after joining constants in "switch state"
// TODO: should you process, switch satate in createLowerToAffinePass to resolve issue?
std::unique_ptr<mlir::Pass> createRelocateConstantPass();

/// GC Pass to replace malloc, realloc, free with GC_malloc, GC_realloc, GC_free
std::unique_ptr<mlir::Pass> createGCPass();

} // end namespace typescript
} // end namespace mlir

#endif // MLIR_TYPESCRIPT_PASSES_H
