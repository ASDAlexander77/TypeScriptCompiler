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
std::unique_ptr<mlir::Pass> createLowerToAffineTSFuncPass(CompileOptions&);
std::unique_ptr<mlir::Pass> createLowerToAffineFuncPass(CompileOptions&);
std::unique_ptr<mlir::Pass> createLowerToAffineModulePass(CompileOptions&);

/// Create a pass for lowering operations the remaining `TypeScript` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass(CompileOptions&);

// to move constant to root of function to avoid "dominating" issue after joining constants in "switch state"
// TODO: should you process, switch satate in createLowerToAffinePass to resolve issue?
std::unique_ptr<mlir::Pass> createRelocateConstantPass();

/// Checks that a slot which takes a reference gives it back on every path out of the function,
/// unwind paths included. Runs at the affine level, where unwind paths are ordinary CFG edges,
/// and in every memory model - the ownership ops survive to there regardless of model.
std::unique_ptr<mlir::Pass> createOwnershipVerifierPass();

/// GC Pass to replace malloc, realloc, free with GC_malloc, GC_realloc, GC_free
std::unique_ptr<mlir::Pass> createGCPass(CompileOptions&);
/// MemAlloc Pass to replace ts_malloc, ts_realloc, ts_free
std::unique_ptr<mlir::Pass> createMemAllocPass(CompileOptions&);

} // end namespace typescript
} // end namespace mlir

#endif // MLIR_TYPESCRIPT_PASSES_H
