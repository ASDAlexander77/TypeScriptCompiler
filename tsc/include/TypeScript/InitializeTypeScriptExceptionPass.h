#ifndef INITIALIZETYPESCRIPTEXCEPTIONPASS__H
#define INITIALIZETYPESCRIPTEXCEPTIONPASS__H

#include "llvm/IR/PassManager.h"

namespace llvm
{

void initializeTypeScriptExceptionPassPass(llvm::PassRegistry &Registry);

namespace typescript
{

void initializeTypeScriptExceptionPassIRTransforms(llvm::PassRegistry &Registry);

} // end namespace typescript

} // end namespace llvm

#endif // INITIALIZETYPESCRIPTEXCEPTIONPASS__H
