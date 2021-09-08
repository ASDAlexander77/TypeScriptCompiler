#ifndef TYPESCRIPTEXCEPTIONPASS__H
#define TYPESCRIPTEXCEPTIONPASS__H

#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

namespace llvm
{

void initializeTypeScriptExceptionPassPass(llvm::PassRegistry &);

namespace typescript
{

llvm::FunctionPass *createTypeScriptExceptionPass();

} // end namespace typescript

} // end namespace llvm

#endif // TYPESCRIPTEXCEPTIONPASS__H
