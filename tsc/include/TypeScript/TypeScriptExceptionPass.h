#ifndef TYPESCRIPTEXCEPTIONPASS__H
#define TYPESCRIPTEXCEPTIONPASS__H

#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

#define TYPESCRIPT_EXCEPTION_PASS_ARG_NAME "typescript-exception"
#define TYPESCRIPT_EXCEPTION_PASS_NAME "TypeScript Exception Pass"

namespace llvm
{

const void *getTypeScriptExceptionPassID();
void initializeTypeScriptExceptionPassPass(llvm::PassRegistry &);

} // end namespace llvm

#endif // TYPESCRIPTEXCEPTIONPASS__H
