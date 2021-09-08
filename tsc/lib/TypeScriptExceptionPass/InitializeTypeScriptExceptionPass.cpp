#include "TypeScript/InitializeTypeScriptExceptionPass.h"
#include "llvm/PassRegistry.h"

namespace llvm
{

namespace typescript
{

void initializeTypeScriptExceptionPassIRTransforms(llvm::PassRegistry &Registry)
{
    llvm::initializeTypeScriptExceptionPassPass(Registry);
}

} // namespace typescript

} // namespace llvm
