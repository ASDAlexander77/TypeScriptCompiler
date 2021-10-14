#ifndef LANDINGPAD_FIX_PASS__H
#define LANDINGPAD_FIX_PASS__H

#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"

#define LANDINGPAD_FIX_PASS_NAME "Landing Pad Fix Pass"

namespace llvm
{

const void *getLandingPadFixPassID();
void initializeLandingPadFixPassPass(llvm::PassRegistry &);

} // end namespace llvm

#endif // LANDINGPAD_FIX_PASS__H
