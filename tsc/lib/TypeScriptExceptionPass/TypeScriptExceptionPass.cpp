#include "TypeScript\TypeScriptExceptionPass.h"
#include "TypeScript\InitializeTypeScriptExceptionPass.h"

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::typescript;
using namespace PatternMatch;

#define DEBUG_TYPE "pass"

namespace
{
struct TypeScriptExceptionPass : public FunctionPass
{
    static char ID;
    TypeScriptExceptionPass() : FunctionPass(ID)
    {
        initializeTypeScriptExceptionPassPass(*PassRegistry::getPassRegistry());
    }

    bool runOnFunction(Function &F) override
    {
        return true;
    }
};
} // namespace

char TypeScriptExceptionPass::ID = 0;
INITIALIZE_PASS_BEGIN(TypeScriptExceptionPass, DEBUG_TYPE, "TypeScript Exception Pass", false, false)
INITIALIZE_PASS_END(TypeScriptExceptionPass, DEBUG_TYPE, "TypeScript Exception Pass", false, false)
