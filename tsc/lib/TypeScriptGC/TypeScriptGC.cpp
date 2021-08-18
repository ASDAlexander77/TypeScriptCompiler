#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptGC.h"

#include "llvm/IR/GCStrategy.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

namespace
{

class TypeScriptGC : public GCStrategy
{
  public:
    TypeScriptGC()
    {
        UseStatepoints = false;
        NeededSafePoints = false;
        // to use TypeScriptGCPrinter
        UsesMetadata = true;
    }
};

} // end anonymous namespace

// Register all the above so that they can be found at runtime.  Note that
// these static initializers are important since the registration list is
// constructed from their storage.
static GCRegistry::Add<TypeScriptGC> TSGC(TYPESCRIPT_GC_NAME, "typescript garbage collector");

// to force linking
void mlir::typescript::registerTypeScriptGC()
{
    mlir::typescript::registerTypeScriptGCPrinter();
}