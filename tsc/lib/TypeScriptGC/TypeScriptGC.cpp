#include "TypeScript/Defines.h"
#include "TypeScript/TypeScriptGC.h"

#include "llvm/IR/GCStrategy.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/StringMap.h"

#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"

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
}

void mlir::typescript::registerTypeScriptGCStrategy()
{
    static GCRegistry::Add<TypeScriptGC> TSGC1(TYPESCRIPT_GC_NAME, "typescript garbage collector");
    mlir::typescript::registerTypeScriptGCPrinter();
}

// Export symbols for the MLIR runner integration. All other symbols are hidden.
#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

extern "C" API void __mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols);

// to support shared_libs
void __mlir_execution_engine_init(llvm::StringMap<void *> &exportSymbols)
{
    // NOT WORKING ANYWAY
    mlir::typescript::registerTypeScriptGCStrategy();
}

extern "C" API void __mlir_execution_engine_destroy()
{
    // nothing todo.
}
