#define DEBUG_TYPE "pass"

#include "mlir/Pass/Pass.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptFunctionPass.h"
#include "TypeScript/Passes.h"
#include "TypeScript/LoadBoundPropertiesInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir_ts = mlir::typescript;

using namespace mlir;
using namespace typescript;

#include "TypeScript/LoadBoundPropertiesOpInterfaces.cpp.inc"

namespace
{
class LoadBoundPropertiesPass : public mlir::PassWrapper<LoadBoundPropertiesPass, TypeScriptFunctionPass>
{
  public:
    void runOnFunction() override
    {
        auto f = getFunction();

        f.walk([&](mlir::Operation *op) {
            if (auto propertyRefOp = dyn_cast<mlir_ts::PropertyRefOp>(op))
            {
                // ...
            }
        });
    }
};
} // end anonymous namespace

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createLoadBoundPropertiesPass()
{
    return std::make_unique<LoadBoundPropertiesPass>();
}
