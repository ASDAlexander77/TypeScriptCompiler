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
            // if PropertyRefOf
            if (auto propertyRefOp = dyn_cast<mlir_ts::PropertyRefOp>(op))
            {
                if (auto refType = propertyRefOp.getType().dyn_cast_or_null<mlir_ts::RefType>())
                {
                    if (auto funcType = refType.getElementType().dyn_cast_or_null<mlir::FunctionType>())
                    {
                        auto boundFunc = mlir_ts::BoundFunctionType::get(funcType.getContext(), funcType);
                        auto refBoundFunc = mlir_ts::RefType::get(boundFunc);
                        propertyRefOp.getResult().setType(refBoundFunc);

                        LLVM_DEBUG(llvm::dbgs() << "users: \n"; for (auto user
                                                                     : propertyRefOp.getResult().getUsers()) llvm::dbgs()
                                                                << *user << "\n";);

                        updateChainDependencies(propertyRefOp.getResult(), boundFunc);
                    }
                }
            }
        });
    }

    void updateChainDependencies(mlir::Value resultValue, mlir::Type boundFunc)
    {
        for (auto user : resultValue.getUsers())
        {
            if (auto loadOp = dyn_cast<mlir_ts::LoadOp>(user))
            {
                loadOp.getResult().setType(boundFunc);
                updateChainDependencies(loadOp.getResult(), boundFunc);
                continue;
            }

            if (auto variableOp = dyn_cast<mlir_ts::VariableOp>(user))
            {
                variableOp.getResult().setType(mlir_ts::RefType::get(boundFunc));
                updateChainDependencies(variableOp.getResult(), boundFunc);
                continue;
            }
        }
    }
};
} // end anonymous namespace

/// Create pass.
std::unique_ptr<mlir::Pass> mlir_ts::createLoadBoundPropertiesPass()
{
    return std::make_unique<LoadBoundPropertiesPass>();
}
