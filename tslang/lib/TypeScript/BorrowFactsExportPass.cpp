#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/Passes.h"
#include "TypeScript/Defines.h"
#include "TypeScript/MLIRLogic/MLIRTypeHelper.h"

#include "StackEscapeFacts.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "borrow-facts-export"

namespace mlir_ts = mlir::typescript;

namespace
{

// Borrow facts (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md): what this
// module promises a program linked statically against it - for each function another module can
// call, the parameters its body only borrows. Computed by the analysis stack promotion runs, but
// trusting no vtable: an importer may subclass any class and override anything. Written whether
// or not this module promotes anything itself, and whatever it exports.
class BorrowFactsExportPass : public mlir::PassWrapper<BorrowFactsExportPass, mlir::OperationPass<mlir::ModuleOp>>
{
    std::string symbolName;

  public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BorrowFactsExportPass)

    explicit BorrowFactsExportPass(std::string symbolName) : symbolName(std::move(symbolName))
    {
    }

    void runOnOperation() override
    {
        auto module = getOperation();
        stack_escape::StackEscapeFacts facts(module, /*trustLocalVTables=*/false);

        llvm::SmallVector<std::pair<llvm::StringRef, std::string>> lines;
        module.walk([&](mlir_ts::FuncOp funcOp) {
            // renamed by @dllname / @linkname: it defines another symbol than its name, which an
            // importer could not tell apart
            if (funcOp.isDeclaration() || funcOp.isPrivate() || funcOp->hasAttr(DLL_NAME))
            {
                return;
            }

            std::string indexes;
            for (auto index : llvm::seq<unsigned>(0, funcOp.getBody().getNumArguments()))
            {
                if (facts.paramBorrowed(funcOp, index))
                {
                    indexes += " " + std::to_string(index);
                }
            }

            if (!indexes.empty())
            {
                lines.push_back({funcOp.getSymName(), indexes});
            }
        });

        if (lines.empty())
        {
            return;
        }

        llvm::sort(lines);
        std::string text;
        for (auto &[name, indexes] : lines)
        {
            text += name.str() + indexes + "\n";
        }

        // The LLVM lowering turns this into the symbol (emitBorrowFacts): an LLVM global in the
        // module before then upsets its lookups of the globals it converts.
        mlir::Builder builder(module.getContext());
        module->setAttr(SHARED_LIB_BORROW_FACTS_ATTR_NAME,
                        builder.getArrayAttr({builder.getStringAttr(symbolName), builder.getStringAttr(text)}));
    }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir_ts::createBorrowFactsExportPass(std::string symbolName)
{
    return std::make_unique<BorrowFactsExportPass>(std::move(symbolName));
}
