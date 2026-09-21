#include "TypeScript/DataStructs.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "gmock/gmock.h"

// Lowering builds its type converter from the module's `llvm.data_layout`, which MLIRGenModule sets
// from the target triple. It used to fall back to LLVM's default layout (8-byte pointers, i64
// aligned at 4) whenever the attribute was absent, which is wrong for i686 and for x64 alike. Both
// ways a module can fail to supply a usable layout must now be diagnosed errors - neither a silent
// fallback nor an abort inside llvm::DataLayout's constructor. The CLI cannot produce such a
// module, so the pass is run here on hand-built ones.
namespace
{

struct LoweringRun
{
    bool succeeded;
    std::string diagnostics;
};

LoweringRun runLoweringOn(mlir::MLIRContext &context, mlir::ModuleOp module)
{
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic &diag) {
        diagnostics += diag.str();
        diagnostics += "\n";
        return mlir::success();
    });

    CompileOptions options{};
    mlir::PassManager pm(&context);
    pm.addPass(mlir::typescript::createLowerToLLVMPass(options));
    auto result = pm.run(module);
    return {mlir::succeeded(result), diagnostics};
}

TEST(LoweringDataLayoutTest, MissingDataLayoutIsAnError)
{
    mlir::MLIRContext context;
    context.loadDialect<mlir::typescript::TypeScriptDialect, mlir::LLVM::LLVMDialect>();
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

    auto run = runLoweringOn(context, module);

    EXPECT_FALSE(run.succeeded);
    EXPECT_THAT(run.diagnostics, ::testing::HasSubstr("llvm.data_layout"));
    EXPECT_THAT(run.diagnostics, ::testing::HasSubstr("MLIRGenModule"));
    module->erase();
}

TEST(LoweringDataLayoutTest, MalformedDataLayoutIsAnErrorNotAnAbort)
{
    mlir::MLIRContext context;
    context.loadDialect<mlir::typescript::TypeScriptDialect, mlir::LLVM::LLVMDialect>();
    mlir::OpBuilder builder(&context);
    auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
    module->setAttr(mlir::LLVM::LLVMDialect::getDataLayoutAttrName(), builder.getStringAttr("e-p:banana"));

    auto run = runLoweringOn(context, module);

    EXPECT_FALSE(run.succeeded);
    EXPECT_THAT(run.diagnostics, ::testing::HasSubstr("invalid data layout 'e-p:banana'"));
    module->erase();
}

} // namespace
