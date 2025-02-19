#include "TypeScript/AsyncDialectTranslation.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace ::typescript;
namespace mlir_ts = mlir::typescript;

/// Add all the MLIR dialects to the provided registry.
void mlir::typescript::registerAsyncDialectTranslation(DialectRegistry &registry)
{
    registry.insert<async::AsyncDialect>();
}

/// Append all the MLIR dialects to the registry contained in the given context.
void mlir::typescript::registerAsyncDialectTranslation(MLIRContext &context)
{
    DialectRegistry registry;
    registerAsyncDialectTranslation(registry);
    context.appendDialectRegistry(registry);
}