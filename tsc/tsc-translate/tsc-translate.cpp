//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllTranslations.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Translation.h"

#include "TypeScript/TypeScriptDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  // TODO: Register typescript translations here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::typescript::TypeScriptDialect>();
  registry.insert<mlir::StandardOpsDialect>();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
