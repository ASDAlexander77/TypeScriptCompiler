#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

using namespace mlir;
using namespace mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScript dialect.
//===----------------------------------------------------------------------===//

void TypeScriptDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TypeScript/TypeScriptOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "TypeScript/TypeScriptOpsTypes.cpp.inc"
      >();     
}

//===----------------------------------------------------------------------===//
// TypeScript Types
//===----------------------------------------------------------------------===//
