#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.cpp.inc"

using namespace mlir;
using namespace mlir::typescript;

//===----------------------------------------------------------------------===//
// xxxxOp
//===----------------------------------------------------------------------===//
