#include "TypeScript/TypeScriptOps.h"
#include "TypeScript/TypeScriptDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.cpp.inc"

using namespace mlir;
using namespace mlir::typescript;

//===----------------------------------------------------------------------===//
// IdentifierReferenceOp
//===----------------------------------------------------------------------===//

IdentifierReferenceOp IdentifierReferenceOp::create(Location location, StringRef name) {
  OperationState state(location, "identRef");
  OpBuilder builder(location->getContext());

  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  return cast<IdentifierReferenceOp>(Operation::create(state));
}
