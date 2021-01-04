#ifndef TYPESCRIPT_TYPESCRIPTOPS_H
#define TYPESCRIPT_TYPESCRIPTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.h.inc"

#endif // TYPESCRIPT_TYPESCRIPTOPS_H
