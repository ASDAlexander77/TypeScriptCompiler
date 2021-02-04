#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "TypeScript/TypeScriptTypeDefs.cpp.inc"
