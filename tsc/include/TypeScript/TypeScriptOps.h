#ifndef TYPESCRIPT_TYPESCRIPTOPS_H
#define TYPESCRIPT_TYPESCRIPTOPS_H

#include "TypeScript/Config.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir
{
namespace typescript
{

/// FieldInfo represents a field in the TupleType(StructType) data type. It is used as a
/// parameter in TestTypeDefs.td.
struct FieldInfo
{
    Attribute id;
    Type type;

    // Custom allocation called from generated constructor code
    FieldInfo allocateInto(TypeStorageAllocator &alloc) const
    {
        // return FieldInfo{alloc.copyInto(name), type};
        return FieldInfo{id, type};
    }
};

void buildTerminatedBody(OpBuilder &builder, Location loc);

} // namespace typescript

} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "TypeScript/TypeScriptOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.h.inc"

#endif // TYPESCRIPT_TYPESCRIPTOPS_H
