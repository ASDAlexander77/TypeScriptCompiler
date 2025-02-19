#ifndef TYPESCRIPT_TYPESCRIPTOPS_H
#define TYPESCRIPT_TYPESCRIPTOPS_H

#include "TypeScript/Config.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/IR/Matchers.h"

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
    bool isConditional;

    FieldInfo() = default;
    FieldInfo(Attribute id, Type type) : id{id}, type{type}, isConditional{false} {};
    FieldInfo(Attribute id, Type type, bool isConditional) : id{id}, type{type}, isConditional{isConditional} {};

    // Custom allocation called from generated constructor code
    FieldInfo allocateInto(TypeStorageAllocator &alloc) const
    {
        // return FieldInfo{alloc.copyInto(name), type};
        return FieldInfo{id, type, false};
    }
};

void buildTerminatedBody(OpBuilder &builder, Location loc);

bool isTrue(mlir::Region &);
bool isEmpty(mlir::Region &);

namespace detail
{
struct ObjectStorageTypeStorage : public ::mlir::TypeStorage
{
    ObjectStorageTypeStorage(FlatSymbolRefAttr name) : name(name), fields({})
    {
    }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = FlatSymbolRefAttr;
    bool operator==(const KeyTy &tblgenKey) const
    {
        if (!(name == tblgenKey))
            return false;
        return true;
    }
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey)
    {
        return ::llvm::hash_combine(tblgenKey);
    }

    /// Define a construction method for creating a new instance of this storage.
    static ObjectStorageTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey)
    {
        auto name = tblgenKey;

        return new (allocator.allocate<ObjectStorageTypeStorage>()) ObjectStorageTypeStorage(name);
    }

    LogicalResult mutate(TypeStorageAllocator &allocator, ::llvm::ArrayRef<::mlir::typescript::FieldInfo> newFields)
    {
        // Cannot set a different body than before.
        llvm::SmallVector<::mlir::typescript::FieldInfo, 4> tmpFields;

        for (size_t i = 0, e = newFields.size(); i < e; ++i)
            tmpFields.push_back(newFields[i].allocateInto(allocator));
        auto copiedFields = allocator.copyInto(ArrayRef<::mlir::typescript::FieldInfo>(tmpFields));

        fields = copiedFields;

        return success();
    }
    
    FlatSymbolRefAttr name;
    ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
};

struct ClassStorageTypeStorage : public ::mlir::TypeStorage
{
    ClassStorageTypeStorage(FlatSymbolRefAttr name) : name(name), fields({})
    {
    }

    /// The hash key is a tuple of the parameter types.
    using KeyTy = FlatSymbolRefAttr;
    bool operator==(const KeyTy &tblgenKey) const
    {
        if (!(name == tblgenKey))
            return false;
        return true;
    }
    static ::llvm::hash_code hashKey(const KeyTy &tblgenKey)
    {
        return ::llvm::hash_combine(tblgenKey);
    }

    /// Define a construction method for creating a new instance of this storage.
    static ClassStorageTypeStorage *construct(::mlir::TypeStorageAllocator &allocator, const KeyTy &tblgenKey)
    {
        auto name = tblgenKey;

        return new (allocator.allocate<ClassStorageTypeStorage>()) ClassStorageTypeStorage(name);
    }

    LogicalResult mutate(TypeStorageAllocator &allocator, ::llvm::ArrayRef<::mlir::typescript::FieldInfo> newFields)
    {
        // Cannot set a different body than before.
        llvm::SmallVector<::mlir::typescript::FieldInfo, 4> tmpFields;

        for (size_t i = 0, e = newFields.size(); i < e; ++i)
            tmpFields.push_back(newFields[i].allocateInto(allocator));
        auto copiedFields = allocator.copyInto(ArrayRef<::mlir::typescript::FieldInfo>(tmpFields));

        fields = copiedFields;

        return success();
    }

    FlatSymbolRefAttr name;
    ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields;
};
} // namespace detail

} // namespace typescript

} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "TypeScript/TypeScriptOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.h.inc"

#endif // TYPESCRIPT_TYPESCRIPTOPS_H
