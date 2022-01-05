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

#ifndef DISABLE_CUSTOM_CLASSSTORETYPE
namespace mlir
{
namespace typescript
{
namespace detail
{
struct ClassStorageTypeStorage;
} // end namespace detail
class ClassStorageType : public ::mlir::Type::TypeBase<ClassStorageType, ::mlir::Type, detail::ClassStorageTypeStorage>
{
  public:
    /// Inherit some necessary constructors from 'TypeBase'.
    using Base::Base;

    /// Return the number of held types.
    size_t size() const;

    /// Iterate over the held elements.
    using iterator = ArrayRef<::mlir::typescript::FieldInfo>::iterator;
    iterator begin() const
    {
        return getFields().begin();
    }
    iterator end() const
    {
        return getFields().end();
    }

    int getIndex(Attribute id)
    {
        auto dist =
            std::distance(begin(), std::find_if(begin(), end(), [&](::mlir::typescript::FieldInfo fldInf) { return id == fldInf.id; }));
        return (signed)dist >= (signed)size() ? -1 : dist;
    }

    /// Return the element type at index 'index'.
    ::mlir::typescript::FieldInfo getFieldInfo(size_t index) const
    {
        assert(index >= 0 && index < size() && "invalid index for tuple type");
        return getFields()[index];
    }

    Attribute getId(size_t index) const
    {
        assert(index >= 0 && index < size() && "invalid index for tuple type");
        return getFields()[index].id;
    }

    Type getType(size_t index) const
    {
        assert(index >= 0 && index < size() && "invalid index for tuple type");
        return getFields()[index].type;
    }

    static ClassStorageType get(::mlir::MLIRContext *context, FlatSymbolRefAttr name,
                                ::llvm::ArrayRef<::mlir::typescript::FieldInfo> fields);
    static constexpr ::llvm::StringLiteral getMnemonic()
    {
        return ::llvm::StringLiteral("class_storage");
    }

    static ::mlir::Type parse(::mlir::MLIRContext *context, ::mlir::DialectAsmParser &parser);
    void print(::mlir::DialectAsmPrinter &printer) const;
    FlatSymbolRefAttr getName() const;
    ::llvm::ArrayRef<::mlir::typescript::FieldInfo> getFields() const;
};

} // namespace typescript

} // namespace mlir
#endif

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.h.inc"

#endif // TYPESCRIPT_TYPESCRIPTOPS_H
