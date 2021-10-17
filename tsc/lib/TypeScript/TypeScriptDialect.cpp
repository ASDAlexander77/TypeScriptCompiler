#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace mlir_ts = mlir::typescript;

//===----------------------------------------------------------------------===//
// TypeScript dialect.
//===----------------------------------------------------------------------===//

LogicalResult verify(mlir_ts::FuncOp op);
LogicalResult verify(mlir_ts::InvokeOp op);
LogicalResult verify(mlir_ts::CastOp op);

#define GET_TYPEDEF_CLASSES
#include "TypeScript/TypeScriptOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TypeScriptInlinerInterface
//===----------------------------------------------------------------------===//

/// This class defines the interface for handling inlining with TypeScript
/// operations.
struct TypeScriptInlinerInterface : public mlir::DialectInlinerInterface
{
    using DialectInlinerInterface::DialectInlinerInterface;

    //===--------------------------------------------------------------------===//
    // Analysis Hooks
    //===--------------------------------------------------------------------===//

    /// All call operations within toy can be inlined.
    bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable, bool wouldBeCloned) const final
    {
        return true;
    }

    /// All operations within toy can be inlined.
    bool isLegalToInline(mlir::Operation *, mlir::Region *, bool, mlir::BlockAndValueMapping &) const final
    {
        return true;
    }

    //===--------------------------------------------------------------------===//
    // Transformation Hooks
    //===--------------------------------------------------------------------===//

    void handleTerminator(mlir::Operation *op, mlir::ArrayRef<Value> valuesToRepl) const final
    {
        // nothing todo
    }

    /// Attempts to materialize a conversion for a type mismatch between a call
    /// from this dialect, and a callable region. This method should generate an
    /// operation that takes 'input' as the only operand, and produces a single
    /// result of 'resultType'. If a conversion can not be generated, nullptr
    /// should be returned.
    mlir::Operation *materializeCallConversion(mlir::OpBuilder &builder, mlir::Value input, mlir::Type resultType,
                                               mlir::Location conversionLoc) const final
    {
        return builder.create<mlir_ts::CastOp>(conversionLoc, resultType, input);
    }
};

void mlir_ts::TypeScriptDialect::initialize()
{
    addOperations<
#define GET_OP_LIST
#include "TypeScript/TypeScriptOps.cpp.inc"
        >();
    addTypes<
#define GET_TYPEDEF_LIST
#include "TypeScript/TypeScriptOpsTypes.cpp.inc"
        >();
    addInterfaces<TypeScriptInlinerInterface>();
}

Type mlir_ts::TypeScriptDialect::parseType(DialectAsmParser &parser) const
{
    llvm::SMLoc typeLoc = parser.getCurrentLocation();

    mlir::Type booleanType;
    if (*generatedTypeParser(getContext(), parser, "boolean", booleanType))
    {
        return booleanType;
    }

    mlir::Type numberType;
    if (*generatedTypeParser(getContext(), parser, "number", numberType))
    {
        return numberType;
    }

    mlir::Type stringType;
    if (*generatedTypeParser(getContext(), parser, "string", stringType))
    {
        return stringType;
    }

    mlir::Type refType;
    if (*generatedTypeParser(getContext(), parser, "ref", refType))
    {
        return refType;
    }

    mlir::Type valueRefType;
    if (*generatedTypeParser(getContext(), parser, "value_ref", valueRefType))
    {
        return valueRefType;
    }

    mlir::Type optionalType;
    if (*generatedTypeParser(getContext(), parser, "optional", optionalType))
    {
        return optionalType;
    }

    mlir::Type enumType;
    if (*generatedTypeParser(getContext(), parser, "enum", enumType))
    {
        return enumType;
    }

    mlir::Type arrayType;
    if (*generatedTypeParser(getContext(), parser, "array", arrayType))
    {
        return arrayType;
    }

    mlir::Type tupleType;
    if (*generatedTypeParser(getContext(), parser, "tuple", tupleType))
    {
        return tupleType;
    }

    parser.emitError(typeLoc, "unknown type in TypeScript dialect");
    return Type();
}

void mlir_ts::TypeScriptDialect::printType(Type type, DialectAsmPrinter &os) const
{
    if (failed(generatedTypePrinter(type, os)))
    {
        llvm_unreachable("unexpected 'TypeScript' type kind");
    }
}

// The functions don't need to be in the header file, but need to be in the mlir
// namespace. Declare them here, then define them immediately below. Separating
// the declaration and definition adheres to the LLVM coding standards.
namespace mlir
{
namespace typescript
{
// FieldInfo is used as part of a parameter, so equality comparison is compulsory.
static bool operator==(const FieldInfo &a, const FieldInfo &b);
// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code hash_value(const FieldInfo &fi);
} // namespace typescript
} // namespace mlir

// FieldInfo is used as part of a parameter, so equality comparison is
// compulsory.
static bool mlir_ts::operator==(const FieldInfo &a, const FieldInfo &b)
{
    return a.id == b.id && a.type == b.type;
}

// FieldInfo is used as part of a parameter, so a hash will be computed.
static llvm::hash_code mlir_ts::hash_value(const FieldInfo &fi)
{
    return llvm::hash_combine(fi.id, fi.type);
}
