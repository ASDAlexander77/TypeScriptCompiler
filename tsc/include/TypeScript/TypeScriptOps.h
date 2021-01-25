#ifndef TYPESCRIPT_TYPESCRIPTOPS_H
#define TYPESCRIPT_TYPESCRIPTOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


namespace mlir {
	namespace typescript {
		namespace detail {
			struct IdentifierReferenceTypeStorage;
		} // end namespace detail
	}
}

#define GET_OP_CLASSES
#include "TypeScript/TypeScriptOps.h.inc"

namespace mlir {
namespace typescript {

//===----------------------------------------------------------------------===//
// TypeScript Types
//===----------------------------------------------------------------------===//

/// This class defines the Typescript identifier reference type.
/// All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (IdentifierReferenceType), the base class to use (Type), and the storage class
/// (IdentifierReferenceTypeStorage).
class IdentifierReferenceType : public mlir::Type::TypeBase<IdentifierReferenceType, mlir::Type,
                                               detail::IdentifierReferenceTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `IdentifierReferenceType` with the given name.
  static IdentifierReferenceType get(mlir::MLIRContext *ctx, llvm::StringRef identifierName);

  /// Returns the element types of this struct type.
  llvm::StringRef getIdentifierName();
};
} // end namespace typescript
} // end namespace mlir

#endif // TYPESCRIPT_TYPESCRIPTOPS_H
