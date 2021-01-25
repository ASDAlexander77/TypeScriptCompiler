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
}

//===----------------------------------------------------------------------===//
// TypeScript Types
//===----------------------------------------------------------------------===//

namespace mlir {
namespace typescript {
namespace detail {
/// This class represents the internal storage of the TypeScript `InterfaceReferenceType`.
struct IdentifierReferenceTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::StringRef;

  /// A constructor for the type storage instance.
  IdentifierReferenceTypeStorage(llvm::StringRef identifierName)
      : identifierName(identifierName) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == identifierName; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage, see the `StructType::get` method.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::StringRef identifierName) {
    return KeyTy(identifierName);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static IdentifierReferenceTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the identifier name from the provided `KeyTy` into the allocator.
    llvm::StringRef identifierName = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<IdentifierReferenceTypeStorage>())
        IdentifierReferenceTypeStorage(identifierName);
  }

  /// The following field contains identifier name.
  llvm::StringRef identifierName;
};
} // end namespace detail
} // end namespace typescript
} // end namespace mlir

/// Create an instance of a `InterfaceReferenceType` with the given identifier name.
IdentifierReferenceType IdentifierReferenceType::get(mlir::MLIRContext *ctx, llvm::StringRef identifierName) {
  return Base::get(ctx, identifierName);
}

/// Returns the identifier name of this type.
llvm::StringRef IdentifierReferenceType::getIdentifierName() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->identifierName;
}
