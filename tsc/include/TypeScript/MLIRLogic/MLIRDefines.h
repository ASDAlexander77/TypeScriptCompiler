
#ifndef MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRDEFINES_H_
#define MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRDEFINES_H_

#include "TypeScript/DOM.h"

#include "llvm/ADT/ScopedHashTable.h"

#define VALIDATE(value, loc)                                                                                                               \
    if (!value)                                                                                                                            \
    {                                                                                                                                      \
        if (!genContext.allowPartialResolve)                                                                                               \
        {                                                                                                                                  \
            emitError(loc, "expression has no result");                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        return mlir::Value();                                                                                                              \
    }                                                                                                                                      \
                                                                                                                                           \
    if (auto unresolved = dyn_cast_or_null<mlir_ts::UnresolvedSymbolRefOp>(value.getDefiningOp()))                                         \
    {                                                                                                                                      \
        if (!genContext.allowPartialResolve)                                                                                               \
        {                                                                                                                                  \
            emitError(value.getDefiningOp()->getLoc(), "can't find variable: ") << unresolved.identifier();                                \
        }                                                                                                                                  \
                                                                                                                                           \
        return mlir::Value();                                                                                                              \
    }

#define VALIDATE_LOGIC(value, loc)                                                                                                         \
    if (!value)                                                                                                                            \
    {                                                                                                                                      \
        if (!genContext.allowPartialResolve)                                                                                               \
        {                                                                                                                                  \
            emitError(loc, "expression has no result");                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        return mlir::failure();                                                                                                            \
    }                                                                                                                                      \
                                                                                                                                           \
    if (auto unresolved = dyn_cast_or_null<mlir_ts::UnresolvedSymbolRefOp>(value.getDefiningOp()))                                         \
    {                                                                                                                                      \
        if (!genContext.allowPartialResolve)                                                                                               \
        {                                                                                                                                  \
            emitError(value.getDefiningOp()->getLoc(), "can't find variable: ") << unresolved.identifier();                                \
        }                                                                                                                                  \
                                                                                                                                           \
        return mlir::failure();                                                                                                            \
    }

#define TEST_LOGIC(value)                                                                                                                  \
    if (!value)                                                                                                                            \
    {                                                                                                                                      \
        return mlir::failure();                                                                                                            \
    }                                                                                                                                      \
                                                                                                                                           \
    if (auto unresolved = dyn_cast_or_null<mlir_ts::UnresolvedSymbolRefOp>(value.getDefiningOp()))                                         \
    {                                                                                                                                      \
        return mlir::failure();                                                                                                            \
    }

#define IS_VALID(value) (!genContext.allowPartialResolve && !value && !isa<mlir_ts::UnresolvedSymbolRefOp>(value.getDefiningOp()))

using VariablePairT = std::pair<mlir::Value, ts::VariableDeclarationDOM::TypePtr>;
using SymbolTableScopeT = llvm::ScopedHashTableScope<StringRef, VariablePairT>;

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRDEFINES_H_