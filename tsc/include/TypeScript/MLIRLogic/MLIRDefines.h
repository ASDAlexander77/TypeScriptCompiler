
#ifndef MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRDEFINES_H_
#define MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRDEFINES_H_

#include "TypeScript/DOM.h"

#include "llvm/ADT/ScopedHashTable.h"

#define EXIT_IF_FAILED_OR_NO_VALUE(value)                                                                                                  \
    if (value.failed_or_no_value())                                                                                                        \
    {                                                                                                                                      \
        return value;                                                                                                                      \
    }                                                                                                                                      

#define EXIT_IF_FAILED(value)                                                                                                              \
    if (value.failed())                                                                                                                    \
    {                                                                                                                                      \
        return mlir::failure();                                                                                                            \
    }

#define VALIDATE(value, loc)                                                                                                               \
    if (!value)                                                                                                                            \
    {                                                                                                                                      \
        if (!genContext.allowPartialResolve)                                                                                               \
        {                                                                                                                                  \
            emitError(loc, "expression has no result");                                                                                    \
        }                                                                                                                                  \
                                                                                                                                           \
        return mlir::failure();                                                                                                            \
    }

using VariablePairT = std::pair<mlir::Value, ts::VariableDeclarationDOM::TypePtr>;
using SymbolTableScopeT = llvm::ScopedHashTableScope<StringRef, VariablePairT>;

#endif // MLIR_TYPESCRIPT_MLIRGENLOGIC_MLIRDEFINES_H_