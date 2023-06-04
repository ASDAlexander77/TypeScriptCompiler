#ifndef MLIR_DIAGNOSTIC_HELPER_H_
#define MLIR_DIAGNOSTIC_HELPER_H_

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

bool compareDiagnostic(const mlir::Diagnostic&, const mlir::Diagnostic&);
void publishDiagnostic(const mlir::Diagnostic &, llvm::StringRef);
void printDiagnostics(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &, llvm::StringRef);
void printLocation(llvm::raw_ostream &, mlir::Location, llvm::StringRef, bool suppressSeparator = false);

#endif // MLIR_DIAGNOSTIC_HELPER_H_
