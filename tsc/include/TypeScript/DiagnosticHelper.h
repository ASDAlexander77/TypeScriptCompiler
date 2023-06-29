#ifndef MLIR_DIAGNOSTIC_HELPER_H_
#define MLIR_DIAGNOSTIC_HELPER_H_

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

class SourceMgrDiagnosticHandlerEx : public mlir::SourceMgrDiagnosticHandler
{
public:
    SourceMgrDiagnosticHandlerEx(llvm::SourceMgr &mgr, mlir::MLIRContext *ctx);
    void emit(mlir::Diagnostic &diag);
};

void printDiagnostics(SourceMgrDiagnosticHandlerEx &, mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &, bool);
void printLocation(llvm::raw_ostream &, mlir::Location, llvm::StringRef, bool suppressSeparator = false);

#endif // MLIR_DIAGNOSTIC_HELPER_H_
