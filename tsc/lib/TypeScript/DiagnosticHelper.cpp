#include "TypeScript/DiagnosticHelper.h"

bool compareDiagnostic(const mlir::Diagnostic& left, const mlir::Diagnostic& right)
{
    if (left.getLocation() != right.getLocation())
    {
        return false;
    }

    if (left.str() != right.str())
    {
        return false;
    }

    return true;
}

void publishDiagnostic(mlir::Diagnostic &diag)
{
    auto printMsg = [](llvm::raw_fd_ostream &os, mlir::Diagnostic &diag, const char *msg) {
        if (!diag.getLocation().isa<mlir::UnknownLoc>())
            os << diag.getLocation() << ": ";
        os << msg;

        // The default behavior for errors is to emit them to stderr.
        os << diag << '\n';
        os.flush();
    };

    switch (diag.getSeverity())
    {
    case mlir::DiagnosticSeverity::Note:
        printMsg(llvm::outs(), diag, "note: ");
        for (auto &note : diag.getNotes())
        {
            printMsg(llvm::outs(), note, "note: ");
        }

        break;
    case mlir::DiagnosticSeverity::Warning:
        printMsg(llvm::outs(), diag, "warning: ");
        break;
    case mlir::DiagnosticSeverity::Error:
        printMsg(llvm::errs(), diag, "error: ");
        break;
    case mlir::DiagnosticSeverity::Remark:
        printMsg(llvm::outs(), diag, "information: ");
        break;
    }
}

void printDiagnostics(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &postponedMessages)
{
    for (auto msgIndex = 0; msgIndex < postponedMessages.size(); msgIndex++)
    {
        auto &diag = postponedMessages[msgIndex];

        // check if unique
        auto unique = true;
        if (msgIndex > 0)
        {
            for (auto msgIndexTest = msgIndex - 1; msgIndexTest >= 0; msgIndexTest--)
            {
                if (compareDiagnostic(*postponedMessages[msgIndexTest].get(), *diag.get()))
                {
                    unique = false;
                    break;
                }
            }
        }

        if (unique)
        {
            // we show messages when they metter
            publishDiagnostic(*diag.get());
        }
    }
}    

