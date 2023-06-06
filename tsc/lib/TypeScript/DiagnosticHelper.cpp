#include "TypeScript/DiagnosticHelper.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"

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

void printLocation(llvm::raw_ostream &os, mlir::Location location, llvm::StringRef path, bool suppressSeparator)
{
    llvm::TypeSwitch<mlir::Location>(location)
        .template Case<mlir::UnknownLoc>([&](auto) {
            // nothing todo
        })
        .template Case<mlir::FileLineColLoc>([&](auto fileLineColLoc) {
            auto filePath = fileLineColLoc.getFilename().getValue();
            if (filePath.starts_with(path))
            {
                filePath = filePath.substr(path.size());
                if (filePath.starts_with("\\") || filePath.starts_with("/"))
                {
                    filePath = filePath.substr(1);
                }
            }

            if (filePath.size() > 0)
            {
                os << filePath << ':';
            }
            
            os << fileLineColLoc.getLine() << ':' << fileLineColLoc.getColumn();
            if (!suppressSeparator)
            {
                os << ':' << ' ';
            }
        })
        .template Case<mlir::FusedLoc>([&](auto fusedLoc) {
            auto notFirst = false;

            auto currentPath = path;
            for (auto loc : fusedLoc.getLocations())
            {
                if (notFirst)
                {
                    os << ' ' << '-' << ' ';
                }

                printLocation(os, loc, currentPath, true);

                if (!notFirst)
                {
                    if (auto fileLineColLoc = loc.template dyn_cast<mlir::FileLineColLoc>())
                    {
                        currentPath = fileLineColLoc.getFilename().getValue();
                    }
                }

                notFirst = true;
            }

            if (!suppressSeparator)
            {
                os << ':' << ' ';
            }
        })        
        .Default([&](auto type) { llvm_unreachable("not implemented"); });
}

void publishDiagnostic(const mlir::Diagnostic &diag, llvm::StringRef path)
{
    auto printMsg = [](llvm::raw_ostream &os, const mlir::Diagnostic &diag, const char *msg) {
        os << msg;
        os << diag << '\n';
        os.flush();
    };

    switch (diag.getSeverity())
    {
    case mlir::DiagnosticSeverity::Note:
        printLocation(llvm::outs(), diag.getLocation(), path);
        printMsg(llvm::WithColor::note(llvm::outs()), diag, "");
        for (auto &note : diag.getNotes())
        {
            printLocation(llvm::outs(), note.getLocation(), path);
            printMsg(llvm::WithColor::note(llvm::outs()), note, "");
        }

        break;
    case mlir::DiagnosticSeverity::Warning:
        printLocation(llvm::outs(), diag.getLocation(), path);
        printMsg(llvm::WithColor::warning(llvm::outs()), diag, "");
        break;
    case mlir::DiagnosticSeverity::Error:
        printLocation(llvm::errs(), diag.getLocation(), path);
        printMsg(llvm::WithColor::error(llvm::errs()), diag, "");
        break;
    case mlir::DiagnosticSeverity::Remark:
        printLocation(llvm::outs(), diag.getLocation(), path);
        printMsg(llvm::WithColor::remark(llvm::outs()), diag, "");
        break;
    }
}

void printDiagnostics(mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &postponedMessages, llvm::StringRef path)
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
            publishDiagnostic(*diag.get(), path);
        }
    }
}    

