#include "TypeScript/DiagnosticHelper.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/Debug.h"

//#define DEBUG_TYPE "mlir"
#define DEBUG_TYPE "llvm"

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
        .template Case<mlir::NameLoc>([&](auto nameLoc) {
            printLocation(os, nameLoc.getChildLoc(), path, suppressSeparator);
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
        .template Case<mlir::CallSiteLoc>([&](auto callSiteLoc) {
            printLocation(os, callSiteLoc.getCaller(), path, suppressSeparator);
        })        
        .template Case<mlir::OpaqueLoc>([&](auto opaqueLoc) {
            printLocation(os, opaqueLoc.getFallbackLocation(), path, suppressSeparator);
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

            if (auto locAsMetadata = fusedLoc.getMetadata().template dyn_cast<mlir::LocationAttr>())
            {
                if (notFirst)
                {
                    os << ' ' << '-' << ' ';
                }

                printLocation(os, locAsMetadata, currentPath, true);
            }

            if (!suppressSeparator)
            {
                os << ':' << ' ';
            }
        })        
        .Default([&](auto loc) { 
            LLVM_DEBUG(llvm::dbgs() << "not impl location type: " << loc << "\n";);
            llvm_unreachable("not implemented"); 
        });
}

void printDiagnostics(SourceMgrDiagnosticHandlerEx &sourceMgrHandler, mlir::SmallVector<std::unique_ptr<mlir::Diagnostic>> &postponedMessages, bool disableWarnings)
{
    for (auto msgIndex = 0; msgIndex < postponedMessages.size(); msgIndex++)
    {
        auto &diag = postponedMessages[msgIndex];

        if (disableWarnings && diag->getSeverity() == mlir::DiagnosticSeverity::Warning)
        {
            continue;
        }

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
            sourceMgrHandler.emit(*diag.get());
        }
    }
}    

