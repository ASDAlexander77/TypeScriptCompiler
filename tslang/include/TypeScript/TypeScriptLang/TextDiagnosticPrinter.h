#ifndef TYPESCRIPT_TEXTDIAGNOSTICPRINTER_H
#define TYPESCRIPT_TEXTDIAGNOSTICPRINTER_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/raw_ostream.h"

namespace clang
{
    class DiagnosticOptions;
    class DiagnosticsEngine;
} // namespace clang

using llvm::IntrusiveRefCntPtr;
using llvm::raw_ostream;

namespace typescript::tslang
{
    class TextDiagnostic;

    class TextDiagnosticPrinter : public clang::DiagnosticConsumer
    {
        raw_ostream &os;
        llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diagOpts;

        /// A string to prefix to error messages.
        std::string prefix;

    public:
        TextDiagnosticPrinter(raw_ostream &os, clang::DiagnosticOptions *diags);
        ~TextDiagnosticPrinter() override;

        /// Set the diagnostic printer prefix string, which will be printed at the
        /// start of any diagnostics. If empty, no prefix string is used.
        void setPrefix(std::string value) { prefix = std::move(value); }

        void HandleDiagnostic(clang::DiagnosticsEngine::Level level,
                              const clang::Diagnostic &info) override;
    };

} // namespace typescript::tslang

#endif // TYPESCRIPT_TEXTDIAGNOSTICPRINTER_H
