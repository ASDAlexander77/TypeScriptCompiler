#include "TypeScript/TypeScriptLang/TextDiagnosticPrinter.h"
#include "TypeScript/TypeScriptLang/TextDiagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace typescript::tslang;

TextDiagnosticPrinter::TextDiagnosticPrinter(raw_ostream &diagOs,
                                             clang::DiagnosticOptions *diags)
    : os(diagOs), diagOpts(diags) {}

TextDiagnosticPrinter::~TextDiagnosticPrinter() {}

void TextDiagnosticPrinter::HandleDiagnostic(
    clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info)
{
    // Default implementation (Warnings/errors count).
    DiagnosticConsumer::HandleDiagnostic(level, info);

    // Render the diagnostic message into a temporary buffer eagerly. We'll use
    // this later as we print out the diagnostic to the terminal.
    llvm::SmallString<100> outStr;
    info.FormatDiagnostic(outStr);

    llvm::raw_svector_ostream diagMessageStream(outStr);

    if (!prefix.empty())
        os << prefix << ": ";

    // We only emit diagnostics in contexts that lack valid source locations.
    assert(!info.getLocation().isValid() &&
           "Diagnostics with valid source location are not supported");

    typescript::tslang::TextDiagnostic::printDiagnosticLevel(os, level,
                                                            diagOpts->ShowColors);
    typescript::tslang::TextDiagnostic::printDiagnosticMessage(
        os,
        /*IsSupplemental=*/level == clang::DiagnosticsEngine::Note,
        diagMessageStream.str(), diagOpts->ShowColors);

    os.flush();
}
