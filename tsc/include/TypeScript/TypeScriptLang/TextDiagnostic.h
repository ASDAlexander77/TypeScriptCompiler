#ifndef TYPESCRIPT_TEXTDIAGNOSTIC_H
#define TYPESCRIPT_TEXTDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace typescript::tslang {

class TextDiagnostic {
public:
  TextDiagnostic();

  ~TextDiagnostic();

  static void printDiagnosticLevel(llvm::raw_ostream &os,
                                   clang::DiagnosticsEngine::Level level,
                                   bool showColors);

  static void printDiagnosticMessage(llvm::raw_ostream &os, bool isSupplemental,
                                     llvm::StringRef message, bool showColors);
};

} // namespace typescript::tslang

#endif
