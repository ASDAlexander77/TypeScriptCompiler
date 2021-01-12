
// Generated from c:\dev\TypeScriptCompiler\tsc\tsc-grammar\TypeScriptParserANTLR.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "TypeScriptParserANTLRVisitor.h"


namespace typescript {

/**
 * This class provides an empty implementation of TypeScriptParserANTLRVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  TypeScriptParserANTLRBaseVisitor : public TypeScriptParserANTLRVisitor {
public:

  virtual antlrcpp::Any visitMain(TypeScriptParserANTLR::MainContext *ctx) override {
    return visitChildren(ctx);
  }


};

}  // namespace typescript
