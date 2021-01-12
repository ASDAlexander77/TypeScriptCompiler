
// Generated from c:\dev\TypeScriptCompiler\tsc\tsc-grammar\TypeScriptParserANTLR.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "TypeScriptParserANTLR.h"


namespace typescript {

/**
 * This interface defines an abstract listener for a parse tree produced by TypeScriptParserANTLR.
 */
class  TypeScriptParserANTLRListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterMain(TypeScriptParserANTLR::MainContext *ctx) = 0;
  virtual void exitMain(TypeScriptParserANTLR::MainContext *ctx) = 0;


};

}  // namespace typescript
