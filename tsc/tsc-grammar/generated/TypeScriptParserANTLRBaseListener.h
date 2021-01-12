
// Generated from c:\dev\TypeScriptCompiler\tsc\tsc-grammar\TypeScriptParserANTLR.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "TypeScriptParserANTLRListener.h"


namespace typescript {

/**
 * This class provides an empty implementation of TypeScriptParserANTLRListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  TypeScriptParserANTLRBaseListener : public TypeScriptParserANTLRListener {
public:

  virtual void enterMain(TypeScriptParserANTLR::MainContext * /*ctx*/) override { }
  virtual void exitMain(TypeScriptParserANTLR::MainContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

}  // namespace typescript
