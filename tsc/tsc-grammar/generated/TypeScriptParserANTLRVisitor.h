
// Generated from c:\dev\TypeScriptCompiler\tsc\tsc-grammar\TypeScriptParserANTLR.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "TypeScriptParserANTLR.h"


namespace typescript {

/**
 * This class defines an abstract visitor for a parse tree
 * produced by TypeScriptParserANTLR.
 */
class  TypeScriptParserANTLRVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by TypeScriptParserANTLR.
   */
    virtual antlrcpp::Any visitMain(TypeScriptParserANTLR::MainContext *context) = 0;


};

}  // namespace typescript
