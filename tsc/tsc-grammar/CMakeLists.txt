# Call macro to add lexer and grammar to your build dependencies.
antlr_target(TypeScriptLexerANTLR TypeScriptLexerANTLR.g4 LEXER
             PACKAGE typescript)
antlr_target(TypeScriptParserANTLR TypeScriptParserANTLR.g4 PARSER
             PACKAGE typescript
             DEPENDS_ANTLR TypeScriptLexerANTLR
             COMPILE_FLAGS -lib ${ANTLR_TypeScriptLexerANTLR_OUTPUT_DIR})

# include generated files in project environment
include_directories(${ANTLR_TypeScriptLexerANTLR_OUTPUT_DIR})
include_directories(${ANTLR_TypeScriptParserANTLR_OUTPUT_DIR})
