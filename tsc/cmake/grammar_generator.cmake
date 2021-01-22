macro(grammar_generator)

# generate lexer
antlr4_generate( 
   TypeScriptLexerANTLR
   "${CMAKE_SOURCE_DIR}/tsc-grammar/TypeScriptLexerANTLR.g4"
   LEXER
   FALSE
   FALSE
   "typescript"
   )
 
# generate parser
antlr4_generate( 
   TypeScriptParserANTLR
   "${CMAKE_SOURCE_DIR}/tsc-grammar/TypeScriptParserANTLR.g4"
   PARSER
   FALSE
   TRUE
   "typescript"
   "${ANTLR4_TOKEN_FILES_TypeScriptLexerANTLR}"
   "${ANTLR4_TOKEN_DIRECTORY_TypeScriptLexerANTLR}"
   )

list(FILTER ANTLR4_SRC_FILES_TypeScriptLexerANTLR INCLUDE REGEX ".*\.cpp$")
list(FILTER ANTLR4_SRC_FILES_TypeScriptParserANTLR INCLUDE REGEX ".*\.cpp$")

endmacro()

macro(set_MSVC_Options)

if(MSVC)
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

endmacro()
