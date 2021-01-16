message(STATUS ">>> lib >>> TypeScript")

# generate files first
# Call macro to add lexer and grammar to your build dependencies.
antlr4_generate(TypeScriptLexerANTLR 
             "${CMAKE_SOURCE_DIR}/tsc-grammar/TypeScriptLexerANTLR.g4"
             LEXER
	     FALSE
             TRUE
             "typescript")
antlr4_generate(TypeScriptParserANTLR 
             "${CMAKE_SOURCE_DIR}/tsc-grammar/TypeScriptParserANTLR.g4" 
             PARSER
	     FALSE
             TRUE
             "typescript"
   	     "${ANTLR4_TOKEN_FILES_TypeScriptLexerANTLR}"
	     "${ANTLR4_TOKEN_DIRECTORY_TypeScriptLexerANTLR}")

message(STATUS "Using TypeScriptLexerANTLR in: ${ANTLR4_INCLUDE_DIR_TypeScriptLexer}")
message(STATUS "Using TypeScriptParserANTLR in: ${ANTLR4_INCLUDE_DIR_TypeScriptParser}")

# include generated files in project environment
include_directories(
    ${ANTLR4_INCLUDE_DIR_TypeScriptLexer}
    ${ANTLR4_INCLUDE_DIR_TypeScriptParser}
    )

set (ANTLR_CXX_OUTPUT_DIRS 
    ${ANTLR4_SRC_FILES_TypeScriptLexer} 
    ${ANTLR4_SRC_FILES_TypeScriptParser}
    CACHE INTERNAL "")

set (ANTLR_OUTPUT_DIRS 
    ${ANTLR4_INCLUDE_DIR_TypeScriptLexer} 
    ${ANTLR4_INCLUDE_DIR_TypeScriptParser}
    CACHE INTERNAL "")

message(STATUS "set ANTLR_OUTPUT_DIRS as: ${ANTLR_OUTPUT_DIRS}")    

add_mlir_dialect_library(MLIRTypeScript
    TypeScriptDialect.cpp
    TypeScriptOps.cpp
    AST.cpp
    MLIRGen.cpp
    LowerToAffineLoops.cpp
    LowerToLLVM.cpp
    ${ANTLR4_SRC_FILES_TypeScriptLexer}
    ${ANTLR4_SRC_FILES_TypeScriptParser}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TypeScript
    ${ANTLR4_INCLUDE_DIR_TypeScriptLexer}
    ${ANTLR4_INCLUDE_DIR_TypeScriptParser}

    DEPENDS
    MLIRTypeScriptOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
