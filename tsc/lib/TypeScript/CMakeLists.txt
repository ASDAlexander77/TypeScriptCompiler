# generate files first
# Call macro to add lexer and grammar to your build dependencies.
antlr_target(TypeScriptLexerANTLR "${CMAKE_SOURCE_DIR}/tsc-grammar/TypeScriptLexerANTLR.g4" LEXER
             PACKAGE typescript)
antlr_target(TypeScriptParserANTLR "${CMAKE_SOURCE_DIR}/tsc-grammar/TypeScriptParserANTLR.g4" PARSER
             PACKAGE typescript
             DEPENDS_ANTLR TypeScriptLexerANTLR
             COMPILE_FLAGS -lib ${ANTLR_TypeScriptLexerANTLR_OUTPUT_DIR})

message(STATUS "Using TypeScriptLexerANTLR in: ${ANTLR_TypeScriptLexerANTLR_OUTPUT_DIR}")
message(STATUS "Using TypeScriptParserANTLR in: ${ANTLR_TypeScriptParserANTLR_OUTPUT_DIR}")

# include generated files in project environment
include_directories(
    ${ANTLR_TypeScriptLexerANTLR_OUTPUT_DIR}
    ${ANTLR_TypeScriptParserANTLR_OUTPUT_DIR}
    )

set (ANTLR_CXX_OUTPUT_DIRS 
    ${ANTLR_TypeScriptLexerANTLR_CXX_OUTPUTS} 
    ${ANTLR_TypeScriptParserANTLR_CXX_OUTPUTS}
    CACHE INTERNAL "")

set (ANTLR_OUTPUT_DIRS 
    ${ANTLR_TypeScriptLexerANTLR_OUTPUT_DIR} 
    ${ANTLR_TypeScriptParserANTLR_OUTPUT_DIR}
    CACHE INTERNAL "")

message(STATUS "set ANTLR_OUTPUT_DIRS as: ${ANTLR_OUTPUT_DIRS}")    

add_mlir_dialect_library(MLIRTypeScript
    TypeScriptDialect.cpp
    TypeScriptOps.cpp
    AST.cpp
    MLIRGen.cpp
    LowerToAffineLoops.cpp
    LowerToLLVM.cpp
    ${ANTLR_TypeScriptLexerANTLR_CXX_OUTPUTS}
    ${ANTLR_TypeScriptParserANTLR_CXX_OUTPUTS}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TypeScript
    ${ANTLR_TypeScriptLexerANTLR_OUTPUT_DIR}
    ${ANTLR_TypeScriptParserANTLR_OUTPUT_DIR}

    DEPENDS
    MLIRTypeScriptOpsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
