message(STATUS ">>> lib >>> TypeScript")

grammar_generator()

set (GRAMMAR_GENERATED_DIRS 
    ${ANTLR4_INCLUDE_DIR_TypeScriptLexerANTLR} 
    ${ANTLR4_INCLUDE_DIR_TypeScriptParserANTLR} 
    CACHE INTERNAL "")

message(STATUS "set GRAMMAR_GENERATED_DIRS as: ${GRAMMAR_GENERATED_DIRS}")    

include_directories(
    ${GRAMMAR_GENERATED_DIRS}
    )

add_mlir_dialect_library(MLIRTypeScript
    TypeScriptDialect.cpp
    TypeScriptOps.cpp
    MLIRGen.cpp
    LowerToAffineLoops.cpp
    LowerToLLVM.cpp
    LowerToLLVMLogic.cpp
    ${ANTLR4_SRC_FILES_TypeScriptLexerANTLR}
    ${ANTLR4_SRC_FILES_TypeScriptParserANTLR}

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/TypeScript
    ${GRAMMAR_GENERATED_DIRS}

    DEPENDS
    MLIRTypeScriptOpsIncGen

    LINK_LIBS PUBLIC
    antlr4_static
    MLIRIR
    )
