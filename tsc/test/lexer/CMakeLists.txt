message(STATUS ">>> test >>> lexer")

set(LIBS
    antlr4_static
    )

grammar_generator()
set_MSVC_Options()

include_directories(
    ${ANTLR4_INCLUDE_DIR_TypeScriptLexerANTLR} 
    ${ANTLR4_INCLUDE_DIR_TypeScriptParserANTLR} 
    )

add_executable(test-lexer test-lexer.cpp ${ANTLR4_SRC_FILES_TypeScriptLexerANTLR} ${ANTLR4_SRC_FILES_TypeScriptParserANTLR})

# add required runtime library
add_dependencies(test-lexer ${LIBS})

target_link_libraries(test-lexer PRIVATE ${LIBS})

######## enable testing ############
add_test(test-lexer test-lexer)