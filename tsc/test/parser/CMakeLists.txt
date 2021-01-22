message(STATUS ">>> test >>> parser")

set(LIBS
    antlr4_static
    )

grammar_generator()
set_MSVC_Options()

include_directories(
    ${ANTLR4_INCLUDE_DIR_TypeScriptLexerANTLR} 
    ${ANTLR4_INCLUDE_DIR_TypeScriptParserANTLR} 
    )

add_executable(test-parser test-parser.cpp ${ANTLR4_SRC_FILES_TypeScriptLexerANTLR} ${ANTLR4_SRC_FILES_TypeScriptParserANTLR})

# add required runtime library
add_dependencies(test-parser ${LIBS})

target_link_libraries(test-parser PRIVATE ${LIBS})

######## enable testing ############
add_test(test-parser test-parser)