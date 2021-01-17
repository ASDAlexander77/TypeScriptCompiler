message(STATUS ">>> test >>> lexer")

set(LIBS
    antlr4_static
    )

grammar_generator()

if(MSVC)
    message(STATUS "Using static MSVC CRT")
    # http://stackoverflow.com/a/32128977/486990
    add_compile_options(
        "$<$<CONFIG:Debug>:/MTd>"
        "$<$<CONFIG:RelWithDebInfo>:/MT>"
        "$<$<CONFIG:Release>:/MT>"
        "$<$<CONFIG:MinSizeRel>:/MT>"
    )

    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
endif()

include_directories(
    ${ANTLR4_INCLUDE_DIR_TypeScriptLexerANTLR} 
    ${ANTLR4_INCLUDE_DIR_TypeScriptParserANTLR} 
    )

add_executable(test-lexer test-lexer.cpp ${ANTLR4_SRC_FILES_TypeScriptLexerANTLR} ${ANTLR4_SRC_FILES_TypeScriptParserANTLR})

# add required runtime library
add_dependencies(test-lexer antlr4_static)

target_link_libraries(test-lexer PRIVATE ${LIBS})

######## enable testing ############

enable_testing()
add_test(test-lexer test-lexer)