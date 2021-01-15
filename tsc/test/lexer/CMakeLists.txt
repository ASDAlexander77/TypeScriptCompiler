include_directories(
    ${ANTLR_OUTPUT_DIRS}
    )

set(LIBS
    antlr4_static
    )

add_executable(test-lexer test-lexer.cpp ${ANTLR_CXX_OUTPUT_DIRS})
target_link_libraries(test-lexer PRIVATE ${LIBS})

######## enable testing ############

enable_testing()
add_test(test-lexer test-lexer)