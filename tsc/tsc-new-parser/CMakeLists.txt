set_MSVC_Options()

add_library(tsc-new-parser-lib parser.cpp node_factory.cpp parenthesizer_rules.cpp scanner.cpp)

add_executable(tsc-new-scanner scanner_run.cpp scanner.cpp)

target_link_libraries(tsc-new-scanner PRIVATE ${LIBS})

add_executable(tsc-new-parser parser_run.cpp parser.cpp node_factory.cpp parenthesizer_rules.cpp scanner.cpp)

target_link_libraries(tsc-new-parser PRIVATE ${LIBS})

