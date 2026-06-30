set_Options_With_FS()

add_library(ts-new-parser-lib parser.cpp node_factory.cpp parenthesizer_rules.cpp scanner.cpp)

add_executable(tslang-new-scanner scanner_run.cpp scanner.cpp)

target_link_libraries(tslang-new-scanner PRIVATE ${LIBS})

add_executable(ts-new-parser parser_run.cpp parser.cpp node_factory.cpp parenthesizer_rules.cpp scanner.cpp)

target_link_libraries(ts-new-parser PRIVATE ${LIBS})

