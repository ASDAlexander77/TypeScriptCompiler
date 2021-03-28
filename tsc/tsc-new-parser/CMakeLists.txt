set_MSVC_Options()

add_executable(tsc-new-scanner scanner_run.cpp scanner.cpp)

target_link_libraries(tsc-new-scanner PRIVATE ${LIBS})

add_executable(tsc-new-parser parser_run.cpp parser.cpp scanner.cpp)

target_link_libraries(tsc-new-parser PRIVATE ${LIBS})

