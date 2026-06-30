set_Options()

include_directories(
    ${PROJECT_SOURCE_DIR}/tsc-new-parser
    )

link_directories(
    ${PROJECT_SOURCE_DIR}/tsc-new-parser
    )

set (LIBS "${LIBS}" tsc-new-parser-lib)

add_executable(printer-tester printer-tester.cpp)
target_link_libraries(printer-tester PRIVATE ${LIBS})
