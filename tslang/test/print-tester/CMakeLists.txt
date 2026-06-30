set_Options()

include_directories(
    ${PROJECT_SOURCE_DIR}/ts-new-parser
    )

link_directories(
    ${PROJECT_SOURCE_DIR}/ts-new-parser
    )

set (LIBS "${LIBS}" ts-new-parser-lib)

add_executable(printer-tester printer-tester.cpp)
target_link_libraries(printer-tester PRIVATE ${LIBS})
