set_MSVC_Options()

add_executable(tsc-new-parser run.cpp scanner.cpp)

target_link_libraries(tsc-new-parser PRIVATE ${LIBS})

