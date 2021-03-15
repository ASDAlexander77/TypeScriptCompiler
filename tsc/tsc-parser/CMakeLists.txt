add_executable(tsc-parser run.cpp scanner.cpp)

target_link_libraries(tsc-parser PRIVATE ${LIBS})

