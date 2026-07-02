cmake_minimum_required(VERSION 3.20)

project(hello_tslang LANGUAGES NONE)

# Make the add_ts_executable() helper available.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(TSLang)

# Compile src/hello.ts into a native "hello" executable using tslang.
add_ts_executable(hello
    "${CMAKE_CURRENT_SOURCE_DIR}/src/hello.ts"
    OUTPUT_NAME hello)

# Convenience target:  cmake --build . --target run
add_custom_target(run
    COMMAND "$<TARGET_PROPERTY:hello,TS_OUTPUT>"
    DEPENDS hello
    USES_TERMINAL
    COMMENT "Running hello")
