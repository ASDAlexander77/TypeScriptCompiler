# Locate your custom compiler
find_program(CMAKE_TSLANG_COMPILER
    NAMES tslang tslang.exe
    HINTS "${CMAKE_SOURCE_DIR}/tools"
    DOC "TSLANG compiler")

mark_as_advanced(CMAKE_TSLANG_COMPILER)

# Which source extensions belong to TSLANG, and the object suffix
set(CMAKE_TSLANG_SOURCE_FILE_EXTENSIONS ts)
if (NOT WIN32)
	set(CMAKE_TSLANG_OUTPUT_EXTENSION .o)
else()
	set(CMAKE_TSLANG_OUTPUT_EXTENSION .obj)   # .o on Linux
endif()
set(CMAKE_TSLANG_COMPILER_ENV_VAR "TSLANG")

# Emit the compiler-id config file CMake expects
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/CMakeTSLANGCompiler.cmake.in
    ${CMAKE_PLATFORM_INFO_DIR}/CMakeTSLANGCompiler.cmake @ONLY)
