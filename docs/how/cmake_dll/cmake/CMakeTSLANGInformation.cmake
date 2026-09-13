# The actual compile command.
# Placeholders CMake substitutes:
#   <CMAKE_TSLANG_COMPILER>  the binary
#   <FLAGS>               per-target flags
#   <SOURCE>              input .ts
#   <OBJECT>              output .obj
#   <DEFINES> <INCLUDES>  optional
if(NOT CMAKE_TSLANG_COMPILE_OBJECT)
    set(CMAKE_TSLANG_COMPILE_OBJECT
        "<CMAKE_TSLANG_COMPILER> <FLAGS> --default-lib-path=${CMAKE_TSLANG_DIR} --emit=obj -o=<OBJECT> <SOURCE>")
endif()

# How CMake links TSLANG objects into an executable/library.
# Reuse the C++ linker so linking with .cpp works out of the box.
if(NOT CMAKE_TSLANG_LINK_EXECUTABLE)
    set(CMAKE_TSLANG_LINK_EXECUTABLE
        "<CMAKE_CXX_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

set(CMAKE_TSLANG_INFORMATION_LOADED 1)
