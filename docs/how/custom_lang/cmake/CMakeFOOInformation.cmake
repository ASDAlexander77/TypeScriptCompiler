# The actual compile command.
# Placeholders CMake substitutes:
#   <CMAKE_FOO_COMPILER>  the binary
#   <FLAGS>               per-target flags
#   <SOURCE>              input .foo
#   <OBJECT>              output .obj
#   <DEFINES> <INCLUDES>  optional
if(NOT CMAKE_FOO_COMPILE_OBJECT)
    set(CMAKE_FOO_COMPILE_OBJECT
        "<CMAKE_FOO_COMPILER> <FLAGS> --output <OBJECT> <SOURCE>")
endif()

# How CMake links FOO objects into an executable/library.
# Reuse the C++ linker so linking with .cpp works out of the box.
if(NOT CMAKE_FOO_LINK_EXECUTABLE)
    set(CMAKE_FOO_LINK_EXECUTABLE
        "<CMAKE_CXX_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

set(CMAKE_FOO_INFORMATION_LOADED 1)
