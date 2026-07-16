# bx propagates -msse4.2 via PUBLIC compile options on GCC/Clang. tslang does
# not accept those flags, so filter them through a small portable wrapper
# invoked via `cmake -P` (works the same under Ninja on Windows and Unix,
# unlike a #!/usr/bin/env bash script which Ninja cannot exec on Windows).
get_filename_component(_TSLANG_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" DIRECTORY)
set(_TSLANG_COMPILE_TEMPLATE
    "<CMAKE_TSLANG_COMPILER> <FLAGS> --default-lib-path=${CMAKE_TSLANG_DIR} --emit=obj --export=none -o=<OBJECT> <SOURCE>")
if(NOT CMAKE_TSLANG_COMPILE_OBJECT)
    set(CMAKE_TSLANG_COMPILE_OBJECT
        "\"${CMAKE_COMMAND}\" -P \"${_TSLANG_CMAKE_DIR}/tslang_filter_flags.cmake\" -- ${_TSLANG_COMPILE_TEMPLATE}")
endif()

if(NOT CMAKE_TSLANG_LINK_EXECUTABLE)
    set(CMAKE_TSLANG_LINK_EXECUTABLE
        "<CMAKE_CXX_COMPILER> <FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
endif()

set(CMAKE_TSLANG_INFORMATION_LOADED 1)
