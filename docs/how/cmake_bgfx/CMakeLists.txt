cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(cmake_bgfx CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TSLANG_ROOT "" CACHE PATH "TypeScriptCompiler __build folder (contains tslang/, llvm/, gc/)")

include(LocateTSLang)
locate_tslang_compiler()
set(CMAKE_TSLANG_COMPILER "${CMAKE_TSLANG_COMPILER}" CACHE FILEPATH "TSLANG compiler" FORCE)

include(FetchContent)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
    GIT_SHALLOW TRUE)

set(BGFX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BGFX_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(BGFX_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    bgfx
    GIT_REPOSITORY https://github.com/bkaradzic/bgfx.cmake.git
    GIT_TAG v1.150.9365-558
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(glfw bgfx)

enable_language(TSLANG)

setup_tslang_link_paths()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_TSLANG_FLAGS "--opt --opt_level=3")
else()
    set(CMAKE_TSLANG_FLAGS "--di --opt_level=0")
endif()

if(WIN32)
else()
    set(CMAKE_TSLANG_FLAGS "${CMAKE_TSLANG_FLAGS} -relocation-model=pic")
endif()

add_executable(${PROJECT_NAME}
    src/main.ts
    src/application.ts
    src/appwindow.ts
    native/bgfx_bridge.cpp
    native/main_entry.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE native)

set(TSLANG_LINK_LIBS
    TypeScriptDefaultLib
    TypeScriptAsyncRuntime
    gc
    LLVMSupport)

if(WIN32)
    list(APPEND TSLANG_LINK_LIBS ntdll)
else()
    find_library(TSLANG_TINFO_LIB NAMES tinfo)
    list(APPEND TSLANG_LINK_LIBS LLVMDemangle stdc++ m pthread dl rt)
    if(TSLANG_TINFO_LIB)
        list(APPEND TSLANG_LINK_LIBS ${TSLANG_TINFO_LIB})
    else()
        list(APPEND TSLANG_LINK_LIBS tinfo)
    endif()
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    glfw
    bgfx
    bx
    bimg
    ${TSLANG_LINK_LIBS})

if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE X11::X11)
endif()

add_custom_target(run
    COMMAND "$<TARGET_FILE:${PROJECT_NAME}>"
    DEPENDS ${PROJECT_NAME}
    USES_TERMINAL
    COMMENT "Running ${PROJECT_NAME}")
