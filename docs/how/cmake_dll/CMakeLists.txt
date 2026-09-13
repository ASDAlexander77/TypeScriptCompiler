cmake_minimum_required(VERSION 3.20)

# Make CMake find ts-language modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(anax_enhancer CXX)

# Enable TS-language
enable_language(TSLANG)

# Include folders
include_directories(${CMAKE_TSLANG_DIR}/defaultlib)

# The compiled default lib is split into per-build subfolders (debug/release) and then per
# memory model (gc/rc/none); pick the pair matching this build, so that the CRT, the allocator
# and the default-lib binaries all agree. A library built for one model cannot be linked into
# a program built for another.
if (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(TSLANG_DEFAULTLIB_BUILD "release")
else()
	set(TSLANG_DEFAULTLIB_BUILD "debug")
endif()

set(TSLANG_MEMORY_MODEL "gc" CACHE STRING "Memory model of compiled code: gc, rc or none")
set_property(CACHE TSLANG_MEMORY_MODEL PROPERTY STRINGS gc rc none)

# Lib folders
link_directories(${CMAKE_TSLANG_DIR} ${CMAKE_TSLANG_DIR}/defaultlib/lib/${TSLANG_DEFAULTLIB_BUILD}/${TSLANG_MEMORY_MODEL})

# set options
if (CMAKE_BUILD_TYPE STREQUAL "Release")
	set(CMAKE_TSLANG_FLAGS "--opt --opt_level=3") # global
else()
	set(CMAKE_TSLANG_FLAGS "--di --opt_level=0") # global
endif()

# The same variable that picked the link directory has to reach the compiler as well, or the
# program is compiled for one model and linked against another model's default lib.
set(CMAKE_TSLANG_FLAGS "${CMAKE_TSLANG_FLAGS} -mm=${TSLANG_MEMORY_MODEL}") # global

if(WIN32)
else()
    set(CMAKE_TSLANG_FLAGS "${CMAKE_TSLANG_FLAGS} -relocation-model=pic") # global
endif()

# .ts files compile with TSLANG command; .cpp with the C++ compiler.
add_library("opengl32" SHARED
    dummy.cpp
    wrapper.ts
)

# required libs
set(TSLANG_LINK_LIBS "TypeScriptDefaultLib" "TypeScriptAsyncRuntime" "LLVMSupport")

# Boehm is only referenced by the gc default lib; the rc and none builds allocate through the
# CRT and must not drag a collector in.
if (TSLANG_MEMORY_MODEL STREQUAL "gc")
    list(APPEND TSLANG_LINK_LIBS "gcdll/gc")
endif()

# ntdll provides RtlGetLastNtStatus (pulled in by LLVMSupport) on Windows
if(WIN32)
    list(APPEND TSLANG_LINK_LIBS "ntdll")
else()
    list(APPEND TSLANG_LINK_LIBS "LLVMDemangle" "stdc++" "m" "pthread" "tinfo" "dl" "rt")
endif()

target_link_libraries("opengl32" ${TSLANG_LINK_LIBS})

# The GC-model default lib depends on gc.dll at runtime; copy it next to the
# built binary so the program can find it without relying on PATH.
if (TSLANG_MEMORY_MODEL STREQUAL "gc")
    add_custom_command(TARGET "opengl32" POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_TSLANG_DIR}/gcdll/gc.dll"
            "$<TARGET_FILE_DIR:opengl32>"
    )
endif()
