project(TypeScriptApp CXX)
cmake_minimum_required(VERSION 3.10.2)
set(PROJECT_VERSION 0.0.0.dev0)

# link_directories interprate relative paths with respect to CMAKE_CURRENT_SOURCE_DIR
cmake_policy(SET CMP0015 NEW)

# let cmake autolink dependencies on windows
# it's specified globally, cause qt libraries requires that on windows and they are also found globally
cmake_policy(SET CMP0020 NEW)

# 3.1 and above
cmake_policy(SET CMP0054 NEW)

# SETUP FOR CPP FILES
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS Off)
set(CMAKE_EXPORT_COMPILE_COMMANDS Off)

# add vulkan
include_directories($ENV{VULKAN_SDK}/Include)
link_directories($ENV{VULKAN_SDK}/Lib)
set (EXTRA_LIBS ${EXTRA_LIBS} "vulkan-1")

# add std lib
if (NOT WIN32)
	set (EXTRA_LIBS ${EXTRA_LIBS} "stdc++")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_definitions(-DVK_USE_PLATFORM_WIN32_KHR -DVC_EXTRALEAN -DVK_PROTOTYPES -D_USE_MATH_DEFINES)
endif()

include("typescript.cmake")

add_tsc_files("src")

add_executable(
    ${PROJECT_NAME} 
    dummy.cpp
    ${TS_SRC}
    native/window.win32.cpp
    native/vulkan.win32.cpp
)

target_link_directories(
    ${PROJECT_NAME} 
    PUBLIC
    ${GCLIBPATH}
)

set (LIBS "gcmt-lib")

target_link_libraries(
    ${PROJECT_NAME} 
    ${LIBS}
    ${EXTRA_LIBS}
)



