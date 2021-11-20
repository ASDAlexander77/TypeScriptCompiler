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

set (TSCPATH "C:\\dev\\TypeScriptCompiler\\__build\\tsc\\bin")
set (LLVMPATH "C:\\dev\\TypeScriptCompiler\\3rdParty\\llvm\\release\\bin")
set (GCLIBPATH "C:\\dev\\TypeScriptCompiler\\3rdParty\\gc\\Release")

find_program(TSC_APP tsc.exe HINTS "${TSCPATH}" DOC "path to tsc")
find_program(LLC_APP llc.exe HINTS "${LLVMPATH}" DOC "path to llc")

if (NOT TSC_APP)
	message(FATAL_ERROR "Can't find tsc.exe")
endif()

if (NOT LLC_APP)
	message(FATAL_ERROR "Can't find llc.exe")
endif()

add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/main.${CMAKE_CXX_OUTPUT_EXTENSION}
  COMMAND "${TSC_APP}" --emit=llvm "${CMAKE_CURRENT_SOURCE_DIR}/src/main.ts" 2>"${CMAKE_CURRENT_BINARY_DIR}/main.ll"
  COMMAND "${LLC_APP}" --filetype=obj "-o=${CMAKE_CURRENT_BINARY_DIR}/main.${CMAKE_CXX_OUTPUT_EXTENSION}" "${CMAKE_CURRENT_BINARY_DIR}/main.ll"
  COMMENT Added custom file
)

add_executable(
    ${PROJECT_NAME} 
    dummy.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/main.${CMAKE_CXX_OUTPUT_EXTENSION}
)

target_link_directories(
    ${PROJECT_NAME} 
    PUBLIC
    ${GCLIBPATH}
)

target_link_libraries(
    ${PROJECT_NAME} 
    gcmt-lib
)



