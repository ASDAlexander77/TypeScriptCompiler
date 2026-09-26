set_Options()

set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_llvm_executable(tsbindgen
  tsbindgen.cpp
  )

llvm_update_compile_flags(tsbindgen)
target_link_libraries(tsbindgen PRIVATE TsClangImporter)

# The last place tsbindgen looks for clang's resource directory (stddef.h, stdint.h, ...): the
# LLVM it was built with. A release package puts one next to tsbindgen instead.
target_compile_definitions(tsbindgen PRIVATE
  TSBINDGEN_CONFIGURED_RESOURCE_DIR="${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")

# the same version tslang reports: the release tag, or the commit a dev build was made from
if(TSLANG_PACKAGE_VERSION)
  set(TSBINDGEN_VERSION "${TSLANG_PACKAGE_VERSION}")
else()
  find_package(Git QUIET)
  if(GIT_EXECUTABLE)
    execute_process(
      COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE TSBINDGEN_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
  endif()
endif()

if(TSBINDGEN_VERSION)
  target_compile_definitions(tsbindgen PRIVATE TSBINDGEN_VERSION="${TSBINDGEN_VERSION}")
endif()
