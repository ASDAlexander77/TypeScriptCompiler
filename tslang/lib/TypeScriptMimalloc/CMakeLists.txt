# The heap a Windows executable built by tslang allocates from: mimalloc (lib/mimalloc, one
# translation unit) behind the process-heap entry points (MimallocHeap.c). exe.cpp links it ahead
# of TypeScriptAsyncRuntime. tslang.exe links the `TypeScriptMimallocCore` half alone, for the JIT.
# The same sources are built for x86 by tslang/runtime/CMakeLists.txt.
if (NOT WIN32)
  return()
endif()

set(MIMALLOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mimalloc")

# mimalloc itself, its symbols prefixed mi_ - nothing else in the process is replaced
add_library(TypeScriptMimallocCore OBJECT "${MIMALLOC_DIR}/src/static.c")
target_include_directories(TypeScriptMimallocCore PUBLIC "${MIMALLOC_DIR}/include")
# not hardened: the guard pages and encoded free lists of MI_SECURE cost what this is for
target_compile_definitions(TypeScriptMimallocCore PRIVATE MI_SECURE=0)
if (MSVC)
  # upstream's own code; its warnings are not ours to fix
  target_compile_options(TypeScriptMimallocCore PRIVATE /W0)
endif()

add_library(TypeScriptMimalloc STATIC
  MimallocHeap.c
  $<TARGET_OBJECTS:TypeScriptMimallocCore>)
target_include_directories(TypeScriptMimalloc PRIVATE "${MIMALLOC_DIR}/include")
# beside TypeScriptAsyncRuntime.lib, where --tslang-lib-path / TSLANG_LIB_PATH point (the generator
# expression keeps a multi-config generator from adding a per-config subfolder)
set_target_properties(TypeScriptMimalloc PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:TypeScriptAsyncRuntime>")
