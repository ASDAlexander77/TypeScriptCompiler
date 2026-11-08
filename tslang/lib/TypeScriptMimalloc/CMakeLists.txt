# The heap a program built by tslang allocates from: mimalloc (lib/mimalloc, one translation unit).
# exe.cpp links it into an executable; tslang links it into itself, for the program the JIT runs.
# The same sources are built for x86 by tslang/runtime/CMakeLists.txt.
#
# Windows: behind the process-heap entry points (MimallocHeap.c), ahead of TypeScriptAsyncRuntime.
# tslang.exe links the `TypeScriptMimallocCore` half alone and serves the JIT from it (jit.cpp).
#
# Linux: as the process's `malloc` (MI_MALLOC_OVERRIDE). An executable that defines malloc, free and
# the rest is where every shared object of the process finds them - the default library's, one
# tslang built, libc's own - so one copy in the executable is the heap of the whole process.
if (WIN32)
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
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(MIMALLOC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mimalloc")

  add_library(TypeScriptMimalloc STATIC "${MIMALLOC_DIR}/src/static.c")
  target_include_directories(TypeScriptMimalloc PUBLIC "${MIMALLOC_DIR}/include")
  # malloc, free, calloc, realloc, posix_memalign, aligned_alloc, malloc_usable_size... defined by it
  target_compile_definitions(TypeScriptMimalloc PRIVATE MI_SECURE=0 MI_MALLOC_OVERRIDE)
  # an executable tslang links may be position independent (-pie)
  set_target_properties(TypeScriptMimalloc PROPERTIES POSITION_INDEPENDENT_CODE ON)
  # upstream's own code; its warnings are not ours to fix
  target_compile_options(TypeScriptMimalloc PRIVATE -w)
else()
  return()
endif()

# beside TypeScriptAsyncRuntime, where --tslang-lib-path / TSLANG_LIB_PATH point (the generator
# expression keeps a multi-config generator from adding a per-config subfolder)
set_target_properties(TypeScriptMimalloc PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:TypeScriptAsyncRuntime>")
