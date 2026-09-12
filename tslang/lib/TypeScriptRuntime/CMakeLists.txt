# Include dir + lib location come from the BDWgc::gc imported target (find_package
# in the top-level CMakeLists), so no hardcoded include_directories/link_directories.
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif()

# On Windows the runtime takes Boehm from gc.dll rather than linking it in: under the JIT this
# DLL shares the process with the default library's DLL and with user shared libraries, and a
# second static collector frees what the others still hold. See the top-level CMakeLists and
# docs/single-gc-collector-design.md. Linux is unchanged until it has been measured.
if (WIN32)
  set(TSLANG_RUNTIME_GC tslang_gc_shared)
else()
  set(TSLANG_RUNTIME_GC BDWgc::gc)
endif()

add_mlir_library(TypeScriptRuntime
  SHARED
  TypeScriptGC.cpp
  gc.cpp
  MemRuntime.cpp
  AsyncRuntime.cpp
  DynamicRuntime.cpp
  mlir_init.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PRIVATE
  ${TSLANG_RUNTIME_GC}
)

if (WIN32)
  # gc.dll beside TypeScriptRuntime.dll (and tslang.exe), where the loader looks for it.
  add_custom_command(TARGET TypeScriptRuntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${TSLANG_GC_SHARED_PREFIX}/bin/gc.dll" "$<TARGET_FILE_DIR:TypeScriptRuntime>")
endif()

if(MSVC)
  # Export the runtime symbols under the names JIT-compiled code expects so the
  # MLIR ExecutionEngine resolves them via the JITDylib path (no init callback,
  # no cross-heap free of the symbol StringMap). See TypeScriptRuntime.def.
  target_link_options(TypeScriptRuntime PRIVATE
    "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/TypeScriptRuntime.def")
endif()
