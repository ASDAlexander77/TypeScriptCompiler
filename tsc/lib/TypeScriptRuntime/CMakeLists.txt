# Include dir + lib location come from the BDWgc::gc imported target (find_package
# in the top-level CMakeLists), so no hardcoded include_directories/link_directories.
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
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
  BDWgc::gc
)

if(MSVC)
  # Export the runtime symbols under the names JIT-compiled code expects so the
  # MLIR ExecutionEngine resolves them via the JITDylib path (no init callback,
  # no cross-heap free of the symbol StringMap). See TypeScriptRuntime.def.
  target_link_options(TypeScriptRuntime PRIVATE
    "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/TypeScriptRuntime.def")
endif()
