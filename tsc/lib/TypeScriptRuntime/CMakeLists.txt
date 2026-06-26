include_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc-8.2.8/include/")

string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
if(MSVC)
  message(STATUS "Using GC Lib: ${PROJECT_SOURCE_DIR}/../3rdParty/gc/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/")

  link_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR")
else()
  message(STATUS "Using GC Lib: ${PROJECT_SOURCE_DIR}/../3rdParty/gc/${CMAKE_BUILD_TYPE_LOWERCASE}/")

  link_directories("${PROJECT_SOURCE_DIR}/../3rdParty/gc/${CMAKE_BUILD_TYPE_LOWERCASE}/")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif()

if(MSVC)
  set(GC_LIB "${GC_LIB} gc")
else()
  set(GC_LIB "${GC_LIB} gc-lib")
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
  ${GC_LIB}
)

if(MSVC)
  # Export the runtime symbols under the names JIT-compiled code expects so the
  # MLIR ExecutionEngine resolves them via the JITDylib path (no init callback,
  # no cross-heap free of the symbol StringMap). See TypeScriptRuntime.def.
  target_link_options(TypeScriptRuntime PRIVATE
    "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/TypeScriptRuntime.def")
endif()
