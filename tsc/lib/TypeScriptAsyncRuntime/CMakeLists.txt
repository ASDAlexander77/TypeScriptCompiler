if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif()

add_mlir_library(TypeScriptAsyncRuntime
  STATIC
  AsyncRuntime.cpp

  EXCLUDE_FROM_LIBMLIR
)
