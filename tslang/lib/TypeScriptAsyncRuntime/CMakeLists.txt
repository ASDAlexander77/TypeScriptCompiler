if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GR")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
endif()

add_mlir_library(TypeScriptAsyncRuntime
  STATIC
  AsyncRuntime.cpp

  EXCLUDE_FROM_LIBMLIR

  # For gc.h only - AsyncGCThreads.h registers the pool's workers with the collector when the
  # program is a `gc` one. The collector itself is linked into the executable, not into here.
  LINK_LIBS PRIVATE
  BDWgc::gc
)
