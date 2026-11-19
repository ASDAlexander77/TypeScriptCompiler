add_mlir_library(TypeScriptAsyncRuntime
  STATIC
  AsyncRuntime.cpp
  AsyncGCThreads.cpp
  ProcessHeap.cpp
  DynamicRuntime.cpp
  UncaughtException.cpp
  StdioDefinitions.cpp
  # f16 conversions, which MSVC's runtime does not have (compiler-rt/README.md)
  HalfTruncSingle.c
  HalfTruncDouble.c
  HalfExtendSingle.c
  HalfExtendDouble.c

  EXCLUDE_FROM_LIBMLIR

  # For gc.h only - AsyncGCThreads.cpp registers the pool's workers with the collector when the
  # program is a `gc` one. The collector itself is linked into the executable, not into here.
  LINK_LIBS PRIVATE
  BDWgc::gc
)
