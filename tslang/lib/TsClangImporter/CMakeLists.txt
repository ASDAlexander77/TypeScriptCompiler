# C header -> tslang bindings, as a library: tsbindgen is a thin command line over it, and v2's C++
# wrapper generation is meant to reuse it. It links clang's frontend; tslang itself does not.
set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_llvm_library(TsClangImporter
  BindingModel.cpp
  TypeMapper.cpp
  HeaderParser.cpp
  BindingPrinter.cpp
  # tslang's @cppname mangler (LLVM-only, no MLIR), to check that the declaration a binding carries
  # mangles to clang's symbol; tslang and TsClangImporter are never linked together
  ../TypeScript/CppMangler.cpp

  LINK_LIBS
  clangTooling
  clangFrontend
  clangAST
  clangLex
  clangBasic
  )
