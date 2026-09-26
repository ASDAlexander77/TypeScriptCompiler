# C header -> tslang bindings, as a library: tsbindgen is a thin command line over it, and v2's C++
# wrapper generation is meant to reuse it. It links clang's frontend; tslang itself does not.
set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_library(TsClangImporter
  BindingModel.cpp
  TypeMapper.cpp
  HeaderParser.cpp
  BindingPrinter.cpp

  LINK_LIBS
  clangTooling
  clangFrontend
  clangAST
  clangLex
  clangBasic
  )
