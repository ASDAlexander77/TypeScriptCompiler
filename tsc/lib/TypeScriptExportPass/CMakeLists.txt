set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  )

add_llvm_library(TypeScriptExportPass
  ExportFixPass.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  intrinsics_gen
  )
