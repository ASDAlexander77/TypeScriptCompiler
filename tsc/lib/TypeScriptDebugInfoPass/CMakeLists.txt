set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  )

add_llvm_library(TypeScriptDebugInfoPatchPass
  DebugInfoPatchPass.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  intrinsics_gen
  )
