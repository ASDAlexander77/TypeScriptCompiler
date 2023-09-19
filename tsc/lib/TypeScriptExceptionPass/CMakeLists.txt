set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  )

add_llvm_library(TypeScriptExceptionPass
  Win32ExceptionPass.cpp
  LandingPadFixPass.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  intrinsics_gen
  )
