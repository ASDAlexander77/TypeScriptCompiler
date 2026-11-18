set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils
  )

add_llvm_library(TypeScriptExceptionPass
  Win32ExceptionPass.cpp
  LandingPadFixPass.cpp
  ItaniumEndCatchPass.cpp
  CxxThrowCallingConvPass.cpp
  WasmExceptionPass.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  intrinsics_gen
  )
