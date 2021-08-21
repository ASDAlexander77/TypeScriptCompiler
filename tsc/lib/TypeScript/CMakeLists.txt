message(STATUS ">>> lib >>> TypeScript")

set_Options()

include_directories(
    ${PROJECT_SOURCE_DIR}/tsc-new-parser
    )

link_directories(
    ${PROJECT_SOURCE_DIR}/tsc-new-parser
    )

add_mlir_dialect_library(MLIRTypeScript
    TypeScriptDialect.cpp
    TypeScriptOps.cpp
    TypeScriptToLLVMIRTranslation.cpp
    MLIRGen.cpp
    LowerToAffineLoops.cpp   
    LowerToLLVM.cpp
    LoadBoundPropertiesPass.cpp
    GCPass.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/tsc-new-parser

    DEPENDS
    MLIRTypeScriptOpsIncGen
    MLIRTypeScriptLoadBoundPropertiesInterfaceIncGen

    LINK_LIBS PUBLIC
    tsc-new-parser-lib
    MLIRIR
    )
