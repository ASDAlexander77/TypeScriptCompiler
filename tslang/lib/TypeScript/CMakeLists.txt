message(STATUS ">>> lib >>> TypeScript")

set_Options()

include_directories(
    ${PROJECT_SOURCE_DIR}/ts-new-parser
    )

link_directories(
    ${PROJECT_SOURCE_DIR}/ts-new-parser
    )

add_mlir_dialect_library(MLIRTypeScript
    TypeScriptDialect.cpp
    TypeScriptOps.cpp
    TypeScriptDialectTranslation.cpp
    AsyncDialectTranslation.cpp
    DiagnosticHelper.cpp
    MLIRGen.cpp
    MLIRGenCast.cpp
    MLIRGenClasses.cpp
    MLIRGenExpressions.cpp
    MLIRGenFunctions.cpp
    MLIRGenGenerics.cpp
    MLIRGenInterfaces.cpp
    MLIRGenModule.cpp
    MLIRGenStatements.cpp
    MLIRGenTypes.cpp
    LowerToAffineLoops.cpp
    LowerToLLVM.cpp
    RelocateConstantPass.cpp
    GCPass.cpp
    ObjDumper.cpp
    DeclarationPrinter.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/ts-new-parser

    DEPENDS
    MLIRTypeScriptOpsIncGen

    LINK_LIBS PUBLIC
    ts-new-parser-lib
    MLIRIR
    )
