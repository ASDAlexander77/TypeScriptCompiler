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
    MLIRGenAccessCall.cpp
    MLIRGenCast.cpp
    MLIRGenClasses.cpp
    MLIRGenExpressions.cpp
    MLIRGenFunctions.cpp
    MLIRGenGenerics.cpp
    MLIRGenInterfaces.cpp
    MLIRGenModule.cpp
    MLIRGenStatements.cpp
    MLIRGenTypes.cpp
    MLIRGenVariables.cpp
    LowerToAffineLoops.cpp
    LowerToLLVM.cpp
    RelocateConstantPass.cpp
    OwnershipVerifierPass.cpp
    OwnershipInferencePass.cpp
    OwnershipInferenceBorrowers.cpp
    OwnershipInferenceClosures.cpp
    OwnershipInferenceDiagnostics.cpp
    OwnershipInferenceEdgeReleases.cpp
    OwnershipInferenceGlobalReads.cpp
    OwnershipInferenceKeptParams.cpp
    OwnershipInferenceMoves.cpp
    OwnershipInferencePlaceReads.cpp
    OwnershipInferenceReadsAfterMove.cpp
    OwnershipInferenceRecords.cpp
    OwnershipSignaturePass.cpp
    OwnershipSignatureCallbacks.cpp
    OwnershipSignatureClosedWorld.cpp
    OwnershipSignatureGenerators.cpp
    OwnershipSignatureGlobals.cpp
    OwnershipSignatureInterfaces.cpp
    OwnershipSignatureRecords.cpp
    OwnershipSignatureResolve.cpp
    OwnedReturnConsumptionPass.cpp
    StackPromotionPass.cpp
    BorrowFactsExportPass.cpp
    GCPass.cpp
    ProcessHeapPass.cpp
    InlinedLocationPass.cpp
    AsyncTargetWidthPass.cpp
    ObjDumper.cpp
    DeclarationPrinter.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/ts-new-parser

    DEPENDS
    MLIRTypeScriptOpsIncGen

    LINK_LIBS PUBLIC
    ts-new-parser-lib
    MLIRIR
    # mlir::parseType: the slot types of the own facts read from an object (--own-facts)
    MLIRAsmParser
    )
