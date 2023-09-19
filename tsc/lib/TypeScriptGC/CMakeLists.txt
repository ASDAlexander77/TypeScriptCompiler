add_llvm_library(TypeScriptGC MODULE
    TypeScriptGC.cpp
    TypeScriptGCPrinter.cpp

    ADDITIONAL_HEADER_DIRS
    ${LLVM_MAIN_INCLUDE_DIR}/llvm/IR

    LINK_LIBS
    ${LLVM_PTHREAD_LIB}

    DEPENDS
    intrinsics_gen

    LINK_COMPONENTS
    AsmPrinter
    Analysis
    BinaryFormat
    CodeGen
    Core
    DebugInfoCodeView
    DebugInfoDWARF
    DebugInfoMSF
    MC
    MCParser
    Remarks
    Support
    SelectionDAG
    Target

    PLUGIN_TOOL
    llc
)

add_mlir_library(MLIRTypeScriptGC
    TypeScriptGC.cpp
    TypeScriptGCPrinter.cpp

    ADDITIONAL_HEADER_DIRS
    ${LLVM_MAIN_INCLUDE_DIR}/llvm/IR

    LINK_LIBS
    ${LLVM_PTHREAD_LIB}

    DEPENDS
    intrinsics_gen

    LINK_COMPONENTS
    AsmPrinter
    Analysis
    BinaryFormat
    CodeGen
    Core
    DebugInfoCodeView
    DebugInfoDWARF
    DebugInfoMSF
    MC
    MCParser
    Remarks
    Support
    Target
)
