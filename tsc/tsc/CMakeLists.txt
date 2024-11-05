set_Options()

set(LLVM_LINK_COMPONENTS
    # for Obj/ASM
    AllTargetsAsmParsers
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsInfos
    # end - Obj/ASM
    Core
    Option
    Support
    TargetParser    
    nativecodegen
    native
    OrcJIT
    )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    clangDriver
    clangBasic    
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRJitRunner
    MLIRLLVMIRTransforms
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRTypeScript
    #MLIRTypeScriptGC
    TypeScriptExceptionPass
    TypeScriptDebugInfoPass
    TypeScriptExportPass
    TypeScriptAliasPass
    TypeScriptMemAllocPass
    )

add_llvm_executable(tsc 
    tsc.cpp 
    compile.cpp 
    transform.cpp 
    dump.cpp 
    jit.cpp 
    obj.cpp 
    exe.cpp 
    TextDiagnostic.cpp 
    TextDiagnosticPrinter.cpp 
    utils.cpp 
    opts.cpp 
    declarationInline.cpp)

llvm_update_compile_flags(tsc)
target_link_libraries(tsc PRIVATE ${LIBS})
target_compile_definitions(tsc PUBLIC TSC_PACKAGE_VERSION="${TSC_PACKAGE_VERSION}")

MESSAGE (STATUS "VERSION = " ${TSC_PACKAGE_VERSION})

mlir_check_all_link_libraries(tsc)
