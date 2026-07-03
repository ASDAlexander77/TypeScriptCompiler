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
    MLIRRegisterAllExtensions
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

add_llvm_executable(tslang 
    tslang.cpp 
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
    declarationInline.cpp
    vscode.cpp
    cmake.cpp
    defaultlib.cpp)

llvm_update_compile_flags(tslang)
target_link_libraries(tslang PRIVATE ${LIBS})
target_compile_definitions(tslang PUBLIC TSLANG_PACKAGE_VERSION="${TSLANG_PACKAGE_VERSION}")

MESSAGE (STATUS "VERSION = " ${TSLANG_PACKAGE_VERSION})

mlir_check_all_link_libraries(tslang)
