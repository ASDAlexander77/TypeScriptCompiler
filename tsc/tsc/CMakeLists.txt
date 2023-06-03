set_Options()

set(LLVM_LINK_COMPONENTS
    Core
    Support
    nativecodegen
    native
    OrcJIT
    )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
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
    )

add_llvm_executable(tsc tsc.cpp jit.cpp obj.cpp utils.cpp)

llvm_update_compile_flags(tsc)
target_link_libraries(tsc PRIVATE ${LIBS})
target_compile_definitions(tsc PUBLIC TSC_PACKAGE_VERSION="${TSC_PACKAGE_VERSION}")

MESSAGE (STATUS "VERSION = " ${TSC_PACKAGE_VERSION})

mlir_check_all_link_libraries(tsc)
