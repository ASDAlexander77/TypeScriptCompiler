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

# TSLANG_PACKAGE_VERSION is only set by a release build (CI passes the tag via
# -DTSLANG_PACKAGE_VERSION / the TSLANG_PACKAGE_VERSION env var); a plain dev
# build leaves it empty, so give `tslang --version` the commit it was built
# from instead of falling back on Version.h's "0.0.0-not-set" placeholder.
find_package(Git QUIET)
if(GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE TSLANG_GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE TSLANG_GIT_RESULT)
    if(NOT TSLANG_GIT_RESULT EQUAL 0)
        set(TSLANG_GIT_COMMIT_HASH "")
    endif()
endif()

if(TSLANG_GIT_COMMIT_HASH)
    target_compile_definitions(tslang PUBLIC TSLANG_GIT_COMMIT_HASH="${TSLANG_GIT_COMMIT_HASH}")
endif()

MESSAGE (STATUS "VERSION = " ${TSLANG_PACKAGE_VERSION})
MESSAGE (STATUS "GIT COMMIT HASH = " ${TSLANG_GIT_COMMIT_HASH})

mlir_check_all_link_libraries(tslang)
