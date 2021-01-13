set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
	antlr4_static
  	MLIRAnalysis
  	MLIREDSC
  	MLIRExecutionEngine
  	MLIRIR
  	MLIRJitRunner
  	MLIRLLVMIR
  	MLIRParser
  	MLIRTargetLLVMIR
  	MLIRSupport
        MLIROptLib
        MLIRTypeScript
        )

add_llvm_executable(tsc tsc.cpp)

llvm_update_compile_flags(tsc)
target_link_libraries(tsc PRIVATE ${LIBS})

mlir_check_all_link_libraries(tsc)
