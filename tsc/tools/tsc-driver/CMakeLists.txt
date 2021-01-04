get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        )
add_llvm_executable(tsc driver.cpp)

llvm_update_compile_flags(tsc)
target_link_libraries(tsc PRIVATE ${LIBS})

mlir_check_all_link_libraries(tsc)
