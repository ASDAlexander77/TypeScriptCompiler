set_Options()

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRTypeScript
        )

set_Libs(LIBS)

add_llvm_executable(tsc-opt tsc-opt.cpp)

llvm_update_compile_flags(tsc-opt)
target_link_libraries(tsc-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(tsc-opt)
