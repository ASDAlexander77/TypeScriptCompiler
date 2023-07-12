add_mlir_unittest(MLIRExampleTests
  Example.cpp
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_link_libraries(MLIRExampleTests
  PRIVATE
  #MLIRArithToLLVM
  ${dialect_libs}

)
