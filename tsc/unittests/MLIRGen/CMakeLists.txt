add_mlir_unittest(MLIRGenTests
  TypeToString.cpp
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

target_include_directories(MLIRGenTests
    PRIVATE
    ${PROJECT_SOURCE_DIR}/tsc-new-parser
    )

target_link_libraries(MLIRGenTests
  PRIVATE
  ${dialect_libs}
  MLIRTypeScript
)
