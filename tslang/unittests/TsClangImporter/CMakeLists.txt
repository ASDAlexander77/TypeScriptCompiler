add_mlir_unittest(TsClangImporterTests
  BindingModelTest.cpp
)

target_link_libraries(TsClangImporterTests
  PRIVATE
  TsClangImporter
)
