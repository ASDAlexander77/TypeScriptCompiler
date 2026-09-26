add_mlir_unittest(TsClangImporterTests
  BindingModelTest.cpp
  TypeMappingTest.cpp
  ParseErrorTest.cpp
)

target_link_libraries(TsClangImporterTests
  PRIVATE
  TsClangImporter
)

# clang's own headers (stdint.h, stddef.h) for the hermetic -ffreestanding -nostdlibinc parses
target_compile_definitions(TsClangImporterTests PRIVATE
  TSBINDGEN_TEST_RESOURCE_DIR="${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
