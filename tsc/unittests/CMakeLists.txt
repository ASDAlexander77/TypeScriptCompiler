message(STATUS ">>> unittests >>> TypeScript Unit Tests")

set_target_properties(MLIRTypeScriptUnitTests PROPERTIES FOLDER "TypeScript Unit Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRTypeScriptUnitTests ${test_dirname} ${ARGN})
  message(STATUS ">>> unittests >>> added ${test_dirname}")

  add_custom_command(
    TARGET ${test_dirname} POST_BUILD
    COMMAND "$<TARGET_FILE:${test_dirname}>"
    VERBATIM)

endfunction()

add_subdirectory(Example)
