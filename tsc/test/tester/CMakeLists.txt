message(STATUS ">>> test >>> runner")

set_MSVC_Options()

add_definitions("-DTEST_VCPATH=\"C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/VC/lib\"")
add_definitions("-DTEST_SDKPATH=\"C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/SDK/lib\"")
add_definitions("-DTEST_EXEPATH=\"C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin\"")
add_definitions("-DTEST_TSCEXEPATH=\"C:/dev/TypeScriptCompiler/__build/tsc/bin\"")

add_executable(test-runner test-runner.cpp)

######## enable testing ############
add_test(NAME test-runner-00-assert COMMAND test-runner "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00assert.ts")
add_test(NAME test-runner-00-if_return COMMAND test-runner "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00if_return.ts")
add_test(NAME test-runner-00-vars COMMAND test-runner "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00vars.ts")
add_test(NAME test-runner-00-globals COMMAND test-runner "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00globals.ts")
add_test(NAME test-runner-01-arguments COMMAND test-runner "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/01arguments.ts")
add_test(NAME test-runner-02-numbers COMMAND test-runner "C:/dev/TypeScriptCompiler/tsc/test/tester/tests/02numbers.ts")
