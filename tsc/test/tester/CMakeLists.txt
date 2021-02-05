message(STATUS ">>> test >>> runner")

set_MSVC_Options()

#add_definitions(-DTEST_VCPATH="C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/VC/lib")
#add_definitions(-DTEST_SDKPATH="C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/SDK/ScopeCppSDK/vc15/SDK/lib")
add_definitions(-DTEST_EXEPATH="C:/dev/TypeScriptCompiler/3rdParty/llvm/debug/bin")
add_definitions(-DTEST_TSCEXEPATH="C:/dev/TypeScriptCompiler/__build/tsc/bin")

# tests 00
add_definitions(-DTEST_FILE="C:/dev/TypeScriptCompiler/tsc/test/tester/tests/00assert.ts")
add_executable(test-runner-00-assert test-runner.cpp)

# tests 01
add_definitions(-DTEST_FILE="C:/dev/TypeScriptCompiler/tsc/test/tester/tests/01arguments.ts")
add_executable(test-runner-01-arguments test-runner.cpp)

######## enable testing ############
add_test(test-runner-00-assert test-runner-00-assert)
add_test(test-runner-01-arguments test-runner-01-arguments)