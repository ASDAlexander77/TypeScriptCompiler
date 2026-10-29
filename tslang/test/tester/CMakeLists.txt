message(STATUS ">>> test >>> runner")

message(STATUS ">>> CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION >>> ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}")
message(STATUS ">>> CMAKE_CXX_COMPILER >>> ${CMAKE_CXX_COMPILER}")

if (WIN32)
    # Find the Windows Kits directory.
    get_filename_component(windows_kits_dir "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots;KitsRoot10]" ABSOLUTE)
    message(STATUS ">>> windows_kits_dir >>> ${windows_kits_dir}")
    set(programfilesx86 "ProgramFiles(x86)")

    find_path(WINDOWS_KITS_DIR_LIB 
        NAMES Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um/${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}/kernel32.lib
        PATHS $ENV{CMAKE_WINDOWS_KITS_10_DIR} "${windows_kits_dir}" "$ENV{ProgramFiles}/Windows Kits/10" "$ENV{${programfilesx86}}/Windows Kits/10")
    if (NOT("${WINDOWS_KITS_DIR_LIB}" STREQUAL "" OR "${WINDOWS_KITS_DIR_LIB}" STREQUAL "WINDOWS_KITS_DIR_LIB-NOTFOUND"))
        set(WINDOWS_KITS_DIR_LIB "${WINDOWS_KITS_DIR_LIB}/Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/um/${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}")
    endif()

    message(STATUS ">>> WINDOWS_KITS_DIR_LIB >>> ${WINDOWS_KITS_DIR_LIB}")

    find_path(WINDOWS_KITS_DIR_UCRT_LIB 
        NAMES Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/ucrt/${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}/libucrt.lib
        PATHS $ENV{CMAKE_WINDOWS_KITS_10_DIR} "${windows_kits_dir}" "$ENV{ProgramFiles}/Windows Kits/10" "$ENV{${programfilesx86}}/Windows Kits/10")
    if (NOT("${WINDOWS_KITS_DIR_UCRT_LIB}" STREQUAL "" OR "${WINDOWS_KITS_DIR_UCRT_LIB}" STREQUAL "WINDOWS_KITS_DIR_UCRT_LIB-NOTFOUND"))
        set(WINDOWS_KITS_DIR_UCRT_LIB "${WINDOWS_KITS_DIR_UCRT_LIB}/Lib/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/ucrt/${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}")
    endif()

    message(STATUS ">>> WINDOWS_KITS_DIR_UCRT_LIB >>> ${WINDOWS_KITS_DIR_UCRT_LIB}")
endif()

# Find the VC Libs
string(FIND "${CMAKE_CXX_COMPILER}" "Hostx86" HAS_Hostx86)
if (NOT "${HAS_Hostx86}" EQUAL "-1")
    set(C_HOST "Hostx86")
else()    
    string(FIND "${CMAKE_CXX_COMPILER}" "HostX86" HAS_HostX86)
    if (NOT "${HAS_HostX86}" EQUAL "-1")
        set(C_HOST "HostX86")
    else()    
        string(FIND "${CMAKE_CXX_COMPILER}" "Hostx64" HAS_Hostx64)
        if (NOT "${HAS_Hostx64}" EQUAL "-1")
            set(C_HOST "Hostx64")
        else()    
            string(FIND "${CMAKE_CXX_COMPILER}" "HostX64" HAS_HostX64)
            if (NOT "${HAS_HostX64}" EQUAL "-1")
                set(C_HOST "HostX64")
            else()    
                # default
                set(C_HOST "Hostx86")
            endif()
        endif()
    endif()
endif()

string(REPLACE "bin/${C_HOST}/${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}/cl.exe" "lib/${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}" VC_LIB_DIR "${CMAKE_CXX_COMPILER}")

message(STATUS ">>> VC_LIB_DIR >>> ${VC_LIB_DIR}")
message(STATUS ">>> CMAKE_BINARY_DIR >>> ${CMAKE_BINARY_DIR}")
message(STATUS ">>> CMAKE_CXX_COMPILER >>> ${CMAKE_CXX_COMPILER}")
message(STATUS ">>> CMAKE_LINKER >>> ${CMAKE_LINKER}")
message(STATUS ">>> CMAKE_C_STANDARD_LIBRARIES >>> ${CMAKE_C_STANDARD_LIBRARIES}")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set_Options()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
message(STATUS ">>> Test-runner is Release")
add_definitions(-DCOMPILE_RELEASE=1)
else()
message(STATUS ">>> Test-runner is Debug")
add_definitions(-DCOMPILE_DEBUG=1)
endif()
add_executable(test-runner test-runner.cpp)
target_link_libraries(test-runner PRIVATE ${LIBS})
string(TOLOWER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_LOWERCASE)
# GC lib dir derived from the BDWgc package (BDWgc_DIR=<prefix>/lib/cmake/bdwgc)
get_filename_component(TEST_GC_LIBDIR "${BDWgc_DIR}/../.." ABSOLUTE)
# A shared Boehm, for the shared-library tests only. Two binaries that each link gc.lib
# statically get two collectors, and the one inside the library then frees objects the
# executable is still holding - item 5ao. Built by scripts/build_gc_release_shared_vs.bat;
# when it is absent the shared-library tests are skipped rather than run wrong.
set(TEST_GC_SHARED_PREFIX "${TSLANG_GC_SHARED_PREFIX}")
if (EXISTS "${TEST_GC_SHARED_PREFIX}/lib/gc.lib" OR EXISTS "${TEST_GC_SHARED_PREFIX}/lib/libgc.so")
    set(TSLANG_HAVE_SHARED_GC TRUE)
else()
    set(TSLANG_HAVE_SHARED_GC FALSE)
    message(STATUS "shared Boehm not found at ${TEST_GC_SHARED_PREFIX} - shared-library tests will be skipped (see scripts/build_gc_release_shared_vs.bat)")
endif()

if (WIN32)
    # The 32-bit (i686) libraries, for `test-runner -x86`. Each path names the directory that
    # holds the libraries themselves: the runner links with lld directly and passes these as
    # /libpath, so there is no tslang lookup rule to add the `x86` segment for it. The gc and
    # runtime libraries use the phase 2 layout, an `x86` subdirectory beside the x64 ones.
    string(REPLACE "bin/${C_HOST}/${CMAKE_VS_PLATFORM_TOOLSET_HOST_ARCHITECTURE}/cl.exe" "lib/x86" TEST_LIBPATH_X86 "${CMAKE_CXX_COMPILER}")
    get_filename_component(TEST_SDKPATH_X86 "${WINDOWS_KITS_DIR_LIB}" DIRECTORY)
    set(TEST_SDKPATH_X86 "${TEST_SDKPATH_X86}/x86")
    get_filename_component(TEST_UCRTPATH_X86 "${WINDOWS_KITS_DIR_UCRT_LIB}" DIRECTORY)
    set(TEST_UCRTPATH_X86 "${TEST_UCRTPATH_X86}/x86")
    set(TEST_GCPATH_X86 "${TEST_GC_LIBDIR}/x86")
    # the 32-bit gc.dll sits beside its import library; there is no bin/x86
    set(TEST_GC_SHARED_LIBPATH_X86 "${TEST_GC_SHARED_PREFIX}/lib/x86")
    get_filename_component(TEST_TSLANG_LIBPATH_X86 "${PROJECT_SOURCE_DIR}/../__build/tslang-runtime/${CMAKE_BUILD_TYPE_LOWERCASE}/x86" ABSOLUTE)

    message(STATUS ">>> TEST_LIBPATH_X86 >>> ${TEST_LIBPATH_X86}")
    message(STATUS ">>> TEST_SDKPATH_X86 >>> ${TEST_SDKPATH_X86}")
    message(STATUS ">>> TEST_UCRTPATH_X86 >>> ${TEST_UCRTPATH_X86}")
    message(STATUS ">>> TEST_GCPATH_X86 >>> ${TEST_GCPATH_X86}")
    message(STATUS ">>> TEST_GC_SHARED_LIBPATH_X86 >>> ${TEST_GC_SHARED_LIBPATH_X86}")
    message(STATUS ">>> TEST_TSLANG_LIBPATH_X86 >>> ${TEST_TSLANG_LIBPATH_X86}")

    target_compile_definitions(test-runner PUBLIC "TEST_LIBPATH_X86=\"${TEST_LIBPATH_X86}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_SDKPATH_X86=\"${TEST_SDKPATH_X86}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_UCRTPATH_X86=\"${TEST_UCRTPATH_X86}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GCPATH_X86=\"${TEST_GCPATH_X86}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GC_SHARED_LIBPATH_X86=\"${TEST_GC_SHARED_LIBPATH_X86}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_TSLANG_LIBPATH_X86=\"${TEST_TSLANG_LIBPATH_X86}\"")

    target_compile_definitions(test-runner PUBLIC "TEST_GC_SHARED_LIBPATH=\"${TEST_GC_SHARED_PREFIX}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GC_SHARED_BINPATH=\"${TEST_GC_SHARED_PREFIX}/bin\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LIBPATH=\"${VC_LIB_DIR}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_SDKPATH=\"${WINDOWS_KITS_DIR_LIB}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_UCRTPATH=\"${WINDOWS_KITS_DIR_UCRT_LIB}\"")
    target_compile_definitions(test-runner PUBLIC "CMAKE_C_STANDARD_LIBRARIES=\"${CMAKE_C_STANDARD_LIBRARIES}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GCPATH=\"${TEST_GC_LIBDIR}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_TSLANG_EXEPATH=\"${CMAKE_BINARY_DIR}/bin\"")
    target_compile_definitions(test-runner PUBLIC "TEST_TSLANG_LIBPATH=\"${CMAKE_BINARY_DIR}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_EXEPATH=\"${LLVM_TOOLS_BINARY_DIR}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_LIBPATH=\"${LLVM_LIBRARY_DIR}\"")
else()
    target_compile_definitions(test-runner PUBLIC "TEST_GC_SHARED_LIBPATH=\"${TEST_GC_SHARED_PREFIX}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GC_SHARED_BINPATH=\"${TEST_GC_SHARED_PREFIX}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GCPATH=\"${TEST_GC_LIBDIR}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_TSLANG_EXEPATH=\"${CMAKE_BINARY_DIR}/bin\"")
    target_compile_definitions(test-runner PUBLIC "TEST_TSLANG_LIBPATH=\"${CMAKE_BINARY_DIR}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_EXEPATH=\"${LLVM_TOOLS_BINARY_DIR}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_LIBPATH=\"${LLVM_LIBRARY_DIR}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_COMPILER=\"${CMAKE_CXX_COMPILER}\"")
endif()

# test-runner caches the per-mode link scripts (compile/jit[d].bat|.sh) via an
# exists() guard and reuses them across runs, so they retain stale lib paths
# (e.g. GC_LIB_PATH) after a path-affecting reconfigure. Remove them on every
# (re)configure so they regenerate with the current TEST_* definitions above.
file(REMOVE
    "${CMAKE_CURRENT_BINARY_DIR}/compile.bat"  "${CMAKE_CURRENT_BINARY_DIR}/jit.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compiled.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitd.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compile.sh"   "${CMAKE_CURRENT_BINARY_DIR}/jit.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compiled.sh"  "${CMAKE_CURRENT_BINARY_DIR}/jitd.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compilefm.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitfm.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compiledfm.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitdfm.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compilefm.sh"  "${CMAKE_CURRENT_BINARY_DIR}/jitfm.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compiledfm.sh" "${CMAKE_CURRENT_BINARY_DIR}/jitdfm.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compilerc.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitrc.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compiledrc.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitdrc.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compilerc.sh"  "${CMAKE_CURRENT_BINARY_DIR}/jitrc.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compiledrc.sh" "${CMAKE_CURRENT_BINARY_DIR}/jitdrc.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compilenone.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitnone.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compilednone.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitdnone.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compilenone.sh"  "${CMAKE_CURRENT_BINARY_DIR}/jitnone.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compilednone.sh" "${CMAKE_CURRENT_BINARY_DIR}/jitdnone.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compileown.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitown.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compiledown.bat" "${CMAKE_CURRENT_BINARY_DIR}/jitdown.bat"
    "${CMAKE_CURRENT_BINARY_DIR}/compileown.sh"  "${CMAKE_CURRENT_BINARY_DIR}/jitown.sh"
    "${CMAKE_CURRENT_BINARY_DIR}/compiledown.sh" "${CMAKE_CURRENT_BINARY_DIR}/jitdown.sh")
# and the 32-bit variants (`-x86`), every one of which ends in "x86"
file(GLOB TSLANG_X86_CACHED_SCRIPTS "${CMAKE_CURRENT_BINARY_DIR}/compile*x86.bat")
if (TSLANG_X86_CACHED_SCRIPTS)
    file(REMOVE ${TSLANG_X86_CACHED_SCRIPTS})
endif()

######## 32-bit (i686) twins ############
# With TSLANG_TEST_X86 on, every `test-runner` registration below that does not pass -jit gets a
# twin that runs the same files as a 32-bit program: `test-x86-<name without "test-">`, which
# runs `test-runner -x86 <the same args>` and is labelled x86, so `ctest -L x86` runs just those.
# -jit registrations get none: the JIT runs code inside the x64 compiler process, and the runner
# refuses `-jit -x86`. A twin that cannot work at 32 bits is listed, with its reason, in
# x86-exclusions.cmake, and is registered DISABLED so ctest reports it rather than hiding it.
# Off by default: the default suite and its test names do not change.
option(TSLANG_TEST_X86 "Also register the 32-bit (i686) twin of every non-JIT test" OFF)

# With TSLANG_TEST_NO_STACK_PROMOTION on, every `test-runner` registration builds with every
# allocation on the heap: run the suite both ways and the pass/fail lists must be the same
# (docs/superpowers/specs/2026-10-18-stack-promotion-design.md section 9).
option(TSLANG_TEST_NO_STACK_PROMOTION "Build every test-runner test with --no-stack-promotion" OFF)

if (TSLANG_TEST_X86)
    if (NOT WIN32)
        message(FATAL_ERROR "TSLANG_TEST_X86: the x86 twins build Windows i686 programs, so they need a Windows host")
    endif()

    # The MSVC and Windows SDK x86 libraries every 32-bit link takes (from the -x86 defines above).
    foreach(x86_system_lib "${TEST_LIBPATH_X86}/libcmt.lib" "${TEST_SDKPATH_X86}/kernel32.lib" "${TEST_UCRTPATH_X86}/libucrt.lib")
        if (NOT EXISTS "${x86_system_lib}")
            message(FATAL_ERROR "TSLANG_TEST_X86: ${x86_system_lib} is missing. "
                                "Install the MSVC x86 build tools and the Windows SDK x86 libraries (Visual Studio Installer).")
        endif()
    endforeach()

    # The x86 libraries the twins link against (phase 2 layout; paths from the -x86 defines above).
    if (NOT EXISTS "${TEST_GCPATH_X86}/gc.lib")
        message(FATAL_ERROR "TSLANG_TEST_X86: the x86 gc.lib is missing at ${TEST_GCPATH_X86}. "
                            "Build it with: prepare_3rdParty.bat ${CMAKE_BUILD_TYPE_LOWERCASE} x86")
    endif()
    foreach(gc_shared_file gc.lib gc.dll)
        if (NOT EXISTS "${TEST_GC_SHARED_LIBPATH_X86}/${gc_shared_file}")
            message(FATAL_ERROR "TSLANG_TEST_X86: the x86 shared ${gc_shared_file} is missing at ${TEST_GC_SHARED_LIBPATH_X86}. "
                                "Build it with: prepare_3rdParty.bat ${CMAKE_BUILD_TYPE_LOWERCASE} x86")
        endif()
    endforeach()
    if (NOT EXISTS "${TEST_TSLANG_LIBPATH_X86}/TypeScriptAsyncRuntime.lib")
        message(FATAL_ERROR "TSLANG_TEST_X86: the x86 TypeScriptAsyncRuntime.lib is missing at ${TEST_TSLANG_LIBPATH_X86}. "
                            "Build it with: scripts\\build_tslang_runtime_${CMAKE_BUILD_TYPE_LOWERCASE}_x86.bat")
    endif()

    include("${CMAKE_CURRENT_SOURCE_DIR}/x86-exclusions.cmake")
    foreach(x86_excluded ${TSLANG_X86_EXCLUDED})
        if ("${TSLANG_X86_EXCLUDED_REASON_${x86_excluded}}" STREQUAL "")
            message(FATAL_ERROR "x86-exclusions.cmake: ${x86_excluded} has no TSLANG_X86_EXCLUDED_REASON_${x86_excluded}")
        endif()
    endforeach()
endif()

# tslang_add_test(NAME <name> COMMAND test-runner <args...>)
# add_test, plus the x86 twin described above. Sets TSLANG_X86_TWIN in the caller to the twin's
# name, or to empty when there is none, so a caller that sets properties on the test can set
# them on the twin too.
function(tslang_add_test)
    cmake_parse_arguments(PARSE_ARGV 0 arg "" "NAME" "COMMAND")
    if (arg_UNPARSED_ARGUMENTS OR "${arg_NAME}" STREQUAL "" OR "${arg_COMMAND}" STREQUAL "")
        message(FATAL_ERROR "tslang_add_test(NAME <name> COMMAND test-runner <args...>): bad arguments: ${ARGV}")
    endif()
    list(GET arg_COMMAND 0 arg_program)
    if (NOT arg_program STREQUAL "test-runner")
        message(FATAL_ERROR "tslang_add_test(${arg_NAME}): the command must be test-runner; use add_test for anything else")
    endif()

    # Checked whether the twins are on or not, so a name that could not have one fails every configure.
    if (NOT arg_NAME MATCHES "^test-")
        message(FATAL_ERROR "tslang_add_test(${arg_NAME}): the name must start with test-")
    endif()

    if (TSLANG_TEST_NO_STACK_PROMOTION)
        list(INSERT arg_COMMAND 1 -no-stack-promotion)
    endif()

    add_test(NAME ${arg_NAME} COMMAND ${arg_COMMAND})

    set(twin "")
    if (TSLANG_TEST_X86)
        if ("-jit" IN_LIST arg_COMMAND)
            set_property(GLOBAL APPEND PROPERTY TSLANG_X86_JIT_SKIPPED ${arg_NAME})
        else()
            string(REGEX REPLACE "^test-" "test-x86-" twin "${arg_NAME}")
            list(SUBLIST arg_COMMAND 1 -1 arg_runner_args)
            add_test(NAME ${twin} COMMAND test-runner -x86 ${arg_runner_args})
            set_tests_properties(${twin} PROPERTIES LABELS x86)
            set_property(GLOBAL APPEND PROPERTY TSLANG_X86_TWINS ${twin})
            if (arg_NAME IN_LIST TSLANG_X86_EXCLUDED)
                set_tests_properties(${twin} PROPERTIES DISABLED TRUE)
                set_property(GLOBAL APPEND PROPERTY TSLANG_X86_EXCLUDED_SEEN ${arg_NAME})
                message(STATUS "x86 twin ${twin} excluded: ${TSLANG_X86_EXCLUDED_REASON_${arg_NAME}}")
            endif()
        endif()
    endif()
    set(TSLANG_X86_TWIN "${twin}" PARENT_SCOPE)
endfunction()

######## enable testing ############
# open __build\tslang   
# ctest.exe -j4 -C Release -T test --output-on-failure
tslang_add_test(NAME test-compile-00-print COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00print.ts")
tslang_add_test(NAME test-compile-00-assert COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00assert.ts")
tslang_add_test(NAME test-compile-00-enums COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00enum.ts")
tslang_add_test(NAME test-compile-00-enums-multiple COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00enum_multiple.ts")
tslang_add_test(NAME test-compile-01-enums COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01enum.ts")
tslang_add_test(NAME test-compile-00-numbers COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00numbers.ts")
tslang_add_test(NAME test-compile-00-equals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00equals.ts")
tslang_add_test(NAME test-compile-00-bool-arith-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00bool_arith_ops.ts")
tslang_add_test(NAME test-compile-00-int64-shift-bitwise COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00int64_shift_bitwise.ts")
tslang_add_test(NAME test-compile-00-int64-mul-literal COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00int64_mul_literal.ts")
tslang_add_test(NAME test-compile-00-int-literal-above-s32 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00int_literal_above_s32.ts")
tslang_add_test(NAME test-compile-00-literal-union-arith COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00literal_union_arith.ts")
tslang_add_test(NAME test-compile-00-literal-union-cast COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00literal_union_cast.ts")
tslang_add_test(NAME test-compile-00-narrowing-and-or COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrowing_and_or.ts")
tslang_add_test(NAME test-compile-00-enum-compare COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00enum_compare.ts")
tslang_add_test(NAME test-compile-00-bool-const-array COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00bool_const_array.ts")
tslang_add_test(NAME test-compile-00-mixed-type-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00mixed_type_ops.ts")
tslang_add_test(NAME test-compile-00-generator-manual-next COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next.ts")
tslang_add_test(NAME test-compile-00-generator-manual-next-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next2.ts")
tslang_add_test(NAME test-compile-00-global-const-object-method COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_const_object_method.ts")
tslang_add_test(NAME test-compile-00-funcs COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs.ts")
tslang_add_test(NAME test-compile-00-funcs-capture COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_capture.ts")
tslang_add_test(NAME test-compile-00-funcs-vararg COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_vararg.ts")
tslang_add_test(NAME test-compile-00-declare-function-twice COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00declare_function_twice.ts")
tslang_add_test(NAME test-compile-00-function-overloads COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00function_overloads.ts")
tslang_add_test(NAME test-compile-00-gc-allocations-distinct COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00gc_allocations_distinct.ts")
tslang_add_test(NAME test-compile-00-assert-message COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00assert_message.ts")
tslang_add_test(NAME test-compile-01-funcs-vararg COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_vararg.ts")
# printf is avaliable in JIT only, C has snprintf
#tslang_add_test(NAME test-compile-02-funcs-vararg COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02funcs_vararg.ts")
tslang_add_test(NAME test-compile-00-funcs-bindings COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_bindings.ts")
tslang_add_test(NAME test-compile-00-funcs-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic.ts")
tslang_add_test(NAME test-compile-01-funcs-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_generic.ts")
tslang_add_test(NAME test-compile-00-funcs-generic-typeof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_with_typeof.ts")
tslang_add_test(NAME test-compile-00-funcs-generic-arrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_arrow.ts")
tslang_add_test(NAME test-compile-00-funcs-generic-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_iterator.ts")
tslang_add_test(NAME test-compile-00-funcs-typeless-function-as-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_typeless_function_as_generic.ts")
tslang_add_test(NAME test-compile-00-nested-implicit-generics COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_implicit_generics.ts")
tslang_add_test(NAME test-compile-01-funcs-generic-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_generic_iterator.ts")
tslang_add_test(NAME test-compile-02-funcs-generic-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02funcs_generic_iterator.ts")
tslang_add_test(NAME test-compile-00-funcs-typed-generic-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_typed_generic_iterator.ts")
tslang_add_test(NAME test-compile-00-funcs-nesting COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting.ts")
tslang_add_test(NAME test-compile-00-funcs-nesting-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_generic.ts")
tslang_add_test(NAME test-compile-00-funcs-nesting-capture COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_capture.ts")
tslang_add_test(NAME test-compile-00-funcs-hybrid-null-this COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_hybrid_null_this.ts")
tslang_add_test(NAME test-compile-00-funcs-c-varargs-storage COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_c_varargs_storage.ts")
tslang_add_test(NAME test-compile-00-typeof-static-fold COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_static_fold.ts")
tslang_add_test(NAME test-compile-00-typeof-function-narrowing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_function_narrowing.ts")
tslang_add_test(NAME test-compile-00-typeof-static-fold-conditions COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_static_fold_conditions.ts")
tslang_add_test(NAME test-compile-00-typeof-union-narrowing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_union_narrowing.ts")
tslang_add_test(NAME test-compile-00-const-array-to-array-elements COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_to_array_elements.ts")
tslang_add_test(NAME test-compile-00-array-widen-elements-now COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_widen_elements_now.ts")
tslang_add_test(NAME test-compile-00-const-array-nested-elements COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_nested_elements.ts")
tslang_add_test(NAME test-compile-00-const-array-mutation COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_mutation.ts")
tslang_add_test(NAME test-compile-00-nested-array-literal-mutation COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_array_literal_mutation.ts")
tslang_add_test(NAME test-compile-00-array-move-wide-elements COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_move_wide_elements.ts")
tslang_add_test(NAME test-compile-00-bigint-struct-layout COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00bigint_struct_layout.ts")
tslang_add_test(NAME test-compile-00-union-member-padding COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_member_padding.ts")
tslang_add_test(NAME test-compile-00-union-global-init COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_global_init.ts")
tslang_add_test(NAME test-compile-00-funcs-expression-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_expression_generic.ts")
tslang_add_test(NAME test-compile-00-funcs-expression-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_expression_iterator.ts")
tslang_add_test(NAME test-compile-00-arrow-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00arrow_generic.ts")
tslang_add_test(NAME test-compile-00-arrow-in-arrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00arrow_in_arrow.ts")
tslang_add_test(NAME test-compile-00-arrow-global COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00arrow_global.ts")
tslang_add_test(NAME test-compile-00-lambdas COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00lambdas.ts")
tslang_add_test(NAME test-compile-01-lambdas COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01lambdas.ts")
tslang_add_test(NAME test-compile-00-lambdas-gemeric-global COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00lambdas_generic_global.ts")
tslang_add_test(NAME test-compile-00-if_return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00if_return.ts")
tslang_add_test(NAME test-compile-00-dowhile COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00dowhile.ts")
tslang_add_test(NAME test-compile-00-while COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00while.ts")
tslang_add_test(NAME test-compile-00-for COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for.ts")
tslang_add_test(NAME test-compile-00-for-infinite-result COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_infinite_result.ts")
tslang_add_test(NAME test-compile-00-break-continue COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00break_continue.ts")
tslang_add_test(NAME test-compile-00-label-chain-loop COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00label_chain_loop.ts")
tslang_add_test(NAME test-compile-00-labelled-scope-exit COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00labelled_scope_exit.ts")
tslang_add_test(NAME test-compile-00-switch-case-scope-exit COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch_case_scope_exit.ts")
tslang_add_test(NAME test-compile-00-using-dispose-order COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_dispose_order.ts")
tslang_add_test(NAME test-compile-00-using-unwind-outer-locals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_unwind_outer_locals.ts")
tslang_add_test(NAME test-compile-00-try-unwind-outer-locals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_unwind_outer_locals.ts")
tslang_add_test(NAME test-compile-00-try-return-through-finally COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_return_through_finally.ts")
tslang_add_test(NAME test-compile-00-try-scope-exit-through-finally COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_scope_exit_through_finally.ts")
tslang_add_test(NAME test-compile-00-try-jump-from-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_jump_from_catch.ts")
tslang_add_test(NAME test-compile-00-vars COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00vars.ts")
tslang_add_test(NAME test-compile-00-var-bindings COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00var_bindings.ts")
tslang_add_test(NAME test-compile-00-globals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals.ts")
tslang_add_test(NAME test-compile-00-globals-entry-point COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals_entry_point.ts")
tslang_add_test(NAME test-compile-00-main-argc-argv COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00main_argc_argv.ts")
tslang_add_test(NAME test-compile-00-globals2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals2.ts")
tslang_add_test(NAME test-compile-00-globals3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals3.ts")
tslang_add_test(NAME test-compile-00-arrays COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array.ts")
tslang_add_test(NAME test-compile-00-arrays2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array2.ts")
tslang_add_test(NAME test-compile-00-array-reference COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_reference.ts")
tslang_add_test(NAME test-compile-00-arrays3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array3.ts")
tslang_add_test(NAME test-compile-00-array-zero-slots COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_zero_slots.ts")
tslang_add_test(NAME test-compile-00-arrays4-push-pop COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array4_push_pop.ts")
tslang_add_test(NAME test-compile-00-array-shift COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_shift.ts")
tslang_add_test(NAME test-compile-00-array-splice COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_splice.ts")
tslang_add_test(NAME test-compile-00-const-array-push COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_push.ts")
tslang_add_test(NAME test-compile-00-array-spread-args COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_spread_args.ts")
tslang_add_test(NAME test-compile-00-arrays5-deconstruct COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array5_deconst.ts")
tslang_add_test(NAME test-compile-00-array-rest-copy COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_rest_copy.ts")
tslang_add_test(NAME test-compile-00-arrays6 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array6.ts")
tslang_add_test(NAME test-compile-00-array-static-nested COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_static_nested.ts")
tslang_add_test(NAME test-compile-00-array-literal-copy COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_literal_copy.ts")
tslang_add_test(NAME test-compile-00-tuple-runtime-arrays COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple_runtime_arrays.ts")
tslang_add_test(NAME test-compile-00-record-to-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00record_to_class.ts")
tslang_add_test(NAME test-compile-00-record-variable-to-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00record_variable_to_class.ts")
tslang_add_test(NAME test-compile-00-array-insert-functions COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_insert_functions.ts")
tslang_add_test(NAME test-compile-00-await-void COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00await_void.ts")
tslang_add_test(NAME test-compile-00-for-await-sequential COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await_sequential.ts")
tslang_add_test(NAME test-compile-00-for-await-control COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await_control.ts")
tslang_add_test(NAME test-compile-00-for-optional-class-condition COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_optional_class_condition.ts")
tslang_add_test(NAME test-compile-00-for-condition-narrowing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_condition_narrowing.ts")
tslang_add_test(NAME test-compile-00-param-assigned-owned COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00param_assigned_owned.ts")
tslang_add_test(NAME test-compile-00-string-empty-falsy COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_empty_falsy.ts")
tslang_add_test(NAME test-compile-00-string-concat-bounds COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_concat_bounds.ts")
tslang_add_test(NAME test-compile-00-parse-number-range COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00parse_number_range.ts")
tslang_add_test(NAME test-compile-00-global-argument-held COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_argument_held.ts")
tslang_add_test(NAME test-compile-00-shared-global-field-argument COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00shared_global_field_argument.ts")
tslang_add_test(NAME test-compile-00-narrowed-assign-other-member COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrowed_assign_other_member.ts")
tslang_add_test(NAME test-compile-00-bound-method-count COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00bound_method_count.ts")
tslang_add_test(NAME test-compile-00-const-view-of-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_view_of_field.ts")
tslang_add_test(NAME test-compile-00-const-view-of-element COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_view_of_element.ts")
tslang_add_test(NAME test-compile-00-const-record-owned-fields COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_record_owned_fields.ts")
tslang_add_test(NAME test-compile-00-generator-temporaries COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_temporaries.ts")
tslang_add_test(NAME test-compile-00-conditional-named-const-release COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00conditional_named_const_release.ts")
tslang_add_test(NAME test-compile-00-array-length-valid COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_length_valid.ts")
tslang_add_test(NAME test-compile-00-const-interface-of-local COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_interface_of_local.ts")
tslang_add_test(NAME test-compile-00-self-assign COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00self_assign.ts")
tslang_add_test(NAME test-compile-00-fresh-objects-folded COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00fresh_objects_folded.ts")
tslang_add_test(NAME test-compile-00-string-view-copy COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_view_copy.ts")
tslang_add_test(NAME test-compile-00-global-copy-counted COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_copy_counted.ts")
tslang_add_test(NAME test-compile-00-global-interface-initializer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_interface_initializer.ts")
tslang_add_test(NAME test-compile-00-interface-cast-unrelated COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_cast_unrelated.ts")
tslang_add_test(NAME test-compile-00-interface-union-cast-unrelated COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_union_cast_unrelated.ts")
tslang_add_test(NAME test-compile-00-c-string-copy COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00c_string_copy.ts")
tslang_add_test(NAME test-compile-00-const-field-read COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_field_read.ts")
tslang_add_test(NAME test-compile-00-const-element-read COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_element_read.ts")
tslang_add_test(NAME test-compile-00-const-global-read COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_global_read.ts")
tslang_add_test(NAME test-compile-00-global-argument-reassigned COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_argument_reassigned.ts")
tslang_add_test(NAME test-compile-00-global-argument-view COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_argument_view.ts")
tslang_add_test(NAME test-compile-00-bound-method-counts-object COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00bound_method_counts_object.ts")
tslang_add_test(NAME test-compile-00-closure-const-read COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00closure_const_read.ts")
tslang_add_test(NAME test-compile-00-element-store-after-value COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00element_store_after_value.ts")
tslang_add_test(NAME test-compile-00-const-interface-field-read COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_interface_field_read.ts")
tslang_add_test(NAME test-compile-00-argument-place-overwritten COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00argument_place_overwritten.ts")
tslang_add_test(NAME test-compile-00-const-bound-method-declared-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_bound_method_declared_type.ts")
tslang_add_test(NAME test-compile-00-nullable-class-to-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullable_class_to_interface.ts")
tslang_add_test(NAME test-compile-00-class-to-interface-null COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_to_interface_null.ts")
tslang_add_test(NAME test-compile-00-null-interface-to-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00null_interface_to_interface.ts")
tslang_add_test(NAME test-compile-00-interface-type-tag-in-vtable COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_type_tag_in_vtable.ts")
tslang_add_test(NAME test-compile-00-nullable-interface-to-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullable_interface_to_interface.ts")
tslang_add_test(NAME test-compile-00-argument-place-aliased COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00argument_place_aliased.ts")
tslang_add_test(NAME test-compile-00-temporary-read-in-finally COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00temporary_read_in_finally.ts")
tslang_add_test(NAME test-compile-00-unowned-local-store COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00unowned_local_store.ts")
tslang_add_test(NAME test-compile-00-array-argument-held COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_argument_held.ts")
tslang_add_test(NAME test-compile-00-argument-slot-assigned COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00argument_slot_assigned.ts")
tslang_add_test(NAME test-compile-00-array-data-string-alias COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_data_string_alias.ts")
tslang_add_test(NAME test-compile-00-arrays7 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array7.ts")
tslang_add_test(NAME test-compile-00-arrays8-tuple-spread COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array8_tuple_spread.ts")
tslang_add_test(NAME test-compile-00-arrays-assignment5 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_assignment5.ts")
tslang_add_test(NAME test-compile-00-array-of COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_of.ts")
tslang_add_test(NAME test-compile-00-array-conditional-access COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_cond_access.ts")
tslang_add_test(NAME test-compile-00-typed-array COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00typed_array.ts")
tslang_add_test(NAME test-compile-00-array-view COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_view.ts")
tslang_add_test(NAME test-compile-00-objects COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object.ts")
tslang_add_test(NAME test-compile-00-objects-global COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_global.ts")
tslang_add_test(NAME test-compile-00-objects-deconstruct COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_deconst.ts")
tslang_add_test(NAME test-compile-00-objects-functions COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_func.ts")
tslang_add_test(NAME test-compile-00-objects-functions-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_func2.ts")
tslang_add_test(NAME test-compile-00-objects-functions-3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_func3.ts")
tslang_add_test(NAME test-compile-00-objects-new COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_new.ts")
tslang_add_test(NAME test-compile-00-objects-accessor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_accessor.ts")
tslang_add_test(NAME test-compile-00-objects-boxed-infra COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_boxed_infra.ts")
tslang_add_test(NAME test-compile-00-objects-ref-semantics COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_ref_semantics.ts")
tslang_add_test(NAME test-compile-00-prefix-postfix COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00prefix_postfix.ts")
tslang_add_test(NAME test-compile-00-cond_expr COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00cond_expr.ts")
tslang_add_test(NAME test-compile-00-switch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch.ts")
tslang_add_test(NAME test-compile-01-switch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01switch.ts")
tslang_add_test(NAME test-compile-00-strings COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00strings.ts")
tslang_add_test(NAME test-compile-00-tuple COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple.ts")
tslang_add_test(NAME test-compile-01-tuple COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01tuple.ts")
tslang_add_test(NAME test-compile-00-tuple-named COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple_named.ts")
tslang_add_test(NAME test-compile-00-tuple-array COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple_with_array.ts")
tslang_add_test(NAME test-compile-00-computed-property-name COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00computedpropertyname.ts")
tslang_add_test(NAME test-compile-00-union-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_type.ts")
tslang_add_test(NAME test-compile-01-union-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01union_type.ts")
tslang_add_test(NAME test-compile-02-union-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02union_type.ts")
tslang_add_test(NAME test-compile-03-union-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/03union_type.ts")
tslang_add_test(NAME test-compile-03-union-type-case-order COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/03union_type_case_order.ts")
tslang_add_test(NAME test-compile-04-union-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/04union_type.ts")
tslang_add_test(NAME test-compile-05-union-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/05union_type.ts")
tslang_add_test(NAME test-compile-00-union-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_ops.ts")
tslang_add_test(NAME test-compile-00-union-bin-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_bin_ops.ts")
tslang_add_test(NAME test-compile-00-union-bin-ops-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_bin_ops2.ts")
tslang_add_test(NAME test-compile-00-union-to-any COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_to_any.ts")
tslang_add_test(NAME test-compile-00-union-errors COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_errors.ts")
tslang_add_test(NAME test-compile-00-union-null-to-string COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_to_string.ts")
tslang_add_test(NAME test-compile-00-union-null-truthiness COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_truthiness.ts")
tslang_add_test(NAME test-compile-00-null-into-tagged-union COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00null_into_tagged_union.ts")
tslang_add_test(NAME test-compile-00-union-null-in-pointer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_in_pointer.ts")
tslang_add_test(NAME test-compile-00-union-null-undefined COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_undefined.ts")
tslang_add_test(NAME test-compile-00-union-null-undefined-nonstrict COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_undefined_nonstrict.ts")
tslang_add_test(NAME test-compile-00-optional-number-compare COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional_number_compare.ts")
tslang_add_test(NAME test-compile-00-optional-loose-null COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional_loose_null.ts")
tslang_add_test(NAME test-compile-00-union-nullable-array COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_nullable_array.ts")
tslang_add_test(NAME test-compile-00-union-nullable-value COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_nullable_value.ts")
tslang_add_test(NAME test-compile-00-opaque-nullable-compare COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00opaque_nullable_compare.ts")
# in TSLANG_CORPUS too (rc, none), and under own here: a @cpp reference is counted by no model
tslang_add_test(NAME test-compile-00-cppvtable-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00cppvtable_class.ts")
tslang_add_test(NAME test-compile-own-00-cppvtable-class COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00cppvtable_class.ts")
tslang_add_test(NAME test-compile-own-00-cpp-slot COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_slot.ts")
tslang_add_test(NAME test-compile-own-00-cpp-overloads COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_overloads.ts")
tslang_add_test(NAME test-compile-00-cpp-slot COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_slot.ts")
tslang_add_test(NAME test-compile-00-cpp-overloads COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_overloads.ts")
tslang_add_test(NAME test-compile-00-union-value-equality COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_value_equality.ts")
tslang_add_test(NAME test-compile-00-strict-equals-mismatched-types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00strict_equals_mismatched_types.ts")
tslang_add_test(NAME test-compile-00-any-number-compare COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_number_compare.ts")
tslang_add_test(NAME test-compile-00-length-with-negative-merge COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00length_with_negative_merge.ts")
tslang_add_test(NAME test-compile-00-length-signed-arithmetic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00length_signed_arithmetic.ts")
tslang_add_test(NAME test-compile-00-optional-fields COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional_fields.ts")
tslang_add_test(NAME test-compile-00-generator-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_return.ts")
tslang_add_test(NAME test-compile-00-return-inference-undefined COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00return_inference_undefined.ts")
tslang_add_test(NAME test-compile-00-return-falls-off-end COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00return_falls_off_end.ts")
tslang_add_test(NAME test-compile-00-nullish-values COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullish_values.ts")
tslang_add_test(NAME test-compile-00-class-optional-fields COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_optional_fields.ts")
tslang_add_test(NAME test-compile-00-signless-int-signed COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00signless_int_signed.ts")
tslang_add_test(NAME test-compile-00-intersection-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00intersection_type.ts")
tslang_add_test(NAME test-compile-00-intersection-type-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00intersection_type_generic.ts")
tslang_add_test(NAME test-compile-00-length COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00length.ts")
tslang_add_test(NAME test-compile-00-undef COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00undef.ts")
tslang_add_test(NAME test-compile-00-str-null COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00str_null.ts")
tslang_add_test(NAME test-compile-00-for-in COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_in.ts")
tslang_add_test(NAME test-compile-00-for-of COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_of.ts")
tslang_add_test(NAME test-compile-00-sizeof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00sizeof.ts")
tslang_add_test(NAME test-compile-01-sizeof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01sizeof.ts")
tslang_add_test(NAME test-compile-02-sizeof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02sizeof.ts")
tslang_add_test(NAME test-compile-00-new-delete COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00new_delete.ts")
tslang_add_test(NAME test-compile-00-new-generic-from-receiver COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00new_generic_from_receiver.ts")
tslang_add_test(NAME test-compile-00-generic-specialization-cast COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_specialization_cast.ts")
tslang_add_test(NAME test-compile-00-owned-locals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_locals.ts")
# The JIT variants are the ones that matter here: this shape was a JIT-only miscompile until
# the catchable-type size was fixed (see the file header). Kept in all three models, since the
# bad size was model-independent.
tslang_add_test(NAME test-compile-00-try-using-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_using_catch.ts")
tslang_add_test(NAME test-compile-00-throw-in-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_in_catch.ts")
tslang_add_test(NAME test-compile-00-throw-inlined COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_inlined.ts")
tslang_add_test(NAME test-compile-00-try-catch-sequence-and-nesting COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_sequence_and_nesting.ts")
tslang_add_test(NAME test-compile-00-try-in-catch-with-finally COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_in_catch_with_finally.ts")
tslang_add_test(NAME test-compile-00-throw-number-untyped-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_number_untyped_catch.ts")
tslang_add_test(NAME test-compile-00-throw-in-block-debug-info COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_in_block_debug_info.ts")
tslang_add_test(NAME test-compile-00-catch-untyped-value COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_untyped_value.ts")
tslang_add_test(NAME test-compile-00-throw-bool-bigint COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_bool_bigint.ts")
tslang_add_test(NAME test-compile-00-int64-remainder COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00int64_remainder.ts")
tslang_add_test(NAME test-compile-00-throw-f32 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_f32.ts")
tslang_add_test(NAME test-compile-00-throw-u32 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_u32.ts")
tslang_add_test(NAME test-compile-00-cast-to-f32 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00cast_to_f32.ts")
tslang_add_test(NAME test-compile-00-instanceof-namespace-narrowing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof_namespace_narrowing.ts")
tslang_add_test(NAME test-compile-00-global-initializer-branches COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_initializer_branches.ts")
tslang_add_test(NAME test-compile-00-using-nested-scopes COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_nested_scopes.ts")
tslang_add_test(NAME test-compile-00-using-unwind-progress COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_unwind_progress.ts")
tslang_add_test(NAME test-compile-00-break-continue-scope-exit COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00break_continue_scope_exit.ts")
tslang_add_test(NAME test-compile-00-owned-fields COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_fields.ts")
tslang_add_test(NAME test-compile-00-owned-elements COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_elements.ts")
tslang_add_test(NAME test-compile-00-owned-literals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_literals.ts")
tslang_add_test(NAME test-compile-00-owned-array-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_array_ops.ts")
tslang_add_test(NAME test-compile-00-owned-inline-records COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_inline_records.ts")
tslang_add_test(NAME test-compile-00-owned-transfer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_transfer.ts")
tslang_add_test(NAME test-compile-00-owned-call-results COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_call_results.ts")
tslang_add_test(NAME test-compile-00-owned-named-consts COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_named_consts.ts")
tslang_add_test(NAME test-compile-00-owned-temporaries COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_temporaries.ts")
tslang_add_test(NAME test-compile-00-owned-interfaces COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_interfaces.ts")
tslang_add_test(NAME test-compile-00-owned-closures COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_closures.ts")
tslang_add_test(NAME test-compile-00-owned-any-boxing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_any_boxing.ts")
tslang_add_test(NAME test-compile-00-owned-strings COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_strings.ts")
tslang_add_test(NAME test-compile-00-string-length-resize COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_length_resize.ts")
tslang_add_test(NAME test-compile-00-narrow-array-null COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrow_array_null.ts")
tslang_add_test(NAME test-compile-00-array-length-grow-zeroed COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_length_grow_zeroed.ts")
tslang_add_test(NAME test-compile-00-rest-args-pack-owned COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00rest_args_pack_owned.ts")
tslang_add_test(NAME test-compile-00-switch-char-string-cases COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch_char_string_cases.ts")
tslang_add_test(NAME test-compile-00-switch-default-fallthrough COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch_default_fallthrough.ts")
tslang_add_test(NAME test-compile-00-generic-class-constructor-inference COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_class_constructor_inference.ts")
# A class inferred from its constructor's arguments is not also made with its defaults:
# `new Pairs([["one", 1]])` made all of Pairs<any, any> beside Pairs<string, si32>.
add_test(NAME test-generic-class-inference-no-default-specialization
         COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib
                 "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_class_constructor_inference.ts")
set_tests_properties(test-generic-class-inference-no-default-specialization
                     PROPERTIES PASS_REGULAR_EXPRESSION "Pairs<!ts[.]string,si32>[.]constructor"
                                FAIL_REGULAR_EXPRESSION "Pairs<!ts[.]any|error:|Stack dump")
tslang_add_test(NAME test-compile-00-generic-ref-inference COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_ref_inference.ts")
tslang_add_test(NAME test-compile-00-generator-done-value-zero COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_done_value_zero.ts")
tslang_add_test(NAME test-compile-00-generator-yield-value COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_yield_value.ts")
tslang_add_test(NAME test-compile-00-any-equality-undefined COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_equality_undefined.ts")
tslang_add_test(NAME test-compile-00-owned-generators COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generators.ts")
tslang_add_test(NAME test-compile-00-owned-globals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_globals.ts")
tslang_add_test(NAME test-compile-00-owned-generator-locals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generator_locals.ts")
tslang_add_test(NAME test-compile-00-owned-iteration COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_iteration.ts")
tslang_add_test(NAME test-compile-00-owned-async COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_async.ts")
tslang_add_test(NAME test-compile-00-owned-nested-captures COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_nested_captures.ts")
tslang_add_test(NAME test-compile-00-owned-unions COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_unions.ts")
tslang_add_test(NAME test-compile-00-owned-delete COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_delete.ts")
tslang_add_test(NAME test-compile-00-owned-early-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_early_return.ts")
tslang_add_test(NAME test-compile-00-owned-construct-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_construct_interface.ts")
tslang_add_test(NAME test-compile-00-void COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00void.ts")
tslang_add_test(NAME test-compile-00-in COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00in.ts")
tslang_add_test(NAME test-compile-00-in-method-names COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00in_method_names.ts")
tslang_add_test(NAME test-compile-00-class-structural-extends COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_structural_extends.ts")
tslang_add_test(NAME test-compile-00-global-constant-concat COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_constant_concat.ts")
tslang_add_test(NAME test-compile-00-global-object-literal COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_object_literal.ts")
tslang_add_test(NAME test-compile-00-instanceof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof.ts")
tslang_add_test(NAME test-compile-00-unbraced-branch-scope COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00unbraced_branch_scope.ts")
tslang_add_test(NAME test-compile-00-const-of-let COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_of_let.ts")
tslang_add_test(NAME test-compile-00-instanceof-union COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof_union.ts")
tslang_add_test(NAME test-compile-00-const-object-literal-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_object_literal_field.ts")
tslang_add_test(NAME test-compile-00-nullable-to-any COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullable_to_any.ts")
tslang_add_test(NAME test-compile-00-ternary-new-or-null COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ternary_new_or_null.ts")
tslang_add_test(NAME test-compile-00-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class.ts")
tslang_add_test(NAME test-compile-00-class-new COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_new.ts")
tslang_add_test(NAME test-compile-00-array-growth COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_growth.ts")
tslang_add_test(NAME test-compile-00-global-from-global COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_from_global.ts")
tslang_add_test(NAME test-compile-00-try-cleanup-then-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_cleanup_then_catch.ts")
tslang_add_test(NAME test-compile-00-try-empty-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_empty_catch.ts")
tslang_add_test(NAME test-compile-00-global-union-from-call COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_union_from_call.ts")
tslang_add_test(NAME test-compile-00-capture-in-new-arguments COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00capture_in_new_arguments.ts")
tslang_add_test(NAME test-compile-00-capture-in-new-untyped-arrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00capture_in_new_untyped_arrow.ts")
tslang_add_test(NAME test-compile-01-class-new COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01class_new.ts")
tslang_add_test(NAME test-compile-00-class-stack COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_stack.ts")
tslang_add_test(NAME test-compile-00-class-static COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static.ts")
tslang_add_test(NAME test-compile-00-class-static-block COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_block.ts")
tslang_add_test(NAME test-compile-00-class-discover-types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_discover_types.ts")
tslang_add_test(NAME test-compile-00-class-accessor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor.ts")
tslang_add_test(NAME test-compile-00-class-accessor-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor2.ts")
tslang_add_test(NAME test-compile-00-class-accessor-super COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_super.ts")
tslang_add_test(NAME test-compile-00-class-accessor-increment COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_increment.ts")
tslang_add_test(NAME test-compile-00-class-accessor-getter-only-abstract COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_getter_only_abstract.ts")
tslang_add_test(NAME test-compile-00-class-accessor-virtual COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_virtual.ts")
tslang_add_test(NAME test-compile-00-class-accessor-implements-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_implements_field.ts")
tslang_add_test(NAME test-compile-00-object-literal-contextual-lambda COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_literal_contextual_lambda.ts")
tslang_add_test(NAME test-compile-00-class-accessor-untyped-setter COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_untyped_setter.ts")
tslang_add_test(NAME test-compile-00-class-super COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super.ts")
tslang_add_test(NAME test-compile-00-class-implicit-constructor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_implicit_constructor.ts")
tslang_add_test(NAME test-compile-00-class-super-static COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super_static.ts")
tslang_add_test(NAME test-compile-00-class-default-constructor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_def_constr.ts")
tslang_add_test(NAME test-compile-00-class-constructor-fields COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_constr_fields.ts")
tslang_add_test(NAME test-compile-00-class-constructor-overloads COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_constructor_overloads.ts")
tslang_add_test(NAME test-compile-00-callback-result-ignored COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00callback_result_ignored.ts")
tslang_add_test(NAME test-compile-00-generic-default-no-inference COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_default_no_inference.ts")
tslang_add_test(NAME test-compile-00-promise-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00promise_type.ts")
tslang_add_test(NAME test-compile-00-class-static-constructor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_constr.ts")
tslang_add_test(NAME test-compile-00-class-abstract COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_abstract.ts")
tslang_add_test(NAME test-compile-00-class-override-inferred-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_override_inferred_return.ts")
tslang_add_test(NAME test-compile-00-class-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_iface.ts")
tslang_add_test(NAME test-compile-00-class-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic.ts")
tslang_add_test(NAME test-compile-00-class-generic-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic_iface.ts")
tslang_add_test(NAME test-compile-00-class-recursive COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_recursive.ts")
tslang_add_test(NAME test-compile-00-class-expression COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression.ts")
tslang_add_test(NAME test-compile-00-class-expression-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression2.ts")
tslang_add_test(NAME test-compile-00-class-expression-3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression3.ts")
tslang_add_test(NAME test-compile-00-class-expression-no-ctor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression_no_ctor.ts")
tslang_add_test(NAME test-compile-00-class-deconst COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_deconst.ts")
tslang_add_test(NAME test-compile-00-class-virtual-call COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_virtual_call.ts")
tslang_add_test(NAME test-compile-00-class-virtual-table COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_virtual_table.ts")
tslang_add_test(NAME test-compile-00-class-local-decl COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_local_decl.ts")
tslang_add_test(NAME test-compile-00-class-nested COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_nested.ts")
tslang_add_test(NAME test-compile-00-class-static-generic-method COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_generic_method.ts")
tslang_add_test(NAME test-compile-00-class-static-generic-method-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_generic_method2.ts")
tslang_add_test(NAME test-compile-00-class-static-generic-method-3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_generic_method3.ts")
tslang_add_test(NAME test-compile-00-class-generic-method COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic_method.ts")
tslang_add_test(NAME test-compile-00-class-generic-method-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic_method2.ts")
tslang_add_test(NAME test-compile-00-class-var-args COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_var_args.ts")
tslang_add_test(NAME test-compile-00-class-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_iterator.ts")
tslang_add_test(NAME test-compile-00-class-iterator-super COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_iterator_super.ts")
tslang_add_test(NAME test-compile-00-class-symbol-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_symbol_iterator.ts")
tslang_add_test(NAME test-compile-00-class-indexer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_indexer.ts")
tslang_add_test(NAME test-compile-00-class-indexer-super COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_indexer_super.ts")
tslang_add_test(NAME test-compile-00-class-from-tuple COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_from_tuple.ts")
tslang_add_test(NAME test-compile-00-class-access-control COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_access_control.ts")
tslang_add_test(NAME test-compile-00-class-protected-constructor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_protected_constr.ts")
tslang_add_test(NAME test-compile-00-to-primitive-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00to_primitive_ops.ts")
tslang_add_test(NAME test-compile-00-namespace COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns.ts")
tslang_add_test(NAME test-compile-00-namespace-enum COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns2.ts")
tslang_add_test(NAME test-compile-00-namespace-import COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns3.ts")
tslang_add_test(NAME test-compile-00-namespace-globals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns4.ts")
tslang_add_test(NAME test-compile-00-namespace-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns5.ts")
tslang_add_test(NAME test-compile-00-namespace-bug COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns_bug.ts")
tslang_add_test(NAME test-compile-00-namespace-member-shadowed COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns_member_shadowed.ts")
tslang_add_test(NAME test-compile-00-interface-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface.ts")
tslang_add_test(NAME test-compile-00-interface-class-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface2.ts")
tslang_add_test(NAME test-compile-00-interface-object COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object.ts")
tslang_add_test(NAME test-compile-00-interface-object-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object2.ts")
tslang_add_test(NAME test-compile-00-interface-object-3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object3.ts")
tslang_add_test(NAME test-compile-00-interface-object-4 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object4.ts")
tslang_add_test(NAME test-compile-00-interface-object-5 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object5.ts")
tslang_add_test(NAME test-compile-00-interface-global-method COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_global_method.ts")
tslang_add_test(NAME test-compile-00-interface-conjunction COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_conjunction.ts")
tslang_add_test(NAME test-compile-00-interface-partial COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_partial.ts")
tslang_add_test(NAME test-compile-00-interface-optional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional.ts")
tslang_add_test(NAME test-compile-00-interface-optional-cast-order COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_cast_order.ts")
tslang_add_test(NAME test-compile-00-interface-function-typed-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field.ts")
tslang_add_test(NAME test-compile-00-interface-function-typed-field-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field_class.ts")
tslang_add_test(NAME test-compile-00-interface-function-typed-field-class-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field_class_field.ts")
tslang_add_test(NAME test-compile-00-interface-function-field-write COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_field_write.ts")
tslang_add_test(NAME test-compile-own-00-interface-function-field-write COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_field_write.ts")
tslang_add_test(NAME test-compile-00-interface-function-typed-field-object-closure COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field_object_closure.ts")
tslang_add_test(NAME test-compile-00-class-function-typed-field-inferred COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_function_typed_field_inferred.ts")
tslang_add_test(NAME test-compile-00-interface-optional-function-typed-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_function_typed_field.ts")
tslang_add_test(NAME test-compile-00-interface-optional-function-typed-field-class-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_function_typed_field_class_field.ts")
tslang_add_test(NAME test-compile-00-interface-captures COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_captures.ts")
tslang_add_test(NAME test-compile-00-object-annotated-method COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method.ts")
tslang_add_test(NAME test-compile-00-object-annotated-method-params COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_params.ts")
tslang_add_test(NAME test-compile-00-object-chained-sibling-method-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_chained_sibling_method_return.ts")
tslang_add_test(NAME test-compile-00-object-annotated-method-interleaved COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_interleaved.ts")
tslang_add_test(NAME test-compile-00-object-annotated-method-extends-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_extends_interface.ts")
tslang_add_test(NAME test-compile-00-object-annotated-method-extends-interface-multilevel COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_extends_interface_multilevel.ts")
tslang_add_test(NAME test-compile-00-interface-optional-extends COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_extends.ts")
tslang_add_test(NAME test-compile-00-interface-optional-method-extends COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_method_extends.ts")
tslang_add_test(NAME test-compile-00-interface-object-array COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object_array.ts")
tslang_add_test(NAME test-compile-00-interface-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_generic.ts")
tslang_add_test(NAME test-compile-00-interface-new COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_new.ts")
tslang_add_test(NAME test-compile-00-interface-indexer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_indexer.ts")
tslang_add_test(NAME test-compile-00-class-or-interface-to-tuple COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_or_interface_to_tuple.ts")
tslang_add_test(NAME test-compile-00-any COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any.ts")
tslang_add_test(NAME test-compile-00-any-unbox-interface-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_unbox_interface_class.ts")
tslang_add_test(NAME test-compile-00-any-to-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_to_interface.ts")
tslang_add_test(NAME test-compile-own-00-any-to-interface COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_to_interface.ts")
tslang_add_test(NAME test-compile-00-interface-instanceof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_instanceof.ts")
tslang_add_test(NAME test-compile-01-any COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01any.ts")
tslang_add_test(NAME test-compile-00-any-types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_types.ts")
tslang_add_test(NAME test-compile-00-any-compare COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_compare.ts")
tslang_add_test(NAME test-compile-00-any-truthiness-objects COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_truthiness_objects.ts")
tslang_add_test(NAME test-compile-00-any-generic-equals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_generic_equals.ts")
tslang_add_test(NAME test-compile-00-generator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator.ts")
tslang_add_test(NAME test-compile-00-generator-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator2.ts")
tslang_add_test(NAME test-compile-00-generator-3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator3.ts")
tslang_add_test(NAME test-compile-00-generator-4 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator4.ts")
tslang_add_test(NAME test-compile-00-generator-5 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator5.ts")
tslang_add_test(NAME test-compile-00-generator-6 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator6.ts")
tslang_add_test(NAME test-compile-00-generator-7 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator7.ts")
tslang_add_test(NAME test-compile-00-safe-cast COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast.ts")
tslang_add_test(NAME test-compile-00-safe-cast-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast2.ts")
tslang_add_test(NAME test-compile-00-safe-cast-typeof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_typeof.ts")
tslang_add_test(NAME test-compile-00-safe-cast-while COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_while.ts")
tslang_add_test(NAME test-compile-01-safe-cast-while COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01safe_cast_while.ts")
tslang_add_test(NAME test-compile-00-safe-cast-field-access COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_field_access.ts")
tslang_add_test(NAME test-compile-00-safe-cast-early-exit COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_early_exit.ts")
tslang_add_test(NAME test-compile-00-union-object-types-access COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_object_types_access.ts")
tslang_add_test(NAME test-compile-00-safe-cast-null-field COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_null_field.ts")
tslang_add_test(NAME test-compile-00-safe-cast-else-scope COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_else_scope.ts")
tslang_add_test(NAME test-compile-00-safe-cast-bug COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_bug.ts")
tslang_add_test(NAME test-compile-00-optional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional.ts")
tslang_add_test(NAME test-compile-01-optional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01optional.ts")
tslang_add_test(NAME test-compile-00-async-await COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00async_await.ts")
tslang_add_test(NAME test-compile-00-async-result-types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00async_result_types.ts")
tslang_add_test(NAME test-compile-00-async-gc-threading COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00async_gc_threading.ts")
tslang_add_test(NAME test-compile-00-for-await COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await.ts")

tslang_add_test(NAME test-compile-00-for-await-yield COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await_yield.ts")
tslang_add_test(NAME test-compile-00-types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types.ts")
tslang_add_test(NAME test-compile-00-try-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch.ts")
tslang_add_test(NAME test-compile-00-catch-value COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_value.ts")
tslang_add_test(NAME test-compile-00-catch-value-minimal COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_value_minimal.ts")
tslang_add_test(NAME test-compile-01-try-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_catch.ts")
tslang_add_test(NAME test-compile-00-try-catch-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_return.ts")
tslang_add_test(NAME test-compile-00-try-finally COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally.ts")
tslang_add_test(NAME test-compile-00-try-finally-throws COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_throws.ts")
tslang_add_test(NAME test-compile-00-function-fewer-parameters COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00function_fewer_parameters.ts")
tslang_add_test(NAME test-compile-00-closure-fewer-parameters COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00closure_fewer_parameters.ts")
tslang_add_test(NAME test-compile-01-try-finally COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_finally.ts")
tslang_add_test(NAME test-compile-00-try-finally-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_return.ts")
tslang_add_test(NAME test-compile-00-try-finally-funclet COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_funclet.ts")
tslang_add_test(NAME test-compile-00-try-finally-break-continue COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_break_continue.ts")
tslang_add_test(NAME test-compile-00-try-catch-rethrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_rethrow.ts")
tslang_add_test(NAME test-compile-00-try-catch-accessor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_accessor.ts")
tslang_add_test(NAME test-compile-00-try-catch-call-in-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_call_in_catch.ts")
tslang_add_test(NAME test-compile-00-try-catch-mismatch-rethrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_mismatch_rethrow.ts")
tslang_add_test(NAME test-compile-00-try-catch-return-dispose COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_return_dispose.ts")
tslang_add_test(NAME test-compile-01-try-catch-return-dispose COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_catch_return_dispose.ts")
tslang_add_test(NAME test-compile-00-property-access-conditional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00property_access_cond.ts")
tslang_add_test(NAME test-compile-00-method-access-conditional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00method_access_cond.ts")
tslang_add_test(NAME test-compile-00-method-this-param COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00method_this_param.ts")
tslang_add_test(NAME test-compile-01-method-access-conditional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01method_access_cond.ts")
tslang_add_test(NAME test-compile-00-question-question COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00question_question.ts")
tslang_add_test(NAME test-compile-00-spread-assignment COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread_assignment.ts")
tslang_add_test(NAME test-compile-01-spread-assignment COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread_assignment.ts")
tslang_add_test(NAME test-compile-00-spread COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread.ts")
tslang_add_test(NAME test-compile-01-spread COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread.ts")
tslang_add_test(NAME test-compile-00-types-unknown1 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_unknown1.ts")
tslang_add_test(NAME test-compile-00-any-to-tuple COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_to_tuple.ts")
tslang_add_test(NAME test-compile-00-bigint-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00bigint_ops.ts")
tslang_add_test(NAME test-compile-00-namespace-forward-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00namespace_forward_class.ts")
tslang_add_test(NAME test-compile-00-any-literal-to-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_literal_to_interface.ts")
tslang_add_test(NAME test-compile-00-interface-method-object-field-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_method_object_field_type.ts")
tslang_add_test(NAME test-compile-00-unknown-round-trip COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00unknown_round_trip.ts")
tslang_add_test(NAME test-compile-00-types-indexed-access-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_indexedaccesstype.ts")
tslang_add_test(NAME test-compile-00-types-mapped-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_mappedtype.ts")
tslang_add_test(NAME test-compile-01-types-mapped-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_mappedtype.ts")
tslang_add_test(NAME test-compile-00-types-template-literal-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_templateliteraltype.ts")
tslang_add_test(NAME test-compile-00-template-literal-escapes COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00template_literal_escapes.ts")
tslang_add_test(NAME test-compile-01-types-template-literal-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_templateliteraltype.ts")
tslang_add_test(NAME test-compile-00-types-utility COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_utility.ts")
tslang_add_test(NAME test-compile-01-types-utility COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_utility.ts")
tslang_add_test(NAME test-compile-01-types-key-of-enum COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_keyof_enum.ts")
tslang_add_test(NAME test-compile-00-type-alias-in-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00type_aliases_in_generics.ts")
tslang_add_test(NAME test-compile-00-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00iterator.ts")
tslang_add_test(NAME test-compile-01-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01iterator.ts")
tslang_add_test(NAME test-compile-02-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02iterator.ts")
tslang_add_test(NAME test-compile-03-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/03iterator.ts")
tslang_add_test(NAME test-compile-00-iterator-bug COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00iterator_bug.ts")
tslang_add_test(NAME test-compile-00-map COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00map.ts")
tslang_add_test(NAME test-compile-00-iterator-methods COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00iterator_methods.ts")
tslang_add_test(NAME test-compile-01-map COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01map.ts")
tslang_add_test(NAME test-compile-00-filter COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00filter.ts")
tslang_add_test(NAME test-compile-00-reduce COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00reduce.ts")
tslang_add_test(NAME test-compile-01-reduce COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01reduce.ts")
tslang_add_test(NAME test-compile-00-every COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00every.ts")
tslang_add_test(NAME test-compile-00-extension COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00extension.ts")
tslang_add_test(NAME test-compile-01-extension COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01extension.ts")
tslang_add_test(NAME test-compile-00-extension-cond-access COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00extension_cond_access.ts")
tslang_add_test(NAME test-compile-00-infer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00infer.ts")
tslang_add_test(NAME test-compile-00-extends COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00extends.ts")
tslang_add_test(NAME test-compile-01-extends COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01extends.ts")
tslang_add_test(NAME test-compile-02-extends COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02extends.ts")
tslang_add_test(NAME test-compile-00-symbol COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00symbol.ts")
tslang_add_test(NAME test-compile-01-symbol COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01symbol.ts")
tslang_add_test(NAME test-compile-00-as COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00as.ts")
tslang_add_test(NAME test-compile-00-as-const COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00as_const.ts")
tslang_add_test(NAME test-compile-00-type-guard-function COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00type_guard_function.ts")
tslang_add_test(NAME test-compile-00-class-cast-structural COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_cast_structural.ts")
tslang_add_test(NAME test-compile-00-names-conflict COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00names_conflict.ts")
tslang_add_test(NAME test-compile-00-generic-arguments-name-conflict COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_arguments_name_conflict.ts")
tslang_add_test(NAME test-compile-00-decorators COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00decorators.ts")
tslang_add_test(NAME test-compile-00-conditional-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00conditional_type.ts")
tslang_add_test(NAME test-compile-00-disposable COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00disposable.ts")
tslang_add_test(NAME test-compile-01-disposable COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01disposable.ts")
tslang_add_test(NAME test-compile-02-disposable COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02disposable.ts")
tslang_add_test(NAME test-compile-03-disposable COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/03disposable.ts")
tslang_add_test(NAME test-compile-04-disposable COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/04disposable.ts")
tslang_add_test(NAME test-compile-00-if-conditional-compile COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00if_conditional_compile.ts")
tslang_add_test(NAME test-compile-01-arguments COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01arguments.ts")
tslang_add_test(NAME test-compile-02-numbers COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02numbers.ts")
tslang_add_test(NAME test-compile-05-strings COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/05strings.ts")
tslang_add_test(NAME test-compile-06-numbercollections COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/06numbercollections.ts")
tslang_add_test(NAME test-compile-07-stringcollections COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/07stringcollections.ts")
tslang_add_test(NAME test-compile-08-stringopertations COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/08stringopertations.ts")
tslang_add_test(NAME test-compile-09-postprefix COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/09postprefix.ts")
tslang_add_test(NAME test-compile-10-arrayincrement COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/10arrayincrement.ts")
tslang_add_test(NAME test-compile-11-equalsoperator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/11equalsoperator.ts")
tslang_add_test(NAME test-compile-12-referencecollection COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/12referencecollection.ts")
tslang_add_test(NAME test-compile-13-actions.ts COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/13actions.ts")
tslang_add_test(NAME test-compile-14-lazyoperations COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/14lazyoperations.ts")
tslang_add_test(NAME test-compile-15-references COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/15references.ts")
tslang_add_test(NAME test-compile-15-references-capture COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/15references_capture.ts")
tslang_add_test(NAME test-compile-17-classes COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/17classes.ts")
tslang_add_test(NAME test-compile-18-enums COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/18enums.ts")
tslang_add_test(NAME test-compile-19-forof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/19forof.ts")
tslang_add_test(NAME test-compile-19-forof-capture COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/19forof_capture.ts")
tslang_add_test(NAME test-compile-20-maps COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/20maps.ts")
tslang_add_test(NAME test-compile-22-lambdas COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/22lambdas.ts")
tslang_add_test(NAME test-compile-23-generics COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/23generics.ts")
tslang_add_test(NAME test-compile-25-lamdacapture COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/25lamdacapture.ts")
tslang_add_test(NAME test-compile-26-staticclasses COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/26staticclasses.ts")
tslang_add_test(NAME test-compile-28-boolcasts COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/28boolcasts.ts")
tslang_add_test(NAME test-compile-29-lazyreferences COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/29lazyreferences.ts")
tslang_add_test(NAME test-compile-32-complexcalls COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/32complexcalls.ts")
tslang_add_test(NAME test-compile-33-inheritance COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/33inheritance.ts")
tslang_add_test(NAME test-compile-34-switch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/34switch.ts")
tslang_add_test(NAME test-compile-35-lambdaparameters COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/35lambdaparameters.ts")
tslang_add_test(NAME test-compile-36-interfaces COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/36interfaces.ts")
tslang_add_test(NAME test-compile-39-objectdestructuring COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/39objectdestructuring.ts")
tslang_add_test(NAME test-compile-40-generics.ts COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/40generics.ts")
tslang_add_test(NAME test-compile-41-anonymoustypes COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/41anonymoustypes.ts")
tslang_add_test(NAME test-compile-42-lambdaproperties COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/42lambdaproperties.ts")
tslang_add_test(NAME test-compile-43-nestednamespace COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/43nestednamespace.ts")
tslang_add_test(NAME test-compile-44-toplevelcode COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/44toplevelcode.ts")
# 44toplevelcode.ts has top-level declarations but wraps its actual test logic in an
# explicit main() - these two exercise executable statements running directly at root
# scope with NO main() anywhere (generateGlobalEntryCode compiles the root statements
# AS the synthesized `main` itself): control flow (for/while/switch/try-catch-finally)
# plus a closure capturing a root-level `let` (module-scope storage, not a stack local).
tslang_add_test(NAME test-compile-00-toplevel-control-flow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00toplevel_control_flow.ts")
tslang_add_test(NAME test-compile-00-toplevel-no-main-with-helpers COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00toplevel_no_main_with_helpers.ts")
tslang_add_test(NAME test-compile-45-enumtostring COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/45enumtostring.ts")
tslang_add_test(NAME test-compile-48-instanceof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/48instanceof.ts")
tslang_add_test(NAME test-compile-51-exceptions COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/51exceptions.ts")
tslang_add_test(NAME test-compile-241-arrayforeach COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/241arrayforeach.ts")
tslang_add_test(NAME test-compile-243-arrayevery COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/243arrayevery.ts")
tslang_add_test(NAME test-compile-244-arraysome COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/244arraysome.ts")
tslang_add_test(NAME test-compile-Grammar_and_types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/Grammar_and_types.ts")
tslang_add_test(NAME test-compile-StackTest COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00stack_test.ts")
if (WIN32)
tslang_add_test(NAME test-compile-Raytrace COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/raytrace.ts")
endif()
tslang_add_test(NAME test-compile-Path COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/path.ts")
tslang_add_test(NAME test-compile-Print-Bug-01 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01print-bug.ts")
tslang_add_test(NAME test-compile-arrayLiterals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayLiterals.ts")
tslang_add_test(NAME test-compile-arrayLiterals-2-ES5 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayLiterals2ES5.ts")
tslang_add_test(NAME test-compile-arrayLiterals-3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayLiterals3.ts")
tslang_add_test(NAME test-compile-Abstract-Property-In-Constructor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/abstractPropertyInConstructor.ts")
tslang_add_test(NAME test-compile-Dependencies COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/dependencies.ts")
tslang_add_test(NAME test-compile-ArrayFakeFlatNoCrashInferenceDeclarations COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayFakeFlatNoCrashInferenceDeclarations.ts")
tslang_add_test(NAME test-compile-noPropertyAccessFromIndexSignature1 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/noPropertyAccessFromIndexSignature1.ts")
tslang_add_test(NAME test-compile-asyncMethodWithSuper_es2017 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/asyncMethodWithSuper_es2017.ts")
tslang_add_test(NAME test-compile-assignmentTypeNarrowing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/assignmentTypeNarrowing.ts")
tslang_add_test(NAME test-compile-additionOperatorWithNumberAndEnum COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/additionOperatorWithNumberAndEnum.ts")
tslang_add_test(NAME test-compile-arithmeticOperatorWithEnum COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/arithmeticOperatorWithEnum.ts")
tslang_add_test(NAME test-compile-arithmeticOperatorWithTypeParameter COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/arithmeticOperatorWithTypeParameter.ts")
tslang_add_test(NAME test-compile-comparisonOperatorWithIdenticalObjects COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/comparisonOperatorWithIdenticalObjects.ts")
tslang_add_test(NAME test-compile-callWithSpread COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/callWithSpread.ts")
tslang_add_test(NAME test-compile-newWithSpread COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/newWithSpread.ts")
tslang_add_test(NAME test-compile-typeGuardOfFormThisMember COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/typeGuardOfFormThisMember.ts")
tslang_add_test(NAME test-compile-typeGuardOfFormTypeOfBoolean COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/typeGuardOfFormTypeOfBoolean.ts")
tslang_add_test(NAME test-compile-typeGuardFunction COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/typeGuardFunction.ts")
tslang_add_test(NAME test-compile-thisTypeInClasses COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/thisTypeInClasses.ts")
tslang_add_test(NAME test-compile-conditionalTypes1 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/conditionalTypes1.ts")
tslang_add_test(NAME test-compile-conditionalTypes2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/conditionalTypes2.ts")
tslang_add_test(NAME test-compile-types-vs-interfaces COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/types_vs_interfaces.ts")
tslang_add_test(NAME test-compile-structural-typing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/structural-typing.ts")
tslang_add_test(NAME test-compile-declarationsAndAssignments COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/declarationsAndAssignments.ts")
tslang_add_test(NAME test-compile-disallowLineTerminatorBeforeArrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/disallowLineTerminatorBeforeArrow.ts")
tslang_add_test(NAME test-compile-emitDefaultParametersFunctionExpression COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/emitDefaultParametersFunctionExpression.ts")
tslang_add_test(NAME test-compile-logicalAssignment5 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/logicalAssignment5.ts")
tslang_add_test(NAME test-compile-logicalAssignment5-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/logicalAssignment5_2.ts")
tslang_add_test(NAME test-compile-logicalAssignment5-3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/logicalAssignment5_3.ts")
tslang_add_test(NAME test-compile-nbody COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/nbody.ts")
tslang_add_test(NAME test-compile-no-main COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/no_main.ts")
tslang_add_test(NAME test-compile-parser COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/parser.ts")
tslang_add_test(NAME test-compile-internals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/internals.ts")
tslang_add_test(NAME test-compile-load-store-decorators COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/load_store_decorators.ts")
tslang_add_test(NAME test-compile-reference-null-bug COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00reference_null_bug.ts")
tslang_add_test(NAME test-compile-reference-index-bug COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00reference_index_bug.ts")
# present on disk but never wired into any add_test - found while auditing for orphaned
# test files; compiles and passes cleanly in both tiers, exercises Ref<T>/Deref semantics
tslang_add_test(NAME test-compile-reference-ref-deref COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00reference_ref_deref.ts")
tslang_add_test(NAME test-compile-uint-compare-bug COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00uint_compare_bug.ts")
tslang_add_test(NAME test-compile-gc-malloc-cse-o3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/gc_malloc_cse_o3.ts")
tslang_add_test(NAME test-compile-fast-math COMMAND test-runner -fast-math "${PROJECT_SOURCE_DIR}/test/tester/tests/fast_math.ts")

tslang_add_test(NAME test-jit-00-print COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00print.ts")
tslang_add_test(NAME test-jit-00-assert COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00assert.ts")
tslang_add_test(NAME test-jit-00-enums COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00enum.ts")
tslang_add_test(NAME test-jit-00-enums-multiple COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00enum_multiple.ts")
tslang_add_test(NAME test-jit-01-enums COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01enum.ts")
tslang_add_test(NAME test-jit-00-numbers COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00numbers.ts")
tslang_add_test(NAME test-jit-00-equals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00equals.ts")
tslang_add_test(NAME test-jit-00-bool-arith-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00bool_arith_ops.ts")
tslang_add_test(NAME test-jit-00-int64-shift-bitwise COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00int64_shift_bitwise.ts")
tslang_add_test(NAME test-jit-00-int64-mul-literal COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00int64_mul_literal.ts")
tslang_add_test(NAME test-jit-00-int-literal-above-s32 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00int_literal_above_s32.ts")
tslang_add_test(NAME test-jit-00-literal-union-arith COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00literal_union_arith.ts")
tslang_add_test(NAME test-jit-00-literal-union-cast COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00literal_union_cast.ts")
tslang_add_test(NAME test-jit-00-narrowing-and-or COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrowing_and_or.ts")
tslang_add_test(NAME test-jit-00-enum-compare COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00enum_compare.ts")
tslang_add_test(NAME test-jit-00-bool-const-array COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00bool_const_array.ts")
tslang_add_test(NAME test-jit-00-mixed-type-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00mixed_type_ops.ts")
tslang_add_test(NAME test-jit-00-generator-manual-next COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next.ts")
tslang_add_test(NAME test-jit-00-generator-manual-next-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next2.ts")
tslang_add_test(NAME test-jit-00-global-const-object-method COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_const_object_method.ts")
tslang_add_test(NAME test-jit-00-funcs COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs.ts")
tslang_add_test(NAME test-jit-00-funcs-capture COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_capture.ts")
tslang_add_test(NAME test-jit-00-funcs-vararg COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_vararg.ts")
tslang_add_test(NAME test-jit-00-declare-function-twice COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00declare_function_twice.ts")
tslang_add_test(NAME test-jit-00-function-overloads COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00function_overloads.ts")
tslang_add_test(NAME test-jit-00-gc-allocations-distinct COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00gc_allocations_distinct.ts")
tslang_add_test(NAME test-jit-00-assert-message COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00assert_message.ts")
tslang_add_test(NAME test-jit-01-funcs-vararg COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_vararg.ts")
tslang_add_test(NAME test-jit-02-funcs-vararg COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02funcs_vararg.ts")
tslang_add_test(NAME test-jit-00-funcs-bindings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_bindings.ts")
tslang_add_test(NAME test-jit-00-funcs-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic.ts")
tslang_add_test(NAME test-jit-01-funcs-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_generic.ts")
tslang_add_test(NAME test-jit-00-funcs-generic-typeof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_with_typeof.ts")
tslang_add_test(NAME test-jit-00-funcs-generic-arrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_arrow.ts")
tslang_add_test(NAME test-jit-00-funcs-typeless-function-as-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_typeless_function_as_generic.ts")
tslang_add_test(NAME test-jit-00-nested-implicit-generics COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_implicit_generics.ts")
tslang_add_test(NAME test-jit-00-funcs-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_iterator.ts")
tslang_add_test(NAME test-jit-01-funcs-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_generic_iterator.ts")
tslang_add_test(NAME test-jit-02-funcs-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02funcs_generic_iterator.ts")
tslang_add_test(NAME test-jit-00-funcs-typed-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_typed_generic_iterator.ts")
tslang_add_test(NAME test-jit-00-funcs-nesting COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting.ts")
tslang_add_test(NAME test-jit-00-funcs-nesting-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_generic.ts")
tslang_add_test(NAME test-jit-00-funcs-nesting-capture COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_capture.ts")
tslang_add_test(NAME test-jit-00-funcs-hybrid-null-this COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_hybrid_null_this.ts")
tslang_add_test(NAME test-jit-00-funcs-c-varargs-storage COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_c_varargs_storage.ts")
tslang_add_test(NAME test-jit-00-typeof-static-fold COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_static_fold.ts")
tslang_add_test(NAME test-jit-00-typeof-function-narrowing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_function_narrowing.ts")
tslang_add_test(NAME test-jit-00-typeof-static-fold-conditions COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_static_fold_conditions.ts")
tslang_add_test(NAME test-jit-00-typeof-union-narrowing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00typeof_union_narrowing.ts")
tslang_add_test(NAME test-jit-00-const-array-to-array-elements COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_to_array_elements.ts")
tslang_add_test(NAME test-jit-00-array-widen-elements-now COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_widen_elements_now.ts")
tslang_add_test(NAME test-jit-00-const-array-nested-elements COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_nested_elements.ts")
tslang_add_test(NAME test-jit-00-const-array-mutation COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_mutation.ts")
tslang_add_test(NAME test-jit-00-nested-array-literal-mutation COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_array_literal_mutation.ts")
tslang_add_test(NAME test-jit-00-array-move-wide-elements COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_move_wide_elements.ts")
tslang_add_test(NAME test-jit-00-funcs-expression-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_expression_generic.ts")
tslang_add_test(NAME test-jit-00-funcs-expression-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_expression_iterator.ts")
tslang_add_test(NAME test-jit-00-arrow-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00arrow_generic.ts")
tslang_add_test(NAME test-jit-00-arrow-in-arrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00arrow_in_arrow.ts")
tslang_add_test(NAME test-jit-00-arrow-global COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00arrow_global.ts")
tslang_add_test(NAME test-jit-00-lambdas COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00lambdas.ts")
tslang_add_test(NAME test-jit-01-lambdas COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01lambdas.ts")
tslang_add_test(NAME test-jit-00-lambdas-gemeric-global COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00lambdas_generic_global.ts")
tslang_add_test(NAME test-jit-00-if_return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00if_return.ts")
tslang_add_test(NAME test-jit-00-dowhile COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00dowhile.ts")
tslang_add_test(NAME test-jit-00-while COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00while.ts")
tslang_add_test(NAME test-jit-00-for COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for.ts")
tslang_add_test(NAME test-jit-00-for-infinite-result COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_infinite_result.ts")
tslang_add_test(NAME test-jit-00-break-continue COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00break_continue.ts")
tslang_add_test(NAME test-jit-00-label-chain-loop COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00label_chain_loop.ts")
tslang_add_test(NAME test-jit-00-labelled-scope-exit COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00labelled_scope_exit.ts")
tslang_add_test(NAME test-jit-00-switch-case-scope-exit COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch_case_scope_exit.ts")
tslang_add_test(NAME test-jit-00-using-dispose-order COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_dispose_order.ts")
tslang_add_test(NAME test-jit-00-using-unwind-outer-locals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_unwind_outer_locals.ts")
tslang_add_test(NAME test-jit-00-try-unwind-outer-locals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_unwind_outer_locals.ts")
tslang_add_test(NAME test-jit-00-try-return-through-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_return_through_finally.ts")
tslang_add_test(NAME test-jit-00-try-scope-exit-through-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_scope_exit_through_finally.ts")
tslang_add_test(NAME test-jit-00-try-jump-from-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_jump_from_catch.ts")
tslang_add_test(NAME test-jit-00-vars COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00vars.ts")
tslang_add_test(NAME test-jit-00-var-bindings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00var_bindings.ts")
tslang_add_test(NAME test-jit-00-globals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals.ts")
tslang_add_test(NAME test-jit-00-globals-entry-point COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals_entry_point.ts")
tslang_add_test(NAME test-jit-00-main-argc-argv COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00main_argc_argv.ts")
# JIT only: a compiled `main(argc: number): number` is called by the C runtime with an int argc
# where it reads a double, and its double result is not an exit code
tslang_add_test(NAME test-jit-00-main-number-result COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00main_number_result.ts")
tslang_add_test(NAME test-jit-00-globals2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals2.ts")
tslang_add_test(NAME test-jit-00-globals3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals3.ts")
tslang_add_test(NAME test-jit-00-arrays COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array.ts")
tslang_add_test(NAME test-jit-00-arrays2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array2.ts")
tslang_add_test(NAME test-jit-00-array-reference COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_reference.ts")
tslang_add_test(NAME test-jit-00-arrays3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array3.ts")
tslang_add_test(NAME test-jit-00-array-zero-slots COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_zero_slots.ts")
tslang_add_test(NAME test-jit-00-arrays4-push-pop COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array4_push_pop.ts")
tslang_add_test(NAME test-jit-00-array-shift COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_shift.ts")
tslang_add_test(NAME test-jit-00-array-splice COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_splice.ts")
tslang_add_test(NAME test-jit-00-const-array-push COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_array_push.ts")
tslang_add_test(NAME test-jit-00-array-spread-args COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_spread_args.ts")
tslang_add_test(NAME test-jit-00-arrays5-deconstruct COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array5_deconst.ts")
tslang_add_test(NAME test-jit-00-array-rest-copy COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_rest_copy.ts")
tslang_add_test(NAME test-jit-00-arrays6 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array6.ts")
tslang_add_test(NAME test-jit-00-array-static-nested COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_static_nested.ts")
tslang_add_test(NAME test-jit-00-array-literal-copy COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_literal_copy.ts")
tslang_add_test(NAME test-jit-00-tuple-runtime-arrays COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple_runtime_arrays.ts")
tslang_add_test(NAME test-jit-00-record-to-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00record_to_class.ts")
tslang_add_test(NAME test-jit-00-record-variable-to-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00record_variable_to_class.ts")
tslang_add_test(NAME test-jit-00-array-insert-functions COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_insert_functions.ts")
tslang_add_test(NAME test-jit-00-await-void COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00await_void.ts")
tslang_add_test(NAME test-jit-00-for-await-sequential COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await_sequential.ts")
tslang_add_test(NAME test-jit-00-for-await-control COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await_control.ts")
tslang_add_test(NAME test-jit-00-for-optional-class-condition COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_optional_class_condition.ts")
tslang_add_test(NAME test-jit-00-for-condition-narrowing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_condition_narrowing.ts")
tslang_add_test(NAME test-jit-00-param-assigned-owned COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00param_assigned_owned.ts")
tslang_add_test(NAME test-jit-00-string-empty-falsy COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_empty_falsy.ts")
tslang_add_test(NAME test-jit-00-string-concat-bounds COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_concat_bounds.ts")
tslang_add_test(NAME test-jit-00-parse-number-range COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00parse_number_range.ts")
tslang_add_test(NAME test-jit-00-global-argument-held COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_argument_held.ts")
tslang_add_test(NAME test-jit-00-shared-global-field-argument COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00shared_global_field_argument.ts")
tslang_add_test(NAME test-jit-00-narrowed-assign-other-member COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrowed_assign_other_member.ts")
tslang_add_test(NAME test-jit-00-bound-method-count COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00bound_method_count.ts")
tslang_add_test(NAME test-jit-00-const-view-of-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_view_of_field.ts")
tslang_add_test(NAME test-jit-00-const-view-of-element COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_view_of_element.ts")
tslang_add_test(NAME test-jit-00-const-record-owned-fields COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_record_owned_fields.ts")
tslang_add_test(NAME test-jit-00-generator-temporaries COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_temporaries.ts")
tslang_add_test(NAME test-jit-00-conditional-named-const-release COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00conditional_named_const_release.ts")
tslang_add_test(NAME test-jit-00-array-length-valid COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_length_valid.ts")
tslang_add_test(NAME test-jit-00-const-interface-of-local COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_interface_of_local.ts")
tslang_add_test(NAME test-jit-00-self-assign COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00self_assign.ts")
tslang_add_test(NAME test-jit-00-fresh-objects-folded COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00fresh_objects_folded.ts")
tslang_add_test(NAME test-jit-00-string-view-copy COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_view_copy.ts")
tslang_add_test(NAME test-jit-00-global-copy-counted COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_copy_counted.ts")
tslang_add_test(NAME test-jit-00-global-interface-initializer COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_interface_initializer.ts")
tslang_add_test(NAME test-jit-00-interface-cast-unrelated COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_cast_unrelated.ts")
tslang_add_test(NAME test-jit-00-interface-union-cast-unrelated COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_union_cast_unrelated.ts")
tslang_add_test(NAME test-jit-00-c-string-copy COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00c_string_copy.ts")
tslang_add_test(NAME test-jit-00-const-field-read COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_field_read.ts")
tslang_add_test(NAME test-jit-00-const-element-read COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_element_read.ts")
tslang_add_test(NAME test-jit-00-const-global-read COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_global_read.ts")
tslang_add_test(NAME test-jit-00-global-argument-reassigned COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_argument_reassigned.ts")
tslang_add_test(NAME test-jit-00-global-argument-view COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_argument_view.ts")
tslang_add_test(NAME test-jit-00-bound-method-counts-object COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00bound_method_counts_object.ts")
tslang_add_test(NAME test-jit-00-closure-const-read COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00closure_const_read.ts")
tslang_add_test(NAME test-jit-00-element-store-after-value COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00element_store_after_value.ts")
tslang_add_test(NAME test-jit-00-const-interface-field-read COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_interface_field_read.ts")
tslang_add_test(NAME test-jit-00-argument-place-overwritten COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00argument_place_overwritten.ts")
tslang_add_test(NAME test-jit-00-const-bound-method-declared-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_bound_method_declared_type.ts")
tslang_add_test(NAME test-jit-00-nullable-class-to-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullable_class_to_interface.ts")
tslang_add_test(NAME test-jit-00-class-to-interface-null COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_to_interface_null.ts")
tslang_add_test(NAME test-jit-00-null-interface-to-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00null_interface_to_interface.ts")
tslang_add_test(NAME test-jit-00-interface-type-tag-in-vtable COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_type_tag_in_vtable.ts")
tslang_add_test(NAME test-jit-00-nullable-interface-to-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullable_interface_to_interface.ts")
tslang_add_test(NAME test-jit-00-argument-place-aliased COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00argument_place_aliased.ts")
tslang_add_test(NAME test-jit-00-temporary-read-in-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00temporary_read_in_finally.ts")
tslang_add_test(NAME test-jit-00-unowned-local-store COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00unowned_local_store.ts")
tslang_add_test(NAME test-jit-00-array-argument-held COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_argument_held.ts")
tslang_add_test(NAME test-jit-00-argument-slot-assigned COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00argument_slot_assigned.ts")
tslang_add_test(NAME test-jit-00-array-data-string-alias COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_data_string_alias.ts")
tslang_add_test(NAME test-jit-00-arrays7 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array7.ts")
tslang_add_test(NAME test-jit-00-arrays8-tuple-spread COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array8_tuple_spread.ts")
tslang_add_test(NAME test-jit-00-arrays-assignment5 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_assignment5.ts")
tslang_add_test(NAME test-jit-00-array-of COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_of.ts")
# Named test-jit- but has no -jit, so it is a compile test, and so is its x86 twin; not renamed, to keep the suite's names.
tslang_add_test(NAME test-jit-00-array-conditional-access COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_cond_access.ts")
tslang_add_test(NAME test-jit-00-typed-array COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00typed_array.ts")
tslang_add_test(NAME test-jit-00-array-view COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_view.ts")
tslang_add_test(NAME test-jit-00-objects COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object.ts")
tslang_add_test(NAME test-jit-00-objects-global COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_global.ts")
tslang_add_test(NAME test-jit-00-objects-functions COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_func.ts")
tslang_add_test(NAME test-jit-00-objects-functions-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_func2.ts")
tslang_add_test(NAME test-jit-00-objects-functions-3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_func3.ts")
tslang_add_test(NAME test-jit-00-objects-deconstruct COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_deconst.ts")
tslang_add_test(NAME test-jit-00-objects-new COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_new.ts")
tslang_add_test(NAME test-jit-00-objects-accessor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_accessor.ts")
tslang_add_test(NAME test-jit-00-objects-boxed-infra COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_boxed_infra.ts")
tslang_add_test(NAME test-jit-00-objects-ref-semantics COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_ref_semantics.ts")
tslang_add_test(NAME test-jit-00-prefix-postfix COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00prefix_postfix.ts")
tslang_add_test(NAME test-jit-00-cond_expr COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00cond_expr.ts")
tslang_add_test(NAME test-jit-00-switch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch.ts")
tslang_add_test(NAME test-jit-01-switch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01switch.ts")
tslang_add_test(NAME test-jit-00-strings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00strings.ts")
tslang_add_test(NAME test-jit-00-tuple COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple.ts")
tslang_add_test(NAME test-jit-01-tuple COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01tuple.ts")
tslang_add_test(NAME test-jit-00-tuple-named COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple_named.ts")
tslang_add_test(NAME test-jit-00-tuple-array COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple_with_array.ts")
tslang_add_test(NAME test-jit-00-computed-property-name COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00computedpropertyname.ts")
tslang_add_test(NAME test-jit-00-union-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_type.ts")
tslang_add_test(NAME test-jit-01-union-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01union_type.ts")
tslang_add_test(NAME test-jit-02-union-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02union_type.ts")
tslang_add_test(NAME test-jit-03-union-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/03union_type.ts")
tslang_add_test(NAME test-jit-03-union-type-case-order COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/03union_type_case_order.ts")
tslang_add_test(NAME test-jit-04-union-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/04union_type.ts")
tslang_add_test(NAME test-jit-05-union-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/05union_type.ts")
tslang_add_test(NAME test-jit-00-union-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_ops.ts")
tslang_add_test(NAME test-jit-00-union-bin-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_bin_ops.ts")
tslang_add_test(NAME test-jit-00-union-bin-ops-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_bin_ops2.ts")
tslang_add_test(NAME test-jit-00-union-to-any COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_to_any.ts")
tslang_add_test(NAME test-jit-00-union-errors COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_errors.ts")
tslang_add_test(NAME test-jit-00-union-null-to-string COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_to_string.ts")
tslang_add_test(NAME test-jit-00-union-null-truthiness COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_truthiness.ts")
tslang_add_test(NAME test-jit-00-null-into-tagged-union COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00null_into_tagged_union.ts")
tslang_add_test(NAME test-jit-00-union-null-in-pointer COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_in_pointer.ts")
tslang_add_test(NAME test-jit-00-union-null-undefined COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_undefined.ts")
tslang_add_test(NAME test-jit-00-union-null-undefined-nonstrict COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_null_undefined_nonstrict.ts")
tslang_add_test(NAME test-jit-00-optional-number-compare COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional_number_compare.ts")
tslang_add_test(NAME test-jit-00-optional-loose-null COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional_loose_null.ts")
tslang_add_test(NAME test-jit-00-union-nullable-array COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_nullable_array.ts")
tslang_add_test(NAME test-jit-00-union-nullable-value COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_nullable_value.ts")
tslang_add_test(NAME test-jit-00-opaque-nullable-compare COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00opaque_nullable_compare.ts")
tslang_add_test(NAME test-jit-00-cppvtable-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00cppvtable_class.ts")
tslang_add_test(NAME test-jit-00-cpp-slot COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_slot.ts")
tslang_add_test(NAME test-jit-00-cpp-overloads COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_overloads.ts")
tslang_add_test(NAME test-jit-own-00-cppvtable-class COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00cppvtable_class.ts")
tslang_add_test(NAME test-jit-own-00-cpp-slot COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_slot.ts")
tslang_add_test(NAME test-jit-own-00-cpp-overloads COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00cpp_overloads.ts")
tslang_add_test(NAME test-jit-00-union-value-equality COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_value_equality.ts")
tslang_add_test(NAME test-jit-00-strict-equals-mismatched-types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00strict_equals_mismatched_types.ts")
tslang_add_test(NAME test-jit-00-any-number-compare COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_number_compare.ts")
tslang_add_test(NAME test-jit-00-length-with-negative-merge COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00length_with_negative_merge.ts")
tslang_add_test(NAME test-jit-00-length-signed-arithmetic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00length_signed_arithmetic.ts")
tslang_add_test(NAME test-jit-00-optional-fields COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional_fields.ts")
tslang_add_test(NAME test-jit-00-generator-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_return.ts")
tslang_add_test(NAME test-jit-00-return-inference-undefined COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00return_inference_undefined.ts")
tslang_add_test(NAME test-jit-00-return-falls-off-end COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00return_falls_off_end.ts")
tslang_add_test(NAME test-jit-00-nullish-values COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullish_values.ts")
tslang_add_test(NAME test-jit-00-class-optional-fields COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_optional_fields.ts")
tslang_add_test(NAME test-jit-00-signless-int-signed COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00signless_int_signed.ts")
tslang_add_test(NAME test-jit-00-union-global-init COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_global_init.ts")
tslang_add_test(NAME test-jit-00-intersection-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00intersection_type.ts")
tslang_add_test(NAME test-jit-00-intersection-type-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00intersection_type_generic.ts")
tslang_add_test(NAME test-jit-00-length COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00length.ts")
tslang_add_test(NAME test-jit-00-undef COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00undef.ts")
tslang_add_test(NAME test-jit-00-str-null COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00str_null.ts")
tslang_add_test(NAME test-jit-00-for-in COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_in.ts")
tslang_add_test(NAME test-jit-00-for-of COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_of.ts")
tslang_add_test(NAME test-jit-00-sizeof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00sizeof.ts")
tslang_add_test(NAME test-jit-01-sizeof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01sizeof.ts")
tslang_add_test(NAME test-jit-02-sizeof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02sizeof.ts")
tslang_add_test(NAME test-jit-00-new-delete COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00new_delete.ts")
tslang_add_test(NAME test-jit-00-new-generic-from-receiver COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00new_generic_from_receiver.ts")
tslang_add_test(NAME test-jit-00-generic-specialization-cast COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_specialization_cast.ts")
tslang_add_test(NAME test-jit-00-owned-locals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_locals.ts")
tslang_add_test(NAME test-jit-00-throw-in-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_in_catch.ts")
tslang_add_test(NAME test-jit-00-throw-inlined COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_inlined.ts")
tslang_add_test(NAME test-jit-00-try-catch-sequence-and-nesting COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_sequence_and_nesting.ts")
tslang_add_test(NAME test-jit-00-try-in-catch-with-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_in_catch_with_finally.ts")
tslang_add_test(NAME test-jit-00-throw-number-untyped-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_number_untyped_catch.ts")
tslang_add_test(NAME test-jit-00-throw-in-block-debug-info COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_in_block_debug_info.ts")
tslang_add_test(NAME test-jit-00-catch-untyped-value COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_untyped_value.ts")
tslang_add_test(NAME test-jit-00-throw-bool-bigint COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_bool_bigint.ts")
tslang_add_test(NAME test-jit-00-int64-remainder COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00int64_remainder.ts")
tslang_add_test(NAME test-jit-00-throw-f32 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_f32.ts")
tslang_add_test(NAME test-jit-00-throw-u32 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_u32.ts")
tslang_add_test(NAME test-jit-00-cast-to-f32 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00cast_to_f32.ts")
tslang_add_test(NAME test-jit-00-instanceof-namespace-narrowing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof_namespace_narrowing.ts")
tslang_add_test(NAME test-jit-00-global-initializer-branches COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_initializer_branches.ts")
tslang_add_test(NAME test-jit-00-using-nested-scopes COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_nested_scopes.ts")
tslang_add_test(NAME test-jit-00-using-unwind-progress COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_unwind_progress.ts")
tslang_add_test(NAME test-jit-00-break-continue-scope-exit COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00break_continue_scope_exit.ts")
tslang_add_test(NAME test-jit-00-owned-fields COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_fields.ts")
tslang_add_test(NAME test-jit-00-owned-elements COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_elements.ts")
tslang_add_test(NAME test-jit-00-owned-literals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_literals.ts")
tslang_add_test(NAME test-jit-00-owned-array-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_array_ops.ts")
tslang_add_test(NAME test-jit-00-owned-inline-records COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_inline_records.ts")
tslang_add_test(NAME test-jit-00-owned-transfer COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_transfer.ts")
tslang_add_test(NAME test-jit-00-owned-call-results COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_call_results.ts")
tslang_add_test(NAME test-jit-00-owned-named-consts COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_named_consts.ts")
tslang_add_test(NAME test-jit-00-owned-temporaries COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_temporaries.ts")
tslang_add_test(NAME test-jit-00-owned-interfaces COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_interfaces.ts")
tslang_add_test(NAME test-jit-00-owned-closures COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_closures.ts")
tslang_add_test(NAME test-jit-00-owned-any-boxing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_any_boxing.ts")
tslang_add_test(NAME test-jit-00-owned-strings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_strings.ts")
tslang_add_test(NAME test-jit-00-string-length-resize COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00string_length_resize.ts")
tslang_add_test(NAME test-jit-00-narrow-array-null COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrow_array_null.ts")
# own sees each branch's array of its literal as fresh
tslang_add_test(NAME test-jit-own-00-narrow-array-null COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrow_array_null.ts")
tslang_add_test(NAME test-compile-own-00-narrow-array-null COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00narrow_array_null.ts")
tslang_add_test(NAME test-jit-00-array-length-grow-zeroed COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_length_grow_zeroed.ts")
tslang_add_test(NAME test-jit-00-rest-args-pack-owned COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00rest_args_pack_owned.ts")
tslang_add_test(NAME test-jit-00-switch-char-string-cases COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch_char_string_cases.ts")
tslang_add_test(NAME test-jit-00-switch-default-fallthrough COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00switch_default_fallthrough.ts")
tslang_add_test(NAME test-jit-00-generic-class-constructor-inference COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_class_constructor_inference.ts")
tslang_add_test(NAME test-jit-00-generic-ref-inference COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_ref_inference.ts")
tslang_add_test(NAME test-jit-00-generator-done-value-zero COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_done_value_zero.ts")
tslang_add_test(NAME test-jit-00-generator-yield-value COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_yield_value.ts")
tslang_add_test(NAME test-jit-00-any-equality-undefined COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_equality_undefined.ts")
tslang_add_test(NAME test-jit-00-owned-generators COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generators.ts")
tslang_add_test(NAME test-jit-00-owned-globals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_globals.ts")
tslang_add_test(NAME test-jit-00-owned-generator-locals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generator_locals.ts")
tslang_add_test(NAME test-jit-00-owned-iteration COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_iteration.ts")
tslang_add_test(NAME test-jit-00-owned-async COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_async.ts")
tslang_add_test(NAME test-jit-00-owned-nested-captures COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_nested_captures.ts")
tslang_add_test(NAME test-jit-00-owned-unions COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_unions.ts")
tslang_add_test(NAME test-jit-00-owned-delete COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_delete.ts")
tslang_add_test(NAME test-jit-00-owned-early-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_early_return.ts")
tslang_add_test(NAME test-jit-00-owned-construct-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_construct_interface.ts")
tslang_add_test(NAME test-jit-00-try-using-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_using_catch.ts")
tslang_add_test(NAME test-jit-00-void COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00void.ts")
tslang_add_test(NAME test-jit-00-in COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00in.ts")
tslang_add_test(NAME test-jit-00-in-method-names COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00in_method_names.ts")
tslang_add_test(NAME test-jit-00-class-structural-extends COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_structural_extends.ts")
tslang_add_test(NAME test-jit-00-global-constant-concat COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_constant_concat.ts")
tslang_add_test(NAME test-jit-00-global-object-literal COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_object_literal.ts")
tslang_add_test(NAME test-jit-00-instanceof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof.ts")
tslang_add_test(NAME test-jit-00-unbraced-branch-scope COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00unbraced_branch_scope.ts")
tslang_add_test(NAME test-jit-00-const-of-let COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_of_let.ts")
tslang_add_test(NAME test-jit-00-instanceof-union COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof_union.ts")
tslang_add_test(NAME test-jit-00-const-object-literal-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_object_literal_field.ts")
tslang_add_test(NAME test-jit-00-nullable-to-any COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nullable_to_any.ts")
tslang_add_test(NAME test-jit-00-ternary-new-or-null COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ternary_new_or_null.ts")
tslang_add_test(NAME test-jit-00-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class.ts")
tslang_add_test(NAME test-jit-00-class-new COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_new.ts")
tslang_add_test(NAME test-jit-00-array-growth COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_growth.ts")
tslang_add_test(NAME test-jit-00-global-from-global COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_from_global.ts")
tslang_add_test(NAME test-jit-00-try-cleanup-then-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_cleanup_then_catch.ts")
tslang_add_test(NAME test-jit-00-try-empty-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_empty_catch.ts")
tslang_add_test(NAME test-jit-00-global-union-from-call COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_union_from_call.ts")
tslang_add_test(NAME test-jit-00-capture-in-new-arguments COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00capture_in_new_arguments.ts")
tslang_add_test(NAME test-jit-00-capture-in-new-untyped-arrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00capture_in_new_untyped_arrow.ts")
tslang_add_test(NAME test-jit-01-class-new COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01class_new.ts")
tslang_add_test(NAME test-jit-00-class-stack COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_stack.ts")
tslang_add_test(NAME test-jit-00-class-static COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static.ts")
tslang_add_test(NAME test-jit-00-class-static-block COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_block.ts")
tslang_add_test(NAME test-jit-00-class-discover-types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_discover_types.ts")
tslang_add_test(NAME test-jit-00-class-accessor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor.ts")
tslang_add_test(NAME test-jit-00-class-accessor-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor2.ts")
tslang_add_test(NAME test-jit-00-class-accessor-super COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_super.ts")
tslang_add_test(NAME test-jit-00-class-accessor-increment COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_increment.ts")
tslang_add_test(NAME test-jit-00-class-accessor-getter-only-abstract COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_getter_only_abstract.ts")
tslang_add_test(NAME test-jit-00-class-accessor-virtual COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_virtual.ts")
tslang_add_test(NAME test-jit-00-class-accessor-implements-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_implements_field.ts")
tslang_add_test(NAME test-jit-00-object-literal-contextual-lambda COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_literal_contextual_lambda.ts")
tslang_add_test(NAME test-jit-00-class-accessor-untyped-setter COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_untyped_setter.ts")
tslang_add_test(NAME test-jit-00-class-super COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super.ts")
tslang_add_test(NAME test-jit-00-class-implicit-constructor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_implicit_constructor.ts")
tslang_add_test(NAME test-jit-00-class-super-static COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super_static.ts")
tslang_add_test(NAME test-jit-00-class-default-constructor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_def_constr.ts")
tslang_add_test(NAME test-jit-00-class-static-constructor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_constr.ts")
tslang_add_test(NAME test-jit-00-class-constructor-fields COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_constr_fields.ts")
tslang_add_test(NAME test-jit-00-class-constructor-overloads COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_constructor_overloads.ts")
tslang_add_test(NAME test-jit-00-callback-result-ignored COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00callback_result_ignored.ts")
tslang_add_test(NAME test-jit-00-generic-default-no-inference COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_default_no_inference.ts")
tslang_add_test(NAME test-jit-00-promise-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00promise_type.ts")
tslang_add_test(NAME test-jit-00-class-abstract COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_abstract.ts")
tslang_add_test(NAME test-jit-00-class-override-inferred-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_override_inferred_return.ts")
tslang_add_test(NAME test-jit-00-class-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_iface.ts")
tslang_add_test(NAME test-jit-00-class-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic.ts")
tslang_add_test(NAME test-jit-00-class-generic-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic_iface.ts")
tslang_add_test(NAME test-jit-00-class-recursive COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_recursive.ts")
tslang_add_test(NAME test-jit-00-class-expression COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression.ts")
tslang_add_test(NAME test-jit-00-class-expression-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression2.ts")
tslang_add_test(NAME test-jit-00-class-expression-3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression3.ts")
tslang_add_test(NAME test-jit-00-class-expression-no-ctor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_expression_no_ctor.ts")
tslang_add_test(NAME test-jit-00-class-deconst COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_deconst.ts")
tslang_add_test(NAME test-jit-00-class-virtual-call COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_virtual_call.ts")
tslang_add_test(NAME test-jit-00-class-virtual-table COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_virtual_table.ts")
tslang_add_test(NAME test-jit-00-class-local-decl COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_local_decl.ts")
tslang_add_test(NAME test-jit-00-class-nested COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_nested.ts")
tslang_add_test(NAME test-jit-00-class-static-generic-method COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_generic_method.ts")
tslang_add_test(NAME test-jit-00-class-static-generic-method-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_generic_method2.ts")
tslang_add_test(NAME test-jit-00-class-static-generic-method-3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_generic_method3.ts")
tslang_add_test(NAME test-jit-00-class-generic-method COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic_method.ts")
tslang_add_test(NAME test-jit-00-class-generic-method-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_generic_method2.ts")
tslang_add_test(NAME test-jit-00-class-var-args COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_var_args.ts")
tslang_add_test(NAME test-jit-00-class-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_iterator.ts")
tslang_add_test(NAME test-jit-00-class-iterator-super COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_iterator_super.ts")
tslang_add_test(NAME test-jit-00-class-symbol-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_symbol_iterator.ts")
tslang_add_test(NAME test-jit-00-class-indexer COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_indexer.ts")
tslang_add_test(NAME test-jit-00-class-indexer-super COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_indexer_super.ts")
tslang_add_test(NAME test-jit-00-class-from-tuple COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_from_tuple.ts")
tslang_add_test(NAME test-jit-00-class-access-control COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_access_control.ts")
tslang_add_test(NAME test-jit-00-class-protected-constructor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_protected_constr.ts")
tslang_add_test(NAME test-jit-00-to-primitive-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00to_primitive_ops.ts")
tslang_add_test(NAME test-jit-00-namespace COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns.ts")
tslang_add_test(NAME test-jit-00-namespace-enum COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns2.ts")
tslang_add_test(NAME test-jit-00-namespace-import COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns3.ts")
tslang_add_test(NAME test-jit-00-namespace-globals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns4.ts")
tslang_add_test(NAME test-jit-00-namespace-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns5.ts")
tslang_add_test(NAME test-jit-00-namespace-bug COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns_bug.ts")
tslang_add_test(NAME test-jit-00-namespace-member-shadowed COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00ns_member_shadowed.ts")
tslang_add_test(NAME test-jit-00-interface-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface.ts")
tslang_add_test(NAME test-jit-00-interface-class-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface2.ts")
tslang_add_test(NAME test-jit-00-interface-object COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object.ts")
tslang_add_test(NAME test-jit-00-interface-object-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object2.ts")
tslang_add_test(NAME test-jit-00-interface-object-3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object3.ts")
tslang_add_test(NAME test-jit-00-interface-object-4 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object4.ts")
tslang_add_test(NAME test-jit-00-interface-object-5 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object5.ts")
tslang_add_test(NAME test-jit-00-interface-global-method COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_global_method.ts")
tslang_add_test(NAME test-jit-00-interface-conjunction COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_conjunction.ts")
tslang_add_test(NAME test-jit-00-interface-partial COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_partial.ts")
tslang_add_test(NAME test-jit-00-interface-optional COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional.ts")
tslang_add_test(NAME test-jit-00-interface-optional-cast-order COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_cast_order.ts")
tslang_add_test(NAME test-jit-00-interface-function-typed-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field.ts")
tslang_add_test(NAME test-jit-00-interface-function-typed-field-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field_class.ts")
tslang_add_test(NAME test-jit-00-interface-function-typed-field-class-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field_class_field.ts")
tslang_add_test(NAME test-jit-00-interface-function-field-write COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_field_write.ts")
tslang_add_test(NAME test-jit-own-00-interface-function-field-write COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_field_write.ts")
tslang_add_test(NAME test-jit-00-interface-function-typed-field-object-closure COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_function_typed_field_object_closure.ts")
tslang_add_test(NAME test-jit-00-class-function-typed-field-inferred COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_function_typed_field_inferred.ts")
tslang_add_test(NAME test-jit-00-interface-optional-function-typed-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_function_typed_field.ts")
tslang_add_test(NAME test-jit-00-interface-optional-function-typed-field-class-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_function_typed_field_class_field.ts")
tslang_add_test(NAME test-jit-00-interface-captures COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_captures.ts")
tslang_add_test(NAME test-jit-00-object-annotated-method COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method.ts")
tslang_add_test(NAME test-jit-00-object-annotated-method-params COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_params.ts")
tslang_add_test(NAME test-jit-00-object-chained-sibling-method-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_chained_sibling_method_return.ts")
tslang_add_test(NAME test-jit-00-object-annotated-method-interleaved COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_interleaved.ts")
tslang_add_test(NAME test-jit-00-object-annotated-method-extends-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_extends_interface.ts")
tslang_add_test(NAME test-jit-00-object-annotated-method-extends-interface-multilevel COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00object_annotated_method_extends_interface_multilevel.ts")
tslang_add_test(NAME test-jit-00-interface-optional-extends COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_extends.ts")
tslang_add_test(NAME test-jit-00-interface-optional-method-extends COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_optional_method_extends.ts")
tslang_add_test(NAME test-jit-00-interface-object-array COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_object_array.ts")
tslang_add_test(NAME test-jit-00-interface-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_generic.ts")
tslang_add_test(NAME test-jit-00-interface-new COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_new.ts")
tslang_add_test(NAME test-jit-00-interface-indexer COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_indexer.ts")
tslang_add_test(NAME test-jit-00-class-or-interface-to-tuple COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_or_interface_to_tuple.ts")
tslang_add_test(NAME test-jit-00-any COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any.ts")
tslang_add_test(NAME test-jit-00-any-unbox-interface-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_unbox_interface_class.ts")
tslang_add_test(NAME test-jit-00-any-to-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_to_interface.ts")
tslang_add_test(NAME test-jit-own-00-any-to-interface COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_to_interface.ts")
tslang_add_test(NAME test-jit-00-interface-instanceof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_instanceof.ts")
tslang_add_test(NAME test-jit-01-any COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01any.ts")
tslang_add_test(NAME test-jit-00-any-types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_types.ts")
tslang_add_test(NAME test-jit-00-any-compare COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_compare.ts")
tslang_add_test(NAME test-jit-00-any-truthiness-objects COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_truthiness_objects.ts")
tslang_add_test(NAME test-jit-00-any-generic-equals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_generic_equals.ts")
tslang_add_test(NAME test-jit-00-generator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator.ts")
tslang_add_test(NAME test-jit-00-generator-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator2.ts")
tslang_add_test(NAME test-jit-00-generator-3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator3.ts")
tslang_add_test(NAME test-jit-00-generator-4 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator4.ts")
tslang_add_test(NAME test-jit-00-generator-5 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator5.ts")
tslang_add_test(NAME test-jit-00-generator-6 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator6.ts")
tslang_add_test(NAME test-jit-00-generator-7 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator7.ts")
tslang_add_test(NAME test-jit-00-safe-cast COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast.ts")
tslang_add_test(NAME test-jit-00-safe-cast-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast2.ts")
tslang_add_test(NAME test-jit-00-safe-cast-typeof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_typeof.ts")
tslang_add_test(NAME test-jit-00-safe-cast-while COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_while.ts")
tslang_add_test(NAME test-jit-01-safe-cast-while COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01safe_cast_while.ts")
tslang_add_test(NAME test-jit-00-safe-cast-field-access COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_field_access.ts")
tslang_add_test(NAME test-jit-00-safe-cast-early-exit COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_early_exit.ts")
tslang_add_test(NAME test-jit-00-union-object-types-access COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_object_types_access.ts")
tslang_add_test(NAME test-jit-00-safe-cast-null-field COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_null_field.ts")
tslang_add_test(NAME test-jit-00-safe-cast-else-scope COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_else_scope.ts")
tslang_add_test(NAME test-jit-00-safe-cast-bug COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00safe_cast_bug.ts")
tslang_add_test(NAME test-jit-00-optional COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00optional.ts")
tslang_add_test(NAME test-jit-01-optional COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01optional.ts")
tslang_add_test(NAME test-jit-00-async-await COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00async_await.ts")
tslang_add_test(NAME test-jit-00-async-result-types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00async_result_types.ts")
tslang_add_test(NAME test-jit-00-async-gc-threading COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00async_gc_threading.ts")
tslang_add_test(NAME test-jit-00-for-await COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await.ts")
tslang_add_test(NAME test-jit-00-for-await-yield COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_await_yield.ts")
tslang_add_test(NAME test-jit-00-try-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch.ts")
tslang_add_test(NAME test-jit-00-catch-value COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_value.ts")
tslang_add_test(NAME test-jit-00-catch-value-minimal COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_value_minimal.ts")
tslang_add_test(NAME test-jit-01-try-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_catch.ts")
tslang_add_test(NAME test-jit-00-try-catch-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_return.ts")
tslang_add_test(NAME test-jit-00-try-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally.ts")
tslang_add_test(NAME test-jit-00-try-finally-throws COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_throws.ts")
tslang_add_test(NAME test-jit-00-function-fewer-parameters COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00function_fewer_parameters.ts")
tslang_add_test(NAME test-jit-00-closure-fewer-parameters COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00closure_fewer_parameters.ts")
tslang_add_test(NAME test-jit-01-try-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_finally.ts")
tslang_add_test(NAME test-jit-00-try-finally-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_return.ts")
tslang_add_test(NAME test-jit-00-try-finally-funclet COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_funclet.ts")
tslang_add_test(NAME test-jit-00-try-finally-break-continue COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_break_continue.ts")
tslang_add_test(NAME test-jit-00-try-catch-rethrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_rethrow.ts")
tslang_add_test(NAME test-jit-00-try-catch-accessor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_accessor.ts")
tslang_add_test(NAME test-jit-00-try-catch-call-in-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_call_in_catch.ts")
# pre-existing JIT limitation (reproduces on unmodified main too): the JIT's
# Windows SEH landing pad/personality plumbing exits with code 3 and no
# output on a typed-catch mismatch that must unwind past a function boundary,
# unrelated to the LowerToAffineLoops.cpp rethrow fix this test targets (that
# fix only affects the non-Windows/Itanium cmpValue path). AOT variant above
# passes and exercises the fix correctly.
#tslang_add_test(NAME test-jit-00-try-catch-mismatch-rethrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_mismatch_rethrow.ts")
tslang_add_test(NAME test-jit-00-try-catch-return-dispose COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_return_dispose.ts")
tslang_add_test(NAME test-jit-01-try-catch-return-dispose COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_catch_return_dispose.ts")
tslang_add_test(NAME test-jit-00-types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types.ts")
tslang_add_test(NAME test-jit-00-property-access-conditional COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00property_access_cond.ts")
tslang_add_test(NAME test-jit-00-method-access-conditional COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00method_access_cond.ts")
tslang_add_test(NAME test-jit-00-method-this-param COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00method_this_param.ts")
tslang_add_test(NAME test-jit-01-method-access-conditional COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01method_access_cond.ts")
tslang_add_test(NAME test-jit-00-question-question COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00question_question.ts")
tslang_add_test(NAME test-jit-00-spread-assignment COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread_assignment.ts")
tslang_add_test(NAME test-jit-01-spread-assignment COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread_assignment.ts")
tslang_add_test(NAME test-jit-00-spread COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread.ts")
tslang_add_test(NAME test-jit-01-spread COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread.ts")
tslang_add_test(NAME test-jit-00-types-unknown1 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_unknown1.ts")
tslang_add_test(NAME test-jit-00-any-to-tuple COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_to_tuple.ts")
tslang_add_test(NAME test-jit-00-bigint-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00bigint_ops.ts")
tslang_add_test(NAME test-jit-00-namespace-forward-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00namespace_forward_class.ts")
tslang_add_test(NAME test-jit-00-any-literal-to-interface COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_literal_to_interface.ts")
tslang_add_test(NAME test-jit-00-interface-method-object-field-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface_method_object_field_type.ts")
tslang_add_test(NAME test-jit-00-unknown-round-trip COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00unknown_round_trip.ts")
tslang_add_test(NAME test-jit-00-types-indexed-access-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_indexedaccesstype.ts")
tslang_add_test(NAME test-jit-00-types-mapped-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_mappedtype.ts")
tslang_add_test(NAME test-jit-01-types-mapped-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_mappedtype.ts")
tslang_add_test(NAME test-jit-00-types-template-literal-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_templateliteraltype.ts")
tslang_add_test(NAME test-jit-00-template-literal-escapes COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00template_literal_escapes.ts")
tslang_add_test(NAME test-jit-01-types-template-literal-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_templateliteraltype.ts")
tslang_add_test(NAME test-jit-00-types-utility COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_utility.ts")
tslang_add_test(NAME test-jit-01-types-utility COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_utility.ts")
tslang_add_test(NAME test-jit-01-types-key-of-enum COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_keyof_enum.ts")
tslang_add_test(NAME test-jit-00-type-alias-in-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00type_aliases_in_generics.ts")
tslang_add_test(NAME test-jit-00-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00iterator.ts")
tslang_add_test(NAME test-jit-01-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01iterator.ts")
tslang_add_test(NAME test-jit-02-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02iterator.ts")
tslang_add_test(NAME test-jit-03-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/03iterator.ts")
tslang_add_test(NAME test-jit-00-iterator-bug COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00iterator_bug.ts")
tslang_add_test(NAME test-jit-00-map COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00map.ts")
tslang_add_test(NAME test-jit-00-iterator-methods COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00iterator_methods.ts")
tslang_add_test(NAME test-jit-01-map COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01map.ts")
tslang_add_test(NAME test-jit-00-filter COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00filter.ts")
tslang_add_test(NAME test-jit-00-reduce COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00reduce.ts")
tslang_add_test(NAME test-jit-01-reduce COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01reduce.ts")
tslang_add_test(NAME test-jit-00-every COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00every.ts")
tslang_add_test(NAME test-jit-00-extension COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00extension.ts")
tslang_add_test(NAME test-jit-01-extension COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01extension.ts")
tslang_add_test(NAME test-jit-00-extension-cond-access COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00extension_cond_access.ts")
tslang_add_test(NAME test-jit-00-infer COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00infer.ts")
tslang_add_test(NAME test-jit-00-extends COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00extends.ts")
tslang_add_test(NAME test-jit-01-extends COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01extends.ts")
tslang_add_test(NAME test-jit-02-extends COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02extends.ts")
tslang_add_test(NAME test-jit-00-symbol COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00symbol.ts")
tslang_add_test(NAME test-jit-01-symbol COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01symbol.ts")
tslang_add_test(NAME test-jit-00-as COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00as.ts")
tslang_add_test(NAME test-jit-00-as-const COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00as_const.ts")
tslang_add_test(NAME test-jit-00-type-guard-function COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00type_guard_function.ts")
tslang_add_test(NAME test-jit-00-class-cast-structural COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_cast_structural.ts")
tslang_add_test(NAME test-jit-00-names-conflict COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00names_conflict.ts")
tslang_add_test(NAME test-jit-00-generic-arguments-name-conflict COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generic_arguments_name_conflict.ts")
tslang_add_test(NAME test-jit-00-decorators COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00decorators.ts")
tslang_add_test(NAME test-jit-00-conditional-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00conditional_type.ts")
tslang_add_test(NAME test-jit-00-disposable COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00disposable.ts")
tslang_add_test(NAME test-jit-01-disposable COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01disposable.ts")
tslang_add_test(NAME test-jit-02-disposable COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02disposable.ts")
tslang_add_test(NAME test-jit-03-disposable COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/03disposable.ts")
tslang_add_test(NAME test-jit-04-disposable COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/04disposable.ts")
tslang_add_test(NAME test-jit-00-if-conditional-compile COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00if_conditional_compile.ts")
tslang_add_test(NAME test-jit-01-arguments COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01arguments.ts")
tslang_add_test(NAME test-jit-02-numbers COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02numbers.ts")
tslang_add_test(NAME test-jit-05-strings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/05strings.ts")
tslang_add_test(NAME test-jit-06-numbercollections COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/06numbercollections.ts")
tslang_add_test(NAME test-jit-07-stringcollections COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/07stringcollections.ts")
tslang_add_test(NAME test-jit-08-stringopertations COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/08stringopertations.ts")
tslang_add_test(NAME test-jit-09-postprefix COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/09postprefix.ts")
tslang_add_test(NAME test-jit-10-arrayincrement COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/10arrayincrement.ts")
tslang_add_test(NAME test-jit-11-equalsoperator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/11equalsoperator.ts")
tslang_add_test(NAME test-jit-12-referencecollection COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/12referencecollection.ts")
tslang_add_test(NAME test-jit-13-actions.ts COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/13actions.ts")
tslang_add_test(NAME test-jit-14-lazyoperations COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/14lazyoperations.ts")
tslang_add_test(NAME test-jit-15-references COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/15references.ts")
tslang_add_test(NAME test-jit-15-references-capture COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/15references_capture.ts")
tslang_add_test(NAME test-jit-17-classes COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/17classes.ts")
tslang_add_test(NAME test-jit-18-enums COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/18enums.ts")
tslang_add_test(NAME test-jit-19-forof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/19forof.ts")
tslang_add_test(NAME test-jit-19-forof-capture COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/19forof_capture.ts")
tslang_add_test(NAME test-jit-20-maps COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/20maps.ts")
tslang_add_test(NAME test-jit-22-lambdas COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/22lambdas.ts")
tslang_add_test(NAME test-jit-23-generics COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/23generics.ts")
tslang_add_test(NAME test-jit-25-lamdacapture COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/25lamdacapture.ts")
tslang_add_test(NAME test-jit-26-staticclasses COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/26staticclasses.ts")
tslang_add_test(NAME test-jit-28-boolcasts COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/28boolcasts.ts")
tslang_add_test(NAME test-jit-29-lazyreferences COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/29lazyreferences.ts")
tslang_add_test(NAME test-jit-32-complexcalls COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/32complexcalls.ts")
tslang_add_test(NAME test-jit-33-inheritance COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/33inheritance.ts")
tslang_add_test(NAME test-jit-34-switch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/34switch.ts")
tslang_add_test(NAME test-jit-35-lambdaparameters COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/35lambdaparameters.ts")
tslang_add_test(NAME test-jit-36-interfaces COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/36interfaces.ts")
tslang_add_test(NAME test-jit-39-objectdestructuring COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/39objectdestructuring.ts")
tslang_add_test(NAME test-jit-40-generics.ts COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/40generics.ts")
tslang_add_test(NAME test-jit-41-anonymoustypes COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/41anonymoustypes.ts")
tslang_add_test(NAME test-jit-42-lambdaproperties COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/42lambdaproperties.ts")
tslang_add_test(NAME test-jit-43-nestednamespace COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/43nestednamespace.ts")
tslang_add_test(NAME test-jit-44-toplevelcode COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/44toplevelcode.ts")
tslang_add_test(NAME test-jit-00-toplevel-control-flow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00toplevel_control_flow.ts")
tslang_add_test(NAME test-jit-00-toplevel-no-main-with-helpers COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00toplevel_no_main_with_helpers.ts")
tslang_add_test(NAME test-jit-45-enumtostring COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/45enumtostring.ts")
tslang_add_test(NAME test-jit-48-instanceof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/48instanceof.ts")
tslang_add_test(NAME test-jit-51-exceptions COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/51exceptions.ts")
tslang_add_test(NAME test-jit-241-arrayforeach COMMAND test-runner  -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/241arrayforeach.ts")
tslang_add_test(NAME test-jit-243-arrayevery COMMAND test-runner  -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/243arrayevery.ts")
tslang_add_test(NAME test-jit-244-arraysome COMMAND test-runner  -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/244arraysome.ts")
tslang_add_test(NAME test-jit-Grammar_and_types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/Grammar_and_types.ts")
tslang_add_test(NAME test-jit-StackTest COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00stack_test.ts")
tslang_add_test(NAME test-jit-Raytrace COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/raytrace.ts")
tslang_add_test(NAME test-jit-Path COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/path.ts")
tslang_add_test(NAME test-jit-Print-Bug-01 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01print-bug.ts")
tslang_add_test(NAME test-jit-arrayLiterals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayLiterals.ts")
tslang_add_test(NAME test-jit-arrayLiterals-2-ES5 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayLiterals2ES5.ts")
tslang_add_test(NAME test-jit-arrayLiterals-3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayLiterals3.ts")
tslang_add_test(NAME test-jit-Abstract-Property-In-Constructor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/abstractPropertyInConstructor.ts")
tslang_add_test(NAME test-jit-Dependencies COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/dependencies.ts")
tslang_add_test(NAME test-jit-ArrayFakeFlatNoCrashInferenceDeclarations COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/arrayFakeFlatNoCrashInferenceDeclarations.ts")
tslang_add_test(NAME test-jit-noPropertyAccessFromIndexSignature1 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/noPropertyAccessFromIndexSignature1.ts")
tslang_add_test(NAME test-jit-asyncMethodWithSuper_es2017 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/asyncMethodWithSuper_es2017.ts")
tslang_add_test(NAME test-jit-assignmentTypeNarrowing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/assignmentTypeNarrowing.ts")
tslang_add_test(NAME test-jit-additionOperatorWithNumberAndEnum COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/additionOperatorWithNumberAndEnum.ts")
tslang_add_test(NAME test-jit-arithmeticOperatorWithEnum COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/arithmeticOperatorWithEnum.ts")
tslang_add_test(NAME test-jit-arithmeticOperatorWithTypeParameter COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/arithmeticOperatorWithTypeParameter.ts")
tslang_add_test(NAME test-jit-comparisonOperatorWithIdenticalObjects COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/comparisonOperatorWithIdenticalObjects.ts")
tslang_add_test(NAME test-jit-callWithSpread COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/callWithSpread.ts")
tslang_add_test(NAME test-jit-newWithSpread COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/newWithSpread.ts")
tslang_add_test(NAME test-jit-typeGuardOfFormThisMember COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/typeGuardOfFormThisMember.ts")
tslang_add_test(NAME test-jit-typeGuardOfFormTypeOfBoolean COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/typeGuardOfFormTypeOfBoolean.ts")
tslang_add_test(NAME test-jit-typeGuardFunction COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/typeGuardFunction.ts")
tslang_add_test(NAME test-jit-thisTypeInClasses COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/thisTypeInClasses.ts")
tslang_add_test(NAME test-jit-conditionalTypes1 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/conditionalTypes1.ts")
tslang_add_test(NAME test-jit-conditionalTypes2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/conditionalTypes2.ts")
tslang_add_test(NAME test-jit-types-vs-interfaces COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/types_vs_interfaces.ts")
tslang_add_test(NAME test-jit-structural-typing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/structural-typing.ts")
tslang_add_test(NAME test-jit-declarationsAndAssignments COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/declarationsAndAssignments.ts")
tslang_add_test(NAME test-jit-disallowLineTerminatorBeforeArrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/disallowLineTerminatorBeforeArrow.ts")
tslang_add_test(NAME test-jit-emitDefaultParametersFunctionExpression COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/emitDefaultParametersFunctionExpression.ts")
tslang_add_test(NAME test-jit-logicalAssignment5 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/logicalAssignment5.ts")
tslang_add_test(NAME test-jit-logicalAssignment5-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/logicalAssignment5_2.ts")
tslang_add_test(NAME test-jit-logicalAssignment5-3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/logicalAssignment5_3.ts")
tslang_add_test(NAME test-jit-nbody COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/nbody.ts")
tslang_add_test(NAME test-jit-no-main COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/no_main.ts")
tslang_add_test(NAME test-jit-parser COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/parser.ts")
tslang_add_test(NAME test-jit-internals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/internals.ts")
tslang_add_test(NAME test-jit-load-store-decorators COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/load_store_decorators.ts")
tslang_add_test(NAME test-jit-reference-null-bug COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00reference_null_bug.ts")
tslang_add_test(NAME test-jit-reference-index-bug COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00reference_index_bug.ts")
tslang_add_test(NAME test-jit-reference-ref-deref COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00reference_ref_deref.ts")
tslang_add_test(NAME test-jit-uint-compare-bug COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00uint_compare_bug.ts")
tslang_add_test(NAME test-jit-gc-malloc-cse-o3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/gc_malloc_cse_o3.ts")
tslang_add_test(NAME test-jit-fast-math COMMAND test-runner -jit -fast-math "${PROJECT_SOURCE_DIR}/test/tester/tests/fast_math.ts")

# tesing include
tslang_add_test(NAME test-compile-include-global-var COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/include_global_var.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/define_global_var.ts")

# imports support only compile mode
tslang_add_test(NAME test-compile-import-component COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/component.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/service.ts")
# A library whose root only declares and initializes a variable must not get an entry point of
# its own: it is compiled with the same --emit=obj as the program linked beside it, and a `main`
# in both fails to link. Only the program half is given --entry-point.
tslang_add_test(NAME test-compile-export-import-root-var COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_root_var.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_root_var.ts")
tslang_add_test(NAME test-compile-export-import-class-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_interface.ts")
# @linkname across a static link - the mode it is named for; the second is the live guard for an
# imported `export declare` binding while its -shared twin is DISABLED (see below)
tslang_add_test(NAME test-compile-export-import-linkname COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_linkname.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_linkname.ts")
tslang_add_test(NAME test-compile-export-import-declare-linkname COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_declare_linkname.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_declare_linkname.ts")
# Item 5al: an imported function's returned reference is taken over rather than retained
# again. Registered statically as well as shared, and under every memory model - the
# static two-module form had no rc/none coverage at all, which is why the leak it fixes
# went unmeasured for so long.
tslang_add_test(NAME test-compile-export-import-owned-returns COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-compile-export-import-const-locals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_locals.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_locals.ts")
tslang_add_test(NAME test-compile-export-import-const-functions COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_functions.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_functions.ts")
tslang_add_test(NAME test-compile-export-import-c-varargs COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_c_varargs.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_c_varargs.ts")
tslang_add_test(NAME test-compile-export-import-generators COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_generators.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_generators.ts")
# `import './export_os_suffix'` takes the target's file, export_os_suffix.win32.ts or
# export_os_suffix.linux.ts: the host's is compiled and linked beside it, and the import's
# declarations for each target are those of its file (the OS function that file declares).
if (WIN32)
    tslang_add_test(NAME test-compile-export-import-os-suffix COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_os_suffix.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_os_suffix.win32.ts")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    tslang_add_test(NAME test-compile-export-import-os-suffix COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_os_suffix.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_os_suffix.linux.ts")
endif()
foreach(os_suffix "win32|x86_64-pc-windows-msvc|GetCurrentProcessId|getpid" "linux|x86_64-unknown-linux-gnu|getpid|GetCurrentProcessId")
    string(REPLACE "|" ";" os_suffix "${os_suffix}")
    list(GET os_suffix 0 os_name)
    list(GET os_suffix 1 os_triple)
    list(GET os_suffix 2 os_function)
    list(GET os_suffix 3 os_other_function)
    add_test(NAME test-compile-import-os-suffix-${os_name}
             COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib -mtriple=${os_triple}
                     "${PROJECT_SOURCE_DIR}/test/tester/tests/import_os_suffix.ts")
    set_tests_properties(test-compile-import-os-suffix-${os_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "ts\\.Func @${os_function} "
                                    FAIL_REGULAR_EXPRESSION "ts\\.Func @${os_other_function} |error:")
endforeach()
tslang_add_test(NAME test-compile-rc-export-import-owned-returns COMMAND test-runner -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-compile-none-export-import-owned-returns COMMAND test-runner -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-compile-export-import-class-extends COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends.ts")
tslang_add_test(NAME test-compile-export-import-throw-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_throw_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_throw_class.ts")
tslang_add_test(NAME test-compile-export-import-class-extends-multilevel COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_multilevel.ts")
tslang_add_test(NAME test-compile-export-import-class-extends-implements-diamond COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_implements_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_implements_diamond.ts")
tslang_add_test(NAME test-compile-export-import-class-abstract COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract.ts")
tslang_add_test(NAME test-compile-export-import-class-static COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_static.ts")
# Cross-module generic class instantiation: FIXED 2026-07-22 (was disabled). Two independent
# bugs, both now fixed:
# (1) A generic class specialization instantiated in the IMPORTING module was incorrectly
#     inheriting `export`/dllexport linkage from the original `export class` declaration, even
#     though the specialization is a fresh, local instantiation. For a multi-type-param generic
#     (Pair<A,B>) the resulting mangled name contains a raw comma
#     (M.Pair<!ts.number,!ts.string>..instanceOf), a linker-directive metacharacter - lld
#     rejected it with "invalid /export:". Fixed in mlirGen(ClassLikeDeclaration): a concrete
#     instantiation now forces isExport/isPublic false regardless of the template's own
#     modifiers (see the comment there).
# (2) The bare generic class declaration itself (`export class Box<T> {...}`) never reached
#     addClassDeclarationToExport at all (registerGenericClass's early-return path skipped it
#     entirely), so nothing about it was ever embedded in the exporting module's __decls
#     global - once a compiled .dll/.lib for the exporting module already existed next to its
#     .ts source, import resolution reads declarations back from that binary artifact instead
#     of reparsing source, and found nothing, so the whole namespace (M) failed to resolve
#     ("can't resolve name: M"). Fixed by re-exporting the generic's own ORIGINAL source text
#     (not a signature-only declaration like concrete classes get - a generic has no compiled
#     body anywhere for the importer to link against, so it needs the real, recompilable
#     source) via a new addGenericClassDeclarationToExport, embedded in ITS OWN
#     "__decls_generic_<file>_<hash>" global (kept separate from the regular
#     "__decls_<file>_<hash>" one) so it can be re-parsed with a plain ".ts" filename instead
#     of ".d.ts" - re-parsing as ".d.ts" implicitly marks everything ambient/external
#     regardless of any per-declaration @dllimport marker, which broke instantiation entirely
#     (crashed the compiler with an unhandled std::out_of_range once the reimported
#     declaration's own re-export attempt hit a stale source-file/AST-position mismatch).
# See class-generic-declaration-export-fix memory for the full account.
tslang_add_test(NAME test-compile-export-import-class-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-compile-export-import-function-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-compile-export-import-type-alias-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
# plain (non-generic) counterparts - every other declaration kind (class, interface, enum, vars)
# already had both a "-generic" and a plain cross-module export/import pair; function and
# type-alias only had "-generic". Confirmed working (not a gap) before adding: default/optional
# params, union/chained-alias/object-shape aliases, and an aliased type used in a function
# signature all round-trip correctly through __decls across all 3 tiers.
tslang_add_test(NAME test-compile-export-import-function COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-compile-export-import-type-alias COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias.ts")
tslang_add_test(NAME test-compile-export-import-interface-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_generic.ts")
tslang_add_test(NAME test-compile-export-import-class-accessor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_accessor.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_accessor.ts")
tslang_add_test(NAME test-compile-export-import-class-indexer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_indexer.ts")
tslang_add_test(NAME test-compile-export-import-interface-indexer COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_indexer.ts")
tslang_add_test(NAME test-compile-export-import-class-implements-interface-multilevel COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_multilevel.ts")
tslang_add_test(NAME test-compile-export-import-class-implements-interface-optional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_optional.ts")
tslang_add_test(NAME test-compile-export-import-class-implements-interface-abstract COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_abstract.ts")
tslang_add_test(NAME test-compile-export-import-class-structural-interface COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_structural_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_structural_interface.ts")

# shared libs tests (dlls/compile-time)
tslang_add_test(NAME test-compile-time-shared-decl-emit-type COMMAND test-runner -shared -compile-time "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_compiletime_func.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-compile-time-shared-decl-emit-class COMMAND test-runner -shared -compile-time "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_compiletime_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")

# shared libs tests (dlls/dynamics)
tslang_add_test(NAME test-compile-shared-component COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/use_shared.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/shared.ts")
tslang_add_test(NAME test-compile-shared-decl-emit-interface COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_interface.ts")
tslang_add_test(NAME test-compile-shared-decl-emit-type COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_type.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-compile-shared-decl-emit-enum COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_enum.ts")
tslang_add_test(NAME test-compile-shared-decl-emit-class COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")

# shared libs tests (exports/imports)
tslang_add_test(NAME test-compile-shared-export-import-class-interface COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_interface.ts")
# a @final class across the boundary: direct calls from the importer, compiled and under the JIT
# own: the importer owns what the generator method's next() gives it (isIteratorNext)
foreach(final_mm gc rc own)
    tslang_add_test(NAME test-compile-shared-export-import-final-class-${final_mm} COMMAND test-runner -shared -mm=${final_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_final_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_final_class.ts")
    tslang_add_test(NAME test-jit-shared-export-import-final-class-${final_mm} COMMAND test-runner -jit -shared -mm=${final_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_final_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_final_class.ts")
endforeach()
# @value across the boundary: the importer knows the class and interfaces from __decls (`Hit`
# printed before the `Ray` it holds), builds and reads values of them, and gives them back
foreach(value_mm gc rc own)
    tslang_add_test(NAME test-compile-shared-export-import-value-types-${value_mm} COMMAND test-runner -shared -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_value_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_value_types.ts")
    tslang_add_test(NAME test-jit-shared-export-import-value-types-${value_mm} COMMAND test-runner -jit -shared -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_value_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_value_types.ts")
endforeach()
# ...and an imported @value class is nominal: an object literal of its shape is not one
add_test(NAME test-compile-shared-import-value-types-structural
         COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_value_types_structural.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_value_types.ts")
set_tests_properties(test-compile-shared-import-value-types-structural PROPERTIES PASS_REGULAR_EXPRESSION "type '.*' cannot be converted to the @value class 'V'"
                                                                                 FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# A forward reference in __decls: an exported interface names one declared after it, and the
# importer reads each declaration again once the others are in
foreach(forward_mm gc rc own)
    tslang_add_test(NAME test-compile-shared-export-import-forward-iface-${forward_mm} COMMAND test-runner -shared -mm=${forward_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_forward_iface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_forward_iface.ts")
    tslang_add_test(NAME test-jit-shared-export-import-forward-iface-${forward_mm} COMMAND test-runner -jit -shared -mm=${forward_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_forward_iface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_forward_iface.ts")
endforeach()
# ...while a type they name and never declare is still an error
add_test(NAME test-compile-shared-import-unresolved-iface
         COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_unresolved_iface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_unresolved_iface.ts")
set_tests_properties(test-compile-shared-import-unresolved-iface PROPERTIES PASS_REGULAR_EXPRESSION "error: can't find type by name: Hidden"
                                                                            FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# @keeps and @borrows across the boundary: the importer reads them from __decls (-mm=own), and the
# other models ignore them
foreach(markers_mm gc rc own)
    tslang_add_test(NAME test-compile-shared-export-import-own-markers-${markers_mm} COMMAND test-runner -shared -mm=${markers_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_markers.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_markers.ts")
    tslang_add_test(NAME test-jit-shared-export-import-own-markers-${markers_mm} COMMAND test-runner -jit -shared -mm=${markers_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_markers.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_markers.ts")
endforeach()
tslang_add_test(NAME test-compile-shared-export-import-own-markers-keeps COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_markers_keeps.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_markers.ts")
tslang_add_test(NAME test-jit-shared-export-import-own-markers-keeps COMMAND test-runner -jit -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_markers_keeps.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_markers.ts")
# ...and an importer that breaks them is told so: they crossed
add_test(NAME test-compile-shared-import-own-markers-moved
         COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_markers_moved.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_markers.ts")
set_tests_properties(test-compile-shared-import-own-markers-moved PROPERTIES PASS_REGULAR_EXPRESSION "is used here after its value was moved"
                                                                           FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
add_test(NAME test-compile-shared-import-own-markers-outlives
         COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_markers_outlives.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_markers.ts")
set_tests_properties(test-compile-shared-import-own-markers-outlives PROPERTIES PASS_REGULAR_EXPRESSION "borrows its generator's arguments and cannot be stored, returned or captured"
                                                                              FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
add_test(NAME test-compile-shared-import-own-markers-lent-outlives
         COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_markers_lent_outlives.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_markers.ts")
set_tests_properties(test-compile-shared-import-own-markers-lent-outlives PROPERTIES PASS_REGULAR_EXPRESSION "borrows a call's result but is used here after it may be released or overwritten"
                                                                                   FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# @record across the boundary (own spec 32): the importer owns the box a borrowing record's
# function returns, and knows it only from the declaration's @record
foreach(record_mm gc rc own)
    tslang_add_test(NAME test-compile-shared-export-import-own-record-${record_mm} COMMAND test-runner -shared -mm=${record_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_record.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_record.ts")
    tslang_add_test(NAME test-jit-shared-export-import-own-record-${record_mm} COMMAND test-runner -jit -shared -mm=${record_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_record.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_record.ts")
endforeach()
# ...a read of the record after an argument it borrows is given a new value is refused
add_test(NAME test-compile-shared-import-own-record-outlives
         COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_record_outlives.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_record.ts")
set_tests_properties(test-compile-shared-import-own-record-outlives PROPERTIES PASS_REGULAR_EXPRESSION "borrows a call's result but is used here after it may be released or overwritten"
                                                                       FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# ...and after a call of a method of a class that is not @final: an override may destroy what it borrows
add_test(NAME test-compile-shared-import-own-record-open-method
         COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_record_open_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_record.ts")
set_tests_properties(test-compile-shared-import-own-record-open-method PROPERTIES PASS_REGULAR_EXPRESSION "borrows a call's result but is used here after it may be released or overwritten"
                                                                           FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# ...and after a call of a @final class's accessor its library does not list as destroying nothing
add_test(NAME test-compile-shared-import-own-record-final-drops
         COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/import_own_record_final_drops.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_own_record.ts")
set_tests_properties(test-compile-shared-import-own-record-final-drops PROPERTIES PASS_REGULAR_EXPRESSION "borrows the result of 'Maker.pairWith' but is used here after it may be released or overwritten"
                                                                           FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# ...and the importer releases each box, which no run can see (own-shared-record.cmake)
if (WIN32)
    add_test(NAME test-own-shared-record-released
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/own-shared-record"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/own-shared-record.cmake")
endif()
# -mm=own across modules compiled into objects of their own: the importer reads what the library's
# object says its functions destroy (--own-facts, own-facts-objects.cmake)
if (WIN32)
    add_test(NAME test-own-facts-objects
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/own-facts-objects"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/own-facts-objects.cmake")
    # ...and a library's keeper: the importer makes what it gives `@immortalizes` immortal
    add_test(NAME test-own-keepers-objects
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/own-keepers-objects"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/own-keepers-objects.cmake")
endif()
# __MEMORY_MODEL__: the branch not taken is not generated (a call of a function defined nowhere there
# would not link)
foreach(model_mm gc rc own none)
    tslang_add_test(NAME test-jit-${model_mm}-memory-model-constant COMMAND test-runner -jit -mm=${model_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/00memory_model_constant.ts")
    tslang_add_test(NAME test-compile-${model_mm}-memory-model-constant COMMAND test-runner -mm=${model_mm} "${PROJECT_SOURCE_DIR}/test/tester/tests/00memory_model_constant.ts")
endforeach()
# the markers mean nothing to the other models
foreach(markers_mm gc rc)
    tslang_add_test(NAME test-jit-${markers_mm}-own-generator-locals COMMAND test-runner -jit -mm=${markers_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_generator_locals.ts")
    tslang_add_test(NAME test-jit-${markers_mm}-own-param-markers COMMAND test-runner -jit -mm=${markers_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_param_markers.ts")
endforeach()
# A generator method of an exported class is exported like any other method: an importer that
# calls it by name (the default library's String[Symbol.iterator] under the JIT) finds it. Checked
# on Windows, where an exported function is `dllexport` in the IR; ELF has no such mark.
if (WIN32)
    add_test(NAME test-export-generator-method
             COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib
                     "${PROJECT_SOURCE_DIR}/test/tester/tests/export_final_class.ts" -o -)
    set_tests_properties(test-export-generator-method PROPERTIES PASS_REGULAR_EXPRESSION "define dllexport ptr @Counter.upTo"
                                                                 FAIL_REGULAR_EXPRESSION "error|Stack dump")
endif()
# ...and linked against its import library (`@static import`), with members from a base its
# library does not export
tslang_add_test(NAME test-compile-time-shared-final-base COMMAND test-runner -shared -compile-time "${PROJECT_SOURCE_DIR}/test/tester/tests/import_final_base_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_final_base.ts")
add_test(NAME test-compile-shared-import-final-extends
         COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_final_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_final_class.ts")
set_tests_properties(test-compile-shared-import-final-extends PROPERTIES PASS_REGULAR_EXPRESSION "class 'X' cannot extend 'Counter': it is @final"
                                                                         FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-with-class-types COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_class_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_class_types.ts")
# Cross-module `class extends` across a real DLL boundary. Formerly disabled as a
# known issue; fixed by three co-operating changes (see
# docs/cross-module-dynamic-import-instanceof-design.md for the full history):
# ClassMethodAccess resolves unregistered dynamic-import members via inline
# SearchForAddressOfSymbolOp (no fragile global registration), the class vtable
# builder emits runtime symbol resolution for slots owned by a dynamic-import base
# (a link-time address of a DLL-resident symbol does not exist), and
# DeclarationPrinter no longer prints a wrong extends target (own name instead of
# the base's) nor the synthetic base-class storage field (which shifted every
# subsequent field's offset in the importer).
tslang_add_test(NAME test-compile-shared-export-import-owned-returns COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
# One collector per process (docs/single-gc-collector-design.md): the library builds strings only
# the importer holds, then churns different ones, so the LIBRARY's collector has to run and has to
# see the importer's array. Under -jit this failed (2000 of 2000 freed) while TypeScriptRuntime.dll
# linked Boehm statically beside a library on gc.dll.
tslang_add_test(NAME test-compile-shared-export-import-gc-single-collector COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_gc_single_collector.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_gc_single_collector.ts")
tslang_add_test(NAME test-compile-shared-export-import-const-locals COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_locals.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_locals.ts")
tslang_add_test(NAME test-compile-shared-export-import-const-functions COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_functions.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_functions.ts")
tslang_add_test(NAME test-compile-shared-export-import-c-varargs COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_c_varargs.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_c_varargs.ts")
tslang_add_test(NAME test-compile-shared-export-import-generators COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_generators.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_generators.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-extends COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends.ts")
tslang_add_test(NAME test-compile-shared-export-import-throw-class COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_throw_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_throw_class.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-extends-implements-diamond COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_implements_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_implements_diamond.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-extends-multilevel COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_multilevel.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-abstract COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-static COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_static.ts")
# FIXED 2026-07-22 (was disabled): mlirGenClassVirtualTableDefinitionForInterface (a class's own
# per-interface vtable, built when it `implements` something) used to always emit a constant
# SymbolRefOp for the implementing method, unlike mlirGenClassVirtualTableDefinition (the class's
# ordinary vtable), which already special-cased a method owned by a dynamic-import class/base and
# resolved it at runtime instead (SearchForAddressOfSymbolOp) since a link-time constant address
# doesn't exist across a DLL boundary. Mirrored that same runtime-resolution pattern into the
# interface-vtable builder's method branch. That alone broke DWARF debug-info generation
# ("DISubprogram attached to more than one function") because this global's initializer can be
# rebuilt lazily from a call site deep inside an unrelated function (a cast expression in main(),
# via mlirGenCreateInterfaceVTableForClass) - two further fixes were needed: (1) GlobalOpLowering
# (LowerToLLVM.cpp) was giving the synthesized __cctor constructor function the triggering call
# site's own "live" location instead of a neutral one, so the debug-info pass attached the wrong
# DISubprogram to it; (2) createFunctionFromRegion's replaceOpWithNewOp<ReturnOp> (LLVMCodeHelper.h)
# implicitly inherited the replaced terminator's (also wrong) location instead of the function's
# own. See class-implements-interface-dynamic-import-fix memory for the full account.
tslang_add_test(NAME test-compile-shared-export-import-class-implements-interface-multilevel COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_multilevel.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-implements-interface-optional COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_optional.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-structural-interface COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_structural_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_structural_interface.ts")
# FIXED 2026-07-23 (was disabled): the "this.area()-returns-0-via-interface" symptom had nothing
# to do with interfaces or `this` binding. DeclarationPrinter dropped the `abstract` modifier when
# embedding the class declaration into the compiled binary (__decls), so the reimporting module
# treated M.Shape as concrete and synthesized a `.new` method for it - a vtable slot the exporting
# module never created (mlirGenClassNew skips `.new` for abstract classes). Every base-class slot
# after `.instanceOf` was thereby shifted by one in the importer's rebuilt vtable, and describe()'s
# compiled-in slot-1 read for area() landed on `.new` instead - a pointer-returning function called
# through a double-returning signature, so the "result" was whatever garbage was left in XMM0.
# That garbage is why the symptom looked interface-specific: a preceding `assert(s.area() == 9)`
# left 9.0 in XMM0, making the direct and base-class-cast calls "pass" by pure luck, while the
# interface path (more string machinery in between) got 0. See the abstract-virtual-dispatch tests
# below for the deterministic minimal regression pair.
tslang_add_test(NAME test-compile-shared-export-import-class-implements-interface-abstract COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_abstract.ts")
# minimal deterministic regression pair for the vtable-slot-shift bug above (single-level abstract,
# describe() called twice before any area() call so XMM0 luck can't hide a regression)
tslang_add_test(NAME test-compile-shared-export-import-class-abstract-virtual-dispatch COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract_virtual_dispatch.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract_virtual_dispatch.ts")
tslang_add_test(NAME test-compile-shared-export-import-cppvtable-class COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_cppvtable_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_cppvtable_class.ts")
# FIXED: see the matching test-compile-export-import-class-generic comment above (2026-07-22).
tslang_add_test(NAME test-compile-shared-export-import-class-generic COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-compile-shared-export-import-function-generic COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-compile-shared-export-import-type-alias-generic COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
tslang_add_test(NAME test-compile-shared-export-import-function COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-compile-shared-export-import-dllname COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_dllname.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_dllname.ts")
tslang_add_test(NAME test-compile-shared-export-import-linkname COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_linkname.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_linkname.ts")
tslang_add_test(NAME test-compile-declare-linkname COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/declare_linkname.ts")
tslang_add_test(NAME test-jit-declare-linkname COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/declare_linkname.ts")
tslang_add_test(NAME test-compile-declare-linkname-shared-symbol COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/declare_linkname_shared_symbol.ts")
tslang_add_test(NAME test-jit-declare-linkname-shared-symbol COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/declare_linkname_shared_symbol.ts")
# `@dllname("strlen") @linkname("strlen")` names the attribute once: two entries of one name are an
# assert in a Debug build ("DictionaryAttr element names must be unique") and print twice in Release.
add_test(NAME test-declare-linkname-attribute-once
         COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib
                 "${PROJECT_SOURCE_DIR}/test/tester/tests/declare_linkname_shared_symbol.ts")
set_tests_properties(test-declare-linkname-attribute-once
                     PROPERTIES PASS_REGULAR_EXPRESSION "dllname = \"strlen\""
                                FAIL_REGULAR_EXPRESSION "dllname = \"strlen\", dllname|Assertion failed|Stack dump")
# DISABLED: importing a module's `export declare function` bound to a C runtime function crashes,
# and did before @linkname (the same with @dllname on a 2026-09-23 build). The importer re-prints
# the declaration as `@dllimport @dllname("strlen")` and looks `strlen` up at startup in the loaded
# libraries (tslang_search_for_address_of_symbol); the exporting DLL never contains it - an unused
# declaration is dropped - so the lookup returns null and the call faults. Enable once that is fixed.
tslang_add_test(NAME test-compile-shared-export-import-declare-linkname COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_declare_linkname.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_declare_linkname.ts")
set_tests_properties(test-compile-shared-export-import-declare-linkname PROPERTIES DISABLED TRUE)
if (TSLANG_X86_TWIN)
    set_tests_properties(${TSLANG_X86_TWIN} PROPERTIES DISABLED TRUE)
endif()
tslang_add_test(NAME test-compile-shared-export-import-type-alias COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias.ts")
tslang_add_test(NAME test-compile-shared-export-import-interface-generic COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_generic.ts")
# FIXED (2026-07-22): the -shared declaration-reconstruction path never printed real
# get/set syntax for a class's accessors, only the underlying get_x/set_x funcOps as
# ordinary methods - so property-style access ("Class member 'celsius' can't be found")
# never populated the reimported class's accessors list. See print(ClassInfo::TypePtr)
# in DeclarationPrinter.cpp.
tslang_add_test(NAME test-compile-shared-export-import-class-accessor COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_accessor.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_accessor.ts")
tslang_add_test(NAME test-compile-shared-export-import-class-indexer COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_indexer.ts")
tslang_add_test(NAME test-compile-shared-export-import-interface-indexer COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_indexer.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-with-interface COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_interface.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-untyped COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-untyped-multi-method COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped_multi_method.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-params COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_params.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_params.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-multi-method COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_multi_method.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-interleaved COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_interleaved.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_interleaved.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-extends-interface COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-extends-interface-multilevel COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_multilevel.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-extends-interface-diamond COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_diamond.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-extends-interface-triple COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_triple.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_triple.ts")
tslang_add_test(NAME test-compile-shared-export-import-object-literal-structural-typed-extends-interface-optional COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_optional.ts")
tslang_add_test(NAME test-compile-shared-export-import-vars COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars.ts")
tslang_add_test(NAME test-compile-shared-export-import-vars-2 COMMAND test-runner -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars2.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars2.ts")
tslang_add_test(NAME test-compile-shared-export-import-enum COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_enum.ts")

# shared libs tests (dlls/dynamics)
tslang_add_test(NAME test-jit-shared-component COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/use_shared.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/shared.ts")
tslang_add_test(NAME test-jit-shared-decl-emit-interface COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_interface.ts")
tslang_add_test(NAME test-jit-shared-decl-emit-type COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_type.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-jit-shared-decl-emit-enum COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_enum.ts")
tslang_add_test(NAME test-jit-shared-decl-emit-class COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")

# shared libs tests (exports/imports)
tslang_add_test(NAME test-jit-shared-export-import-class-interface COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_interface.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-with-class-types COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_class_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_class_types.ts")
tslang_add_test(NAME test-jit-shared-export-import-owned-returns COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-jit-shared-export-import-gc-single-collector COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_gc_single_collector.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_gc_single_collector.ts")
tslang_add_test(NAME test-jit-shared-export-import-const-locals COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_locals.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_locals.ts")
tslang_add_test(NAME test-jit-shared-export-import-const-functions COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_functions.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_functions.ts")
tslang_add_test(NAME test-jit-shared-export-import-c-varargs COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_c_varargs.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_c_varargs.ts")
tslang_add_test(NAME test-jit-shared-export-import-generators COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_generators.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_generators.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-extends COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends.ts")
tslang_add_test(NAME test-jit-shared-export-import-throw-class COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_throw_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_throw_class.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-extends-multilevel COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_multilevel.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-extends-implements-diamond COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_implements_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_implements_diamond.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-abstract COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-static COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_static.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-implements-interface-multilevel COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_multilevel.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-implements-interface-optional COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_optional.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-structural-interface COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_structural_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_structural_interface.ts")
# FIXED 2026-07-23 (was disabled): see the matching test-compile-shared-export-import-class-implements-interface-abstract comment above (DeclarationPrinter dropped `abstract`, shifting the reimported vtable's slots).
tslang_add_test(NAME test-jit-shared-export-import-class-implements-interface-abstract COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_abstract.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-abstract-virtual-dispatch COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract_virtual_dispatch.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract_virtual_dispatch.ts")
tslang_add_test(NAME test-jit-shared-export-import-cppvtable-class COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_cppvtable_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_cppvtable_class.ts")
# FIXED: see the matching test-compile-export-import-class-generic comment above (2026-07-22).
tslang_add_test(NAME test-jit-shared-export-import-class-generic COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-jit-shared-export-import-function-generic COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-jit-shared-export-import-type-alias-generic COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
tslang_add_test(NAME test-jit-shared-export-import-function COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-jit-shared-export-import-type-alias COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias.ts")
tslang_add_test(NAME test-jit-shared-export-import-interface-generic COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_generic.ts")
# FIXED: see the matching test-compile-shared-export-import-class-accessor comment above (2026-07-22).
tslang_add_test(NAME test-jit-shared-export-import-class-accessor COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_accessor.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_accessor.ts")
tslang_add_test(NAME test-jit-shared-export-import-class-indexer COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_indexer.ts")
tslang_add_test(NAME test-jit-shared-export-import-interface-indexer COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_indexer.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-with-interface COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_interface.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-untyped COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-untyped-multi-method COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped_multi_method.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-params COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_params.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_params.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-multi-method COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_multi_method.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-interleaved COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_interleaved.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_interleaved.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-extends-interface COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-extends-interface-multilevel COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_multilevel.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-extends-interface-diamond COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_diamond.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-extends-interface-triple COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_triple.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_triple.ts")
tslang_add_test(NAME test-jit-shared-export-import-object-literal-structural-typed-extends-interface-optional COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_optional.ts")
tslang_add_test(NAME test-jit-shared-export-import-vars COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars.ts")
tslang_add_test(NAME test-jit-shared-export-import-vars-2 COMMAND test-runner -jit -shared -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars2.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars2.ts")
tslang_add_test(NAME test-jit-shared-export-import-enum COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_enum.ts")

# -mm=rc builds every allocation with a live reference count in the block header, generates
# the reference-counting routines, and now calls them: a local holding a heap reference takes
# one at its declaration and gives it back at every exit from its scope. A representative set
# rather than the whole suite, since the cost is a second full compile per test.
#
# These are the only tests ownership insertion can break. The ops it emits erase whole under a
# collector, so the rest of the suite is immune to where they land, not merely expected to
# survive it.
tslang_add_test(NAME test-jit-rc-strings COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00strings.ts")
tslang_add_test(NAME test-jit-rc-str-null COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00str_null.ts")
tslang_add_test(NAME test-jit-rc-array COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00array.ts")
tslang_add_test(NAME test-jit-rc-array-push-pop COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00array4_push_pop.ts")
tslang_add_test(NAME test-jit-rc-array-splice COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_splice.ts")
tslang_add_test(NAME test-jit-rc-any COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00any.ts")
tslang_add_test(NAME test-jit-rc-any-compare COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_compare.ts")
tslang_add_test(NAME test-jit-rc-any-types COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_types.ts")
tslang_add_test(NAME test-jit-rc-class COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00class.ts")
tslang_add_test(NAME test-jit-rc-union-type COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00union_type.ts")
tslang_add_test(NAME test-jit-rc-tuple COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00tuple.ts")
tslang_add_test(NAME test-jit-rc-interface COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00interface.ts")
tslang_add_test(NAME test-jit-rc-generator COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator.ts")
tslang_add_test(NAME test-jit-rc-new-delete COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00new_delete.ts")
tslang_add_test(NAME test-jit-rc-try-catch COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch.ts")
tslang_add_test(NAME test-jit-rc-for-of COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00for_of.ts")
tslang_add_test(NAME test-jit-rc-print COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00print.ts")
tslang_add_test(NAME test-jit-rc-owned-locals COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_locals.ts")
tslang_add_test(NAME test-jit-rc-throw-in-catch COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_in_catch.ts")
tslang_add_test(NAME test-jit-rc-throw-inlined COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_inlined.ts")
tslang_add_test(NAME test-jit-none-throw-inlined COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_inlined.ts")
tslang_add_test(NAME test-jit-rc-using-nested-scopes COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_nested_scopes.ts")
tslang_add_test(NAME test-jit-none-using-nested-scopes COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_nested_scopes.ts")
tslang_add_test(NAME test-jit-rc-using-unwind-progress COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_unwind_progress.ts")
tslang_add_test(NAME test-jit-none-using-unwind-progress COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00using_unwind_progress.ts")
tslang_add_test(NAME test-jit-rc-break-continue-scope-exit COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00break_continue_scope_exit.ts")
tslang_add_test(NAME test-jit-none-break-continue-scope-exit COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00break_continue_scope_exit.ts")
tslang_add_test(NAME test-jit-rc-owned-fields COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_fields.ts")
tslang_add_test(NAME test-jit-none-owned-fields COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_fields.ts")
tslang_add_test(NAME test-jit-rc-owned-elements COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_elements.ts")
tslang_add_test(NAME test-jit-none-owned-elements COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_elements.ts")
tslang_add_test(NAME test-jit-rc-owned-literals COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_literals.ts")
tslang_add_test(NAME test-jit-none-owned-literals COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_literals.ts")
tslang_add_test(NAME test-jit-rc-owned-array-ops COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_array_ops.ts")
tslang_add_test(NAME test-jit-none-owned-array-ops COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_array_ops.ts")
tslang_add_test(NAME test-jit-rc-owned-inline-records COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_inline_records.ts")
tslang_add_test(NAME test-jit-none-owned-inline-records COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_inline_records.ts")
tslang_add_test(NAME test-jit-rc-owned-transfer COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_transfer.ts")
tslang_add_test(NAME test-jit-none-owned-transfer COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_transfer.ts")
tslang_add_test(NAME test-jit-rc-owned-call-results COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_call_results.ts")
tslang_add_test(NAME test-jit-none-owned-call-results COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_call_results.ts")
tslang_add_test(NAME test-jit-rc-owned-named-consts COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_named_consts.ts")
tslang_add_test(NAME test-jit-none-owned-named-consts COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_named_consts.ts")
tslang_add_test(NAME test-jit-rc-owned-temporaries COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_temporaries.ts")
tslang_add_test(NAME test-jit-none-owned-temporaries COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_temporaries.ts")
tslang_add_test(NAME test-jit-rc-owned-interfaces COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_interfaces.ts")
tslang_add_test(NAME test-jit-none-owned-interfaces COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_interfaces.ts")
tslang_add_test(NAME test-jit-rc-owned-closures COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_closures.ts")
tslang_add_test(NAME test-jit-none-owned-closures COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_closures.ts")
tslang_add_test(NAME test-jit-rc-owned-any-boxing COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_any_boxing.ts")
tslang_add_test(NAME test-jit-none-owned-any-boxing COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_any_boxing.ts")
tslang_add_test(NAME test-jit-rc-owned-strings COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_strings.ts")
tslang_add_test(NAME test-jit-none-owned-strings COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_strings.ts")
tslang_add_test(NAME test-jit-rc-owned-generators COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generators.ts")
tslang_add_test(NAME test-jit-none-owned-generators COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generators.ts")
tslang_add_test(NAME test-jit-rc-owned-globals COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_globals.ts")
tslang_add_test(NAME test-jit-none-owned-globals COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_globals.ts")
tslang_add_test(NAME test-jit-rc-owned-generator-locals COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generator_locals.ts")
tslang_add_test(NAME test-jit-none-owned-generator-locals COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_generator_locals.ts")
tslang_add_test(NAME test-jit-rc-owned-iteration COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_iteration.ts")
tslang_add_test(NAME test-jit-none-owned-iteration COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_iteration.ts")
tslang_add_test(NAME test-jit-rc-owned-async COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_async.ts")
tslang_add_test(NAME test-jit-none-owned-async COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_async.ts")
tslang_add_test(NAME test-jit-rc-owned-nested-captures COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_nested_captures.ts")
tslang_add_test(NAME test-jit-none-owned-nested-captures COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_nested_captures.ts")
tslang_add_test(NAME test-jit-rc-owned-unions COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_unions.ts")
tslang_add_test(NAME test-jit-none-owned-unions COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_unions.ts")
tslang_add_test(NAME test-jit-rc-owned-delete COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_delete.ts")
tslang_add_test(NAME test-jit-none-owned-delete COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_delete.ts")
tslang_add_test(NAME test-jit-rc-owned-early-return COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_early_return.ts")
tslang_add_test(NAME test-jit-none-owned-early-return COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_early_return.ts")
tslang_add_test(NAME test-jit-rc-owned-construct-interface COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_construct_interface.ts")
tslang_add_test(NAME test-jit-none-owned-construct-interface COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_construct_interface.ts")
tslang_add_test(NAME test-jit-rc-try-using-catch COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_using_catch.ts")
tslang_add_test(NAME test-jit-none-try-using-catch COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_using_catch.ts")
tslang_add_test(NAME test-jit-rc-disposable-scopes COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/04disposable.ts")
tslang_add_test(NAME test-jit-none-disposable-scopes COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/04disposable.ts")
tslang_add_test(NAME test-jit-rc-disposable-unwind COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/03disposable.ts")
tslang_add_test(NAME test-jit-none-disposable-unwind COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/03disposable.ts")

# Allocating inside a Win64 catch/finally funclet - the shape that only faults once the model
# stops going through the collector, so it needs both non-gc variants to be worth anything.
tslang_add_test(NAME test-compile-00-alloc-in-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00alloc_in_catch.ts")
tslang_add_test(NAME test-jit-00-alloc-in-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00alloc_in_catch.ts")
tslang_add_test(NAME test-jit-rc-alloc-in-catch COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00alloc_in_catch.ts")
tslang_add_test(NAME test-jit-none-alloc-in-catch COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00alloc_in_catch.ts")

# A try/catch inside a catch clause - crashed in every model at every optimisation level.
tslang_add_test(NAME test-compile-00-nested-catch COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_catch.ts")
tslang_add_test(NAME test-compile-00-try-mismatch-in-catch-clause COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_mismatch_in_catch_clause.ts")
tslang_add_test(NAME test-compile-00-try-mismatch-in-catch-clause-using COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_mismatch_in_catch_clause_using.ts")
tslang_add_test(NAME test-jit-00-nested-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_catch.ts")
tslang_add_test(NAME test-jit-00-try-mismatch-in-catch-clause COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_mismatch_in_catch_clause.ts")
tslang_add_test(NAME test-jit-00-try-mismatch-in-catch-clause-using COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_mismatch_in_catch_clause_using.ts")
tslang_add_test(NAME test-jit-rc-nested-catch COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_catch.ts")
tslang_add_test(NAME test-jit-none-nested-catch COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_catch.ts")

# `-mm=none` - leak everything - had no coverage at all before the rename that gave it a name
# of its own. One test, so a future change to the model plumbing cannot silently break it.
tslang_add_test(NAME test-jit-none-strings COMMAND test-runner -jit -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/00strings.ts")

# ============================================================================
# The corpus, under the models that are not the default
#
# `-mm=rc` ran 42 files of this corpus and `-mm=none` ran 24, and neither ran a
# single one ahead of time. That is why three faults in a row were found by an
# out-of-tree sweep instead of here, and why the last of them was invisible
# twice over - once for the model, and once for the build mode. Every file the
# default model runs in a tier is run again below, in the same tier, under both
# other models. See docs/reference-counting-evaluation.md, plan item 5y.
#
# A file that does not pass yet under a model is listed in the matching BROKEN
# set below, and registered but disabled - so what is broken is written down in
# the build rather than in a document somebody has to find.
# ============================================================================

# every single-file test the default model runs, in either tier
set(TSLANG_CORPUS
    00any_truthiness_objects.ts
    00add_promotes_both_operands.ts
    00conditional_owned_result.ts
    00const_array_nested_elements.ts
    00nested_array_literal_mutation.ts
    00const_array_to_array_elements.ts
    00array_widen_elements_now.ts
    00owned_array_splice.ts
    00alloc_in_catch.ts
    00any_compare.ts
    00any_generic_equals.ts
    00any_types.ts
    00any_to_interface.ts
    00any.ts
    00any_equality_undefined.ts
    00array_assignment5.ts
    00array_cond_access.ts
    00array_length_grow_zeroed.ts
    00array_move_wide_elements.ts
    00array_of.ts
    00array_shift.ts
    00array_splice.ts
    00array_spread_args.ts
    00array_view.ts
    00const_array_push.ts
    00array.ts
    00array2.ts
    00array_reference.ts
    00array3.ts
    00array_zero_slots.ts
    00array4_push_pop.ts
    00array5_deconst.ts
    00array_rest_copy.ts
    00array6.ts
    00array_static_nested.ts
    00array_literal_copy.ts
    00tuple_runtime_arrays.ts
    00record_to_class.ts
    00record_variable_to_class.ts
    00array_insert_functions.ts
    00await_void.ts
    00for_await_sequential.ts
    00for_await_control.ts
    00for_optional_class_condition.ts
    00for_condition_narrowing.ts
    00param_assigned_owned.ts
    00string_empty_falsy.ts
    00string_concat_bounds.ts
    00parse_number_range.ts
    00global_argument_held.ts
    00shared_global_field_argument.ts
    00narrowed_assign_other_member.ts
    00bound_method_count.ts
    00const_view_of_field.ts
    00const_view_of_element.ts
    00const_record_owned_fields.ts
    00generator_temporaries.ts
    00conditional_named_const_release.ts
    00array_length_valid.ts
    00const_interface_of_local.ts
    00self_assign.ts
    00string_view_copy.ts
    00global_copy_counted.ts
    00global_interface_initializer.ts
    00interface_cast_unrelated.ts
    00interface_union_cast_unrelated.ts
    00c_string_copy.ts
    00const_field_read.ts
    00const_element_read.ts
    00const_global_read.ts
    00global_argument_reassigned.ts
    00global_argument_view.ts
    00bound_method_counts_object.ts
    00closure_const_read.ts
    00element_store_after_value.ts
    00const_interface_field_read.ts
    00argument_place_overwritten.ts
    00const_bound_method_declared_type.ts
    00nullable_class_to_interface.ts
    00class_to_interface_null.ts
    00null_interface_to_interface.ts
    00interface_type_tag_in_vtable.ts
    00nullable_interface_to_interface.ts
    00argument_place_aliased.ts
    00temporary_read_in_finally.ts
    00unowned_local_store.ts
    00array_argument_held.ts
    00argument_slot_assigned.ts
    00array_data_string_alias.ts
    00array7.ts
    00array8_tuple_spread.ts
    00arrow_generic.ts
    00arrow_global.ts
    00arrow_in_arrow.ts
    00as_const.ts
    00as.ts
    00assert.ts
    00async_await.ts
    00async_gc_threading.ts
    00async_result_types.ts
    00bool_arith_ops.ts
    00int64_shift_bitwise.ts
    00int64_mul_literal.ts
    00int_literal_above_s32.ts
    00literal_union_arith.ts
    00literal_union_cast.ts
    00narrowing_and_or.ts
    00enum_compare.ts
    00bool_const_array.ts
    00break_continue_scope_exit.ts
    00break_continue.ts
    00label_chain_loop.ts
    00labelled_scope_exit.ts
    00switch_case_scope_exit.ts
    00using_dispose_order.ts
    00using_unwind_outer_locals.ts
    00try_unwind_outer_locals.ts
    00try_return_through_finally.ts
    00try_scope_exit_through_finally.ts
    00try_jump_from_catch.ts
    00capture_in_new_arguments.ts
    00capture_in_new_untyped_arrow.ts
    00cast_to_f32.ts
    00catch_untyped_value.ts
    00catch_value.ts
    00catch_value_minimal.ts
    00class_abstract.ts
    00class_override_inferred_return.ts
    00class_access_control.ts
    00class_accessor_super.ts
    00class_accessor_increment.ts
    00class_accessor_getter_only_abstract.ts
    00class_accessor_virtual.ts
    00class_accessor_implements_field.ts
    00object_literal_contextual_lambda.ts
    00class_accessor_untyped_setter.ts
    00class_accessor.ts
    00class_accessor2.ts
    00class_constr_fields.ts
    00class_constructor_overloads.ts
    00callback_result_ignored.ts
    00generic_default_no_inference.ts
    00promise_type.ts
    00class_deconst.ts
    00class_def_constr.ts
    00class_discover_types.ts
    00class_expression.ts
    00class_expression2.ts
    00class_expression3.ts
    00class_expression_no_ctor.ts
    00class_from_tuple.ts
    00class_generic_iface.ts
    00class_generic_method.ts
    00class_generic_method2.ts
    00class_generic.ts
    00class_iface.ts
    00class_indexer_super.ts
    00class_indexer.ts
    00class_iterator_super.ts
    00class_iterator.ts
    00class_local_decl.ts
    00class_nested.ts
    00class_new.ts
    00array_growth.ts
    00class_or_interface_to_tuple.ts
    00class_protected_constr.ts
    00class_recursive.ts
    00class_stack.ts
    00class_static_block.ts
    00class_static_constr.ts
    00class_static_generic_method.ts
    00class_static_generic_method2.ts
    00class_static_generic_method3.ts
    00class_structural_extends.ts
    00class_super_static.ts
    00class_super.ts
    00class_implicit_constructor.ts
    00class_symbol_iterator.ts
    00class_var_args.ts
    00class_virtual_call.ts
    00class_virtual_table.ts
    00class.ts
    00computedpropertyname.ts
    00cond_expr.ts
    00conditional_type.ts
    00decorators.ts
    00disposable.ts
    00dowhile.ts
    00enum_multiple.ts
    00enum.ts
    00equals.ts
    00every.ts
    00extends.ts
    00extension_cond_access.ts
    00extension.ts
    00filter.ts
    00for_await_yield.ts
    00for_await.ts
    00for_in.ts
    00for_infinite_result.ts
    00for_of.ts
    00for.ts
    00funcs_bindings.ts
    00funcs_c_varargs_storage.ts
    00funcs_capture.ts
    00funcs_expression_generic.ts
    00funcs_expression_iterator.ts
    00funcs_generic_arrow.ts
    00funcs_generic_iterator.ts
    00funcs_generic_with_typeof.ts
    00funcs_generic.ts
    00funcs_hybrid_null_this.ts
    00funcs_nesting_capture.ts
    00funcs_nesting_generic.ts
    00funcs_nesting.ts
    00funcs_typed_generic_iterator.ts
    00funcs_typeless_function_as_generic.ts
    00nested_implicit_generics.ts
    00funcs_vararg.ts
    00function_overloads.ts
    00gc_allocations_distinct.ts
    00assert_message.ts
    00funcs.ts
    00generator_manual_next.ts
    00generator_manual_next2.ts
    00generator.ts
    00generator2.ts
    00generator3.ts
    00generator4.ts
    00generator5.ts
    00generator6.ts
    00generator7.ts
    00generator_done_value_zero.ts
    00generator_yield_value.ts
    00generic_arguments_name_conflict.ts
    00generic_class_constructor_inference.ts
    00generic_ref_inference.ts
    00global_const_object_method.ts
    00globals.ts
    00globals2.ts
    00globals3.ts
    00globals_entry_point.ts
    00if_conditional_compile.ts
    00if_return.ts
    00in_method_names.ts
    00in.ts
    00global_from_global.ts
    00try_cleanup_then_catch.ts
    00try_empty_catch.ts
    00global_object_literal.ts
    00infer.ts
    00const_object_literal_field.ts
    00global_union_from_call.ts
    00instanceof.ts
    00unbraced_branch_scope.ts
    00global_constant_concat.ts
    00const_of_let.ts
    00instanceof_namespace_narrowing.ts
    00global_initializer_branches.ts
    00instanceof_union.ts
    00nullable_to_any.ts
    00ternary_new_or_null.ts
    00interface_captures.ts
    00interface_conjunction.ts
    00interface_function_typed_field.ts
    00interface_function_typed_field_class.ts
    00interface_function_typed_field_class_field.ts
    00interface_function_field_write.ts
    00interface_function_typed_field_object_closure.ts
    00class_function_typed_field_inferred.ts
    00interface_generic.ts
    00interface_global_method.ts
    00interface_indexer.ts
    00interface_new.ts
    00interface_object_array.ts
    00interface_object.ts
    00interface_object2.ts
    00interface_object3.ts
    00interface_object4.ts
    00interface_object5.ts
    00interface_optional_cast_order.ts
    00interface_optional_extends.ts
    00interface_optional_function_typed_field.ts
    00interface_optional_function_typed_field_class_field.ts
    00interface_optional_method_extends.ts
    00interface_optional.ts
    00interface_partial.ts
    00interface.ts
    00interface2.ts
    00intersection_type_generic.ts
    00intersection_type.ts
    00iterator_bug.ts
    00iterator_methods.ts
    00iterator.ts
    00lambdas_generic_global.ts
    00lambdas.ts
    00length.ts
    00map.ts
    00method_access_cond.ts
    00method_this_param.ts
    00mixed_type_ops.ts
    00names_conflict.ts
    00nested_catch.ts
    00try_mismatch_in_catch_clause.ts
    00try_mismatch_in_catch_clause_using.ts
    00new_delete.ts
    00new_generic_from_receiver.ts
    00generic_specialization_cast.ts
    00ns_bug.ts
    00ns.ts
    00ns2.ts
    00ns3.ts
    00ns4.ts
    00ns5.ts
    00ns_member_shadowed.ts
    00numbers.ts
    00object_accessor.ts
    00object_annotated_method_extends_interface_multilevel.ts
    00object_annotated_method_extends_interface.ts
    00object_annotated_method_interleaved.ts
    00object_annotated_method_params.ts
    00object_annotated_method.ts
    00object_boxed_infra.ts
    00object_chained_sibling_method_return.ts
    00object_deconst.ts
    00object_func.ts
    00object_func2.ts
    00object_func3.ts
    00object_global.ts
    00object_new.ts
    00object_ref_semantics.ts
    00object.ts
    00optional.ts
    00owned_any_boxing.ts
    00owned_array_ops.ts
    00owned_async.ts
    00owned_call_results.ts
    00owned_closures.ts
    00owned_construct_interface.ts
    00owned_delete.ts
    00owned_early_return.ts
    00owned_elements.ts
    00owned_fields.ts
    00owned_generator_locals.ts
    00owned_generators.ts
    00owned_globals.ts
    00owned_inline_records.ts
    00owned_interfaces.ts
    00owned_iteration.ts
    00owned_literals.ts
    00owned_locals.ts
    00owned_named_consts.ts
    00owned_nested_captures.ts
    00owned_strings.ts
    00owned_temporaries.ts
    00owned_unions.ts
    00owned_transfer.ts
    00prefix_postfix.ts
    00print.ts
    00property_access_cond.ts
    00question_question.ts
    00reduce.ts
    00reference_index_bug.ts
    00reference_null_bug.ts
    00reference_ref_deref.ts
    00rest_args_pack_owned.ts
    00safe_cast_bug.ts
    00safe_cast_field_access.ts
    00safe_cast_early_exit.ts
    00union_object_types_access.ts
    00safe_cast_null_field.ts
    00safe_cast_else_scope.ts
    00safe_cast_typeof.ts
    00safe_cast_while.ts
    00safe_cast.ts
    00safe_cast2.ts
    00sizeof.ts
    00spread_assignment.ts
    00spread.ts
    00stack_test.ts
    00str_null.ts
    00string_length_resize.ts
    00strings.ts
    00switch.ts
    00switch_default_fallthrough.ts
    00switch_char_string_cases.ts
    00symbol.ts
    00throw_in_catch.ts
    00throw_bool_bigint.ts
    00int64_remainder.ts
    00throw_f32.ts
    00throw_u32.ts
    00throw_in_block_debug_info.ts
    00throw_inlined.ts
    00throw_number_untyped_catch.ts
    00to_primitive_ops.ts
    00toplevel_control_flow.ts
    00toplevel_no_main_with_helpers.ts
    00try_catch_mismatch_rethrow.ts
    00try_catch_rethrow.ts
    00try_catch_accessor.ts
    00try_catch_call_in_catch.ts
    00try_catch_return_dispose.ts
    00try_catch_return.ts
    00try_catch_sequence_and_nesting.ts
    00try_catch.ts
    00try_finally_break_continue.ts
    00try_finally_return.ts
    00try_finally_funclet.ts
    00try_in_catch_with_finally.ts
    00try_finally.ts
    00try_finally_throws.ts
    00function_fewer_parameters.ts
    00closure_fewer_parameters.ts
    00try_using_catch.ts
    00tuple_named.ts
    00tuple_with_array.ts
    00tuple.ts
    00type_aliases_in_generics.ts
    00type_guard_function.ts
    00class_cast_structural.ts
    00typed_array.ts
    00typeof_function_narrowing.ts
    00typeof_static_fold.ts
    00typeof_static_fold_conditions.ts
    00typeof_union_narrowing.ts
    00types_indexedaccesstype.ts
    00types_keyof_enum.ts
    00types_mappedtype.ts
    00types_templateliteraltype.ts
    00template_literal_escapes.ts
    00types_unknown1.ts
    00any_to_tuple.ts
    00bigint_ops.ts
    00namespace_forward_class.ts
    00any_literal_to_interface.ts
    00interface_method_object_field_type.ts
    00unknown_round_trip.ts
    00types_utility.ts
    00types.ts
    00uint_compare_bug.ts
    00undef.ts
    00union_bin_ops.ts
    00union_bin_ops2.ts
    00union_errors.ts
    00union_null_to_string.ts
    00union_null_truthiness.ts
    00null_into_tagged_union.ts
    00union_null_in_pointer.ts
    00union_ops.ts
    00union_to_any.ts
    00union_type.ts
    00using_nested_scopes.ts
    00using_unwind_progress.ts
    00var_bindings.ts
    00vars.ts
    00void.ts
    00while.ts
    01any.ts
    01arguments.ts
    01class_new.ts
    01disposable.ts
    01enum.ts
    01extends.ts
    01extension.ts
    01funcs_generic_iterator.ts
    01funcs_generic.ts
    01funcs_vararg.ts
    01iterator.ts
    01lambdas.ts
    01map.ts
    01method_access_cond.ts
    01optional.ts
    01print-bug.ts
    01reduce.ts
    01safe_cast_while.ts
    01sizeof.ts
    01spread_assignment.ts
    01spread.ts
    01switch.ts
    01symbol.ts
    01try_catch_return_dispose.ts
    01try_catch.ts
    01try_finally.ts
    01tuple.ts
    01types_mappedtype.ts
    01types_templateliteraltype.ts
    01types_utility.ts
    01union_type.ts
    02disposable.ts
    02extends.ts
    02funcs_generic_iterator.ts
    02funcs_vararg.ts
    02iterator.ts
    02numbers.ts
    02sizeof.ts
    02union_type.ts
    03disposable.ts
    03iterator.ts
    03union_type.ts
    03union_type_case_order.ts
    04disposable.ts
    04union_type.ts
    05strings.ts
    05union_type.ts
    06numbercollections.ts
    07stringcollections.ts
    08stringopertations.ts
    09postprefix.ts
    10arrayincrement.ts
    11equalsoperator.ts
    12referencecollection.ts
    13actions.ts
    14lazyoperations.ts
    15references_capture.ts
    15references.ts
    17classes.ts
    18enums.ts
    19forof_capture.ts
    19forof.ts
    20maps.ts
    22lambdas.ts
    23generics.ts
    241arrayforeach.ts
    243arrayevery.ts
    244arraysome.ts
    25lamdacapture.ts
    26staticclasses.ts
    28boolcasts.ts
    29lazyreferences.ts
    32complexcalls.ts
    33inheritance.ts
    34switch.ts
    35lambdaparameters.ts
    36interfaces.ts
    39objectdestructuring.ts
    40generics.ts
    41anonymoustypes.ts
    42lambdaproperties.ts
    43nestednamespace.ts
    44toplevelcode.ts
    45enumtostring.ts
    48instanceof.ts
    51exceptions.ts
    abstractPropertyInConstructor.ts
    additionOperatorWithNumberAndEnum.ts
    arithmeticOperatorWithEnum.ts
    arithmeticOperatorWithTypeParameter.ts
    arrayFakeFlatNoCrashInferenceDeclarations.ts
    arrayLiterals.ts
    arrayLiterals2ES5.ts
    arrayLiterals3.ts
    assignmentTypeNarrowing.ts
    asyncMethodWithSuper_es2017.ts
    callWithSpread.ts
    comparisonOperatorWithIdenticalObjects.ts
    conditionalTypes1.ts
    conditionalTypes2.ts
    declarationsAndAssignments.ts
    dependencies.ts
    disallowLineTerminatorBeforeArrow.ts
    emitDefaultParametersFunctionExpression.ts
    gc_malloc_cse_o3.ts
    Grammar_and_types.ts
    internals.ts
    load_store_decorators.ts
    logicalAssignment5_2.ts
    logicalAssignment5_3.ts
    logicalAssignment5.ts
    nbody.ts
    newWithSpread.ts
    no_main.ts
    noPropertyAccessFromIndexSignature1.ts
    parser.ts
    path.ts
    raytrace.ts
    structural-typing.ts
    thisTypeInClasses.ts
    typeGuardFunction.ts
    typeGuardOfFormThisMember.ts
    typeGuardOfFormTypeOfBoolean.ts
    types_vs_interfaces.ts
    00any_number_compare.ts
    00narrow_array_null.ts
    00opaque_nullable_compare.ts
    00cppvtable_class.ts
    00cpp_slot.ts
    00cpp_overloads.ts
    )

# the default model runs these ahead of time only, so the other models do too
set(TSLANG_CORPUS_NO_JIT
    00array_cond_access.ts
    00try_catch_mismatch_rethrow.ts
    )

# and this one in the JIT only
set(TSLANG_CORPUS_NO_AOT
    02funcs_vararg.ts
    )

# Files with a hand-written `-mm=rc` JIT entry further up. They keep the names the
# evaluation document cites, so the generated pass skips them rather than running
# each of them twice.
set(TSLANG_CORPUS_RC_NAMED
    00alloc_in_catch.ts
    00any_compare.ts
    00any_types.ts
    00any.ts
    00array_splice.ts
    00array.ts
    00array4_push_pop.ts
    00break_continue_scope_exit.ts
    00class.ts
    00for_of.ts
    00generator.ts
    00interface.ts
    00nested_catch.ts
    00new_delete.ts
    00owned_any_boxing.ts
    00owned_array_ops.ts
    00owned_async.ts
    00owned_call_results.ts
    00owned_closures.ts
    00owned_construct_interface.ts
    00owned_delete.ts
    00owned_early_return.ts
    00owned_elements.ts
    00owned_fields.ts
    00owned_generator_locals.ts
    00owned_generators.ts
    00owned_globals.ts
    00owned_inline_records.ts
    00owned_interfaces.ts
    00owned_iteration.ts
    00owned_literals.ts
    00owned_locals.ts
    00owned_named_consts.ts
    00owned_nested_captures.ts
    00owned_strings.ts
    00owned_temporaries.ts
    00owned_unions.ts
    00owned_transfer.ts
    00print.ts
    00str_null.ts
    00strings.ts
    00throw_in_catch.ts
    00throw_inlined.ts
    00try_catch.ts
    00try_using_catch.ts
    00tuple.ts
    00union_type.ts
    00using_nested_scopes.ts
    00using_unwind_progress.ts
    03disposable.ts
    04disposable.ts
    )

# the same, for `-mm=none`
set(TSLANG_CORPUS_NONE_NAMED
    00alloc_in_catch.ts
    00break_continue_scope_exit.ts
    00nested_catch.ts
    00owned_any_boxing.ts
    00owned_array_ops.ts
    00owned_async.ts
    00owned_call_results.ts
    00owned_closures.ts
    00owned_construct_interface.ts
    00owned_delete.ts
    00owned_early_return.ts
    00owned_elements.ts
    00owned_fields.ts
    00owned_generator_locals.ts
    00owned_generators.ts
    00owned_globals.ts
    00owned_inline_records.ts
    00owned_interfaces.ts
    00owned_iteration.ts
    00owned_literals.ts
    00owned_named_consts.ts
    00owned_nested_captures.ts
    00owned_strings.ts
    00owned_temporaries.ts
    00owned_unions.ts
    00owned_transfer.ts
    00strings.ts
    00throw_inlined.ts
    00try_using_catch.ts
    00using_nested_scopes.ts
    00using_unwind_progress.ts
    03disposable.ts
    04disposable.ts
    )

# Nothing is known broken. The lists are kept because they are how a reference-counting fault
# gets written down while it is being worked on: a file that does not pass yet under a model
# goes here and is registered but DISABLED, so ctest counts it out loud and the list of what is
# broken lives in the build rather than in a document somebody has to find.
#
# Not WILL_FAIL, which was tried first and does not hold - a corrupted heap does not always
# land, so a file can fail six runs in six on its own and come up clean in a suite run, where
# twelve tests at a time give the allocator a different history, and a WILL_FAIL test that
# passes is a red suite.
#
# Six files came off these lists. Section 9.48 took `00mixed_type_ops.ts` (a boxing cast
# reported no memory effects, so CSE merged two boxes over one constant and the block was freed
# twice), section 9.49 took `nbody.ts` and with it the two that wandered, `13actions.ts` and
# `44toplevelcode.ts` (a global never took a reference to what was stored into it), and section
# 9.50 took `00spread.ts` (a generator's locals live in its state object, which released what
# nothing had retained, and a by-value capture took a reference to the cell instead of to the
# value the box would release).
set(TSLANG_CORPUS_BROKEN_JIT_RC
    )

set(TSLANG_CORPUS_BROKEN_JIT_NONE
    )

set(TSLANG_CORPUS_BROKEN_AOT_RC
    )

set(TSLANG_CORPUS_BROKEN_AOT_NONE
    )

foreach(corpus_file ${TSLANG_CORPUS})
    string(REGEX REPLACE "\\.ts$" "" corpus_slug "${corpus_file}")
    string(REGEX REPLACE "[_.]" "-" corpus_slug "${corpus_slug}")
    set(corpus_path "${PROJECT_SOURCE_DIR}/test/tester/tests/${corpus_file}")

    foreach(corpus_model rc none)
        string(TOUPPER "${corpus_model}" corpus_MODEL)

        if(NOT corpus_file IN_LIST TSLANG_CORPUS_NO_JIT AND
           NOT corpus_file IN_LIST TSLANG_CORPUS_${corpus_MODEL}_NAMED)
            set(corpus_test "test-jit-${corpus_model}-corpus-${corpus_slug}")
            tslang_add_test(NAME ${corpus_test}
                            COMMAND test-runner -jit -mm=${corpus_model} "${corpus_path}")
            if(corpus_file IN_LIST TSLANG_CORPUS_BROKEN_JIT_${corpus_MODEL})
                # a -jit registration has no x86 twin
                set_tests_properties(${corpus_test} PROPERTIES DISABLED TRUE)
            endif()
        endif()

        if(NOT corpus_file IN_LIST TSLANG_CORPUS_NO_AOT)
            set(corpus_test "test-compile-${corpus_model}-corpus-${corpus_slug}")
            tslang_add_test(NAME ${corpus_test}
                            COMMAND test-runner -mm=${corpus_model} "${corpus_path}")
            if(corpus_file IN_LIST TSLANG_CORPUS_BROKEN_AOT_${corpus_MODEL})
                # ${TSLANG_X86_TWIN} is the x86 twin tslang_add_test just registered, or empty
                set_tests_properties(${corpus_test} ${TSLANG_X86_TWIN} PROPERTIES DISABLED TRUE)
            endif()
        endif()
    endforeach()
endforeach()

# `fast_math.ts` is the one corpus file the loop leaves out. It needs `-fast-math`, which
# would give the runner a fourth and fifth cached link script (`jitfmrc`, `jitfmnone`, and
# their compile and debug variants) for a test about floating point, where the memory model
# has nothing to do with the answer.

# The ownership verifier over every corpus file. It is the only check here that reads the IR
# rather than the program's output, which is what makes it worth running: a reference nobody
# gives back changes no answer, so nothing else in this suite can see one. See
# verify-ownership.cmake for what it asks and why it is sharded.
#
# The tests from here to the ownership verifier are cmake scripts, not `test-runner` runs, so
# they use plain add_test and have no twin via tslang_add_test: the rc debug-info check, the
# collector-linkage check, the foreign-target import checks and the ownership verifier shards.
# The default-library collector tests are the one exception - they get a hand-registered x86
# twin (test-x86-compile-gc-defaultlib-collector, below) once the default library, built in the
# separate TypeScriptCompilerDefaultLib repository, has an x86 tree (TSLANG_ARCH=x86).
# Reference-counted programs built WITH debug info - see debug-info-rc.cmake for why this needs
# a test of its own rather than a `test-runner` registration.
add_test(NAME test-compile-rc-debug-info
         COMMAND ${CMAKE_COMMAND}
                 -DTSLANG=$<TARGET_FILE:tslang>
                 -DTESTS_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/debug-info-rc.cmake)

# Debug info together with optimisation, under every model - see debug-info-opt.cmake.
add_test(NAME test-compile-debug-info-opt
         COMMAND ${CMAKE_COMMAND}
                 -DTSLANG=$<TARGET_FILE:tslang>
                 -DTESTS_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/debug-info-opt.cmake)

# A throw under the JIT without TypeScriptRuntime (which test-runner always passes), under every
# model but gc - see jit-throw-without-runtime.cmake. Windows only: the symbol is MSVC's type_info vtable.
if (WIN32)
    add_test(NAME test-jit-throw-without-runtime
             COMMAND ${CMAKE_COMMAND}
                     -DTSLANG=$<TARGET_FILE:tslang>
                     -DTESTS_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests
                     -P ${CMAKE_CURRENT_SOURCE_DIR}/jit-throw-without-runtime.cmake)
endif()

# An async program linked by tslang itself (test-runner links with gc.lib whatever the model) under
# every model but gc - see async-without-collector.cmake. Every platform: the static async runtime
# is an archive on Linux too, and before the split it failed there the same way.
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(async_without_collector_opt "--opt;--opt_level=3")
else()
    set(async_without_collector_opt "--di;--opt_level=0")
endif()
add_test(NAME test-compile-async-without-collector
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DTESTS_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/async-without-collector"
                 "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                 "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                 "-DOPT=${async_without_collector_opt}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/async-without-collector.cmake")

# tslang choosing the collector's linkage itself under -mm=gc - see shared-collector-auto.cmake.
if (WIN32)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(gc_shared_auto_opt "--opt;--opt_level=3")
    else()
        set(gc_shared_auto_opt "--di;--opt_level=0")
    endif()
    add_test(NAME test-compile-gc-shared-auto
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DTESTS_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/gc-shared-auto"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     "-DGC_SHARED_LIB=${TEST_GC_SHARED_PREFIX}/lib"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     "-DOPT=${gc_shared_auto_opt}"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/shared-collector-auto.cmake")
endif()

# Importing a DLL for a target other than the host, whose declarations are read from the file -
# see foreign-target-import.cmake. Always on, with no x86 libraries: it is what covers that path
# when the x86 twins are off.
if (WIN32)
    foreach(foreign_import_mode read errors)
        add_test(NAME test-compile-foreign-target-import-${foreign_import_mode}
                 COMMAND ${CMAKE_COMMAND}
                         "-DMODE=${foreign_import_mode}"
                         "-DTSLANG=$<TARGET_FILE:tslang>"
                         "-DTESTS_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests"
                         "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/foreign-target-import-${foreign_import_mode}"
                         "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                         "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                         -P "${CMAKE_CURRENT_SOURCE_DIR}/foreign-target-import.cmake")
    endforeach()
endif()

# The CRT functions a program calls follow its target, not the host - see target-crt.cmake. IR
# only, so every host runs it.
add_test(NAME test-compile-target-crt
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/target-crt"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/target-crt.cmake")

# The default library's folder per target, build and model - see default-lib-dir.cmake.
add_test(NAME test-default-lib-dir
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/default-lib-dir.cmake")

# Compiling and linking for Android - see android-link.cmake. `codegen` needs no NDK and always
# runs; `link` is skipped (not failed) without the NDK and the Android libraries.
foreach(android_mode codegen link)
    add_test(NAME test-compile-android-${android_mode}
             COMMAND ${CMAKE_COMMAND}
                     "-DMODE=${android_mode}"
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/android-${android_mode}"
                     "-DREPO_DIR=${PROJECT_SOURCE_DIR}/.."
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/android-link.cmake")
    set_tests_properties(test-compile-android-${android_mode} PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")
endforeach()

# A gc shared library called from a non-tslang host's own threads - see foreign-thread-gc.cmake.
# Windows needs gc.dll (a shared library takes its collector from it); ELF links libgc.a in.
if (NOT WIN32 OR TSLANG_HAVE_SHARED_GC)
    add_executable(foreign-thread-gc-host foreign-thread-gc-host.cpp)
    if (NOT WIN32)
        target_link_libraries(foreign-thread-gc-host PRIVATE ${CMAKE_DL_LIBS} pthread)
    endif()
    if (WIN32)
        set(gc_foreign_thread_gc_shared "-DGC_SHARED_LIB=${TEST_GC_SHARED_PREFIX}/lib")
    else()
        set(gc_foreign_thread_gc_shared "")
    endif()
    add_test(NAME test-compile-foreign-thread-gc
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DHOST=$<TARGET_FILE:foreign-thread-gc-host>"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/foreign-thread-gc"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     ${gc_foreign_thread_gc_shared}
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/foreign-thread-gc.cmake")
endif()

# One collector per process with the real default library in it - see defaultlib-collector.cmake.
# Skipped (not failed) when no default library has been built; the release workflows run them after
# building one. The sources live outside tests/: without the default library they do not compile,
# and every file in tests/ is compiled alone with --no-default-lib by the ownership verifier.
if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(defaultlib_collector_opt "--opt;--opt_level=3")
else()
    set(defaultlib_collector_opt "--di;--opt_level=0")
endif()
set(defaultlib_collector_llvm_lib "${LLVM_LIBRARY_DIR}")
if (WIN32)
    set(defaultlib_collector_gc_shared "-DGC_SHARED_LIB=${TEST_GC_SHARED_PREFIX}/lib")
else()
    set(defaultlib_collector_gc_shared "")
endif()
foreach(defaultlib_collector_mode jit compile)
    add_test(NAME test-${defaultlib_collector_mode}-gc-defaultlib-collector
             COMMAND ${CMAKE_COMMAND}
                     "-DMODE=${defaultlib_collector_mode}"
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DTSLANG_BIN=${CMAKE_BINARY_DIR}/bin"
                     "-DTESTS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/defaultlib"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/gc-defaultlib-collector-${defaultlib_collector_mode}"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     ${defaultlib_collector_gc_shared}
                     "-DLLVM_LIB=${defaultlib_collector_llvm_lib}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     "-DOPT=${defaultlib_collector_opt}"
                     "-DDEFAULT_LIB_CANDIDATES=${PROJECT_SOURCE_DIR}/../TypeScriptCompilerDefaultLib/__build;${PROJECT_SOURCE_DIR}/../../TypeScriptCompilerDefaultLib/__build"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/defaultlib-collector.cmake")
    set_tests_properties(test-${defaultlib_collector_mode}-gc-defaultlib-collector PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")
endforeach()

# -mm=own with the real default library: its facts reach the program, under the JIT as for an exe
# (see own-defaultlib-facts.cmake). Skipped when no own default library has been built.
foreach(own_defaultlib_mode jit compile)
    add_test(NAME test-${own_defaultlib_mode}-own-defaultlib-facts
             COMMAND ${CMAKE_COMMAND}
                     "-DMODE=${own_defaultlib_mode}"
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DTSLANG_BIN=${CMAKE_BINARY_DIR}/bin"
                     "-DTEST=${CMAKE_CURRENT_SOURCE_DIR}/defaultlib/own_default_lib_facts.ts"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/own-defaultlib-facts-${own_defaultlib_mode}"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     "-DLLVM_LIB=${defaultlib_collector_llvm_lib}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     "-DOPT=${defaultlib_collector_opt}"
                     "-DDEFAULT_LIB_CANDIDATES=${PROJECT_SOURCE_DIR}/../TypeScriptCompilerDefaultLib/__build;${PROJECT_SOURCE_DIR}/../../TypeScriptCompilerDefaultLib/__build"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/own-defaultlib-facts.cmake")
    set_tests_properties(test-${own_defaultlib_mode}-own-defaultlib-facts PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")
endforeach()

# x86 twin of the compile-mode default-library collector test above (the x86 default library,
# spec Phase 4, As built). No JIT twin: the JIT refuses an x86 target by design (see jit.cpp).
# GC_LIB and TSLANG_LIB stay ROOT paths - tslang appends the `x86` segment itself for an i686
# target (getTargetLibDir), the same rule the -x86 test-runner twins document above - except
# TSLANG_LIB, which for the x64 registration above is this build's own lib/ (no x86 runtime
# there); the i686 TypeScriptAsyncRuntime.lib lives in the separate tslang-runtime build instead
# (see TEST_TSLANG_LIBPATH_X86), so this twin points TSLANG_LIB at that tree's root - derived from
# TEST_TSLANG_LIBPATH_X86 itself (which names the `x86` subdirectory) rather than re-spelling the
# tslang-runtime path a second time, so the two can't drift apart.
if (TSLANG_TEST_X86)
    get_filename_component(defaultlib_collector_tslang_runtime_root "${TEST_TSLANG_LIBPATH_X86}" DIRECTORY)
    add_test(NAME test-x86-compile-gc-defaultlib-collector
             COMMAND ${CMAKE_COMMAND}
                     "-DMODE=compile"
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DTSLANG_BIN=${CMAKE_BINARY_DIR}/bin"
                     "-DTESTS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/defaultlib"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/gc-defaultlib-collector-x86-compile"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     "-DGC_SHARED_LIB=${TEST_GC_SHARED_PREFIX}/lib"
                     "-DLLVM_LIB=${defaultlib_collector_llvm_lib}"
                     "-DTSLANG_LIB=${defaultlib_collector_tslang_runtime_root}"
                     "-DOPT=${defaultlib_collector_opt};-mtriple=i686-pc-windows-msvc"
                     "-DDEFAULT_LIB_CANDIDATES=${PROJECT_SOURCE_DIR}/../TypeScriptCompilerDefaultLib/__build;${PROJECT_SOURCE_DIR}/../../TypeScriptCompilerDefaultLib/__build"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/defaultlib-collector.cmake")
    set_tests_properties(test-x86-compile-gc-defaultlib-collector PROPERTIES
                          SKIP_REGULAR_EXPRESSION "SKIPPED:"
                          LABELS x86)
endif()

# A shared library reading the default library's data from its DLL - see
# defaultlib-data-import.cmake. Skipped (not failed) when no default library has been built, like the
# collector tests above, whose options it shares.
add_test(NAME test-compile-shared-defaultlib-data
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DTESTS_DIR=${CMAKE_CURRENT_SOURCE_DIR}/defaultlib"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/shared-defaultlib-data"
                 "-DLLVM_LIB=${defaultlib_collector_llvm_lib}"
                 "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                 "-DOPT=${defaultlib_collector_opt}"
                 "-DDEFAULT_LIB_CANDIDATES=${PROJECT_SOURCE_DIR}/../TypeScriptCompilerDefaultLib/__build;${PROJECT_SOURCE_DIR}/../../TypeScriptCompilerDefaultLib/__build"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/defaultlib-data-import.cmake")
set_tests_properties(test-compile-shared-defaultlib-data PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")

# tsbindgen: bindings generated from test/bindgen/fixture.h call the C library compiled from
# fixture.c, linked (--emit=exe) and under the JIT; `namespace` does it again through
# --namespace/--strip-prefix, which @linkname binds; `cpp` calls the C++ classes of
# fixture_cpp.hpp through the @cpp classes generated for them. And the command line's exit codes.
foreach(bindgen_mode plain namespace cpp)
    add_test(NAME test-bindgen-${bindgen_mode}
             COMMAND ${CMAKE_COMMAND}
                     "-DMODE=${bindgen_mode}"
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DTSBINDGEN=$<TARGET_FILE:tsbindgen>"
                     "-DCLANG=${LLVM_TOOLS_BINARY_DIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/bindgen"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/bindgen-${bindgen_mode}"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     "-DLLVM_LIB=${defaultlib_collector_llvm_lib}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     "-DTSLANG_BIN=${CMAKE_BINARY_DIR}/bin"
                     "-DOPT=${defaultlib_collector_opt}"
                     -P "${PROJECT_SOURCE_DIR}/test/bindgen/run-bindgen-test.cmake")
endforeach()
add_test(NAME test-bindgen-cli
         COMMAND ${CMAKE_COMMAND}
                 "-DTSBINDGEN=$<TARGET_FILE:tsbindgen>"
                 "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/bindgen"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/bindgen-cli"
                 -P "${PROJECT_SOURCE_DIR}/test/bindgen/run-cli-test.cmake")

set(TSLANG_OWNERSHIP_SHARDS 8)
math(EXPR ownership_last "${TSLANG_OWNERSHIP_SHARDS} - 1")
foreach(ownership_shard RANGE ${ownership_last})
    add_test(NAME test-ownership-verifier-${ownership_shard}
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DTESTS_DIR=${PROJECT_SOURCE_DIR}/test/tester/tests"
                     "-DSHARD=${ownership_shard}"
                     "-DSHARDS=${TSLANG_OWNERSHIP_SHARDS}"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/verify-ownership.cmake")
endforeach()

# A rest parameter of a number, string or boolean type is an error at the parameter: it reached the
# MLIR verifier, which failed the module at 1:1.
add_test(NAME test-compile-diagnostic-location-rest-parameter-type
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/rest_parameter_type.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-rest-parameter-type.obj")
set_tests_properties(test-compile-diagnostic-location-rest-parameter-type
                     PROPERTIES PASS_REGULAR_EXPRESSION "rest_parameter_type\\.ts:6:12: error: A rest parameter must be of an array type, not 's32'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|rest_parameter_type\\.ts:1:|rest_parameter_type\\.ts:4:|verification")

# An optional function-typed interface field cannot be written through the interface: the write would
# have to set the instance's presence flag as well.
add_test(NAME test-compile-interface-function-field-write
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/interface-errors/function_field_write.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/interface-function-field-write.obj")
set_tests_properties(test-compile-interface-function-field-write
                     PROPERTIES PASS_REGULAR_EXPRESSION "function_field_write\\.ts:9:5: error: an optional function-typed interface field can't be assigned through the interface"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# A call with fewer arguments than the callee's required parameters is an error (TS2554),
# not padded with undef; so is one with more arguments than the callee has parameters
# (too_many_*), which went silently into a slot the callee does not have (a rest parameter
# takes any number). Each file in call-arity/ makes one bad call; the test passes when
# the compiler prints the expected message. The counts are what the user wrote, so `this`
# (a method, a constructor) and a closure's capture are not counted. Plain add_test: these
# compile nothing, so they need no x86 twin.
set(call_arity_cases
    "plain|Expected 2 arguments, but got 1\\."
    "optional|Expected 2\\-3 arguments, but got 1\\."
    "method|Expected 2 arguments, but got 1\\."
    "method_this|Expected 2 arguments, but got 1\\."
    "constructor|Expected 2 arguments, but got 1\\."
    "closure|Expected 2 arguments, but got 1\\."
    "declared|Expected 2 arguments, but got 1\\."
    "varargs|Expected at least 1 arguments, but got 0\\."
    "optional_call|Expected 2 arguments, but got 1\\."
    "optional_call_value|Expected 2 arguments, but got 1\\."
    "too_many_plain|Expected 2 arguments, but got 3\\."
    "too_many_optional|Expected 2\\-3 arguments, but got 4\\."
    "too_many_method|Expected 2 arguments, but got 3\\."
    "too_many_constructor|Expected 2 arguments, but got 3\\."
    "too_many_closure|Expected 2 arguments, but got 3\\."
    "too_many_optional_call|Expected 2 arguments, but got 3\\."
    "too_many_value|Expected 1 arguments, but got 2\\."
    "too_many_param|Expected 1 arguments, but got 2\\.")
foreach(call_arity_case ${call_arity_cases})
    string(FIND "${call_arity_case}" "|" call_arity_sep)
    string(SUBSTRING "${call_arity_case}" 0 ${call_arity_sep} call_arity_name)
    math(EXPR call_arity_sep "${call_arity_sep} + 1")
    string(SUBSTRING "${call_arity_case}" ${call_arity_sep} -1 call_arity_message)
    add_test(NAME test-compile-call-arity-${call_arity_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/call-arity/${call_arity_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/call-arity-${call_arity_name}.obj")
    # PASS_REGULAR_EXPRESSION alone ignores the exit code, so a crash after printing the
    # expected message would still pass; FAIL_REGULAR_EXPRESSION catches that.
    set_tests_properties(test-compile-call-arity-${call_arity_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${call_arity_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# `@final` (docs/superpowers/specs/2026-10-19-final-classes-design.md): a class nobody may extend.
# Each file in final/ breaks one rule; the test passes when the compiler prints the message.
set(final_error_cases
    "extends_final|class 'X' cannot extend 'B': it is @final"
    "extends_declared_final|class 'E' cannot extend 'D': it is @final"
    "final_abstract|class 'A' cannot be both @final and abstract"
    "final_cpp|class 'P' cannot be both @final and @cpp")
foreach(final_error_case ${final_error_cases})
    string(FIND "${final_error_case}" "|" final_error_sep)
    string(SUBSTRING "${final_error_case}" 0 ${final_error_sep} final_error_name)
    math(EXPR final_error_sep "${final_error_sep} + 1")
    string(SUBSTRING "${final_error_case}" ${final_error_sep} -1 final_error_message)
    add_test(NAME test-compile-final-${final_error_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/final/${final_error_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/final-${final_error_name}.obj")
    set_tests_properties(test-compile-final-${final_error_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${final_error_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# `@value` (docs/superpowers/specs/2026-10-24-value-types-design.md): a class or interface whose
# values are copied, not shared. Each file in value/ breaks one rule; the test passes when the
# compiler prints the message.
set(value_error_cases
    "value_cpp|class 'V' cannot be both @value and @cpp"
    "value_abstract|class 'V' cannot be both @value and abstract"
    "value_final|class 'V' cannot be both @value and @final"
    "value_extends|@value class 'V' cannot extend 'B'"
    "extends_value|class 'X' cannot extend 'V': it is @value"
    "value_implements|@value class 'V' cannot implement an interface"
    "value_generic|@value class 'V' cannot have type parameters"
    "value_iface_extends|@value interface 'R' cannot extend 'I': it is not @value"
    "value_iface_extends_class|@value interface 'R' cannot extend 'C': it is not @value"
    "value_iface_generic|@value interface 'R' cannot have type parameters"
    "value_iface_mutable|field 'f' of @value interface 'R' must be readonly"
    "value_iface_optional|field 'f' of @value interface 'R' cannot be optional"
    "value_iface_method|@value interface 'R' can only have fields"
    "value_mutable_field|field 'x' of @value class 'V' must be readonly"
    "value_setter|@value class 'V' cannot have a setter"
    "value_instanceof|instanceof can't be used with the @value class 'V': its values have no type information"
    "value_to_any|the @value class 'V' cannot be converted to 'any'"
    "value_from_literal|type '.*' cannot be converted to the @value class 'V'"
    "value_from_iface|type 'R' cannot be converted to the @value class 'V'"
    "value_from_class|type 'C' cannot be converted to the @value class 'V'"
    # `.` matches the message's `;`, which a CMake list cannot hold
    "value_identity|a @value type has no identity. compare its fields"
    "value_iface_identity|a @value type has no identity. compare its fields"
    "value_switch_identity|a @value type has no identity. compare its fields"
    "value_missing_identity|a @value type has no identity. compare its fields"
    "value_compare_number|a @value type has no identity. compare its fields"
    "value_optional_member_plain_field|field 'f' must be declared 'f\\?: Hit' to implement the optional @value member 'f' of 'J'"
    "value_optional_member_literal|field 'f' must be declared 'f\\?: Hit' to implement the optional @value member 'f' of 'J'"
    "value_null_plain|a @value type has no null. write 'Hit \\| null'"
    "value_ctor_closure_escape_global|a closure that captures 'this' of the @value class 'V' can only be called inside its constructor"
    "value_ctor_closure_escape_field|a closure that captures 'this' of the @value class 'V' can only be called inside its constructor"
    "value_ctor_closure_escape_arg|a closure that captures 'this' of the @value class 'V' can only be called inside its constructor"
    "value_ctor_closure_escape_nested_store|a closure that captures 'this' of the @value class 'V' can only be called inside its constructor"
    "value_ctor_closure_escape_nested_return|a closure that captures 'this' of the @value class 'V' can only be called inside its constructor"
    "value_readonly_nested_ctor|cannot assign to 'node': the fields of a @value type are readonly"
    "value_readonly_method|cannot assign to 'node': the fields of a @value type are readonly"
    "value_readonly_outside|cannot assign to 'x': the fields of a @value type are readonly"
    "value_readonly_outside_iface|cannot assign to 'start': the fields of a @value type are readonly"
    "value_readonly_incdec|cannot assign to 'x': the fields of a @value type are readonly"
    "value_ctor_this_const|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_ctor_this_closure|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_ctor_this_arg|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_ctor_this_push|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_ctor_this_arrow|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_ctor_this_literal|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_ctor_this_spread|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_self_iface|@value type 'L' contains itself through 'next'. a value cannot hold itself"
    "value_self_class|@value type 'N' contains itself through 'next'. a value cannot hold itself"
    "value_self_mutual|@value type 'A' contains itself through 'b.a'. a value cannot hold itself"
    "value_self_array|@value type 'L' contains itself through 'next'. a value cannot hold itself"
    "value_self_generic|@value type 'L' reaches itself through 'box' in the type arguments of 'Box<L>'. a @value type cannot hold a generic type made from it"
    "value_self_generic_class|@value type 'L' reaches itself through 'box' in the type arguments of 'Box<L>'. a @value type cannot hold a generic type made from it"
    "value_self_generic_mutual|@value type 'A' reaches itself through 'p.a' in the type arguments of 'Box<B>'. a @value type cannot hold a generic type made from it"
    "value_self_generic_pair|@value type 'A' reaches itself through 'p.a' in the type arguments of 'Pair<B, number>'. a @value type cannot hold a generic type made from it"
    "value_generic_unknown|can't find type by name: Nope"
    "value_ctor_this_init|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'"
    "value_ctor_this_init_noctor|'this' of the @value class 'V' is not a value until its constructor returns. read 'this.f'")
foreach(value_error_case ${value_error_cases})
    string(FIND "${value_error_case}" "|" value_error_sep)
    string(SUBSTRING "${value_error_case}" 0 ${value_error_sep} value_error_name)
    math(EXPR value_error_sep "${value_error_sep} + 1")
    string(SUBSTRING "${value_error_case}" ${value_error_sep} -1 value_error_message)
    add_test(NAME test-compile-value-${value_error_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/value/${value_error_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/value-${value_error_name}.obj")
    set_tests_properties(test-compile-value-${value_error_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${value_error_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# A literal of a @value interface is a value: built in the interface's field order, read, passed,
# and converted to `any` and a type literal.
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-interface-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_interface.ts")
    tslang_add_test(NAME test-jit-value-interface-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_interface.ts")
endforeach()
# ...nested in another literal, two levels out of order, and converted to an ordinary interface
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-interface-own-gaps-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_interface_own_gaps.ts")
    set(value_interface_gaps_x86_twin "${TSLANG_X86_TWIN}")
    tslang_add_test(NAME test-jit-value-interface-own-gaps-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_interface_own_gaps.ts")
    # own rejects both (spec 36 "Not here"): a literal nested in another holding new vectors (an
    # owning tuple moved into a tuple's field, #679's limit), and a record that borrows temporaries
    # converted to an ordinary interface (its box would own what the record borrows)
    if (value_mm STREQUAL "own")
        set_tests_properties(test-compile-value-interface-own-gaps-own test-jit-value-interface-own-gaps-own ${value_interface_gaps_x86_twin}
                             PROPERTIES DISABLED TRUE)
    endif()
endforeach()
# `new` of a @value class makes a value: constructor, static helpers and field, a method and a
# getter taking `this` by value; stored in an ordinary class's field and in an array, and copied.
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-class-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_class.ts")
    tslang_add_test(NAME test-jit-value-class-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_class.ts")
endforeach()
# ...and its optimized code allocates nothing for it
add_test(NAME test-value-no-alloc
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/value/value_no_alloc.ts"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/value-no-alloc.cmake")
# `T | null`, `T | undefined` and `v?: T` of a @value type are an optional value: returned,
# compared with null and undefined, switched on, and read after a check or through `?.`.
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-missing-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_missing.ts")
    tslang_add_test(NAME test-jit-value-missing-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_missing.ts")
endforeach()
# ...and `g?.ray.dir.x` (and `.len()`) of a missing one skips the rest of the chain: undefined, as is a
# type literal's
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-optional-chain-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_optional_chain.ts")
    tslang_add_test(NAME test-jit-value-optional-chain-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_optional_chain.ts")
endforeach()
# ...and a field read, a getter call or a conversion to the plain type of a missing one throws
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-missing-read-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_missing_read.ts")
    tslang_add_test(NAME test-jit-value-missing-read-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_missing_read.ts")
endforeach()
# Methods of an ordinary class calling each other before all are declared, returning a @value class:
# discovery's unresolved results are not conversions to report.
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-mutual-methods-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_mutual_methods.ts")
    tslang_add_test(NAME test-jit-value-mutual-methods-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_mutual_methods.ts")
endforeach()
# A @value interface holding an interface whose method returns it, declared before it: the interface
# and the class implementing it see the @value type with all its fields.
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-iface-cycle-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_iface_cycle.ts")
    tslang_add_test(NAME test-jit-value-iface-cycle-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_iface_cycle.ts")
endforeach()
# A @value interface reaching itself through an ordinary class (it holds itself in place, it is an
# error: value_self_*). Not under own: `l.v` of `const l = b.box.l` after `l !== null` is "borrows a
# field but is used here after it may be released or overwritten", an own rule outside this one.
foreach(value_mm gc rc none)
    tslang_add_test(NAME test-compile-value-self-through-class-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_self_through_class.ts")
    tslang_add_test(NAME test-jit-value-self-through-class-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_self_through_class.ts")
endforeach()
# ...and one holding a generic of another @value type declared after it, the generic declared last or
# first: typed once that type is laid out, not as the unspecialized generic
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-generic-field-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_generic_field.ts")
    tslang_add_test(NAME test-jit-value-generic-field-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_generic_field.ts")
    tslang_add_test(NAME test-compile-value-generic-field-first-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_generic_field_first.ts")
    tslang_add_test(NAME test-jit-value-generic-field-first-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_generic_field_first.ts")
endforeach()
# `c ? { ... } : null` into `T | null` of a @value type: the literal is built as the T
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-ternary-null-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_ternary_null.ts")
    tslang_add_test(NAME test-jit-value-ternary-null-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_ternary_null.ts")
endforeach()
# A @value value holding references copies them as a tuple does (rc retains each one): copies out of
# an array, through a generic and into closures keep their Nodes after the originals are overwritten.
# A closure captures a @value local's binding, `reduce` works on copies, and a closure in a @value
# constructor reads `this`. Not under own: most of it copies a value that holds an owned Node and reads
# the copy after the original's Node is overwritten, which single ownership rejects (spec 36).
foreach(value_mm gc rc none)
    tslang_add_test(NAME test-compile-value-refs-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_refs.ts")
    tslang_add_test(NAME test-jit-value-refs-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_refs.ts")
endforeach()
# ...every retain of it paired with a release on every path (the ownership verifier)
add_test(NAME test-rc-verify-ownership-value-refs
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=rc --verify-ownership
                 "${PROJECT_SOURCE_DIR}/test/tester/value/value_refs.ts")
set_tests_properties(test-rc-verify-ownership-value-refs PROPERTIES FAIL_REGULAR_EXPRESSION "error|Stack dump")
# ...a field of a value in a class's field, read into a const or a literal, is retained as a type
# literal's is
add_test(NAME test-value-rc-field-retain
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/value/value_rc_field_retain.ts"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/value-rc-field-retain.cmake")
# ...and the box of a closure capturing `this` in a @value constructor borrows the slot `new` made
add_test(NAME test-value-ctor-capture
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/value/value_ctor_capture.ts"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/value-ctor-capture.cmake")
# ...and `this` in a @value constructor read through its members - a method, a closure, `this["a"]` -
# sees the fields stored so far (used whole, it is an error: value_ctor_this_* above)
foreach(value_mm gc rc none own)
    tslang_add_test(NAME test-compile-value-ctor-this-method-${value_mm} COMMAND test-runner -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_ctor_this_method.ts")
    tslang_add_test(NAME test-jit-value-ctor-this-method-${value_mm} COMMAND test-runner -jit -mm=${value_mm} "${PROJECT_SOURCE_DIR}/test/tester/value/value_ctor_this_method.ts")
endforeach()

# A call on a @final class is direct, and the program runs the same under every model.
foreach(final_mm gc rc none own)
    tslang_add_test(NAME test-compile-final-dispatch-${final_mm} COMMAND test-runner -mm=${final_mm} "${PROJECT_SOURCE_DIR}/test/tester/final/final_dispatch.ts")
    tslang_add_test(NAME test-jit-final-dispatch-${final_mm} COMMAND test-runner -jit -mm=${final_mm} "${PROJECT_SOURCE_DIR}/test/tester/final/final_dispatch.ts")
endforeach()
# ...with no vtable call left on a @final receiver, while one through a non-final base keeps it
add_test(NAME test-final-ir-direct
         COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib
                 "${PROJECT_SOURCE_DIR}/test/tester/final/final_ir_direct.ts")
set_tests_properties(test-final-ir-direct PROPERTIES PASS_REGULAR_EXPRESSION "ThisSymbolRef"
                                                     FAIL_REGULAR_EXPRESSION "VirtualSymbolRef|error|Stack dump")
add_test(NAME test-final-ir-base
         COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib
                 "${PROJECT_SOURCE_DIR}/test/tester/final/final_ir_base.ts")
set_tests_properties(test-final-ir-base PROPERTIES PASS_REGULAR_EXPRESSION "ThisVirtualSymbolRef"
                                                   FAIL_REGULAR_EXPRESSION "error|Stack dump")

# A `continue` or `break` naming a label that no statement around it has is an error (a label
# outside the function doesn't count); it compiled to a jump with nowhere to go.
add_test(NAME test-compile-diagnostic-location-continue-label
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/continue_label.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-continue-label.obj")
set_tests_properties(test-compile-diagnostic-location-continue-label
                     PROPERTIES PASS_REGULAR_EXPRESSION "continue_label\\.ts:7:13: error: A 'continue' statement can only jump to a label of an enclosing statement"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|continue_label\\.ts:6:")
add_test(NAME test-compile-diagnostic-location-break-label
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/break_label.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-break-label.obj")
set_tests_properties(test-compile-diagnostic-location-break-label
                     PROPERTIES PASS_REGULAR_EXPRESSION "break_label\\.ts:7:17: error: A 'break' statement can only jump to a label of an enclosing statement"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|break_label\\.ts:10:")

# Storing a fraction into a variable whose type was inferred from an integer literal (`let s = 0`
# is an s32) drops it, and is warned about; the inference itself stays.
foreach(truncation_case "global|3:1" "local|5:9")
    string(REPLACE "|" ";" truncation_parts "${truncation_case}")
    list(GET truncation_parts 0 truncation_name)
    list(GET truncation_parts 1 truncation_position)
    add_test(NAME test-compile-integer-literal-truncation-${truncation_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/integer-literal-truncation/${truncation_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/integer-literal-truncation-${truncation_name}.obj")
    set_tests_properties(test-compile-integer-literal-truncation-${truncation_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${truncation_name}\\.ts:${truncation_position}: warning: a 'number' is stored in a 's32', so any fraction is dropped"
                                    FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
endforeach()
add_test(NAME test-compile-integer-literal-truncation-quiet
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/integer-literal-truncation/quiet.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/integer-literal-truncation-quiet.obj")
set_tests_properties(test-compile-integer-literal-truncation-quiet
                     PROPERTIES FAIL_REGULAR_EXPRESSION "warning:|error:|Stack dump|Assertion failed")

# A type predicate whose constraint fails (the DefaultLib's `__is<V extends T, T>` / `__is_any`)
# folds to `false` without a warning; a real constraint violation warns once, not also from the
# inference pass ("Type 1 ..." beside "Type s32 ...").
add_test(NAME test-compile-constraint-warning-type-predicate
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/constraint-warning/type_predicate.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/constraint-warning-type-predicate.obj")
set_tests_properties(test-compile-constraint-warning-type-predicate
                     PROPERTIES FAIL_REGULAR_EXPRESSION "does not satisfy the constraint|warning:|error:|Stack dump|Assertion failed")
add_test(NAME test-compile-constraint-warning-violation
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/constraint-warning/violation.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/constraint-warning-violation.obj")
set_tests_properties(test-compile-constraint-warning-violation
                     PROPERTIES PASS_REGULAR_EXPRESSION "violation\\.ts:4:25: warning: Type s32 does not satisfy the constraint string\\."
                                FAIL_REGULAR_EXPRESSION "Type 1 does not satisfy|Stack dump|Assertion failed")

# A switch's case of another kind of primitive (a string for a number) is an error, as it can never
# match: the switch's value was cast to the case's type instead.
add_test(NAME test-compile-diagnostic-location-switch-case-type
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/switch_case_type.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-switch-case-type.obj")
set_tests_properties(test-compile-diagnostic-location-switch-case-type
                     PROPERTIES PASS_REGULAR_EXPRESSION "switch_case_type\\.ts:8:14: error: Type '\"a\"' is not comparable to type '5'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|switch_case_type\\.ts:7:")

# A store to an indexed Ref<T> (`bytes[i] = x`) names the form that stores, `Deref(bytes[i]) = x`,
# instead of "saving to constant object".
add_test(NAME test-compile-ref-index-store
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/ref-index-store/store.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/ref-index-store.obj")
set_tests_properties(test-compile-ref-index-store
                     PROPERTIES PASS_REGULAR_EXPRESSION "store\\.ts:8:5: error: indexing a 'Ref<T>' gives a reference to the element; store through it with 'Deref\\(p\\[i\\]\\) = value'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# `in` with a number, a boolean or a bigint on the right is an error: it was quietly false. A
# string on the right stays an extension (its `length`).
add_test(NAME test-compile-diagnostic-location-in-primitive
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/in_primitive.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-in-primitive.obj")
set_tests_properties(test-compile-diagnostic-location-in-primitive
                     PROPERTIES PASS_REGULAR_EXPRESSION "in_primitive\\.ts:7:18: error: The right-hand side of an 'in' expression must not be a primitive: '5'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|in_primitive\\.ts:6:")

# A number, string, boolean or array passed where an interface with members is expected is an
# error; it compiled to a cast lowering could not make, and the callee read garbage. An interface
# without members still takes any value.
foreach(interface_cast_name number string boolean array)
    add_test(NAME test-compile-interface-cast-${interface_cast_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/interface-cast/${interface_cast_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/interface-cast-${interface_cast_name}.obj")
    set_tests_properties(test-compile-interface-cast-${interface_cast_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "can't be cast to interface 'IKind': it has none of the interface's members"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()
add_test(NAME test-compile-interface-cast-empty
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/interface-cast/empty.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/interface-cast-empty.obj")
set_tests_properties(test-compile-interface-cast-empty
                     PROPERTIES FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")

# An interface cast to another one it does not extend gets a vtable built from its own one, slot for
# slot (00interface_cast_unrelated.ts); a member that is a method in one and a field in the other, or a
# field of another type, has no slot to copy.
set(interface_cast_error_cases
    "method_field|interface 'F' can't be cast to interface 'M': member 'g' is a method of one and a field of the other"
    "field_type|interface 'S1' can't be cast to interface 'T1': member 'b' has another type"
    "method_object_narrow_field|field 'count' of the object is 's32' but interface 'Counter' declares 'number'")
foreach(interface_cast_error_case ${interface_cast_error_cases})
    string(FIND "${interface_cast_error_case}" "|" interface_cast_error_sep)
    string(SUBSTRING "${interface_cast_error_case}" 0 ${interface_cast_error_sep} interface_cast_error_name)
    math(EXPR interface_cast_error_sep "${interface_cast_error_sep} + 1")
    string(SUBSTRING "${interface_cast_error_case}" ${interface_cast_error_sep} -1 interface_cast_error_message)
    add_test(NAME test-compile-interface-cast-${interface_cast_error_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/interface-cast/${interface_cast_error_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/interface-cast-${interface_cast_error_name}.obj")
    set_tests_properties(test-compile-interface-cast-${interface_cast_error_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${interface_cast_error_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# `instanceof` with a number, a string, a boolean or a bigint value on the right is an error: the
# test compared the two types. A type there (`type str = TypeOf<"">`) stays an extension.
add_test(NAME test-compile-diagnostic-location-instanceof-primitive
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/instanceof_primitive.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-instanceof-primitive.obj")
set_tests_properties(test-compile-diagnostic-location-instanceof-primitive
                     PROPERTIES PASS_REGULAR_EXPRESSION "instanceof_primitive\\.ts:8:24: error: The right-hand side of an 'instanceof' expression must be a class, not '3'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|instanceof_primitive\\.ts:7:")

# A @cpp class names the slots of a vtable made by C++ (tests/00cppvtable_class.ts), so what
# would add a slot, run code of its own or ask the object what it is is an error: a slot off by one
# would call the wrong C++ method, with no error at all. One whose methods have bodies implements
# C++ methods (cppvtable/abi_*.ts' Counter), with limits of its own.
set(cppvtable_errors
    "instanceof|instanceof can't be used with a @cpp class"
    "box_any|can't cast 'Base' to 'any': a @cpp class's objects are C\\+\\+ objects"
    "unbox_any|can't cast 'any' to 'Base': a @cpp class's objects are C\\+\\+ objects"
    "to_interface|can't cast 'Base' to 'IFirst': a @cpp class's objects are C\\+\\+ objects"
    "extends_plain|the @cpp class 'Base' can only extend a @cpp class, not 'Plain'"
    "plain_extends|class 'Plain' can't extend the @cpp class 'Base': it has to be @cpp too"
    "implements|a @cpp class can't implement an interface"
    "method_body|a method implementing a C\\+\\+ method \\(a @cpp class's, with a body\\) can't return a struct yet"
    "export_implementation|a @cpp class implementing C\\+\\+ methods can't be exported"
    "static_member|a @cpp class can't have static members"
    "constructor|a @cpp class can't have a constructor"
    "get_accessor|a @cpp class can only have methods and fields"
    "field_initializer|a field of a @cpp class can't have an initializer"
    "slot_old_name|@cppvtable is now @cpp, and a virtual method needs @slot\\(N\\)"
    "slot_missing|'second' has no @slot, so it isn't virtual"
    "slot_abstract|the abstract method 'first' of a @cpp class needs @slot\\(N\\), its vtable slot"
    "slot_conflict|@slot: 'second' and 'first' are both in slot 1"
    "slot_override|@slot: 'first' overrides slot 0 of a base class, not slot 2"
    "slot_argument|@slot takes the slot number, an integer from 0"
    "slot_plain_class|@slot is for a method of a @cpp class"
    "overload_no_match|no overload of 'write' takes these arguments \\(string\\). it has:"
    "overload_ambiguous|the call of 'write' is ambiguous: its arguments \\(s32, s32\\) fit more than one of:"
    "overload_override|'write' is overloaded in a base class: an override names the slot it overrides, @slot\\(N\\)"
    "nonvirtual_struct|a C\\+\\+ non-virtual method \\(a @cpp class's, without @slot\\) can't return a struct yet")
foreach(cppvtable_error ${cppvtable_errors})
    string(REPLACE "|" ";" cppvtable_error "${cppvtable_error}")
    list(GET cppvtable_error 0 cppvtable_name)
    list(GET cppvtable_error 1 cppvtable_message)
    add_test(NAME test-compile-cppvtable-${cppvtable_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/${cppvtable_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/cppvtable-${cppvtable_name}.obj")
    set_tests_properties(test-compile-cppvtable-${cppvtable_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${cppvtable_name}\\.ts:[0-9]+:[0-9]+: error: ${cppvtable_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# A @cpp reference is a C++ pointer, with no count in front of the object: every model compiles it,
# and rc and own count none of its copies (test-jit-cppvtable-abi-* below run it).
foreach(cppvtable_mm rc own none)
    add_test(NAME test-compile-cppvtable-memory-model-${cppvtable_mm}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=${cppvtable_mm}
                     "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/memory_model.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/cppvtable-memory-model-${cppvtable_mm}.obj")
    set_tests_properties(test-compile-cppvtable-memory-model-${cppvtable_mm}
                         PROPERTIES FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
endforeach()

# A method returning a record: tests/00cppvtable_hidden_return.ts's slots, made by hand, return it as
# MSVC does, through a pointer after `this`, so they run on Windows only; the Itanium ABI's way is
# tested against C++ below (test-jit-cppvtable-abi).
if (WIN32)
    tslang_add_test(NAME test-compile-00-cppvtable-hidden-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00cppvtable_hidden_return.ts")
    tslang_add_test(NAME test-jit-00-cppvtable-hidden-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00cppvtable_hidden_return.ts")
    tslang_add_test(NAME test-compile-shared-export-import-cppvtable-hidden-return COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_cppvtable_hidden_return.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_cppvtable_hidden_return.ts")
    tslang_add_test(NAME test-jit-shared-export-import-cppvtable-hidden-return COMMAND test-runner -jit -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_cppvtable_hidden_return.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_cppvtable_hidden_return.ts")
endif()
add_test(NAME test-compile-cppvtable-hidden-return-target
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none -mtriple=i686-unknown-linux-gnu
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/hidden_return_target.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/cppvtable-hidden-return-target.obj")
set_tests_properties(test-compile-cppvtable-hidden-return-target
                     PROPERTIES PASS_REGULAR_EXPRESSION "hidden_return_target\\.ts:7:5: error: a method of a @cpp class returning a struct needs the MSVC ABI, or the Itanium ABI on x86-64 or AArch64"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# x86-64 Itanium: a trivial record tslang can't return as the SysV classification does, unmarked
add_test(NAME test-compile-cppvtable-itanium-record
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none -mtriple=x86_64-unknown-linux-gnu
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/itanium_record.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/cppvtable-itanium-record.obj")
set_tests_properties(test-compile-cppvtable-itanium-record
                     PROPERTIES PASS_REGULAR_EXPRESSION "itanium_record\\.ts:7:5: error: on this target a method of a @cpp class returns a struct in registers by its fields.*mark the method @cppnontrivial"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
add_test(NAME test-compile-cppvtable-nontrivial-no-struct
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/nontrivial_no_struct.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/cppvtable-nontrivial-no-struct.obj")
set_tests_properties(test-compile-cppvtable-nontrivial-no-struct
                     PROPERTIES PASS_REGULAR_EXPRESSION "nontrivial_no_struct\\.ts:[0-9]+:[0-9]+: error: @cppnontrivial is for a method of a @cpp class returning a struct"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# The Itanium calls' argument attributes, on any host: the hidden result pointer `sret` (X8 on
# AArch64), narrow integers extended as a C++ caller extends them (clang's code relies on it)
foreach(cppvtable_itanium_triple x86_64-unknown-linux-gnu aarch64-unknown-linux-android29)
    add_test(NAME test-compile-cppvtable-itanium-call-${cppvtable_itanium_triple}
             COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none -mtriple=${cppvtable_itanium_triple}
                     "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_itanium.ts" -o -)
    set_tests_properties(test-compile-cppvtable-itanium-call-${cppvtable_itanium_triple}
                         PROPERTIES PASS_REGULAR_EXPRESSION "i1 zeroext %[0-9]+, i8 zeroext %[0-9]+, i8 signext %[0-9]+\\).*\\(ptr sret\\(\\{ ptr \\}\\) %[0-9]+, ptr %[0-9]+, i32 3\\)"
                                    FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
    # C++ methods implemented here (abi_itanium.ts' Counter): the vtable is the C++ one's slots and
    # nothing else, the two destructors left null, a bool result is extended for C++ callers, and the
    # two overloads of pick are in their slots, in declaration order
    add_test(NAME test-compile-cppvtable-itanium-implementation-${cppvtable_itanium_triple}
             COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none -mtriple=${cppvtable_itanium_triple}
                     "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_itanium.ts" -o -)
    set_tests_properties(test-compile-cppvtable-itanium-implementation-${cppvtable_itanium_triple}
                         PROPERTIES PASS_REGULAR_EXPRESSION "@Counter\\.\\.vtbl = private global \\{ ptr, ptr, ptr, ptr, ptr, ptr, ptr \\} \\{ ptr null, ptr null, ptr @Counter\\.message, ptr @Counter\\.count, ptr @Counter\\.isEven, ptr @Counter\\.pick, ptr @\"Counter\\.pick#1\" \\}.*define zeroext i1 @Counter\\.isEven"
                                    FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
endforeach()

# @cppname: one C++ declaration, the symbol of each target's C++ compiler (cppvtable/cppname.ts)
set(cppname_symbols_x86_64-pc-windows-msvc
    "\\?scaled@Methods@@QEAAHH@Z" "\\?scaled@Methods@@QEAAHN@Z" "\\?get@Methods@@QEBAHXZ"
    "\\?total@Methods@@QEAA_J_J@Z" "\\?sum@cppname@@YAHHH@Z")
set(cppname_symbols_x86_64-unknown-linux-gnu
    "_ZN7Methods6scaledEi" "_ZN7Methods6scaledEd" "_ZNK7Methods3getEv" "_ZN7Methods5totalEl" "_ZN7cppname3sumEii")
set(cppname_symbols_x86_64-pc-windows-gnu
    "_ZN7Methods6scaledEi" "_ZN7Methods6scaledEd" "_ZNK7Methods3getEv" "_ZN7Methods5totalEx" "_ZN7cppname3sumEii")
foreach(cppname_triple x86_64-pc-windows-msvc x86_64-unknown-linux-gnu x86_64-pc-windows-gnu)
    set(cppname_index 0)
    foreach(cppname_symbol ${cppname_symbols_${cppname_triple}})
        add_test(NAME test-compile-cppname-${cppname_triple}-${cppname_index}
                 COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none -mtriple=${cppname_triple}
                         "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/cppname.ts" -o -)
        set_tests_properties(test-compile-cppname-${cppname_triple}-${cppname_index}
                             PROPERTIES PASS_REGULAR_EXPRESSION "declare [^\n]*@\"?${cppname_symbol}\"?\\("
                                        FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
        math(EXPR cppname_index "${cppname_index} + 1")
    endforeach()
endforeach()

# Records passed by value as each ABI passes them (abi_host.cpp's byValue: 4, 8, 12, 16, 24 bytes,
# then an int), on any host: MSVC x64 integers of 4 and 8 bytes, then pointers to copies; SysV the
# 4- and 8-byte ones in a register each, the 12-byte one in two, the 16- and 24-byte ones on the
# stack (`byval`: no two registers remain); AAPCS64 registers in pairs, the 24-byte one by a pointer
add_test(NAME test-compile-cppvtable-records-msvc
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none -mtriple=x86_64-pc-windows-msvc
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_msvc.ts" -o -)
set_tests_properties(test-compile-cppvtable-records-msvc
                     PROPERTIES PASS_REGULAR_EXPRESSION "call i64 %[0-9]+\\(ptr %[0-9]+, i32 1, i32 %[0-9]+, i64 %[0-9]+, ptr %[0-9]+, ptr %[0-9]+, ptr %[0-9]+, i32 16\\)"
                                FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
add_test(NAME test-compile-cppvtable-records-x86_64-unknown-linux-gnu
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none -mtriple=x86_64-unknown-linux-gnu
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_itanium.ts" -o -)
set_tests_properties(test-compile-cppvtable-records-x86_64-unknown-linux-gnu
                     PROPERTIES PASS_REGULAR_EXPRESSION "call i64 %[0-9]+\\(ptr %[0-9]+, i32 1, i64 %[0-9]+, i64 %[0-9]+, i64 %[0-9]+, i64 %[0-9]+, ptr byval\\(\\[2 x i64\\]\\) align 8 %[0-9]+, ptr byval\\(\\[3 x i64\\]\\) align 8 %[0-9]+, i32 16\\).*call \\{ i64, i64 \\} %[0-9]+\\(ptr %[0-9]+, i32 7\\)"
                                FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
add_test(NAME test-compile-cppvtable-records-aarch64-unknown-linux-android29
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none -mtriple=aarch64-unknown-linux-android29
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_itanium.ts" -o -)
set_tests_properties(test-compile-cppvtable-records-aarch64-unknown-linux-android29
                     PROPERTIES PASS_REGULAR_EXPRESSION "call i64 %[0-9]+\\(ptr %[0-9]+, i32 1, i64 %[0-9]+, i64 %[0-9]+, \\[2 x i64\\] %[0-9]+, \\[2 x i64\\] %[0-9]+, ptr %[0-9]+, i32 16\\)"
                                FAIL_REGULAR_EXPRESSION "error:|Stack dump|Assertion failed")
# x86-64 Itanium: a record of floats by value (SSE registers) is an error, not a wrong call
add_test(NAME test-compile-cppvtable-itanium-float-param
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none -mtriple=x86_64-unknown-linux-gnu
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/itanium_float_param.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/cppvtable-itanium-float-param.obj")
set_tests_properties(test-compile-cppvtable-itanium-float-param
                     PROPERTIES PASS_REGULAR_EXPRESSION "error: a record with floating-point fields passed by value to a C\\+\\+ method isn't supported on this target yet"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# A @cpp class calling the methods of an object C++ made (cppvtable/abi_host.cpp), built by
# the host's compiler: its vtable, and the way its methods return records, are that compiler's ABI -
# MSVC's (abi_msvc.ts), or the Itanium one of g++ and clang on x86-64 and AArch64 (abi_itanium.ts) -
# not a vtable made by hand in tslang, which follows tslang's own idea of the ABI.
if (CMAKE_SIZEOF_VOID_P EQUAL 8 AND (MSVC OR (NOT WIN32 AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|aarch64|arm64)$")))
    add_library(tslang_cppvtable_abi SHARED cppvtable/abi_host.cpp)
    if (MSVC)
        set(cppvtable_abi msvc)
    else()
        set(cppvtable_abi itanium)
    endif()
    # under every model: under rc and own nothing writes in front of the C++ object, which C++ deletes
    add_test(NAME test-jit-cppvtable-abi
             COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib -mm=none
                     --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                     --shared-libs=$<TARGET_FILE:tslang_cppvtable_abi>
                     "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_${cppvtable_abi}.ts")
    set_tests_properties(test-jit-cppvtable-abi
                         PROPERTIES PASS_REGULAR_EXPRESSION "done\\."
                                    FAIL_REGULAR_EXPRESSION "error:|Stack dump|Segmentation|Assertion failed")
    foreach(cppvtable_mm rc own)
        add_test(NAME test-jit-cppvtable-abi-${cppvtable_mm}
                 COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib -mm=${cppvtable_mm}
                         --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                         --shared-libs=$<TARGET_FILE:tslang_cppvtable_abi>
                         "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_${cppvtable_abi}.ts")
        set_tests_properties(test-jit-cppvtable-abi-${cppvtable_mm}
                             PROPERTIES PASS_REGULAR_EXPRESSION "done\\."
                                        FAIL_REGULAR_EXPRESSION "error:|Stack dump|Segmentation|Assertion failed")
        add_test(NAME test-${cppvtable_mm}-verify-ownership-cppvtable-abi
                 COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${cppvtable_mm} --verify-ownership
                         "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/abi_${cppvtable_abi}.ts" -o -)
        set_tests_properties(test-${cppvtable_mm}-verify-ownership-cppvtable-abi
                             PROPERTIES FAIL_REGULAR_EXPRESSION "error:|warning:|Stack dump|Assertion failed")
    endforeach()
endif()

# MSVC member functions are __thiscall on 32-bit x86, which a class method call isn't.
add_test(NAME test-compile-cppvtable-thiscall-target
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none -mtriple=i686-pc-windows-msvc
                 "${PROJECT_SOURCE_DIR}/test/tester/cppvtable/memory_model.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/cppvtable-thiscall-target.obj")
set_tests_properties(test-compile-cppvtable-thiscall-target
                     PROPERTIES PASS_REGULAR_EXPRESSION "memory_model\\.ts:3:1: error: a @cpp class can't be used on this target: its C\\+\\+ methods are __thiscall"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# An error inside a cast function the compiler generates (here: null in a union cast to string) is
# reported where the cast is - line 6 - and not at the generated text's offset taken as a position
# in the user's file, which put it inside the comment on line 1.
add_test(NAME test-compile-diagnostic-location-generated-cast
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/generated_cast.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-generated-cast.obj")
set_tests_properties(test-compile-diagnostic-location-generated-cast
                     PROPERTIES PASS_REGULAR_EXPRESSION "generated_cast\\.ts:6:23: error: can't cast from 'null' to 'string'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|generated_cast\\.ts:1:")

# `delete` of a local holding a number is an error: it compiled, as the error made for it was
# dropped. A class instance (or an array, or a value made by `new`) can still be deleted.
add_test(NAME test-compile-diagnostic-location-delete-local
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/delete_local.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-delete-local.obj")
set_tests_properties(test-compile-diagnostic-location-delete-local
                     PROPERTIES PASS_REGULAR_EXPRESSION "delete_local\\.ts:9:5: error: The operand of a 'delete' operator must be a property reference"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|delete_local\\.ts:7:")

# An error in a field initializer, or in a derived class's implicit constructor, is reported at the
# field or the class - not at 1:1, where the code the compiler generates for them had put it.
add_test(NAME test-compile-diagnostic-location-field-initializer
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/field_initializer.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-field-initializer.obj")
set_tests_properties(test-compile-diagnostic-location-field-initializer
                     PROPERTIES PASS_REGULAR_EXPRESSION "field_initializer\\.ts:6:5: error: can't cast from 'null' to 'Reference<string>'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|field_initializer\\.ts:1:")
# A static field's initializer is checked like an instance field's: it compiled with no error at all.
add_test(NAME test-compile-diagnostic-location-static-field-initializer
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/static_field_initializer.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-static-field-initializer.obj")
set_tests_properties(test-compile-diagnostic-location-static-field-initializer
                     PROPERTIES PASS_REGULAR_EXPRESSION "static_field_initializer\\.ts:6:32: error: can't cast from 'null' to 'Reference<string>' in 'strict null mode'"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|static_field_initializer\\.ts:1:")
add_test(NAME test-compile-diagnostic-location-implicit-constructor
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/implicit_constructor.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-implicit-constructor.obj")
set_tests_properties(test-compile-diagnostic-location-implicit-constructor
                     PROPERTIES PASS_REGULAR_EXPRESSION "implicit_constructor\\.ts:9:1: error: Class constructor is not accessable"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|implicit_constructor\\.ts:1:")

# An enum member after a string member needs an initializer: it took the count of the members
# before it.
add_test(NAME test-compile-diagnostic-location-enum-member-initializer
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/enum_member_initializer.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-enum-member-initializer.obj")
set_tests_properties(test-compile-diagnostic-location-enum-member-initializer
                     PROPERTIES PASS_REGULAR_EXPRESSION "enum_member_initializer\\.ts:7:5: error: Enum member 'B' must have initializer"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|enum_member_initializer\\.ts:5:")

# An error in the code the compiler makes for a statement (a for...of's or for...in's loop) is
# reported at the statement, and one in a `using`'s dispose call at the declaration - not at 1:1,
# as a node the compiler makes has no position of its own.
set(diagnostic_location_statement_cases
    "for_of|for_of\\.ts:6:5: error: Can't resolve property 'length' of type s32"
    "for_in|for_in\\.ts:6:5: error: Can't resolve property 'length' of type s32"
    "using|using\\.ts:5:11: error: Can't resolve property 'dispose' of type s32"
    "bigint_literal_out_of_range|bigint_literal_out_of_range\\.ts:3:15: error: bigint literal '12345678901234567890n' is out of range")
foreach(diagnostic_location_case ${diagnostic_location_statement_cases})
    string(REPLACE "|" ";" diagnostic_location_parts "${diagnostic_location_case}")
    list(GET diagnostic_location_parts 0 diagnostic_location_name)
    list(GET diagnostic_location_parts 1 diagnostic_location_error)
    add_test(NAME test-compile-diagnostic-location-${diagnostic_location_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/diagnostic-location/${diagnostic_location_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/diagnostic-location-${diagnostic_location_name}.obj")
    set_tests_properties(test-compile-diagnostic-location-${diagnostic_location_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${diagnostic_location_error}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|${diagnostic_location_name}\\.ts:1:")
endforeach()

# `ns.name` is the namespace's member or an error: a local or an outer namespace's `name` isn't it.
foreach(namespace_member_case missing_member outer_member)
    add_test(NAME test-compile-namespace-member-${namespace_member_case}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/namespace-member/${namespace_member_case}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/namespace-member-${namespace_member_case}.obj")
endforeach()
set_tests_properties(test-compile-namespace-member-missing_member
                     PROPERTIES PASS_REGULAR_EXPRESSION "missing_member\\.ts:12:11: error: Can't resolve property 'missing' of type ns"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
set_tests_properties(test-compile-namespace-member-outer_member
                     PROPERTIES PASS_REGULAR_EXPRESSION "outer_member\\.ts:15:11: error: Can't resolve property 'outer' of type ns"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# A call to the collector's API (GC_gcollect, or the weak-link functions DefaultLib's WeakRef uses)
# is an error under every model but gc, which alone links the collector - it used to reach the
# linker as an unresolved symbol. Under gc the same file compiles.
foreach(gc_api_model rc none own)
    add_test(NAME test-compile-gc-api-outside-gc-${gc_api_model}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=${gc_api_model}
                     "${PROJECT_SOURCE_DIR}/test/tester/collector-api/gc_collect.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/gc-api-${gc_api_model}.obj")
    set_tests_properties(test-compile-gc-api-outside-gc-${gc_api_model}
                         PROPERTIES PASS_REGULAR_EXPRESSION "'GC_gcollect' is part of the garbage collector, which -mm=${gc_api_model} does not have"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()
add_test(NAME test-compile-gc-api-gc
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=gc
                 "${PROJECT_SOURCE_DIR}/test/tester/collector-api/gc_collect.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/gc-api-gc.obj")

# A type alias that contains itself is one error, not a stack overflow (issue #370). The follow-on
# messages ("can't find type", "generic type ... can't be found") would only repeat it. A stack
# overflow prints nothing, so it fails the PASS expression.
foreach(type_alias_circular_name tuple_reference object object_optional declared_param mutual exported
                                 generic union_array namespaced)
    add_test(NAME test-compile-type-alias-circular-${type_alias_circular_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/type-alias-circular/${type_alias_circular_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/type-alias-circular-${type_alias_circular_name}.obj")
    set_tests_properties(test-compile-type-alias-circular-${type_alias_circular_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "type alias '[A-Za-z]+' circularly references itself"
                                    FAIL_REGULAR_EXPRESSION "can't find type|can't be found|can't resolve name|Stack dump|Assertion failed")
endforeach()
tslang_add_test(NAME test-jit-type-alias-not-circular COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/type-alias-circular/positive.ts")
tslang_add_test(NAME test-compile-type-alias-not-circular COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/type-alias-circular/positive.ts")

# An object literal with a field the receiver's type cannot hold (`kind: "ci"` for `kind: "sq"`) is an
# error (issue #513). The field's failed cast was dropped: the literal had no value, an assignment of it
# stored nothing, and the cast's error, postponed like every message, was never shown.
foreach(literal_field_mismatch_name assign declare argument return)
    add_test(NAME test-compile-literal-field-mismatch-${literal_field_mismatch_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/literal-field-mismatch/${literal_field_mismatch_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/literal-field-mismatch-${literal_field_mismatch_name}.obj")
    set_tests_properties(test-compile-literal-field-mismatch-${literal_field_mismatch_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "can't cast from literal type: '\"ci\"' to '\"sq\"'"
                                    FAIL_REGULAR_EXPRESSION "Expected 1 arguments|Stack dump|Assertion failed")
endforeach()
tslang_add_test(NAME test-jit-literal-field-match COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/literal-field-mismatch/positive.ts")
tslang_add_test(NAME test-compile-literal-field-match COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/literal-field-mismatch/positive.ts")

# An error reported by a pass that still returns success - a lowering pattern that cannot build a
# value, says so and carries on - fails the compile. It used to be printed and then ignored: the object
# file was written, and under --emit=jit the program ran and crashed. Two tests because one
# property cannot say both things: the message is the error we meant, and the exit code is not 0.
set(lowering_error_file "${PROJECT_SOURCE_DIR}/test/tester/lowering-errors/array_literal_element_type.ts")
add_test(NAME test-compile-lowering-error-message
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none "${lowering_error_file}"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/lowering-error-message.obj")
set_tests_properties(test-compile-lowering-error-message
                     PROPERTIES PASS_REGULAR_EXPRESSION "cast from array type .* to array type .* is not supported"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
add_test(NAME test-compile-lowering-error-exit
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none "${lowering_error_file}"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/lowering-error-exit.obj")
set_tests_properties(test-compile-lowering-error-exit PROPERTIES WILL_FAIL TRUE)

# `main`'s argv is `Ref<string>` (C's `char **`) only. A `string[]` argv would read that `char **`
# as an array header, so it is a compile error that says to use `Ref<string>`, under the JIT (where
# `main` is called, so a program that wrongly compiled would run instead of matching) and ahead of
# time. A DLL has no `main` entry point, so the same file must not draw the error there. Any
# array-typed parameter is refused, so `main(args: string[])` - one parameter, which takes argc's
# place and compiled to a program that faulted - is checked too (main-args-string-array).
set(main_argv_file "${PROJECT_SOURCE_DIR}/test/tester/lowering-errors/main_argv_string_array.ts")
foreach(argv_case main-argv-string-array main-args-string-array)
    string(REPLACE "-" "_" argv_case_file "${argv_case}")
    set(argv_case_path "${PROJECT_SOURCE_DIR}/test/tester/lowering-errors/${argv_case_file}.ts")
    add_test(NAME test-jit-${argv_case}-error
             COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib "${argv_case_path}")
    set_tests_properties(test-jit-${argv_case}-error
                         PROPERTIES PASS_REGULAR_EXPRESSION "takes argv as Ref<string>"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
    foreach(argv_emit exe obj)
        add_test(NAME test-compile-${argv_case}-error-${argv_emit}
                 COMMAND $<TARGET_FILE:tslang> --emit=${argv_emit} --no-default-lib "${argv_case_path}"
                         -o "${CMAKE_CURRENT_BINARY_DIR}/${argv_case}.${argv_emit}")
        set_tests_properties(test-compile-${argv_case}-error-${argv_emit}
                             PROPERTIES PASS_REGULAR_EXPRESSION "takes argv as Ref<string>"
                                        FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
    endforeach()
endforeach()
# The DLL is linked, so it is given what it links from the build tree rather than the environment:
# -mm=none needs only the runtime library (gc would need the shared collector too), and the check
# does not depend on the model.
add_test(NAME test-compile-main-argv-string-array-dll-no-error
         COMMAND $<TARGET_FILE:tslang> --emit=dll --no-default-lib -mm=none
                 "--tslang-lib-path=${CMAKE_BINARY_DIR}/lib" "${main_argv_file}"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/main-argv-string-array.dll")
set_tests_properties(test-compile-main-argv-string-array-dll-no-error
                     PROPERTIES FAIL_REGULAR_EXPRESSION "takes argv as Ref<string>|Stack dump|Assertion failed")

# -mm=own is accepted. Phase 0: a program that allocates nothing compiles and runs under it.
add_test(NAME test-own-flag-accepted
         COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib -mm=own
                 "${PROJECT_SOURCE_DIR}/test/tester/tests/00assert.ts")
set_tests_properties(test-own-flag-accepted PROPERTIES FAIL_REGULAR_EXPRESSION "error|Stack dump")

# Under own MLIRGen emits exactly what it emits under rc: an awaited, deleted or captured value
# still carries its ownership ops for the inference pass to judge. Checked on `delete`, whose
# claim (MLIRGenExpressions.cpp) is one of the four MLIRGen sites.
add_test(NAME test-own-mlirgen-matches-rc
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/own/own_mlirgen_delete.ts"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/own-mlirgen-matches-rc.cmake")

# The lowering backstop: a retain that reaches LLVM lowering under own is an error, never
# silently erased. Checked with the inference pass switched off, which only this test does.
add_test(NAME test-own-lowering-backstop
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=own --own-skip-inference
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_print_number.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/own-lowering-backstop.ll")
set_tests_properties(test-own-lowering-backstop
                     PROPERTIES PASS_REGULAR_EXPRESSION "ownership inference left a retain behind"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# After inference under own, no retain survives in a program of single owners.
add_test(NAME test-own-inference-erases-birth-takes
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=own
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_print_number.ts")
set_tests_properties(test-own-inference-erases-birth-takes
                     PROPERTIES PASS_REGULAR_EXPRESSION "ts.Release"
                                FAIL_REGULAR_EXPRESSION "ts\\.Retain|error|Stack dump")

# A getter read is settled like a call: its temporary released, its receiver taking it over.
foreach(getter_mm rc gc none)
    tslang_add_test(NAME test-jit-${getter_mm}-getter-temporary COMMAND test-runner -jit -mm=${getter_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/rc_getter_temporary.ts")
    tslang_add_test(NAME test-compile-${getter_mm}-getter-temporary COMMAND test-runner -mm=${getter_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/rc_getter_temporary.ts")
endforeach()
# Under rc, a call through an interface carries the result its implementation retained: no
# `ts.CallIndirect` returning a C through an interface slot is left without `__owned_result`
# (`__owned_result_named` alone only says it initialises a `const`).
tslang_add_test(NAME test-jit-rc-interface-call-owned-result COMMAND test-runner -jit -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/own/rc_interface_call_owned_result.ts")
tslang_add_test(NAME test-compile-rc-interface-call-owned-result COMMAND test-runner -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/own/rc_interface_call_owned_result.ts")
add_test(NAME test-rc-interface-call-owned-result-marked
         COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib -mm=rc
                 "${PROJECT_SOURCE_DIR}/test/tester/own/rc_interface_call_owned_result.ts")
set_tests_properties(test-rc-interface-call-owned-result-marked
                     PROPERTIES PASS_REGULAR_EXPRESSION "__owned_result"
                                FAIL_REGULAR_EXPRESSION "CallIndirect\"\\([^)]*\\) ({__owned_result_named} )?: \\(!ts\\.func<!ts\\.opaque, !ts\\.class<@C|error|Stack dump")

# -mm=own, phase 0 (docs/superpowers/plans/2026-09-28-own-phase-0.md). Positive programs run
# under JIT and ahead of time; each owns every block once.
foreach(own_test own_fresh_string own_fresh_array own_return_new own_try_local own_print_number own_literals
                 own_move_const_branch own_move_let own_move_field own_move_return own_move_loop_body
                 own_move_reassign
                 own_literal_reuse own_move_control_flow
                 own_borrow_inner_let own_borrow_loop own_borrow_branch
                 own_borrow_two_lets own_borrow_nullable
                 own_union_nullable own_union_tagged own_union_optional
                 own_field_borrow own_element_borrow own_container_loop
                 own_optional_access own_borrow_merge
                 own_field_let_borrow own_element_let_borrow own_borrow_merge_after_call
                 own_call_no_drops own_getter_borrow own_param_kept own_any_box
                 own_string_length_borrow
                 own_closure_borrow own_closure_escape own_generator own_generator_loop_local own_interface_view
                 own_async own_record_return own_generator_borrow own_global_script
                 own_global_init own_constant_data own_record_local
                 own_nullable_view own_declared_constant own_spread_build own_interface_field_call
                 own_string_copy_field own_string_copy_after_move own_string_copy_nullable
                 own_string_copy_exported own_string_copy_written own_string_copy_mixed_return
                 own_string_copy_branch own_array_param_push
                 own_declared_no_drops own_return_paths own_string_concat_element
                 own_final_fresh_result own_final_field_lend own_null_narrowing_di
                 own_param_markers own_generator_locals own_string_param_assigned own_lends
                 own_immortal_global own_immortal_global_region own_global_loop_store
                 own_literal_moves own_record_literal_given
                 own_field_chain_let own_read_after_move own_conditional_move
                 own_interface_facts own_declared_scalar_no_drops
                 own_borrowing_record own_borrowing_record_arg
                 own_reduce own_fresh_call_result own_ternary_outer_loop own_destructure_elements
                 own_tuple_record_values own_value_record own_value_loop own_value_temporaries
                 own_call_field_drops own_immortal_keepers
                 own_generator_method_this own_string_copy_call own_param_kept_some_paths own_method_callbacks
                 own_loop_read_again own_length_drops_elements own_tuple_param_kept
                 own_for_of_borrowed_iterator own_generator_keeps own_generator_keeps_bounded
                 own_generator_this_param own_copy_within_returned_param own_swap_elements
                 own_callback_forwarded own_optional_record_read own_param_default_kept own_literal_into_array
                 own_generator_take own_rest_array_borrows own_string_copies
                 own_captured_string_callback own_callback_and_call own_callback_captured own_literal_closures own_literal_variable_closures own_temporary_field_take own_union_narrowed_calls own_named_field_take own_let_field_take
                 own_callback_shapes own_callback_extended own_callback_interface
                 own_generic_specializations own_callback_cast own_callback_interface_slot)
    tslang_add_test(NAME test-jit-own-${own_test} COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/own/${own_test}.ts")
    tslang_add_test(NAME test-compile-own-${own_test} COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/own/${own_test}.ts")
    # The promise: no counting at all in what own emits.
    add_test(NAME test-own-no-counting-${own_test}
             COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=own
                     "${PROJECT_SOURCE_DIR}/test/tester/own/${own_test}.ts" -o -)
    set_tests_properties(test-own-no-counting-${own_test}
                         PROPERTIES PASS_REGULAR_EXPRESSION "define"
                                    FAIL_REGULAR_EXPRESSION "__tslang_inc_ref|__tslang_dec_ref|error:|Stack dump")
endforeach()

# A global never reassigned is immortal under own (spec 25); the same program under every other model.
foreach(immortal_mm rc gc none)
    foreach(immortal_test own_immortal_global own_immortal_global_region own_global_loop_store own_immortal_keepers)
        tslang_add_test(NAME test-jit-${immortal_mm}-${immortal_test} COMMAND test-runner -jit -mm=${immortal_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${immortal_test}.ts")
        tslang_add_test(NAME test-compile-${immortal_mm}-${immortal_test} COMMAND test-runner -mm=${immortal_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${immortal_test}.ts")
    endforeach()
endforeach()

# A for...of over an iterator read out of a place borrows it under own; the same program under every
# other model, where its cursor stays a counted `let`.
foreach(for_of_mm rc gc none)
    tslang_add_test(NAME test-jit-${for_of_mm}-own_for_of_borrowed_iterator COMMAND test-runner -jit -mm=${for_of_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_for_of_borrowed_iterator.ts")
    tslang_add_test(NAME test-compile-${for_of_mm}-own_for_of_borrowed_iterator COMMAND test-runner -mm=${for_of_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_for_of_borrowed_iterator.ts")
endforeach()

# A generator whose `@keeps` makes its state own its arguments (spec 38), and one given a generator
# that borrows, and one over its `this` parameter; strings copied within a parameter the result borrows;
# elements and fields swapped; a callback given on through a parameter; the same programs under every
# other model.
foreach(keeps_mm rc gc none)
    foreach(keeps_test own_generator_keeps own_generator_keeps_bounded own_generator_this_param
                       own_copy_within_returned_param own_swap_elements own_callback_forwarded
                       own_optional_record_read own_param_default_kept own_literal_into_array
                       own_generator_take own_rest_array_borrows own_string_copies
                       own_captured_string_callback own_callback_and_call own_callback_captured own_literal_closures own_literal_variable_closures own_temporary_field_take own_union_narrowed_calls own_named_field_take own_let_field_take
                       own_callback_shapes own_callback_extended own_callback_interface
                       own_generic_specializations own_callback_cast own_callback_interface_slot)
        tslang_add_test(NAME test-jit-${keeps_mm}-${keeps_test} COMMAND test-runner -jit -mm=${keeps_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${keeps_test}.ts")
        tslang_add_test(NAME test-compile-${keeps_mm}-${keeps_test} COMMAND test-runner -mm=${keeps_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${keeps_test}.ts")
    endforeach()
endforeach()

# A value read out of an assigned global, lent to a call that assigns only other globals, outlives the
# call; own refuses a call that may assign that global or one it cannot name.
foreach(other_global_mm own rc gc none)
    tslang_add_test(NAME test-jit-${other_global_mm}-own_global_argument_other_global COMMAND test-runner -jit -mm=${other_global_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_argument_other_global.ts")
    tslang_add_test(NAME test-compile-${other_global_mm}-own_global_argument_other_global COMMAND test-runner -mm=${other_global_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_argument_other_global.ts")
endforeach()

# A global whose one store runs in a loop is mortal (spec 25): its stores are not marked, so each
# releases the value before it, which an immortal header would leak. The `if` arm's store runs once.
add_test(NAME test-own-global-loop-store-mortal
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=own
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_loop_store.ts")
set_tests_properties(test-own-global-loop-store-mortal
                     PROPERTIES PASS_REGULAR_EXPRESSION "global_name = @T[.]w}> {__own_immortal}"
                                FAIL_REGULAR_EXPRESSION "global_name = @[KS][.]w}> {__own_immortal}|error|Stack dump")

# The other write guards of an immortal global (spec 25), one program each: an exported global, a
# static written through its class's vtable (`k: typeof M`), a second store in a static block, and a
# store in a static method leave `M` (`M.w`, `MK.m`) mortal; `I` (`I.w`) is immortal.
foreach(mortal_test export typeof static_twice static_method)
    string(REPLACE "_" "-" mortal_name "${mortal_test}")
    add_test(NAME test-own-global-mortal-${mortal_name}
             COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=own
                     "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_mortal_${mortal_test}.ts")
    set_tests_properties(test-own-global-mortal-${mortal_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "global_name = @I([.]w)?}> {__own_immortal}"
                                    FAIL_REGULAR_EXPRESSION "global_name = @M[^}]*}> {__own_immortal}|error|Stack dump")
endforeach()

# An object literal that owns its fields moves anywhere a fresh object does under own (spec 27);
# the same programs under every other model.
foreach(owning_record_mm rc gc none)
    foreach(owning_record_test own_literal_moves own_record_literal_given)
        tslang_add_test(NAME test-jit-${owning_record_mm}-${owning_record_test} COMMAND test-runner -jit -mm=${owning_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${owning_record_test}.ts")
        tslang_add_test(NAME test-compile-${owning_record_mm}-${owning_record_test} COMMAND test-runner -mm=${owning_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${owning_record_test}.ts")
    endforeach()
endforeach()

# ...and built with debug info, whose records of narrowed variables are ops of their own
add_test(NAME test-own-di-null-narrowing
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --di --opt_level=0 --no-default-lib -mm=own
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_null_narrowing_di.ts")
set_tests_properties(test-own-di-null-narrowing PROPERTIES PASS_REGULAR_EXPRESSION "ts.Func"
                                                           FAIL_REGULAR_EXPRESSION "error|Stack dump")

# ...and a for...of over an element of the array an outer for...of walks, whose local the debugger
# records too
add_test(NAME test-own-di-for-of-nested
         COMMAND $<TARGET_FILE:tslang> --emit=jit --di --opt_level=0 --no-default-lib -mm=own
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_for_of_nested_di.ts")
set_tests_properties(test-own-di-for-of-nested PROPERTIES PASS_REGULAR_EXPRESSION "done[.]"
                                                          FAIL_REGULAR_EXPRESSION "error|Stack dump")

# A caught value given to a call own has no facts for, and read after it (spec 44)
add_test(NAME test-own-caught-value-call
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=own
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_caught_value_call.ts")
set_tests_properties(test-own-caught-value-call PROPERTIES PASS_REGULAR_EXPRESSION "ts.Func"
                                                           FAIL_REGULAR_EXPRESSION "error|Stack dump")

# A field read through an interface is a place read under own (spec 28); the same program under
# every other model.
foreach(field_chain_mm rc gc none)
    tslang_add_test(NAME test-jit-${field_chain_mm}-own_field_chain_let COMMAND test-runner -jit -mm=${field_chain_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_field_chain_let.ts")
    tslang_add_test(NAME test-compile-${field_chain_mm}-own_field_chain_let COMMAND test-runner -mm=${field_chain_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_field_chain_let.ts")
endforeach()

# A read after a move, through the new owner, under own (spec 29); the same program under every
# other model.
foreach(read_after_move_mm rc gc none)
    tslang_add_test(NAME test-jit-${read_after_move_mm}-own_read_after_move COMMAND test-runner -jit -mm=${read_after_move_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_read_after_move.ts")
    tslang_add_test(NAME test-compile-${read_after_move_mm}-own_read_after_move COMMAND test-runner -mm=${read_after_move_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_read_after_move.ts")
endforeach()

# A value moved out of a local on some paths only, released on the others (spec 30); the same
# program under every other model.
foreach(conditional_move_mm rc gc none)
    tslang_add_test(NAME test-jit-${conditional_move_mm}-own_conditional_move COMMAND test-runner -jit -mm=${conditional_move_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_conditional_move.ts")
    tslang_add_test(NAME test-compile-${conditional_move_mm}-own_conditional_move COMMAND test-runner -mm=${conditional_move_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_conditional_move.ts")
endforeach()

# `c ? h : h2` bound in a loop, of locals made before it, borrows the one it chooses (own); the
# same program under every other model.
foreach(ternary_loop_mm rc gc none)
    tslang_add_test(NAME test-jit-${ternary_loop_mm}-own_ternary_outer_loop COMMAND test-runner -jit -mm=${ternary_loop_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_ternary_outer_loop.ts")
    tslang_add_test(NAME test-compile-${ternary_loop_mm}-own_ternary_outer_loop COMMAND test-runner -mm=${ternary_loop_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_ternary_outer_loop.ts")
endforeach()

# `let [a, b] = pair` reads each element behind a length test; under own the local is a borrow of the
# element, as `let a = pair[0]` is. The same program under every other model, and the corpus file
# whose destructuring own refused.
foreach(destructure_mm rc gc none)
    tslang_add_test(NAME test-jit-${destructure_mm}-own_destructure_elements COMMAND test-runner -jit -mm=${destructure_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_destructure_elements.ts")
    tslang_add_test(NAME test-compile-${destructure_mm}-own_destructure_elements COMMAND test-runner -mm=${destructure_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_destructure_elements.ts")
endforeach()
tslang_add_test(NAME test-jit-own-00-owned-locals COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_locals.ts")
tslang_add_test(NAME test-compile-own-00-owned-locals COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_locals.ts")

# A tuple of owned values built or returned as a value, and a record a call returned read through a
# `const` or destructured: own builds the tuple in a slot as an object literal, and a field read out of
# the record is a read of it. The same program under every other model.
foreach(tuple_record_mm rc gc none)
    tslang_add_test(NAME test-jit-${tuple_record_mm}-own_tuple_record_values COMMAND test-runner -jit -mm=${tuple_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_tuple_record_values.ts")
    tslang_add_test(NAME test-compile-${tuple_record_mm}-own_tuple_record_values COMMAND test-runner -mm=${tuple_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_tuple_record_values.ts")
endforeach()

# A call through an interface closed in its module gets its implementations' facts (spec 31); the
# same program under every other model.
foreach(interface_facts_mm rc gc none)
    foreach(interface_facts_test own_interface_facts own_declared_scalar_no_drops own_interface_exe_exports)
        tslang_add_test(NAME test-jit-${interface_facts_mm}-${interface_facts_test} COMMAND test-runner -jit -mm=${interface_facts_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${interface_facts_test}.ts")
        tslang_add_test(NAME test-compile-${interface_facts_mm}-${interface_facts_test} COMMAND test-runner -mm=${interface_facts_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${interface_facts_test}.ts")
    endforeach()
endforeach()

# An object literal that borrows some of what it holds - a borrowing record, returned or a temporary -
# under own (spec 32); the same programs under every other model.
foreach(borrowing_record_mm rc gc none)
    foreach(borrowing_record_test own_borrowing_record own_borrowing_record_arg)
        tslang_add_test(NAME test-jit-${borrowing_record_mm}-${borrowing_record_test} COMMAND test-runner -jit -mm=${borrowing_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${borrowing_record_test}.ts")
        tslang_add_test(NAME test-compile-${borrowing_record_mm}-${borrowing_record_test} COMMAND test-runner -mm=${borrowing_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${borrowing_record_test}.ts")
    endforeach()
endforeach()

# A tuple that holds borrows and owns nothing - a @value record returned through an interface, held
# in a local the loop reassigns, a temporary given to a call - under own (spec 36); the same programs
# under every other model.
foreach(value_record_mm rc gc none)
    foreach(value_record_test own_value_record own_value_loop own_value_temporaries)
        tslang_add_test(NAME test-jit-${value_record_mm}-${value_record_test} COMMAND test-runner -jit -mm=${value_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${value_record_test}.ts")
        tslang_add_test(NAME test-compile-${value_record_mm}-${value_record_test} COMMAND test-runner -mm=${value_record_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${value_record_test}.ts")
    endforeach()
endforeach()

# `reduce` takes its accumulator, and a parameter returned on some paths only is kept (spec 33);
# the same program under every other model.
foreach(reduce_mm rc gc none)
    tslang_add_test(NAME test-jit-${reduce_mm}-own_reduce COMMAND test-runner -jit -mm=${reduce_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_reduce.ts")
    tslang_add_test(NAME test-compile-${reduce_mm}-own_reduce COMMAND test-runner -mm=${reduce_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_reduce.ts")
endforeach()

# A direct call's result is the caller's own, not a use of a borrow given to it (spec 34); the same
# program under every other model.
foreach(fresh_call_mm rc gc none)
    tslang_add_test(NAME test-jit-${fresh_call_mm}-own_fresh_call_result COMMAND test-runner -jit -mm=${fresh_call_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_fresh_call_result.ts")
    tslang_add_test(NAME test-compile-${fresh_call_mm}-own_fresh_call_result COMMAND test-runner -mm=${fresh_call_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_fresh_call_result.ts")
endforeach()

# The ray tracer, own's copy of the benchmark at 64x64 (spec 34), under every model. Its surfaces are
# Shared handles, which count, so it is not in own_test (its no-counting check). It exports its
# `Surfaces` globals, which an object file opens to other modules (spec 31), so under own the JIT's
# entry module and `tslang --emit=exe` build it (own-exe-run.cmake), not test-runner's object file.
foreach(raytrace_mm gc none rc)
    tslang_add_test(NAME test-jit-${raytrace_mm}-own_raytrace COMMAND test-runner -jit -mm=${raytrace_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_raytrace.ts")
    tslang_add_test(NAME test-compile-${raytrace_mm}-own_raytrace COMMAND test-runner -mm=${raytrace_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_raytrace.ts")
endforeach()
tslang_add_test(NAME test-jit-own-own_raytrace COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/own/own_raytrace.ts")
add_test(NAME test-compile-own-own_raytrace
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DSOURCE=${PROJECT_SOURCE_DIR}/test/tester/own/own_raytrace.ts"
                 "-DEXPECT=checksum: 606318"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/own-raytrace"
                 "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                 "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/own-exe-run.cmake")

# An executable's exports leave its interfaces closed (spec 31, USER decision): under own the JIT's
# entry module and `tslang --emit=exe` compile own_interface_exe_exports; an object file and a DLL
# do not (own-exe-exports.cmake). test-runner's own AOT build is an object file, so it is not run.
tslang_add_test(NAME test-jit-own-own_interface_exe_exports COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/own/own_interface_exe_exports.ts")
add_test(NAME test-own-exe-exports
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DSOURCE=${PROJECT_SOURCE_DIR}/test/tester/own/own_interface_exe_exports.ts"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/own-exe-exports"
                 "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                 "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/own-exe-exports.cmake")

# A global declared in top-level code owns what it is given, under every model that counts.
foreach(global_mm rc none gc)
    tslang_add_test(NAME test-jit-${global_mm}-own-global-script COMMAND test-runner -jit -mm=${global_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_script.ts")
    tslang_add_test(NAME test-compile-${global_mm}-own-global-script COMMAND test-runner -mm=${global_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_script.ts")
endforeach()

# A global made by a call that keeps an argument made for it owns that argument through what it was
# given; the same program under every model.
foreach(kept_mm own rc none gc)
    tslang_add_test(NAME test-jit-${kept_mm}-own-global-kept-argument COMMAND test-runner -jit -mm=${kept_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_kept_argument.ts")
    tslang_add_test(NAME test-compile-${kept_mm}-own-global-kept-argument COMMAND test-runner -mm=${kept_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_kept_argument.ts")
endforeach()

# A read of a global used after calls that give only other globals new values; the same program under
# every model.
foreach(read_mm own rc none gc)
    tslang_add_test(NAME test-jit-${read_mm}-own-global-read-callee-other COMMAND test-runner -jit -mm=${read_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_read_callee_other.ts")
    tslang_add_test(NAME test-compile-${read_mm}-own-global-read-callee-other COMMAND test-runner -mm=${read_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_read_callee_other.ts")
endforeach()
# An interface-typed global given a fresh class object owns it; the same program under every model.
foreach(iface_mm own rc none gc)
    tslang_add_test(NAME test-jit-${iface_mm}-own-global-interface-fresh COMMAND test-runner -jit -mm=${iface_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_interface_fresh.ts")
    tslang_add_test(NAME test-compile-${iface_mm}-own-global-interface-fresh COMMAND test-runner -mm=${iface_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_global_interface_fresh.ts")
endforeach()

# An interface-or-null global given a class object or null holds an interface over the object, or
# null; the same program under every model (own takes only a global's initializer for now).
foreach(iface_mm own rc none gc)
    tslang_add_test(NAME test-jit-${iface_mm}-own-nullable-class-interface COMMAND test-runner -jit -mm=${iface_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_nullable_class_interface.ts")
    tslang_add_test(NAME test-compile-${iface_mm}-own-nullable-class-interface COMMAND test-runner -mm=${iface_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_nullable_class_interface.ts")
endforeach()
# a null class object given to an interface (also in TSLANG_CORPUS: rc, none)
tslang_add_test(NAME test-jit-own-00-class-to-interface-null COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_to_interface_null.ts")
tslang_add_test(NAME test-compile-own-00-class-to-interface-null COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_to_interface_null.ts")
# an interface value seen as another interface, or as one inside a union, is the block it shows:
# own moves and borrows it as it does the object (also in TSLANG_CORPUS: rc, none)
foreach(iface_cast_test 00interface_cast_unrelated 00interface_union_cast_unrelated 00nullable_interface_to_interface
                        00null_interface_to_interface)
    string(REPLACE "_" "-" iface_cast_name "${iface_cast_test}")
    tslang_add_test(NAME test-jit-own-${iface_cast_name} COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/${iface_cast_test}.ts")
    tslang_add_test(NAME test-compile-own-${iface_cast_name} COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/${iface_cast_test}.ts")
endforeach()

# A `let` given a new value after its value moved owns the new one (#528); the same program under
# rc and gc (not none: nothing is freed there, and the loops would hold about 800 MB).
foreach(reassign_mm rc gc)
    tslang_add_test(NAME test-jit-${reassign_mm}-own_move_reassign COMMAND test-runner -jit -mm=${reassign_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_move_reassign.ts")
    tslang_add_test(NAME test-compile-${reassign_mm}-own_move_reassign COMMAND test-runner -mm=${reassign_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_move_reassign.ts")
endforeach()

# Strings as values (spec 22): no model but own makes a copy, and each program means the same
# under every model.
foreach(strings_mm rc none gc)
    foreach(strings_test own_string_copy_field own_string_copy_after_move own_string_copy_nullable own_string_copy_exported
                         own_string_copy_mixed_return own_string_copy_branch)
        tslang_add_test(NAME test-jit-${strings_mm}-${strings_test} COMMAND test-runner -jit -mm=${strings_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${strings_test}.ts")
        tslang_add_test(NAME test-compile-${strings_mm}-${strings_test} COMMAND test-runner -mm=${strings_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${strings_test}.ts")
    endforeach()
endforeach()
# A string stored on one branch only is copied in the store's block, not at rc's birth retain, where
# the copy would leak on the path that skips the store; no run can see that leak, so the IR is read.
add_test(NAME test-own-string-copy-in-store-block
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/own/own_string_copy_branch.ts"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/own-string-copy-block.cmake")

# A local declared in a generator's loop gives back what the last turn stored in its field. The
# program means the same under every model; no run can see the leak, so the IR is read.
foreach(loop_local_mm rc none gc)
    tslang_add_test(NAME test-jit-${loop_local_mm}-own_generator_loop_local COMMAND test-runner -jit -mm=${loop_local_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_generator_loop_local.ts")
    tslang_add_test(NAME test-compile-${loop_local_mm}-own_generator_loop_local COMMAND test-runner -mm=${loop_local_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_generator_loop_local.ts")
endforeach()
add_test(NAME test-rc-generator-loop-local-release
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/own/own_generator_loop_local.ts"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/generator-loop-local-release.cmake")

# A `const` record literal's slot is released under rc (#485); no run can see the leak, so the IR is read.
add_test(NAME test-rc-const-record-release
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/tests/00const_record_owned_fields.ts"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/const-record-release.cmake")

# A `const` chosen by a conditional expression - `b ? h : h2` - is released at the end of its block
# under rc; no run can see the leak, so the IR is read.
add_test(NAME test-rc-conditional-named-const-release
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/tests/00conditional_named_const_release.ts"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/conditional-named-const-release.cmake")

# A fresh object's field read after another allocation is the value just stored: the optimized IR
# loads nothing back, under none, rc and own (the allocator is declared as LLVM declares malloc).
add_test(NAME test-fresh-objects-folded
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/tests/00fresh_objects_folded.ts"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/fresh-objects-folded.cmake")

# A generator body's temporaries are released before the yield that ends the call, where no resume
# point comes between, and the file runs under own as well.
add_test(NAME test-rc-generator-temporaries
         COMMAND ${CMAKE_COMMAND} -DTSLANG=$<TARGET_FILE:tslang>
                 "-DFILE=${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_temporaries.ts"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/generator-temporaries.cmake")
tslang_add_test(NAME test-jit-own-00-generator-temporaries COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_temporaries.ts")
tslang_add_test(NAME test-compile-own-00-generator-temporaries COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_temporaries.ts")

# A const interface made over a local keeps the object when the local is assigned again, under own too.
tslang_add_test(NAME test-jit-own-00-const-interface-of-local COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_interface_of_local.ts")
tslang_add_test(NAME test-compile-own-00-const-interface-of-local COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00const_interface_of_local.ts")

# A local given its own value keeps it under own: `n = n` stores nothing.
tslang_add_test(NAME test-jit-own-00-self-assign COMMAND test-runner -jit -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00self_assign.ts")
tslang_add_test(NAME test-compile-own-00-self-assign COMMAND test-runner -mm=own "${PROJECT_SOURCE_DIR}/test/tester/tests/00self_assign.ts")

# Shared<T> (spec 23): counted handles. These are not in the own_test list, whose
# test-own-no-counting check forbids __tslang_inc_ref: a handle is counted under own by design.
# own_shared_user_named: a program's own class named Shared wins over the built-in (spec 23.1).
set(shared_models gc none rc own)
foreach(shared_mm ${shared_models})
    foreach(shared_test own_shared_basic own_shared_user_named own_shared_graph own_shared_containers own_shared_cycle own_shared_union
                        own_shared_global own_shared_unlink own_shared_global_init
                        own_shared_place_two_uses own_shared_global_field own_shared_global_copy own_shared_argument_assigned
                        own_shared_global_argument own_shared_global_argument_throw own_shared_param_assigned
                        own_shared_elements own_shared_counted_reads)
        tslang_add_test(NAME test-jit-${shared_mm}-${shared_test} COMMAND test-runner -jit -mm=${shared_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${shared_test}.ts")
        tslang_add_test(NAME test-compile-${shared_mm}-${shared_test} COMMAND test-runner -mm=${shared_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/${shared_test}.ts")
    endforeach()
endforeach()
# Shared.count is a count only under rc and own.
foreach(shared_count_mm rc own)
    tslang_add_test(NAME test-jit-${shared_count_mm}-own_shared_count COMMAND test-runner -jit -mm=${shared_count_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_count.ts")
    tslang_add_test(NAME test-compile-${shared_count_mm}-own_shared_count COMMAND test-runner -mm=${shared_count_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_count.ts")
    tslang_add_test(NAME test-jit-${shared_count_mm}-own_shared_yield_count COMMAND test-runner -jit -mm=${shared_count_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_yield_count.ts")
    tslang_add_test(NAME test-compile-${shared_count_mm}-own_shared_yield_count COMMAND test-runner -mm=${shared_count_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_yield_count.ts")
endforeach()
# --verify-ownership is quiet on every positive (spec 23.5): handles keep rc's pairing.
foreach(shared_verify_mm own rc)
    foreach(shared_test own_shared_basic own_shared_graph own_shared_count own_shared_containers own_shared_cycle own_shared_union
                        own_shared_global own_shared_unlink own_shared_global_init
                        own_shared_place_two_uses own_shared_global_field own_shared_global_copy own_shared_argument_assigned
                        own_shared_global_argument own_shared_global_argument_throw own_shared_param_assigned
                        own_shared_elements own_shared_counted_reads)
        add_test(NAME test-${shared_verify_mm}-verify-ownership-${shared_test}
                 COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${shared_verify_mm} --verify-ownership
                         "${PROJECT_SOURCE_DIR}/test/tester/own/${shared_test}.ts")
        set_tests_properties(test-${shared_verify_mm}-verify-ownership-${shared_test}
                             PROPERTIES FAIL_REGULAR_EXPRESSION "error|Stack dump")
    endforeach()
endforeach()
# Shared<I> of an object literal (spec 26), global and local: the literal moves into the handle.
# Counted like every Shared test, so not in own_test (its no-counting check).
foreach(shared_literal_mm gc none rc own)
    tslang_add_test(NAME test-jit-${shared_literal_mm}-own_shared_literal COMMAND test-runner -jit -mm=${shared_literal_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_literal.ts")
    tslang_add_test(NAME test-compile-${shared_literal_mm}-own_shared_literal COMMAND test-runner -mm=${shared_literal_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_literal.ts")
endforeach()
foreach(shared_literal_verify_mm own rc)
    add_test(NAME test-${shared_literal_verify_mm}-verify-ownership-own_shared_literal
             COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${shared_literal_verify_mm} --verify-ownership
                     "${PROJECT_SOURCE_DIR}/test/tester/own/own_shared_literal.ts")
    set_tests_properties(test-${shared_literal_verify_mm}-verify-ownership-own_shared_literal
                         PROPERTIES FAIL_REGULAR_EXPRESSION "error|Stack dump")
endforeach()

# A function-typed field of an interface closed in its module, called through a global and a local
# Shared<I> of a literal and through a parameter (spec 31). Counted like every Shared test, so not in
# own_test (its no-counting check).
foreach(fn_field_mm gc none rc own)
    tslang_add_test(NAME test-jit-${fn_field_mm}-own_interface_fn_field COMMAND test-runner -jit -mm=${fn_field_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_interface_fn_field.ts")
    tslang_add_test(NAME test-compile-${fn_field_mm}-own_interface_fn_field COMMAND test-runner -mm=${fn_field_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_interface_fn_field.ts")
endforeach()
foreach(fn_field_verify_mm own rc)
    add_test(NAME test-${fn_field_verify_mm}-verify-ownership-own_interface_fn_field
             COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${fn_field_verify_mm} --verify-ownership
                     "${PROJECT_SOURCE_DIR}/test/tester/own/own_interface_fn_field.ts")
    set_tests_properties(test-${fn_field_verify_mm}-verify-ownership-own_interface_fn_field
                         PROPERTIES FAIL_REGULAR_EXPRESSION "error|Stack dump")
endforeach()

# Across a library boundary: the library exports its functions' `__own_no_drops`, and a borrow held
# across a call to one survives. Run as a -shared pair (the library loaded at run time) and, on
# Windows, checked against a library that lists nothing (own-shared-no-drops.cmake).
tslang_add_test(NAME test-jit-own-shared-no-drops COMMAND test-runner -jit -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/own/import_own_no_drops.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/export_own_no_drops.ts")
tslang_add_test(NAME test-compile-own-shared-no-drops COMMAND test-runner -shared -mm=own "${PROJECT_SOURCE_DIR}/test/tester/own/import_own_no_drops.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/export_own_no_drops.ts")
# An importer builds and uses objects of an own library's classes; under every model, since the pair
# is ordinary TypeScript.
foreach(class_mm own rc none gc)
    tslang_add_test(NAME test-jit-${class_mm}-shared-import-class COMMAND test-runner -jit -shared -mm=${class_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/import_own_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/export_own_class.ts")
    tslang_add_test(NAME test-compile-${class_mm}-shared-import-class COMMAND test-runner -shared -mm=${class_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/import_own_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/export_own_class.ts")
endforeach()
if (WIN32)
    add_test(NAME test-own-shared-no-drops-listed
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/tester/own"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/own-shared-no-drops"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/own-shared-no-drops.cmake")
endif()

# An override of a method of a class another module can name, or defines, is called from that module
# through the base's vtable (own spec 31): under own each library or importer below is an error
# (own_error_cases); the pairs are ordinary TypeScript under every other model.
foreach(override_mm rc none gc)
    tslang_add_test(NAME test-compile-${override_mm}-shared-override-nameable-base COMMAND test-runner -shared -mm=${override_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_override_nameable_main.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/own_err_override_nameable_base.ts")
    tslang_add_test(NAME test-jit-${override_mm}-shared-override-imported-base COMMAND test-runner -jit -shared -mm=${override_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_err_override_imported_base.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/own_override_base_lib.ts")
    tslang_add_test(NAME test-compile-${override_mm}-shared-override-imported-base COMMAND test-runner -shared -mm=${override_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_err_override_imported_base.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/own_override_base_lib.ts")
    tslang_add_test(NAME test-compile-${override_mm}-shared-override-imported-class COMMAND test-runner -shared -mm=${override_mm} "${PROJECT_SOURCE_DIR}/test/tester/own/own_err_override_imported_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/own/own_override_base_lib.ts")
endforeach()

# The ownership verifier runs under own too, on the IR before inference erases its birth takes.
add_test(NAME test-own-verify-ownership
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=own --verify-ownership
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_fresh_array.ts")
set_tests_properties(test-own-verify-ownership PROPERTIES FAIL_REGULAR_EXPRESSION "error|Stack dump")

# Under --di a diagnostic names the variable: MLIRGen puts a NameLoc inside each local's
# scope-fused location, and the error reads it.
add_test(NAME test-own-err-names-the-variable
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=own --di
                 "${PROJECT_SOURCE_DIR}/test/tester/own/own_err_const_alias_field.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/own-err-names-the-variable.obj")
set_tests_properties(test-own-err-names-the-variable
                     PROPERTIES PASS_REGULAR_EXPRESSION "'a' is moved here while 'b' borrows it"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# Each negative program has one shape phase 0 cannot prove, and must be rejected with this text.
set(own_error_cases
    "own_err_assign_after_move|:12:16: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_borrow_escape|:9:5: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_move_some_paths_field|:8:9: error: '[^']*' is moved here on some paths only"
    "own_err_borrow_and_move|:8:5: error: '[^']*' is moved here while '[^']*' borrows it"
    "own_err_borrow_bound_method|:20:17: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_const_bound_method_reassigned|:20:11: error: '[^']*' takes a second reference"
    "own_err_borrow_catch_local|:31:20: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_borrow_finally_local|:30:20: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_borrow_derived_ref|:28:5: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_const_alias_move|:12:5: error: '[^']*' borrows '[^']*' and cannot be assigned"
    "own_err_const_alias_field|:15:5: error: '[^']*' is moved here while '[^']*' borrows it"
    "own_err_field_alias|:9:11: error: '[^']*' is used here after its value was moved"
    "own_err_call_in_try_drops|:30:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_closure_escape_used|:9:11: error: '[^']*' is captured by a closure that escapes, and is still used here"
    "own_err_closure_escape_twice|:12:19: error: '[^']*' is captured by a closure that escapes, and is still used here"
    "own_err_closure_escape_param|:7:12: error: a closure that escapes owns what it captures, but -mm=own cannot move this variable into it: its value is the caller's"
    "own_err_closure_escape_inherited|:9:21: error: a closure that escapes owns what it captures, but -mm=own cannot move this variable into it: it is captured from an enclosing function"
    "own_err_closure_escape_loop|:16:23: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_closure_escape_some_paths|:17:23: error: '[^']*' is moved here on some paths only"
    "own_err_closure_outlives_cell|:13:11: error: '[^']*' borrows '[^']*', which it captures, but runs here after '[^']*' is released"
    "own_err_closure_copy_moved|:13:11: error: '[^']*' borrows '[^']*', which it captures, but runs here after '[^']*' is released"
    "own_err_closure_param_assigned|:9:5: error: '[^']*' is a captured parameter and cannot be assigned under -mm=own: its value is the caller's"
    "own_err_closure_assigns_param|:8:15: error: '[^']*' is a captured parameter and cannot be assigned under -mm=own: its value is the caller's"
    "own_err_cell_assigned_in_frame|:16:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_cell_assigned_in_closure|:18:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_closure_assigns_captured_field|:19:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_unknown_producer|:5:11: error: '[^']*' takes a second reference"
    "own_err_interface_two_views|:20:11: error: '[^']*' takes a second reference"
    "own_err_field_function_reassigned|:7:17: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_generator_outlives_arg|:12:11: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_generator_bounded_escapes|:8:17: error: '[^']*' borrows its generator's arguments and cannot be stored, returned or captured"
    "own_err_literal_two_records|:11:16: error: '[^']*' is used here after its value was moved"
    "own_err_generator_yields_borrow|borrows the generator's argument and cannot be stored, returned or captured"
    "own_err_generator_map_outlives|:7:11: error: '[^']*' borrows 'the generator's source' but is used here after 'the generator's source' is released or overwritten"
    "own_err_generator_place_overwritten|:15:16: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_two_owners|:8:13: error: '[^']*' is used here after its value was moved"
    "own_err_reassign_some_paths|:10:5: error: '[^']*' is moved here on some paths only"
    "own_err_push_let|:7:9: error: '[^']*' is used here after its value was moved"
    "own_err_push_loop|:6:33: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_field_loop|:10:9: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_union_moved|:10:11: error: '[^']*' is used here after its value was moved"
    "own_err_field_read_overwritten|:28:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_element_read_overwritten|:22:11: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_element_read_popped|:26:11: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_element_read_loop_pop|:25:15: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_field_read_owner_assigned|:27:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_field_read_owner_moved|:33:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_field_read_call|:31:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_param_read_call|:33:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_global_read_assigned|:29:11: error: '[^']*' borrows 'g[.]c' but is used here after it may be released or overwritten"
    "own_err_merge_read_overwritten|:28:5: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_alias_read_popped|:28:9: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_alias_read_stale|:31:19: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_merge_root_assigned|:36:12: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_container_read_stored|:27:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_field_let_overwritten|:28:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_field_let_assigned|:26:5: error: '[^']*' borrows a field and cannot be assigned"
    "own_err_call_drops_indirectly|:24:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_call_given_borrow|:21:5: error: '[^']*' borrows 'g[.]c' but is used here after it may be released or overwritten"
    "own_err_call_this_borrow|:19:5: error: '[^']*' borrows 'g[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_drops|:15:16: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_call_field_drops_owner|:30:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_call_field_drops_under|:30:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_exported_virtual_call|:11:16: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_borrowed_result_stored|borrows the result of 'firstOf' and cannot be stored, returned or captured"
    "own_err_borrowed_result_outlives|:19:11: error: '[^']*' borrows the result of 'firstOf' but is used here after it may be released or overwritten"
    "own_err_borrowed_result_let_outlives|borrows the result of 'H.first' but is used here after it may be released or overwritten"
    "own_err_borrowed_result_escapes|could return a borrow or keep a parameter, but it is used other than by a call"
    "own_err_borrowed_result_family|an override in its class family disagrees"
    "own_err_param_kept_used|:19:11: error: '[^']*' is used here after its value was moved"
    "own_err_any_unboxed_outlives|:11:11: error: '[^']*' borrows the result of '___unbox<!ts[.]class<C, !ts[.]class_storage<C>>>' but is used here after it may be released or overwritten"
    "own_err_param_kept_loop|:19:9: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_param_kept_not_owned|is given to 'keep', which keeps it, but -mm=own cannot move it here"
    "own_err_delete|'delete' is not supported by -mm=own yet"
    "own_err_mixed_return|:13:28: error: '[^']*' is used here after its value was moved"
    "own_err_mixed_return_join_use|:14:15: error: '[^']*' is moved here on some paths only"
    "own_err_callback_escapes|cannot make the callers of 'pick' give it up"
    "own_err_callback_disagrees|a function passed in its place disagrees"
    "own_err_callback_promoted_used|:25:26: error: '[^']*' is used here after its value was moved"
    "own_err_kept_param_let_overwritten|:30:12: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_kept_param_field_overwritten|:34:12: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_kept_and_borrowed_arg|:41:14: error: '[^']*' is used here after its value was moved"
    "own_err_kept_and_borrowed_arg_slot|:44:14: error: '[^']*' is used here after its value was moved"
    "own_err_kept_and_borrowed_arg_field|:41:14: error: '[^']*' is used here after its value was moved"
    "own_err_kept_twice|:38:14: error: '[^']*' is used here after its value was moved"
    "own_err_kept_twice_field|:42:14: error: '[^']*' is used here after its value was moved"
    "own_err_kept_twice_push|:36:14: error: '[^']*' is used here after its value was moved"
    "own_err_kept_arg_assigned_in_call|:39:20: error: '[^']*' is used here after its value was moved"
    "own_err_kept_and_borrowed_arg_assigned_in_call|:37:23: error: '[^']*' is used here after its value was moved"
    "own_err_kept_arg_assigned_on_arm|:40:29: error: '[^']*' is used here after its value was moved"
    "own_err_kept_arg_assigned_in_and|:40:35: error: '[^']*' is used here after its value was moved"
    "own_err_call_result_bounded|:43:5: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_call_result_record|:47:12: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_param_returned_exported|is a parameter, returned here, but -mm=own cannot tell the callers of 'identity' that the result borrows it"
    "own_err_param_kept_exported|is a parameter, kept here, but -mm=own cannot make the callers of 'Animal.constructor' give it up"
    "own_err_param_assigned|:7:5: error: '[^']*' is a parameter and cannot be assigned under -mm=own yet"
    "own_err_local_owns_nothing|:8:5: error: '[^']*' owns nothing under -mm=own [(]it is declared without a value, or in a catch or finally clause[)], so it cannot be given a new value here"
    "own_err_merged_result|:7:5: error: '[^']*' is merged from several branches [(]`[?]:`, `&&`, `[|][|]`, `[?][?]`, a loop[)]"
    "own_err_global_borrowed_result|the initializer of global 'c' holds the result of 'child', which borrows its argument"
    "own_err_global_borrowed_temporary|:11:11: error: the initializer of global 'c' holds the result of 'id', which borrows its argument"
    "own_err_global_kept_not_fresh|the initializer of global 'n' gives 'N.constructor' a value it keeps, but -mm=own cannot move it there"
    "own_err_global_returned|borrows the global 'V.def' and cannot be stored, returned or captured"
    "own_err_immortal_borrowed_init|:9:5: error: '[^']*' borrows 'K[.]list\\[\\]' and cannot be stored, returned or captured"
    "own_err_immortal_field_overwritten|:15:11: error: '[^']*' borrows 'K[.]n[.]c' but is used here after it may be released or overwritten"
    "own_err_immortal_region_borrow|borrows the global 'c' and cannot be stored, returned or captured"
    "own_err_immortal_region_call|:10:11: error: the initializer of global 'c' holds the result of 'id', which borrows its argument"
    "own_err_global_loop_store_static|:18:5: error: '[^']*' borrows the global 'K[.]w' and cannot be stored, returned or captured"
    "own_err_global_loop_store_script|:16:5: error: '[^']*' borrows the global 'K[.]w' and cannot be stored, returned or captured"
    "own_err_global_const_store_static|:21:28: error: '[^']*' borrows 'K[.]w' but is used here after 'K[.]w' is released or overwritten"
    "own_err_global_const_store_script|:21:20: error: '[^']*' borrows 'K[.]w' but is used here after 'K[.]w' is released or overwritten"
    "own_err_global_const_store|:19:16: error: '[^']*' borrows 'G' but is used here after 'G' is released or overwritten"
    "own_err_global_record_static|:30:23: error: '[^']*' borrows 'K[.]w' but is used here after 'K[.]w' is released or overwritten"
    "own_err_global_record|:26:11: error: '[^']*' borrows 'G' but is used here after 'G' is released or overwritten"
    "own_err_global_let_holder_static|:30:28: error: '[^']*' borrows 'K[.]w' but is used here after 'K[.]w' is released or overwritten"
    "own_err_global_let_destructure|:30:20: error: '[^']*' borrows 'K[.]w' but is used here after 'K[.]w' is released or overwritten"
    "own_err_global_let_holder|:27:16: error: '[^']*' borrows 'G' but is used here after 'G' is released or overwritten"
    "own_err_global_outer_holder|:32:15: error: '[^']*' borrows 'G' but is used here after 'G' is released or overwritten"
    "own_err_global_outer_holder_static|:33:23: error: '[^']*' borrows 'K[.]w' but is used here after 'K[.]w' is released or overwritten"
    "own_err_global_uninit_let|:28:19: error: '[^']*' borrows 'G' but is used here after 'G' is released or overwritten"
    "own_err_global_uninit_let_static|:28:23: error: '[^']*' borrows 'K[.]w' but is used here after 'K[.]w' is released or overwritten"
    "own_err_global_argument_assigned|:19:5: error: a value read out of the global 'G' is lent to a call that may release or overwrite it"
    "own_err_global_method_assigns|:15:11: error: a value read out of the global 'G' is lent to a call that may release or overwrite it"
    "own_err_global_argument_transitive|:23:5: error: a value read out of the global 'G' is lent to a call that may release or overwrite it"
    "own_err_global_read_callee_assigns|borrows 'G' but is used here after 'G' is released or overwritten"
    "own_err_global_read_callee_assigns_transitive|borrows 'G' but is used here after 'G' is released or overwritten"
    "own_err_argument_assigned|is lent to 'g', but is given a new value before the call takes it"
    "own_err_argument_assigned_conditional|is lent to 'g', but is given a new value before the call takes it"
    "own_err_argument_assigned_closure|is lent to 'g', but is given a new value before the call takes it"
    "own_err_global_method_transitive|:19:11: error: a value read out of the global 'G' is lent to a call that may release or overwrite it"
    "own_err_global_this_field_move|a value read out of the global 'GK' is lent to a call that may release or overwrite it"
    "own_err_global_argument_field_let|:21:5: error: a value read out of the global 'G' is lent to a call that may release or overwrite it"
    "own_err_global_kept_field|:25:5: error: the value read from 'g' is kept where its end at an assignment of 'g' cannot be checked"
    "own_err_global_argument_indirect|:20:11: error: a value read out of the global 'G' is lent to a call that may release or overwrite it"
    "own_err_global_let_call|borrows the global 'names', but -mm=own cannot prove the global keeps its value"
    "own_err_global_from_global|:8:5: error: the initializer of global 'k' reads a value another owner keeps"
    "own_err_global_from_field|:12:7: error: the initializer of global 'k' reads a value another owner keeps"
    "own_err_global_kept_typeof|:15:9: error: the value read from 'M[.]w' is kept where its end at an assignment of 'M[.]w' cannot be checked"
    "own_err_record_literal_nested|:8:24: error: an object literal that holds a value it owns is moved only into a local or returned as it is"
    "own_err_record_literal_let_field|:12:5: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_captured_returned|:6:9: error: '[^']*' borrows a captured variable and cannot be stored, returned or captured"
    "own_err_literal_moved_twice|:14:16: error: '[^']*' is used here after its value was moved"
    "own_err_interface_keeps_literal|is a parameter, kept here, but -mm=own cannot make the callers of 'KS.take' give it up"
    "own_err_interface_keeps_param|is a parameter, kept here, but -mm=own cannot make the callers of 'KV.take' give it up"
    "own_err_interface_borrow_param|:9:44: error: '[^']*' is a parameter, returned here, but -mm=own cannot tell the callers of 'G[.]get' that the result borrows it"
    "own_err_interface_borrow_element|:9:66: error: '[^']*' borrows an element and cannot be stored, returned or captured"
    "own_err_interface_borrow_field|:9:59: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_interface_exported_impl|:15:36: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_family_disagrees|is a parameter, returned here, but -mm=own cannot tell the callers of 'GA.get' that the result borrows it"
    "own_err_interface_member_drops|:13:12: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_from_interface|:9:65: error: '[^']*' is a parameter, returned here, but -mm=own cannot tell the callers of 'G[.]get' that the result borrows it"
    "own_err_interface_result_outlives|borrows the result of 'First.first' but is used here after it may be released or overwritten"
    "own_err_interface_fn_field_drops|:12:12: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_fn_field_unknown|:8:27: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_fn_field_exported|:6:27: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_fn_any_param|:9:41: error: '[^']*' is a parameter, returned here, but -mm=own cannot tell the callers of 'G[.]take' that the result borrows it"
    "own_err_interface_fn_any_result|:9:41: error: '[^']*' is a parameter, returned here, but -mm=own cannot tell the callers of 'G[.]take' that the result borrows it"
    "own_err_interface_fn_any_drops|:11:70: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_public_exported|:9:58: error: '[^']*' is a parameter, returned here, but -mm=own cannot tell the callers of 'PA[.]pick' that the result borrows it"
    "own_err_interface_public_reachable|:8:35: error: '[^']*' is a parameter, returned here, but -mm=own cannot tell the callers of 'PA[.]pick' that the result borrows it"
    "own_err_interface_member_types|:12:70: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_member_methods|:13:90: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_declared_callback|:12:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_global|:11:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_method|:10:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_tuple|:11:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_indirect|:11:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_wrapper|:11:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_class_method|:10:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_interface_field|:15:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_stored_global|:12:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_stored_reference|:12:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_cpp_class|:12:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_field_reference|:13:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_declared_callback_intrinsic|:13:61: error: '[^']*' borrows 'gh[.]c' but is used here after it may be released or overwritten"
    "own_err_interface_member_class|:12:70: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_interface_member_any|:10:70: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_override_nameable_base|:6:63: error: '[^']*' is a parameter, kept here, but -mm=own cannot make the callers of 'Der[.]keep' give it up"
    "own_err_override_imported_base|:5:63: error: '[^']*' is a parameter, kept here, but -mm=own cannot make the callers of 'Der[.]keep' give it up"
    "own_err_override_imported_class|:5:60: error: '[^']*' is a parameter, kept here, but -mm=own cannot make the callers of 'Der[.]keep' give it up"
    "own_err_string_kept_param_known|:12:11: error: '[^']*' is used here after its value was moved"
    "own_err_keeps_mismatch|'@keeps' of 'Pair.constructor' names 'a', but its body keeps 'b'"
    "own_err_borrows_not_generator|'@borrows' of 'copyOf' names 'w', but it neither makes a generator whose state borrows its arguments nor returns one"
    "own_err_borrows_mixed_return|'@borrows' of 'pick' names 'w', but it neither makes a generator whose state borrows its arguments nor returns one"
    "own_err_marker_unknown_param|'@keeps' names 'd', which is not a parameter of this function"
    "own_err_marker_override|'Derived.set' overrides 'Base.set', and has to say the same '@keeps', '@borrows', '@lends' and '@record'"
    "own_err_lends_missing|'firstOf' could return a borrow or keep a parameter, but it can be called from another module. '@lends' or '@keeps' would say it in its signature"
    "own_err_lends_mismatch|'@lends' of 'pick' names 'a', but its body returns a borrow of 'b'"
    "own_err_lends_not_borrow|'@lends' of 'made' names 'l', but its body does not return a borrow of it"
    "own_err_lends_and_borrows|'firstOf' cannot say both '@lends' and '@borrows'"
    "own_err_lends_two_params|'@lends' of 'pick' names 'a', 'b', but its body returns a borrow of 'a'"
    "own_err_lends_index_64|'@lends' of 'K[.]pick' names 'p63', but a method's '@lends' can name only its first 64 arguments, the receiver among them"
    "own_err_lends_outlives|:20:11: error: '[^']*' borrows the result of 'firstOf' but is used here after it may be released or overwritten"
    "own_err_generator_method_this|'this value' is a parameter, kept here, but -mm=own cannot make the callers of 'S.chars' give it up"
    "own_err_generator_method_reassigned|:17:5: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_callback_method_overridden|:12:53: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_string_pushed_twice|:9:5: error: '[^']*' is used here after its value was moved"
    "own_err_shared_borrow_write|:12:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_shared_borrow_call|:19:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_shared_borrow_push|:17:11: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_shared_moved_in|:10:11: error: '[^']*' is used here after its value was moved"
    "own_err_shared_borrow_other_type|:29:16: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_shared_borrow_push_call|:20:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_shared_borrow_reassigned|:11:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_shared_union_owned|a Shared<Node> in a union with 'string', which -mm=own does not count, is not supported"
    "own_err_shared_place_arg|:29:5: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_shared_place_two_uses|:32:5: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_shared_place_literal|:31:15: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_value_record_place_dropped|:44:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_value_record_stored|:33:9: error: '[^']*' borrows the result of 'mk' and cannot be stored, returned or captured"
    "own_err_value_record_let_reassigned|:35:11: error: '[^']*' borrows the result of 'mk' but is used here after it may be released or overwritten"
    "own_err_value_record_mixed_local|:21:5: error: a value that borrows what it holds and owns nothing [(]a tuple that holds a borrow[)] meets a value that owns a block here"
    "own_err_value_record_field_written|:20:5: error: '[^']*' borrows the result of 'mk' and cannot be stored, returned or captured"
    "own_err_value_record_param_field_kept|:8:37: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_value_record_returned|:26:22: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_value_record_captured|:28:11: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_value_record_field_stored|:32:5: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_value_record_global|:30:5: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_value_refs_copy|:32:11: error: '[^']*' borrows an element and cannot be stored, returned or captured"
    "own_err_shared_place_tuple|:31:39: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_shared_literal_borrowed|the initializer of global 'b' takes a second reference"
    "own_err_iface_field_overwritten|:23:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_iface_field_view_overwritten|:26:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_iface_field_call|:27:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_iface_field_result|borrows the result of 'dirOf' but is used here after it may be released or overwritten"
    "own_err_read_after_move_overwritten|:10:11: error: '[^']*' is used here after its value was moved"
    "own_err_read_after_move_view|:12:11: error: '[^']*' is used here after its value was moved"
    "own_err_read_after_move_call|:11:11: error: '[^']*' is used here after its value was moved"
    "own_err_read_after_move_loop|:25:14: error: '[^']*' is used here after its value was moved"
    "own_err_read_after_move_after_loop|:14:11: error: '[^']*' is used here after its value was moved"
    "own_err_read_after_move_global|:42:15: error: '[^']*' is used here after its value was moved"
    "own_err_field_let_global_overwritten|:42:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_conditional_move_read|:20:11: error: '[^']*' is used here after its value was moved"
    "own_err_conditional_move_outer|:19:13: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_ternary_loop_stored|:12:25: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_ternary_loop_returned|:11:23: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_ternary_loop_captured|:17:31: error: '[^']*' is moved inside a loop but was made outside it"
    "own_err_conditional_move_read_after_join|:18:9: error: '[^']*' is moved here on some paths only"
    "own_err_conditional_move_field_borrow|:22:9: error: '[^']*' is moved here on some paths only"
    "own_err_conditional_move_catch|:24:13: error: '[^']*' is moved here on some paths only"
    "own_err_conditional_move_bound_method|:43:9: error: '[^']*' is moved here on some paths only"
    "own_err_conditional_move_view|:43:9: error: '[^']*' is moved here on some paths only"
    "own_err_conditional_move_old_view|:43:9: error: '[^']*' is moved here on some paths only"
    "own_err_borrowing_record_stored|:27:5: error: '[^']*' borrows the result of 'Th[.]hit' and cannot be stored, returned or captured"
    "own_err_borrowing_record_outlives|:26:11: error: '[^']*' borrows the result of 'A[.]hit' but is used here after it may be released or overwritten"
    "own_err_borrowing_record_kept_arg|:24:21: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_borrowing_record_let_outlives|:24:11: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_borrowing_record_returned_local|:17:20: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_borrowing_record_field_written|:15:9: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_borrowing_record_cell|:25:16: error: '[^']*' borrows a captured variable but is used here after it may be released or overwritten"
    "own_err_borrowing_record_kept_param|:30:12: error: '[^']*' borrows '[^']*' but is used here after it may be released or overwritten"
    "own_err_borrowing_record_place_overwritten|:28:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_borrowing_record_open|:17:31: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_borrowing_record_borrowed_again|:29:11: error: '[^']*' borrows the result of 'A[.]hit' but is used here after it may be released or overwritten"
    "own_err_destructure_element_reassigned|:10:11: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_destructure_element_returned|:9:5: error: '[^']*' borrows an element and cannot be stored, returned or captured"
    "own_err_record_field_returned|:12:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_destructured_field_returned|:16:5: error: '[^']*' borrows '[^']*' and cannot be stored, returned or captured"
    "own_err_record_field_after_move|:24:11: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_borrowing_record_const_field|:20:11: error: '[^']*' borrows the result of 'hit' but is used here after it may be released or overwritten"
    "own_err_record_without_lends|:11:1: error: '@record' of 'pairOf' says its result is a borrowing record, but no '@lends' names what it borrows"
    "own_err_record_not_record|:7:1: error: '@record' of 'first' says its result is a record that borrows 'a', but its body returns a borrow of it as it is"
    "own_err_record_exported_unsaid|:12:1: error: 'pairOf' returns a record that borrows 'a', 'b', which another module can learn only from '@record' beside its '@lends'"
    "own_err_value_export_borrowing|:13:1: error: 'wrap' returns a tuple that borrows 'n' and owns nothing, which -mm=own cannot tell another module yet"
    "own_err_keeper_unknown_value|:11:5: error: -mm=own cannot make the value kept in 'retained' immortal here: it is not made here, nor a parameter"
    "own_err_keeper_exported|:7:5: error: 'Registry[.]register' keeps '#1' for good, but -mm=own cannot tell its callers in other modules. '@immortalizes' would say it in its signature"
    "own_err_keeper_declared|:5:1: error: '@immortalizes' of 'keep' names 'a', but its body keeps for good 'b'"
    "own_err_keeper_called_through_value|:6:1: error: 'keep' keeps '#0' for good, but -mm=own cannot make every caller make it immortal: the function is called through a value"
    "own_err_shared_elements_object|:13:21: error: -mm=own: 'Bag' cannot hold 'Node' [(]its 'T'[)]: an element has to be a value, a string or a Shared<T> handle. use 'Shared<Node>'"
    "own_err_shared_elements_any|:8:21: error: -mm=own: 'Bag' cannot hold 'any' [(]its 'T'[)]: an element has to be a value, a string or a Shared<T> handle. give it one of those types"
    "own_err_shared_elements_inferred|:12:17: error: -mm=own: 'Bag' cannot hold 'Node' [(]its 'T'[)]"
    "own_err_keeper_popped|:14:9: error: argument '[^']*' is given to 'Registry[.]register', which keeps it, but -mm=own cannot move it here"
    "own_err_generator_keeps_missing|:9:1: error: '@keeps' of 'mapped<[^']*' names 'iter', but its body keeps 'iter', 'f'"
    "own_err_generator_keeps_used_after|:15:11: error: '[^']*' is used here after its value was moved"
    "own_err_generator_keeps_bounded|:14:17: error: '[^']*' borrows its generator's arguments and cannot be stored, returned or captured"
    "own_err_generator_keeps_bounded_outlives|:20:11: error: '[^']*' borrows '[^']*' but is used here after '[^']*' is released or overwritten"
    "own_err_generator_this_param_returned|:11:26: error: '[^']*' borrows its generator's arguments and cannot be stored, returned or captured"
    "own_err_swap_borrow_used_after|:11:11: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_swap_call_borrow_used_after|:15:11: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_callback_forwarded_drops|:15:17: error: 'this value' borrows an element but is used here after it may be released or overwritten"
    "own_err_callback_forwarded_keeps|:32:18: error: '[^']*' keeps '#0' for good, but -mm=own cannot make every caller make it immortal: the function is called through a value"
    "own_err_optional_record_field_kept|:15:12: error: '[^']*' borrows the result of 'findIn' and cannot be stored, returned or captured"
    "own_err_generator_take_used_after|:14:9: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_captured_string_drops|:20:5: error: '[^']*' borrows a field but is used here after it may be released or overwritten"
    "own_err_callback_captured_drops|:14:9: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_callback_field_drops|:16:13: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_callback_returned_drops|:21:9: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_callback_extended_drops|:16:13: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_callback_interface_param|:34:11: error: a value read out of the global 'pool' is lent to a call that may release or overwrite it"
    "own_err_callback_interface_field|:46:11: error: a value read out of the global 'pool' is lent to a call that may release or overwrite it"
    "own_err_callback_cast_field_drops|:16:13: error: '[^']*' borrows an element but is used here after it may be released or overwritten"
    "own_err_callback_interface_slot_drops|:41:11: error: a value read out of the global 'pool' is lent to a call that may release or overwrite it"
    "own_err_callback_interface_slot_targets|:47:12: error: a value read out of the global 'pool' is lent to a call that may release or overwrite it"
    "own_err_union_narrowed_moved|:26:14: error: '[^']*' is used here after its value was moved"
    "own_err_named_field_read_again|:28:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_named_field_method_after|:28:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_named_field_object_moved|:28:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_named_field_borrow_after|:29:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_let_field_global|:15:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_let_field_read_again|:17:5: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_let_field_loop|:17:9: error: '[^']*' borrows a field and cannot be stored, returned or captured"
    "own_err_let_field_captured|:18:5: error: '[^']*' borrows a field and cannot be stored, returned or captured")
foreach(own_error_case ${own_error_cases})
    string(FIND "${own_error_case}" "|" own_error_sep)
    string(SUBSTRING "${own_error_case}" 0 ${own_error_sep} own_error_name)
    math(EXPR own_error_sep "${own_error_sep} + 1")
    string(SUBSTRING "${own_error_case}" ${own_error_sep} -1 own_error_message)
    add_test(NAME test-own-err-${own_error_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=own
                     "${PROJECT_SOURCE_DIR}/test/tester/own/${own_error_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/own-err-${own_error_name}.obj")
    set_tests_properties(test-own-err-${own_error_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${own_error_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# @dllname and @linkname are one decorator under two names, and a symbol is bound once: each
# case is a program the compiler must reject with the given message instead of binding
# something else or making the name unique.
set(linkname_cases
    "conflicting-decorators|conflicting @dllname/@linkname: 'strlen' vs 'wcslen'"
    "conflicting-decorators-variable|conflicting @dllname/@linkname: 'linkname_var_a' vs 'linkname_var_b'"
    "type-mismatch|binds symbol 'strlen' with type"
    "two-definitions|both define symbol 'linkname_dup'"
    "not-a-function|binds symbol 'linkname_counter', which is not a function"
    "chain|'a' binds symbol 'linkname_chain_b', which @dllname/@linkname renames to 'linkname_chain_c'"
    "cppname-unparsable|@cppname: expected '\\)' at the end"
    "cppname-variable|@cppname: is for a function or a method"
    "cppname-const-function|@cppname: only a method with 'this' can be const or volatile"
    "cppname-method-unparsable|cppname-method-unparsable\\.ts:5:5: error: @cppname: expected '\\)' at the end")
foreach(linkname_case ${linkname_cases})
    string(FIND "${linkname_case}" "|" linkname_sep)
    string(SUBSTRING "${linkname_case}" 0 ${linkname_sep} linkname_name)
    math(EXPR linkname_sep "${linkname_sep} + 1")
    string(SUBSTRING "${linkname_case}" ${linkname_sep} -1 linkname_message)
    add_test(NAME test-compile-linkname-${linkname_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/linkname/${linkname_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/linkname-${linkname_name}.obj")
    # PASS_REGULAR_EXPRESSION alone ignores the exit code, so a crash after printing the
    # expected message would still pass; FAIL_REGULAR_EXPRESSION catches that.
    set_tests_properties(test-compile-linkname-${linkname_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${linkname_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# `export declare function` makes a declaration visible to importers; it must not mark the C
# function it binds dllexport, which made an exe re-export the CRT's strlen. ExportFixPass
# adds dllexport only for MSVC targets, so off Windows this passes either way.
add_test(NAME test-compile-linkname-declare-not-exported
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/linkname/export-declare.ts" -o -)
set_tests_properties(test-compile-linkname-declare-not-exported
                     PROPERTIES PASS_REGULAR_EXPRESSION "declare i64 @strlen\\(ptr\\)"
                                FAIL_REGULAR_EXPRESSION "declare dllexport|Stack dump|Assertion failed")

# C code expects an 8- or 16-bit integer (and a bool) extended to 32 bits by whoever hands it
# over: the caller for an argument, the callee for a result. On x86-64 SysV clang relies on it,
# so without `signext`/`zeroext` a call like take_s8(-5) passed garbage upper bits (tsbindgen's
# `narrow` row read 251). A `declare`d function gets the attributes; `i32` is left alone.
set(small_int_ext_names s8 s16 u8 u16 bool s32 result_s8 result_u16)
set(small_int_ext_patterns
    "declare i32 @take_s8\\(i8 signext\\)"
    "declare i32 @take_s16\\(i16 signext\\)"
    "declare i32 @take_u8\\(i8 zeroext\\)"
    "declare i32 @take_u16\\(i16 zeroext\\)"
    "declare i32 @take_bool\\(i1 zeroext\\)"
    "declare i32 @take_s32\\(i32\\)"
    "declare signext i8 @give_s8\\(\\)"
    "declare zeroext i16 @give_u16\\(\\)")
foreach(small_int_ext_name small_int_ext_pattern IN ZIP_LISTS small_int_ext_names small_int_ext_patterns)
    add_test(NAME test-compile-small-int-ext-${small_int_ext_name}
             COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/small-int-ext/declare.ts" -o -)
    set_tests_properties(test-compile-small-int-ext-${small_int_ext_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${small_int_ext_pattern}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# A file reached a second way is generated once: a .d.ts two `/// <reference path>`s lead to, one
# the program and a module it imports both reference, a module imported directly and through
# another. Each used to be generated again, and the module verifier rejected the second copy of
# every symbol ("redefinition of symbol named ..."). A reference cycle or a file referencing
# itself never finished loading, and an import cycle never finished generating: the TIMEOUT is
# what fails those if that comes back.
# nested_reference: a reference is relative to the file it is in, not to the program.
foreach(reference_path_name diamond twice spelled cycle self program nested_reference)
    string(REPLACE "_" "-" reference_path_test_name "${reference_path_name}")
    tslang_add_test(NAME test-jit-reference-path-${reference_path_test_name}
                    COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/reference-path/${reference_path_name}.ts")
    tslang_add_test(NAME test-compile-reference-path-${reference_path_test_name}
                    COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/reference-path/${reference_path_name}.ts")
    set_tests_properties(test-jit-reference-path-${reference_path_test_name} test-compile-reference-path-${reference_path_test_name}
                         PROPERTIES TIMEOUT 120)
endforeach()

# A referenced file that is not there is an error at its reference, and the compile fails. It
# used to print "can't open file" naming the main file, and compile the program anyway.
set(reference_path_missing_names missing missing_nested)
set(reference_path_missing_patterns
    "missing\\.ts:3:1: error: can't open file: no_such_file\\.d\\.ts"
    "missing_inner\\.d\\.ts:1:1: error: can't open file: not_here\\.d\\.ts")
foreach(reference_path_name reference_path_pattern IN ZIP_LISTS reference_path_missing_names reference_path_missing_patterns)
    string(REPLACE "_" "-" reference_path_test_name "${reference_path_name}")
    add_test(NAME test-compile-reference-path-${reference_path_test_name}-error
             COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib
                     "${PROJECT_SOURCE_DIR}/test/tester/reference-path/${reference_path_name}.ts" -o -)
    set_tests_properties(test-compile-reference-path-${reference_path_test_name}-error
                         PROPERTIES PASS_REGULAR_EXPRESSION "${reference_path_pattern}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
    add_test(NAME test-compile-reference-path-${reference_path_test_name}-fails
             COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib
                     "${PROJECT_SOURCE_DIR}/test/tester/reference-path/${reference_path_name}.ts" -o -)
    set_tests_properties(test-compile-reference-path-${reference_path_test_name}-fails PROPERTIES WILL_FAIL TRUE)
endforeach()

# tslang_add_import_tests(<name> <dir> <program> <modules>...): <program> linked against the
# modules it imports - each an object of its own, and all of them in the one library
# test-runner -shared links them into, compiled and under the JIT.
function(tslang_add_import_tests name dir program)
    set(module_files "")
    foreach(module ${ARGN})
        list(APPEND module_files "${dir}/${module}.ts")
    endforeach()
    tslang_add_test(NAME test-compile-${name} COMMAND test-runner "${dir}/${program}.ts" ${module_files})
    tslang_add_test(NAME test-compile-shared-${name} COMMAND test-runner -shared "${dir}/${program}.ts" ${module_files})
    tslang_add_test(NAME test-jit-shared-${name} COMMAND test-runner -jit -shared "${dir}/${program}.ts" ${module_files})
    set_tests_properties(test-compile-${name} test-compile-shared-${name} test-jit-shared-${name} PROPERTIES TIMEOUT 120)
endfunction()

# In the one library a module imported by another used to be declared twice: the importer's
# __decls carried what it imported as well as its own.
set(reference_path_dir "${PROJECT_SOURCE_DIR}/test/tester/reference-path")
tslang_add_import_tests(reference-path-import-and-reference "${reference_path_dir}" import_and_reference module_left)
tslang_add_import_tests(reference-path-two-imports "${reference_path_dir}" two_imports module_left module_right)
tslang_add_import_tests(reference-path-import-diamond "${reference_path_dir}" import_diamond plain_module via_module)
tslang_add_import_tests(reference-path-import-cycle "${reference_path_dir}" import_cycle cycle_module_a cycle_module_b)
# Through a -shared library of these two modules the importer read Node twice ("redefinition of
# symbol named 'Node..new'"), and then called its methods through the wrong vtable slots - see
# shared-class-members below.
tslang_add_import_tests(reference-path-import-cycle-class "${reference_path_dir}" import_cycle_class cycle_class_node cycle_class_text)

# A -shared library's class, imported: declared twice when another module of the library extends
# it, and with its vtable built in another order than the library's when an accessor comes before
# a method.
set(shared_class_members_dir "${PROJECT_SOURCE_DIR}/test/tester/shared-class-members")
tslang_add_import_tests(shared-class-members "${shared_class_members_dir}" class_members base_module derived_module)
# The first attempt at an import in a cycle fails and is tried again; its errors were printed
# although the compile succeeded.
add_test(NAME test-compile-reference-path-import-cycle-quiet
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib
                 "${reference_path_dir}/cycle_class_node.ts" -o "${CMAKE_CURRENT_BINARY_DIR}/cycle_class_node.obj")
set_tests_properties(test-compile-reference-path-import-cycle-quiet PROPERTIES FAIL_REGULAR_EXPRESSION "error:")

# Under the JIT an import pointing to a .ts file includes it with its bodies: a program run on its
# own found no `S..new` for the imported class, which was only declared. Compiled, the import is a
# declaration and the module's own object is linked beside it.
set(import_source_dir "${PROJECT_SOURCE_DIR}/test/tester/import-source")
tslang_add_test(NAME test-jit-import-source-new-imported-class COMMAND test-runner -jit "${import_source_dir}/new_imported_class.ts")
tslang_add_import_tests(import-source-new-imported-class "${import_source_dir}" new_imported_class imported_class_module)
# A module with top-level statements, imported: they run before main, from a global constructor;
# generated outside any function, they failed with "Global is referenced by parentless
# instruction", and a module linked beside the program got a `main` of its own (#447).
tslang_add_test(NAME test-jit-import-source-toplevel COMMAND test-runner -jit "${import_source_dir}/import_toplevel.ts")
tslang_add_import_tests(import-source-toplevel "${import_source_dir}" import_toplevel toplevel_module)
# An exported const of a module with top-level statements, imported: the importer could not resolve
# it under -shared ("can't resolve name: k"), and it was a `let` with no initializer otherwise (#448).
tslang_add_test(NAME test-jit-import-source-export-const COMMAND test-runner -jit "${import_source_dir}/import_export_const.ts")
tslang_add_import_tests(import-source-export-const "${import_source_dir}" import_export_const export_const_module)
# the library declares k to its importer as `@dllimport let k: s32;`, defined in the library: no
# "'let' does not have initializer" warning for it
set_tests_properties(test-compile-shared-import-source-export-const test-jit-shared-import-source-export-const
                     PROPERTIES FAIL_REGULAR_EXPRESSION "does not have initializer|Stack dump|Assertion failed")

# The JIT cache (--emit=jit, on by default): each .ts module imported is an object of its own,
# loaded beside the program's - imported through two others and initialized first; and a program
# in an import cycle, compiled as one module. jit-cache.cmake: the cache is used, and compiled
# again when a source changes.
set(jit_cache_dir "${PROJECT_SOURCE_DIR}/test/tester/jit-cache")
tslang_add_test(NAME test-jit-cache-imports-diamond COMMAND test-runner -jit "${jit_cache_dir}/imports_diamond.ts")
tslang_add_test(NAME test-jit-cache-import-cycle COMMAND test-runner -jit "${jit_cache_dir}/cycle_main.ts")
add_test(NAME test-jit-cache-reuse-and-invalidate
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DSOURCE_DIR=${jit_cache_dir}"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/jit-cache-reuse"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/jit-cache.cmake")
# Two modules compiled in one run, both naming built-in types (i32, f32): the second one used to
# get the types made in the first one's MLIRContext, destroyed by then, and crash the compiler.
add_test(NAME test-jit-cache-embedded-types
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DSOURCE=${jit_cache_dir}/embedded_types.ts"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/jit-cache-embedded-types"
                 "-DEXPECTED=done."
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/jit-cache-empty.cmake")
# --jit-native: the cache's objects linked into a DLL with a PDB and loaded by the system loader,
# for WinDbg and Visual Studio - see jit-native.cmake. Windows only. With debug info under either
# configuration, and the C runtime of the configuration's own libraries. cdb, if it is installed,
# checks that a breakpoint on a source line is hit.
if (WIN32)
    if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        set(jit_native_opt "--opt;--di")
    else()
        set(jit_native_opt "--di;--opt_level=0")
    endif()
    find_program(TSLANG_TEST_CDB NAMES cdb cdbX64
                 PATHS "C:/Program Files (x86)/Windows Kits/10/Debuggers/x64" "$ENV{LOCALAPPDATA}/Microsoft/WindowsApps")
    if (TSLANG_TEST_CDB)
        set(jit_native_cdb "-DCDB=${TSLANG_TEST_CDB}")
    else()
        set(jit_native_cdb "")
    endif()
    add_test(NAME test-jit-native-dll
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DSOURCE_DIR=${jit_cache_dir}"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/jit-native"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     "-DOPT=${jit_native_opt}"
                     ${jit_native_cdb}
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/jit-native.cmake")
endif()

# `import { a as b }` and `import * as M` - each used to be "can't resolve name".
set(import_bindings_dir "${PROJECT_SOURCE_DIR}/test/tester/import-bindings")
tslang_add_import_tests(import-bindings-renamed "${import_bindings_dir}" import_renamed names_module)
tslang_add_import_tests(import-bindings-namespace "${import_bindings_dir}" import_namespace names_module)

# A generic class specialized while its module was imported lost its members (#231).
set(import_specialization_dir "${PROJECT_SOURCE_DIR}/test/tester/import-specialization")
# Through a -shared library the specialization was declared by its internal name
# (`class Box<!ts.class<Tree, ...>>`), and before the generic it is made from ("generic type Box
# can't be found"). On Linux the name had an `@` (`@Tree`), which ld reads as a symbol version.
tslang_add_import_tests(import-specialization "${import_specialization_dir}" import_specialization box_module)

# The same for a shared library: imported directly and through a source module - see
# shared-import-reimport.cmake.
if (WIN32)
    add_test(NAME test-compile-reference-path-shared-reimport
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/tester/reference-path"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/reference-path-shared-reimport"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/shared-import-reimport.cmake")
endif()

# A statement's condition that produces no value is an error, not a statement generated
# without its test (an `if` that vanished, an endless `for`, a segfault in `do`, an "empty
# block" error from `while`). The void call
# is the ordinary way to get one; the case that mattered was a failing `instanceof` in the
# generated `___unbox<C>`, which made `<C>anyValue` always throw - see conditionHasValue.
foreach(condition_value_name if while do for)
    add_test(NAME test-compile-condition-value-${condition_value_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/condition-value/${condition_value_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/condition-value-${condition_value_name}.obj")
    set_tests_properties(test-compile-condition-value-${condition_value_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "the condition has no value"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# The calls the check must still accept: an omitted optional, default-valued or void
# parameter, and varargs with and without extra arguments.
tslang_add_test(NAME test-compile-call-arity-positive COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/call-arity/positive.ts")

# A C variadic function (`@varargs`) called through a value of it - a `let`, a `typeof printf`
# parameter - is a variadic call, as a call by name is. Lowered as a fixed-arity call, a double
# printed 0.000000 on Win64; since the arity check it was "Expected 1 arguments, but got 2.".
# The arguments to `...` get C's default promotions (an f32 printed 0.000000 even by name).
add_test(NAME test-jit-c-varargs-value-call
         COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib
                 --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                 "${PROJECT_SOURCE_DIR}/test/tester/c-varargs/value_call.ts")
set_tests_properties(test-jit-c-varargs-value-call
                     PROPERTIES PASS_REGULAR_EXPRESSION "direct 1\\.500000.*let 1\\.500000.*no extra arguments.*param 1\\.500000.*mixed 7 2\\.500000 9 0\\.250000.*param 1\\.500000.*mixed 7 2\\.500000 9 0\\.250000.*promoted 1\\.500000 -3 1.*promoted 1\\.500000 -3 1"
                                FAIL_REGULAR_EXPRESSION "Stack dump|error:|0\\.000000")

# A C variadic function type and a plain one do not stand in for each other, either way.
foreach(c_varargs_name to_plain from_plain)
    add_test(NAME test-compile-c-varargs-${c_varargs_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/c-varargs/${c_varargs_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/c-varargs-${c_varargs_name}.obj")
    set_tests_properties(test-compile-c-varargs-${c_varargs_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "only one of them is a C variadic function"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# A declared function with a second signature: one external symbol has one signature, and the
# error says so (it was "redefinition of symbol").
add_test(NAME test-compile-declare-overloads-error
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/declare-overloads/overload.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/declare-overloads.obj")
set_tests_properties(test-compile-declare-overloads-error
                     PROPERTIES PASS_REGULAR_EXPRESSION "'abs' is declared again with another signature"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed|redefinition of symbol")

# A cast between two specializations of one generic class whose type arguments don't extend each
# other is an error - it compiled, and then read one field type as another.
add_test(NAME test-compile-specialization-cast-box-error
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/specialization-cast/box.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/specialization-cast-box.obj")
set_tests_properties(test-compile-specialization-cast-box-error
                     PROPERTIES PASS_REGULAR_EXPRESSION "Box<!ts.number> is not assignable to type Box<!ts.string>"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

add_test(NAME test-compile-specialization-cast-pair-error
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/specialization-cast/pair.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/specialization-cast-pair.obj")
set_tests_properties(test-compile-specialization-cast-pair-error
                     PROPERTIES PASS_REGULAR_EXPRESSION "is not assignable to type Pair<!ts.number,!ts.string>"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# A failing assert shows a message known only at run time, after the output printed before it.
# The JIT reports it itself on every platform, so on Linux abort() never reaches the crash handler.
add_test(NAME test-jit-assert-message-failing
         COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib
                 --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                 "${PROJECT_SOURCE_DIR}/test/tester/assert-message/failing.ts")
set_tests_properties(test-jit-assert-message-failing
                     PROPERTIES PASS_REGULAR_EXPRESSION "printed before.*assertion failed: n is 7"
                                FAIL_REGULAR_EXPRESSION "Stack dump|unlinked")

# An invalid array length stops the program with a message, after the output printed before it,
# under every model (#483), when there is no default library: with one it throws a RangeError. The JIT reports it itself on every platform, as for a failing assert.
foreach(array_length_name negative huge fraction negative_integer new_negative)
    foreach(array_length_model gc rc none own)
        add_test(NAME test-jit-${array_length_model}-array-length-${array_length_name}
                 COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib -mm=${array_length_model}
                         --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                         "${PROJECT_SOURCE_DIR}/test/tester/array-length/${array_length_name}.ts")
        set_tests_properties(test-jit-${array_length_model}-array-length-${array_length_name}
                             PROPERTIES PASS_REGULAR_EXPRESSION "printed before.*assertion failed: Invalid array length"
                                        FAIL_REGULAR_EXPRESSION "Stack dump|PLEASE submit|length: ")
    endforeach()
endforeach()

# A bigint divided by 0n or raised to a negative power stops the program with a message, after the
# output printed before it, under every model, when there is no default library: with one it throws a
# RangeError a `catch` takes (DefaultLib's range_error_caught).
foreach(bigint_range_name division remainder power)
    if(bigint_range_name STREQUAL "power")
        set(bigint_range_message "Exponent must be non-negative")
    else()
        set(bigint_range_message "Division by zero")
    endif()
    foreach(bigint_range_model gc rc none own)
        add_test(NAME test-jit-${bigint_range_model}-bigint-range-${bigint_range_name}
                 COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib -mm=${bigint_range_model}
                         --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                         "${PROJECT_SOURCE_DIR}/test/tester/bigint-range/${bigint_range_name}.ts")
        set_tests_properties(test-jit-${bigint_range_model}-bigint-range-${bigint_range_name}
                             PROPERTIES PASS_REGULAR_EXPRESSION "printed before.*assertion failed: ${bigint_range_message}"
                                        FAIL_REGULAR_EXPRESSION "Stack dump|PLEASE submit|result: ")
    endforeach()
endforeach()

# A write through an interface to its function-typed field stops the program with a message when
# the object behind it fills the field with a method or is an object literal: only a class field
# holding the function has storage to write. Which object it is is known only at run time.
foreach(function_field_write_name method literal)
    foreach(function_field_write_model gc rc none own)
        add_test(NAME test-jit-${function_field_write_model}-interface-function-field-write-${function_field_write_name}
                 COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib -mm=${function_field_write_model}
                         --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                         "${PROJECT_SOURCE_DIR}/test/tester/interface-function-field-write/${function_field_write_name}.ts")
        set_tests_properties(test-jit-${function_field_write_model}-interface-function-field-write-${function_field_write_name}
                             PROPERTIES PASS_REGULAR_EXPRESSION "printed before.*assertion failed: an interface's function field can't be assigned when the object fills it with a method or is an object literal"
                                        FAIL_REGULAR_EXPRESSION "Stack dump|PLEASE submit|not reached")
    endforeach()
endforeach()

# --export=all exports the module's variables, not the locals of its functions.
add_test(NAME test-compile-export-all-no-locals
         COMMAND $<TARGET_FILE:tslang> --emit=mlir --no-default-lib --export=all
                 "${PROJECT_SOURCE_DIR}/test/tester/export-all/locals.ts")
set_tests_properties(test-compile-export-all-no-locals
                     PROPERTIES PASS_REGULAR_EXPRESSION "let moduleLevel"
                                FAIL_REGULAR_EXPRESSION "namespace [.]f_|Stack dump|error:")

# A cast between unrelated classes whose fields do not fit is an error; it compiled and crashed.
add_test(NAME test-compile-class-cast-unrelated-error
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/class-cast/unrelated.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/class-cast-unrelated.obj")
set_tests_properties(test-compile-class-cast-unrelated-error
                     PROPERTIES PASS_REGULAR_EXPRESSION "type A is not assignable to type B: 'v' is number, not string"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# ...and so is any other class cast to a @final one, whose calls are direct
add_test(NAME test-compile-class-cast-to-final-error
         COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                 "${PROJECT_SOURCE_DIR}/test/tester/class-cast/to_final.ts"
                 -o "${CMAKE_CURRENT_BINARY_DIR}/class-cast-to-final.obj")
set_tests_properties(test-compile-class-cast-to-final-error
                     PROPERTIES PASS_REGULAR_EXPRESSION "type Y is not assignable to type F: 'F' is @final"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")

# An object literal with methods or accessors assigned to a class is an error: a class instance
# takes its methods from its class. The cast reinterpreted the pointer, so the class read garbage
# fields and crashed on a method call (#494).
foreach(object_literal_to_class_name method accessor parameter return)
    add_test(NAME test-compile-object-literal-to-class-${object_literal_to_class_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/object-literal-to-class/${object_literal_to_class_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/object-literal-to-class-${object_literal_to_class_name}.obj")
    set_tests_properties(test-compile-object-literal-to-class-${object_literal_to_class_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "an object literal with methods or accessors cannot be assigned to class 'C'"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# A value a builtin cannot cast to what it takes is the cast's error; the failed cast's null value
# went on to the op's builder and crashed the compiler (#438).
set(builtin_cast_cases
    "push|can't assign '[(][)] => s32' to '[(][)] => number'"
    "unshift|can't assign '[(][)] => s32' to '[(][)] => number'"
    "splice|can't assign '[(][)] => s32' to '[(][)] => number'"
    "push_string|invalid cast from [(][)] => s32 to string"
    "parse_int|invalid cast from [(][)] => s32 to string"
    "is_nan|invalid cast from [(][)] => s32 to number")
foreach(builtin_cast_case ${builtin_cast_cases})
    string(FIND "${builtin_cast_case}" "|" builtin_cast_sep)
    string(SUBSTRING "${builtin_cast_case}" 0 ${builtin_cast_sep} builtin_cast_name)
    math(EXPR builtin_cast_sep "${builtin_cast_sep} + 1")
    string(SUBSTRING "${builtin_cast_case}" ${builtin_cast_sep} -1 builtin_cast_message)
    add_test(NAME test-compile-builtin-cast-${builtin_cast_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/builtin-cast/${builtin_cast_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/builtin-cast-${builtin_cast_name}.obj")
    set_tests_properties(test-compile-builtin-cast-${builtin_cast_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${builtin_cast_message}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# A property that does not resolve is named in the error, in a condition too: the run inferring the
# function's return type gave no value and no message, and an `if` reported only "the condition has
# no value" (#442).
foreach(property_error_name if const while conditional)
    add_test(NAME test-compile-property-error-${property_error_name}
             COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=none
                     "${PROJECT_SOURCE_DIR}/test/tester/property-error/${property_error_name}.ts"
                     -o "${CMAKE_CURRENT_BINARY_DIR}/property-error-${property_error_name}.obj")
    set_tests_properties(test-compile-property-error-${property_error_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "Can't resolve property 'foo' of type [{]x:s32[}]"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()

# A yield inside an expression is an error, not a crash: what the expression computed before the
# yield would be an SSA value of the turn that returned, gone when the generator resumes.
foreach(yield_mm gc rc none)
    foreach(yield_case binary call_argument yield_star_value)
        add_test(NAME test-compile-${yield_mm}-yield-in-expression-${yield_case}
                 COMMAND $<TARGET_FILE:tslang> --emit=obj --no-default-lib -mm=${yield_mm}
                         "${PROJECT_SOURCE_DIR}/test/tester/yield-in-expression/${yield_case}.ts"
                         -o "${CMAKE_CURRENT_BINARY_DIR}/yield-in-expression-${yield_mm}-${yield_case}.obj")
        if (yield_case STREQUAL "yield_star_value")
            set(yield_message "the value of yield[*] is not supported")
        else()
            set(yield_message "a yield inside an expression is not supported")
        endif()
        set_tests_properties(test-compile-${yield_mm}-yield-in-expression-${yield_case}
                             PROPERTIES PASS_REGULAR_EXPRESSION "${yield_message}"
                                        FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
    endforeach()
endforeach()

# The shared-component tier under the other two models. A shared library records the model
# it was built under, so both halves of a pair are built with the same flag - which is what
# these run. The file pairs are the default model's, verbatim.
tslang_add_test(NAME test-compile-time-rc-shared-decl-emit-type COMMAND test-runner -shared -mm=rc -compile-time "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_compiletime_func.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-compile-time-none-shared-decl-emit-type COMMAND test-runner -shared -mm=none -compile-time "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_compiletime_func.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-compile-time-rc-shared-decl-emit-class COMMAND test-runner -shared -mm=rc -compile-time "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_compiletime_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")
tslang_add_test(NAME test-compile-time-none-shared-decl-emit-class COMMAND test-runner -shared -mm=none -compile-time "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_compiletime_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")
tslang_add_test(NAME test-compile-rc-shared-component COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/use_shared.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/shared.ts")
tslang_add_test(NAME test-compile-none-shared-component COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/use_shared.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/shared.ts")
tslang_add_test(NAME test-compile-rc-shared-decl-emit-interface COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_interface.ts")
tslang_add_test(NAME test-compile-none-shared-decl-emit-interface COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_interface.ts")
tslang_add_test(NAME test-compile-rc-shared-decl-emit-type COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_type.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-compile-none-shared-decl-emit-type COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_type.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-compile-rc-shared-decl-emit-enum COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_enum.ts")
tslang_add_test(NAME test-compile-none-shared-decl-emit-enum COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_enum.ts")
tslang_add_test(NAME test-compile-rc-shared-decl-emit-class COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")
tslang_add_test(NAME test-compile-none-shared-decl-emit-class COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-interface COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_interface.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-interface COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_interface.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-with-class-types COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_class_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_class_types.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-with-class-types COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_class_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_class_types.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-owned-returns COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-owned-returns COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-extends COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-throw-class COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_throw_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_throw_class.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-extends COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-throw-class COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_throw_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_throw_class.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-extends-implements-diamond COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_implements_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_implements_diamond.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-extends-implements-diamond COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_implements_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_implements_diamond.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-extends-multilevel COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_multilevel.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-extends-multilevel COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_multilevel.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-abstract COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-abstract COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-static COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_static.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-static COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_static.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-implements-interface-multilevel COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_multilevel.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-implements-interface-multilevel COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_multilevel.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-implements-interface-optional COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_optional.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-implements-interface-optional COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_optional.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-structural-interface COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_structural_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_structural_interface.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-structural-interface COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_structural_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_structural_interface.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-implements-interface-abstract COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_abstract.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-implements-interface-abstract COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_abstract.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-abstract-virtual-dispatch COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract_virtual_dispatch.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract_virtual_dispatch.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-abstract-virtual-dispatch COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract_virtual_dispatch.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract_virtual_dispatch.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-generic COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-generic COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-function-generic COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-function-generic COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-type-alias-generic COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-type-alias-generic COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-function COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-function COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-type-alias COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-type-alias COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-interface-generic COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_generic.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-interface-generic COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_generic.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-accessor COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_accessor.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_accessor.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-accessor COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_accessor.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_accessor.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-class-indexer COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_indexer.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-class-indexer COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_indexer.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-interface-indexer COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_indexer.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-interface-indexer COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_indexer.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-with-interface COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_interface.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-with-interface COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_interface.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-untyped COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-untyped COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-untyped-multi-method COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped_multi_method.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-untyped-multi-method COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped_multi_method.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-params COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_params.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_params.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-params COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_params.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_params.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-multi-method COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_multi_method.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-multi-method COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_multi_method.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-interleaved COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_interleaved.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_interleaved.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-interleaved COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_interleaved.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_interleaved.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-extends-interface COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-extends-interface COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-extends-interface-multilevel COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_multilevel.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-extends-interface-multilevel COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_multilevel.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-extends-interface-diamond COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_diamond.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-extends-interface-diamond COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_diamond.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-extends-interface-triple COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_triple.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_triple.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-extends-interface-triple COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_triple.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_triple.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-object-literal-structural-typed-extends-interface-optional COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_optional.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-object-literal-structural-typed-extends-interface-optional COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_optional.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-vars COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-vars COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-vars-2 COMMAND test-runner -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars2.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars2.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-vars-2 COMMAND test-runner -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars2.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars2.ts")
tslang_add_test(NAME test-compile-rc-shared-export-import-enum COMMAND test-runner -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_enum.ts")
tslang_add_test(NAME test-compile-none-shared-export-import-enum COMMAND test-runner -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_enum.ts")
tslang_add_test(NAME test-jit-rc-shared-component COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/use_shared.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/shared.ts")
tslang_add_test(NAME test-jit-none-shared-component COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/use_shared.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/shared.ts")
tslang_add_test(NAME test-jit-rc-shared-decl-emit-interface COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_interface.ts")
tslang_add_test(NAME test-jit-none-shared-decl-emit-interface COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_interface.ts")
tslang_add_test(NAME test-jit-rc-shared-decl-emit-type COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_type.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-jit-none-shared-decl-emit-type COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_type.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_type.ts")
tslang_add_test(NAME test-jit-rc-shared-decl-emit-enum COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_enum.ts")
tslang_add_test(NAME test-jit-none-shared-decl-emit-enum COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_enum.ts")
tslang_add_test(NAME test-jit-rc-shared-decl-emit-class COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")
tslang_add_test(NAME test-jit-none-shared-decl-emit-class COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/emit_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/decl_class.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-interface COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_interface.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-interface COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_interface.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-with-class-types COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_class_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_class_types.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-with-class-types COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_class_types.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_class_types.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-owned-returns COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-owned-returns COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-extends COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-throw-class COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_throw_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_throw_class.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-extends COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-throw-class COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_throw_class.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_throw_class.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-extends-multilevel COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_multilevel.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-extends-multilevel COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_multilevel.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-extends-implements-diamond COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_implements_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_implements_diamond.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-extends-implements-diamond COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_extends_implements_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_extends_implements_diamond.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-abstract COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-abstract COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-static COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_static.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-static COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_static.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_static.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-implements-interface-multilevel COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_multilevel.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-implements-interface-multilevel COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_multilevel.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-implements-interface-optional COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_optional.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-implements-interface-optional COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_optional.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-structural-interface COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_structural_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_structural_interface.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-structural-interface COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_structural_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_structural_interface.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-implements-interface-abstract COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_abstract.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-implements-interface-abstract COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_implements_interface_abstract.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_implements_interface_abstract.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-abstract-virtual-dispatch COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract_virtual_dispatch.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract_virtual_dispatch.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-abstract-virtual-dispatch COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_abstract_virtual_dispatch.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_abstract_virtual_dispatch.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-generic COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-generic COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-function-generic COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-function-generic COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-type-alias-generic COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-type-alias-generic COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-function COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-function COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-type-alias COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-type-alias COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-interface-generic COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_generic.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-interface-generic COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_generic.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-accessor COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_accessor.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_accessor.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-accessor COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_accessor.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_accessor.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-class-indexer COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_indexer.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-class-indexer COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_indexer.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-interface-indexer COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_indexer.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-interface-indexer COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_interface_indexer.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_interface_indexer.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-with-interface COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_interface.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-with-interface COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_with_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_with_interface.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-untyped COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-untyped COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-untyped-multi-method COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped_multi_method.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-untyped-multi-method COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_untyped_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_untyped_multi_method.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-params COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_params.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_params.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-params COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_params.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_params.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-multi-method COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_multi_method.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-multi-method COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_multi_method.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_multi_method.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-interleaved COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_interleaved.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_interleaved.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-interleaved COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_interleaved.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_interleaved.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-extends-interface COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-extends-interface COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-extends-interface-multilevel COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_multilevel.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-extends-interface-multilevel COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_multilevel.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_multilevel.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-extends-interface-diamond COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_diamond.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-extends-interface-diamond COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_diamond.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_diamond.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-extends-interface-triple COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_triple.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_triple.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-extends-interface-triple COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_triple.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_triple.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-object-literal-structural-typed-extends-interface-optional COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_optional.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-object-literal-structural-typed-extends-interface-optional COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_object_literal_structural_typed_extends_interface_optional.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_object_literal_structural_typed_extends_interface_optional.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-vars COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-vars COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-vars-2 COMMAND test-runner -jit -shared -mm=rc -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars2.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars2.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-vars-2 COMMAND test-runner -jit -shared -mm=none -gctors-as-method "${PROJECT_SOURCE_DIR}/test/tester/tests/import_vars2.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_vars2.ts")
tslang_add_test(NAME test-jit-rc-shared-export-import-enum COMMAND test-runner -jit -shared -mm=rc "${PROJECT_SOURCE_DIR}/test/tester/tests/import_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_enum.ts")
tslang_add_test(NAME test-jit-none-shared-export-import-enum COMMAND test-runner -jit -shared -mm=none "${PROJECT_SOURCE_DIR}/test/tester/tests/import_enum.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_enum.ts")

# Without the shared Boehm (see TSLANG_HAVE_SHARED_GC above) every -shared test fails to link
# with "could not open 'gc.lib'", so disable them instead. Only Windows links the shared
# collector. Every test named "-shared-" passes -shared and no other test does.
if (WIN32 AND NOT TSLANG_HAVE_SHARED_GC)
    get_property(tslang_all_tests DIRECTORY PROPERTY TESTS)
    foreach(tslang_test ${tslang_all_tests})
        if (tslang_test MATCHES "-shared-")
            set_tests_properties(${tslang_test} PROPERTIES DISABLED TRUE)
        endif()
    endforeach()
endif()

if (TSLANG_TEST_X86)
    # an exclusion that names no registered twin is a typo or a stale entry
    get_property(x86_excluded_seen GLOBAL PROPERTY TSLANG_X86_EXCLUDED_SEEN)
    foreach(x86_excluded ${TSLANG_X86_EXCLUDED})
        if (NOT x86_excluded IN_LIST x86_excluded_seen)
            message(FATAL_ERROR "x86-exclusions.cmake: ${x86_excluded} is not a registered test with an x86 twin")
        endif()
    endforeach()
    get_property(x86_twins GLOBAL PROPERTY TSLANG_X86_TWINS)
    get_property(x86_jit_skipped GLOBAL PROPERTY TSLANG_X86_JIT_SKIPPED)
    list(LENGTH x86_twins x86_twin_count)
    list(LENGTH x86_jit_skipped x86_jit_skipped_count)
    list(LENGTH TSLANG_X86_EXCLUDED x86_excluded_count)
    message(STATUS "x86 twins: ${x86_twin_count} registered (${x86_excluded_count} excluded), "
                   "${x86_jit_skipped_count} -jit registrations without a twin")
endif()

# An untyped catch of a C++ exception tslang did not throw (Itanium only: the box thunk field of
# a __pointer_type_info is a Linux mechanism). foreign_throw.cpp throws a `const char*`, a
# pointer to a C++ struct and an int; the catch must box each without calling into the type_info.
if (NOT WIN32)
    add_library(tslang_foreign_throw SHARED foreign-throw/foreign_throw.cpp)
    add_dependencies(tslang_foreign_throw TypeScriptRuntime)
    add_test(NAME test-jit-foreign-throw-catch
             COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib
                     --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                     --shared-libs=$<TARGET_FILE:tslang_foreign_throw>
                     "${PROJECT_SOURCE_DIR}/test/tester/foreign-throw/catch.ts")
    set_tests_properties(test-jit-foreign-throw-catch
                         PROPERTIES PASS_REGULAR_EXPRESSION "done\\."
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Segmentation|assertion failed")
endif()

# An exception nothing catches ends a JIT run with an error that says what was thrown (a class by its
# name), not with LLVM's crash handler reporting a crash of the compiler. Windows only: that is where
# the JIT catches it.
if (WIN32)
    set(uncaught_expected_throw_string "error: uncaught exception: not caught")
    set(uncaught_expected_throw_class "error: uncaught exception: Failure")
    set(uncaught_expected_throw_number "error: uncaught exception: 2.5")
    foreach(uncaught_name throw_string throw_class throw_number)
        foreach(uncaught_model gc rc none)
            add_test(NAME test-jit-${uncaught_model}-uncaught-${uncaught_name}
                     COMMAND $<TARGET_FILE:tslang> --emit=jit --no-default-lib -mm=${uncaught_model}
                             --shared-libs=$<TARGET_FILE:TypeScriptRuntime>
                             "${PROJECT_SOURCE_DIR}/test/tester/uncaught/${uncaught_name}.ts")
            set_tests_properties(test-jit-${uncaught_model}-uncaught-${uncaught_name}
                                 PROPERTIES PASS_REGULAR_EXPRESSION "${uncaught_expected_${uncaught_name}}"
                                            FAIL_REGULAR_EXPRESSION "Stack dump|PLEASE submit")
        endforeach()
    endforeach()
endif()

# Stack promotion (docs/superpowers/specs/2026-10-18-stack-promotion-design.md). Run tests: every
# model, JIT and ahead of time; the output must not depend on where an object lives.
set(stack_promotion_run_tests class_call_same_as_new literal_boxed_local leaf_local leaf_returned
    ctor_local borrow_uses pass_to_reader pass_to_keeper method_keeps_this virtual_one_keeps ctor_keeps_this
    to_interface to_any pass_to_print loop_carried
    fields_string fields_object field_overwrite return_field try_region conditional_alloc shared_handle
    too_big frame_budget recursive async_local generator_local thrown
    array_local_fixed array_local_push array_returned closure_local closure_escapes
    virtual_exported_subclass recursive_closure loop_exit_use closure_direct_call closure_nested recursive_mutual
    try_catch_reads)
# Programs -mm=own rejects on its own (a keeper it cannot move into, a move on some paths only):
# they say nothing about promotion under own.
set(stack_promotion_not_own method_keeps_this virtual_one_keeps ctor_keeps_this loop_carried)
foreach(stack_mm gc rc own none)
    foreach(stack_test ${stack_promotion_run_tests})
        if (stack_mm STREQUAL "own" AND stack_test IN_LIST stack_promotion_not_own)
            continue()
        endif()
        # async under rc and own fails to link in the JIT without promotion too
        if ((stack_mm STREQUAL "rc" OR stack_mm STREQUAL "own") AND stack_test STREQUAL "async_local")
            continue()
        endif()
        tslang_add_test(NAME test-jit-${stack_mm}-stack-${stack_test} COMMAND test-runner -jit -mm=${stack_mm} "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/${stack_test}.ts")
        tslang_add_test(NAME test-compile-${stack_mm}-stack-${stack_test} COMMAND test-runner -mm=${stack_mm} "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/${stack_test}.ts")
    endforeach()
endforeach()

# The report reaches stderr, and the pass runs before --emit=mlir-affine stops the pipeline: every
# decision test below depends on both.
add_test(NAME test-stack-promotion-report-reaches-stderr
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=none --stack-promotion-report
                 "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/class_call_same_as_new.ts")
set_tests_properties(test-stack-promotion-report-reaches-stderr
                     PROPERTIES PASS_REGULAR_EXPRESSION "remark: stack-promotion: "
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
add_test(NAME test-stack-promotion-off
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=none --stack-promotion-report --no-stack-promotion
                 "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/class_call_same_as_new.ts")
set_tests_properties(test-stack-promotion-off
                     PROPERTIES PASS_REGULAR_EXPRESSION "ts\\.Func"
                                FAIL_REGULAR_EXPRESSION "stack-promotion:|Stack dump|Assertion failed")
tslang_add_test(NAME test-jit-gc-stack-off-class-call COMMAND test-runner -jit -mm=gc -no-stack-promotion "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/class_call_same_as_new.ts")

# Decision tests: `<file>|<regex>`; the remark at the allocation site says stack, or heap and why.
set(stack_promotion_decision_tests
    "literal_boxed_local|literal_boxed_local\.ts:6:19: remark: stack-promotion: stack"
    "leaf_local|leaf_local\.ts:11:19: remark: stack-promotion: stack"
    "leaf_returned|leaf_returned\.ts:8:15: remark: stack-promotion: heap: returned"
    "ctor_local|ctor_local\.ts:14:19: remark: stack-promotion: stack"
    "borrow_uses|borrow_uses\.ts:17:15: remark: stack-promotion: stack.*borrow_uses\.ts:18:15: remark: stack-promotion: stack"
    "pass_to_reader|pass_to_reader\.ts:18:19: remark: stack-promotion: stack"
    "pass_to_keeper|pass_to_keeper\.ts:16:15: remark: stack-promotion: heap: (argument|kept for good)"
    "method_keeps_this|method_keeps_this\.ts:15:15: remark: stack-promotion: heap: argument"
    "virtual_one_keeps|virtual_one_keeps\.ts:26:18: remark: stack-promotion: heap: argument"
    "ctor_keeps_this|ctor_keeps_this\.ts:13:15: remark: stack-promotion: heap: argument"
    "to_interface|to_interface\.ts:14:15: remark: stack-promotion: heap: converted"
    "to_any|to_any\.ts:10:15: remark: stack-promotion: heap: converted"
    "pass_to_print|pass_to_print\.ts:13:15: remark: stack-promotion: stack"
    "loop_carried|loop_carried\.ts:13:19: remark: stack-promotion: heap: loop-carried"
    "fields_string|fields_string\.ts:15:19: remark: stack-promotion: stack"
    "fields_object|fields_object\.ts:17:15: remark: stack-promotion: stack.*fields_object\.ts:17:25: remark: stack-promotion: heap: argument"
    "field_overwrite|field_overwrite\.ts:10:15: remark: stack-promotion: stack"
    "return_field|return_field\.ts:11:15: remark: stack-promotion: stack"
    "conditional_alloc|conditional_alloc\.ts:12:19: remark: stack-promotion: stack"
    "too_big|too_big\.ts:47:15: remark: stack-promotion: heap: too-big"
    "frame_budget|frame_budget\.ts:67:17: remark: stack-promotion: stack.*frame_budget\.ts:69:17: remark: stack-promotion: heap: frame-budget"
    "recursive|recursive\.ts:16:15: remark: stack-promotion: heap: recursive"
    "async_local|async_local\.ts:10:15: remark: stack-promotion: heap: coroutine"
    "generator_local|generator_local\.ts:11:19: remark: stack-promotion: heap: coroutine"
    "thrown|thrown\.ts:10:15: remark: stack-promotion: heap: thrown"
    "array_local_fixed|array_local_fixed\.ts:3:15: remark: stack-promotion: stack"
    "array_local_push|array_local_push\.ts:3:25: remark: stack-promotion: heap: growable"
    "array_returned|array_returned\.ts:3:15: remark: stack-promotion: heap: returned"
    "virtual_exported_subclass|virtual_exported_subclass\.ts:23:18: remark: stack-promotion: heap: unknown-callee"
    "recursive_closure|recursive_closure\.ts:17:15: remark: stack-promotion: heap: recursive"
    "loop_exit_use|loop_exit_use\.ts:12:19: remark: stack-promotion: stack.*loop_exit_use\.ts:23:19: remark: stack-promotion: stack"
    "recursive_mutual|recursive_mutual\.ts:11:15: remark: stack-promotion: heap: recursive.*recursive_mutual\.ts:16:15: remark: stack-promotion: heap: recursive.*recursive_mutual\.ts:21:15: remark: stack-promotion: stack")
foreach(stack_mm gc rc own none)
    foreach(stack_case ${stack_promotion_decision_tests})
        string(FIND "${stack_case}" "|" stack_sep)
        string(SUBSTRING "${stack_case}" 0 ${stack_sep} stack_name)
        math(EXPR stack_sep "${stack_sep} + 1")
        string(SUBSTRING "${stack_case}" ${stack_sep} -1 stack_regex)
        if (stack_mm STREQUAL "own" AND stack_name IN_LIST stack_promotion_not_own)
            continue()
        endif()
        add_test(NAME test-stack-decision-${stack_mm}-${stack_name}
                 COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${stack_mm} --stack-promotion-report
                         "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/${stack_name}.ts")
        set_tests_properties(test-stack-decision-${stack_mm}-${stack_name}
                             PROPERTIES PASS_REGULAR_EXPRESSION "${stack_regex}"
                                        FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
    endforeach()
endforeach()
# Borrow facts (docs/superpowers/specs/2026-10-18-borrow-facts-across-modules-design.md): the
# library's __tsborrow_ constant lists each function's parameters its body only reads. `this.twice(a)`
# is a virtual call: a library never trusts its vtables, so Holder.peek lists nothing.
set(stack_facts_listed "@__tsborrow_facts_lib_[0-9]+ = (dllexport )?constant [[][0-9]+ x i8[]] c\"[^\"]*Geo[.]len 0.0A[^\"]*mixed 0.0A[^\"]*sumOf 0.0A")
# viaCallback lists its callback (1), never the array it hands to it (0).
set(stack_facts_unlisted "__tsborrow_[^\"]*c\"[^\"]*(keep [0-9]|same [0-9]|viaCallback 0|useOther [0-9]|Holder[.]peek [0-9]|mixed 0 [0-9]|lenOf [0-9]|renamedLen)|Stack dump|Assertion failed")
foreach(stack_mm gc rc none)
    add_test(NAME test-stack-facts-lib-${stack_mm}
             COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=${stack_mm} -o -
                     "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/facts_lib.ts")
    set_tests_properties(test-stack-facts-lib-${stack_mm}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${stack_facts_listed}"
                                    FAIL_REGULAR_EXPRESSION "${stack_facts_unlisted}")
endforeach()
# -mm=own refuses a function that keeps or returns its parameter, so its library has the rest
add_test(NAME test-stack-facts-lib-own
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib -mm=own -o -
                 "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/facts_lib_own.ts")
set_tests_properties(test-stack-facts-lib-own
                     PROPERTIES PASS_REGULAR_EXPRESSION "@__tsborrow_facts_lib_own_[0-9]+ = (dllexport )?constant [[][0-9]+ x i8[]] c\"[^\"]*Geo[.]len 0.0A[^\"]*sumOf 0.0A"
                                FAIL_REGULAR_EXPRESSION "${stack_facts_unlisted}")
# what the module promises importers, whatever it promotes itself or exports
add_test(NAME test-stack-facts-lib-no-promotion
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib --no-stack-promotion --export=none -o -
                 "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/facts_lib.ts")
set_tests_properties(test-stack-facts-lib-no-promotion
                     PROPERTIES PASS_REGULAR_EXPRESSION "${stack_facts_listed}"
                                FAIL_REGULAR_EXPRESSION "${stack_facts_unlisted}")
# a library built before facts existed: no symbol at all
add_test(NAME test-stack-facts-lib-suppressed
         COMMAND $<TARGET_FILE:tslang> --emit=llvm --no-default-lib --no-borrow-facts-export -o -
                 "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/facts_lib.ts")
set_tests_properties(test-stack-facts-lib-suppressed
                     PROPERTIES PASS_REGULAR_EXPRESSION "define [a-z ]*double @sumOf"
                                FAIL_REGULAR_EXPRESSION "__tsborrow_|Stack dump|Assertion failed")
# ...read back from the object, a static library holding it, and not from an import library; and
# read by a program linked against the static library
foreach(stack_mm gc rc own none)
    add_test(NAME test-borrow-facts-${stack_mm}
             COMMAND ${CMAKE_COMMAND}
                     "-DTSLANG=$<TARGET_FILE:tslang>"
                     "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/tester/stack-promotion"
                     "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/borrow-facts"
                     "-DAR=${CMAKE_AR}"
                     "-DMM=${stack_mm}"
                     "-DGC_LIB=${TEST_GC_LIBDIR}"
                     "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                     "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                     -P "${CMAKE_CURRENT_SOURCE_DIR}/borrow-facts.cmake")
endforeach()
# ...and from the real default library, when one built with facts is there
add_test(NAME test-borrow-facts-defaultlib
         COMMAND ${CMAKE_COMMAND}
                 "-DTSLANG=$<TARGET_FILE:tslang>"
                 "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}/test/tester/stack-promotion"
                 "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/borrow-facts-defaultlib"
                 "-DAR=${CMAKE_AR}"
                 "-DGC_LIB=${TEST_GC_LIBDIR}"
                 "-DLLVM_LIB=${LLVM_LIBRARY_DIR}"
                 "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                 "-DDEFAULT_LIB_CANDIDATES=${PROJECT_SOURCE_DIR}/../TypeScriptCompilerDefaultLib/__build;${PROJECT_SOURCE_DIR}/../../TypeScriptCompilerDefaultLib/__build"
                 -P "${CMAKE_CURRENT_SOURCE_DIR}/borrow-facts-defaultlib.cmake")
set_tests_properties(test-borrow-facts-defaultlib PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:")
# The same site is a stack site where nothing counts, and a heap site where a Shared count would see it.
foreach(stack_case "gc|stack" "none|stack" "rc|heap: shared-handle" "own|heap: shared-handle")
    string(FIND "${stack_case}" "|" stack_sep)
    string(SUBSTRING "${stack_case}" 0 ${stack_sep} stack_mm)
    math(EXPR stack_sep "${stack_sep} + 1")
    string(SUBSTRING "${stack_case}" ${stack_sep} -1 stack_expected)
    add_test(NAME test-stack-decision-${stack_mm}-shared_handle
             COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${stack_mm} --stack-promotion-report
                     "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/shared_handle.ts")
    set_tests_properties(test-stack-decision-${stack_mm}-shared_handle
                         PROPERTIES PASS_REGULAR_EXPRESSION "shared_handle\.ts:14:14: remark: stack-promotion: ${stack_expected}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()
# The object limit is a flag.
add_test(NAME test-stack-decision-max-object
         COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=gc --stack-promotion-report --stack-promotion-max-object=512
                 "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/too_big.ts")
set_tests_properties(test-stack-decision-max-object
                     PROPERTIES PASS_REGULAR_EXPRESSION "too_big\.ts:47:15: remark: stack-promotion: stack"
                                FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
# Decisions that depend on the model, `<model>|<file>|<line:col: remark>`. A capture box: on the
# stack where nothing counts, and only while its closures stay in the function - called by name or
# passed to a reader, nested in another closure. A function with a try region: refused only where
# an unwind path would have to give back what the object owns.
foreach(stack_case "gc|closure_local|10:11: remark: stack-promotion: stack"
                   "none|closure_local|10:11: remark: stack-promotion: stack"
                   "rc|closure_local|10:11: remark: stack-promotion: heap: counted-capture"
                   "own|closure_local|10:11: remark: stack-promotion: heap: counted-capture"
                   "gc|closure_escapes|4:12: remark: stack-promotion: heap: returned"
                   "none|closure_escapes|4:12: remark: stack-promotion: heap: returned"
                   "gc|closure_direct_call|4:15: remark: stack-promotion: stack"
                   "none|closure_direct_call|4:15: remark: stack-promotion: stack"
                   "gc|closure_nested|11:11: remark: stack-promotion: stack"
                   "none|closure_nested|11:11: remark: stack-promotion: stack"
                   "gc|try_region|11:15: remark: stack-promotion: stack"
                   "none|try_region|11:15: remark: stack-promotion: stack"
                   "rc|try_region|11:15: remark: stack-promotion: heap: try-region"
                   "own|try_region|11:15: remark: stack-promotion: heap: try-region"
                   "gc|try_catch_reads|19:19: remark: stack-promotion: stack"
                   "none|try_catch_reads|19:19: remark: stack-promotion: stack")
    string(REPLACE "|" ";" stack_parts "${stack_case}")
    list(GET stack_parts 0 stack_mm)
    list(GET stack_parts 1 stack_name)
    list(GET stack_parts 2 stack_expected)
    add_test(NAME test-stack-decision-${stack_mm}-${stack_name}
             COMMAND $<TARGET_FILE:tslang> --emit=mlir-affine --no-default-lib -mm=${stack_mm} --stack-promotion-report
                     "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/${stack_name}.ts")
    set_tests_properties(test-stack-decision-${stack_mm}-${stack_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${stack_name}\.ts:${stack_expected}"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()
# ts.ReleaseContents nulls each field it gives back: the release call is followed by a null store.
foreach(stack_mm rc own)
    add_test(NAME test-stack-release-contents-nulls-${stack_mm}
             COMMAND $<TARGET_FILE:tslang> --emit=mlir-llvm --no-default-lib -mm=${stack_mm}
                     "${PROJECT_SOURCE_DIR}/test/tester/stack-promotion/fields_string.ts")
    set_tests_properties(test-stack-release-contents-nulls-${stack_mm}
                         PROPERTIES PASS_REGULAR_EXPRESSION "llvm\\.call @tsrel_[0-9]+\\(%[0-9]+\\) : \\(!llvm\\.ptr\\) -> \\(\\)[\r\n]+ +%[0-9]+ = llvm\\.mlir\\.zero : !llvm\\.ptr[\r\n]+ +llvm\\.store"
                                    FAIL_REGULAR_EXPRESSION "Stack dump|Assertion failed")
endforeach()
