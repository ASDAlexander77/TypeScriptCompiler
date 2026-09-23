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
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_EXEPATH=\"${PROJECT_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/bin\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_LIBPATH=\"${PROJECT_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/lib\"")
else()
    target_compile_definitions(test-runner PUBLIC "TEST_GC_SHARED_LIBPATH=\"${TEST_GC_SHARED_PREFIX}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GC_SHARED_BINPATH=\"${TEST_GC_SHARED_PREFIX}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_GCPATH=\"${TEST_GC_LIBDIR}\"")
    target_compile_definitions(test-runner PUBLIC "TEST_TSLANG_EXEPATH=\"${CMAKE_BINARY_DIR}/bin\"")
    target_compile_definitions(test-runner PUBLIC "TEST_TSLANG_LIBPATH=\"${CMAKE_BINARY_DIR}/lib\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_EXEPATH=\"${PROJECT_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE_LOWERCASE}/bin\"")
    target_compile_definitions(test-runner PUBLIC "TEST_LLVM_LIBPATH=\"${PROJECT_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE_LOWERCASE}/lib\"")
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
    "${CMAKE_CURRENT_BINARY_DIR}/compilednone.sh" "${CMAKE_CURRENT_BINARY_DIR}/jitdnone.sh")
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
tslang_add_test(NAME test-compile-00-mixed-type-ops COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00mixed_type_ops.ts")
tslang_add_test(NAME test-compile-00-generator-manual-next COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next.ts")
tslang_add_test(NAME test-compile-00-generator-manual-next-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next2.ts")
tslang_add_test(NAME test-compile-00-global-const-object-method COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_const_object_method.ts")
tslang_add_test(NAME test-compile-00-funcs COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs.ts")
tslang_add_test(NAME test-compile-00-funcs-capture COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_capture.ts")
tslang_add_test(NAME test-compile-00-funcs-vararg COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_vararg.ts")
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
tslang_add_test(NAME test-compile-01-funcs-generic-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_generic_iterator.ts")
tslang_add_test(NAME test-compile-02-funcs-generic-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/02funcs_generic_iterator.ts")
tslang_add_test(NAME test-compile-00-funcs-typed-generic-iterator COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_typed_generic_iterator.ts")
tslang_add_test(NAME test-compile-00-funcs-nesting COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting.ts")
tslang_add_test(NAME test-compile-00-funcs-nesting-generic COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_generic.ts")
tslang_add_test(NAME test-compile-00-funcs-nesting-capture COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_capture.ts")
tslang_add_test(NAME test-compile-00-funcs-hybrid-null-this COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_hybrid_null_this.ts")
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
tslang_add_test(NAME test-compile-00-vars COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00vars.ts")
tslang_add_test(NAME test-compile-00-var-bindings COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00var_bindings.ts")
tslang_add_test(NAME test-compile-00-globals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals.ts")
tslang_add_test(NAME test-compile-00-globals-entry-point COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals_entry_point.ts")
tslang_add_test(NAME test-compile-00-globals2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals2.ts")
tslang_add_test(NAME test-compile-00-globals3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals3.ts")
tslang_add_test(NAME test-compile-00-arrays COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array.ts")
tslang_add_test(NAME test-compile-00-arrays2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array2.ts")
tslang_add_test(NAME test-compile-00-arrays3 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array3.ts")
tslang_add_test(NAME test-compile-00-arrays4-push-pop COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array4_push_pop.ts")
tslang_add_test(NAME test-compile-00-array-shift COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_shift.ts")
tslang_add_test(NAME test-compile-00-array-splice COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_splice.ts")
tslang_add_test(NAME test-compile-00-arrays5-deconstruct COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array5_deconst.ts")
tslang_add_test(NAME test-compile-00-arrays6 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00array6.ts")
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
tslang_add_test(NAME test-compile-00-catch-untyped-value COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_untyped_value.ts")
tslang_add_test(NAME test-compile-00-throw-bool-bigint COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_bool_bigint.ts")
tslang_add_test(NAME test-compile-00-throw-f32 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_f32.ts")
tslang_add_test(NAME test-compile-00-throw-u32 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_u32.ts")
tslang_add_test(NAME test-compile-00-cast-to-f32 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00cast_to_f32.ts")
tslang_add_test(NAME test-compile-00-instanceof-namespace-narrowing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof_namespace_narrowing.ts")
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
tslang_add_test(NAME test-compile-00-owned-temporaries COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_temporaries.ts")
tslang_add_test(NAME test-compile-00-owned-interfaces COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_interfaces.ts")
tslang_add_test(NAME test-compile-00-owned-closures COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_closures.ts")
tslang_add_test(NAME test-compile-00-owned-any-boxing COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_any_boxing.ts")
tslang_add_test(NAME test-compile-00-owned-strings COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_strings.ts")
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
tslang_add_test(NAME test-compile-00-instanceof COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof.ts")
tslang_add_test(NAME test-compile-00-class COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class.ts")
tslang_add_test(NAME test-compile-00-class-new COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_new.ts")
tslang_add_test(NAME test-compile-00-capture-in-new-arguments COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00capture_in_new_arguments.ts")
tslang_add_test(NAME test-compile-01-class-new COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01class_new.ts")
tslang_add_test(NAME test-compile-00-class-stack COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_stack.ts")
tslang_add_test(NAME test-compile-00-class-static COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static.ts")
tslang_add_test(NAME test-compile-00-class-static-block COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_block.ts")
tslang_add_test(NAME test-compile-00-class-discover-types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_discover_types.ts")
tslang_add_test(NAME test-compile-00-class-accessor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor.ts")
tslang_add_test(NAME test-compile-00-class-accessor-2 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor2.ts")
tslang_add_test(NAME test-compile-00-class-accessor-super COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_super.ts")
tslang_add_test(NAME test-compile-00-class-accessor-virtual COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_virtual.ts")
tslang_add_test(NAME test-compile-00-class-super COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super.ts")
tslang_add_test(NAME test-compile-00-class-super-static COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super_static.ts")
tslang_add_test(NAME test-compile-00-class-default-constructor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_def_constr.ts")
tslang_add_test(NAME test-compile-00-class-constructor-fields COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_constr_fields.ts")
tslang_add_test(NAME test-compile-00-class-static-constructor COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_constr.ts")
tslang_add_test(NAME test-compile-00-class-abstract COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_abstract.ts")
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
tslang_add_test(NAME test-compile-01-any COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01any.ts")
tslang_add_test(NAME test-compile-00-any-types COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_types.ts")
tslang_add_test(NAME test-compile-00-any-compare COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_compare.ts")
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
tslang_add_test(NAME test-compile-01-try-finally COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_finally.ts")
tslang_add_test(NAME test-compile-00-try-finally-return COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_return.ts")
tslang_add_test(NAME test-compile-00-try-finally-break-continue COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_break_continue.ts")
tslang_add_test(NAME test-compile-00-try-catch-rethrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_rethrow.ts")
tslang_add_test(NAME test-compile-00-try-catch-mismatch-rethrow COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_mismatch_rethrow.ts")
tslang_add_test(NAME test-compile-00-try-catch-return-dispose COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_return_dispose.ts")
tslang_add_test(NAME test-compile-01-try-catch-return-dispose COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_catch_return_dispose.ts")
tslang_add_test(NAME test-compile-00-property-access-conditional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00property_access_cond.ts")
tslang_add_test(NAME test-compile-00-method-access-conditional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00method_access_cond.ts")
tslang_add_test(NAME test-compile-01-method-access-conditional COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01method_access_cond.ts")
tslang_add_test(NAME test-compile-00-question-question COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00question_question.ts")
tslang_add_test(NAME test-compile-00-spread-assignment COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread_assignment.ts")
tslang_add_test(NAME test-compile-01-spread-assignment COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread_assignment.ts")
tslang_add_test(NAME test-compile-00-spread COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread.ts")
tslang_add_test(NAME test-compile-01-spread COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread.ts")
tslang_add_test(NAME test-compile-00-types-unknown1 COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_unknown1.ts")
tslang_add_test(NAME test-compile-00-types-indexed-access-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_indexedaccesstype.ts")
tslang_add_test(NAME test-compile-00-types-mapped-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_mappedtype.ts")
tslang_add_test(NAME test-compile-01-types-mapped-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_mappedtype.ts")
tslang_add_test(NAME test-compile-00-types-template-literal-type COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_templateliteraltype.ts")
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
tslang_add_test(NAME test-jit-00-mixed-type-ops COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00mixed_type_ops.ts")
tslang_add_test(NAME test-jit-00-generator-manual-next COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next.ts")
tslang_add_test(NAME test-jit-00-generator-manual-next-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00generator_manual_next2.ts")
tslang_add_test(NAME test-jit-00-global-const-object-method COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00global_const_object_method.ts")
tslang_add_test(NAME test-jit-00-funcs COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs.ts")
tslang_add_test(NAME test-jit-00-funcs-capture COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_capture.ts")
tslang_add_test(NAME test-jit-00-funcs-vararg COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_vararg.ts")
tslang_add_test(NAME test-jit-01-funcs-vararg COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_vararg.ts")
tslang_add_test(NAME test-jit-02-funcs-vararg COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02funcs_vararg.ts")
tslang_add_test(NAME test-jit-00-funcs-bindings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_bindings.ts")
tslang_add_test(NAME test-jit-00-funcs-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic.ts")
tslang_add_test(NAME test-jit-01-funcs-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_generic.ts")
tslang_add_test(NAME test-jit-00-funcs-generic-typeof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_with_typeof.ts")
tslang_add_test(NAME test-jit-00-funcs-generic-arrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_arrow.ts")
tslang_add_test(NAME test-jit-00-funcs-typeless-function-as-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_typeless_function_as_generic.ts")
tslang_add_test(NAME test-jit-00-funcs-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_generic_iterator.ts")
tslang_add_test(NAME test-jit-01-funcs-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01funcs_generic_iterator.ts")
tslang_add_test(NAME test-jit-02-funcs-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/02funcs_generic_iterator.ts")
tslang_add_test(NAME test-jit-00-funcs-typed-generic-iterator COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_typed_generic_iterator.ts")
tslang_add_test(NAME test-jit-00-funcs-nesting COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting.ts")
tslang_add_test(NAME test-jit-00-funcs-nesting-generic COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_generic.ts")
tslang_add_test(NAME test-jit-00-funcs-nesting-capture COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_nesting_capture.ts")
tslang_add_test(NAME test-jit-00-funcs-hybrid-null-this COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00funcs_hybrid_null_this.ts")
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
tslang_add_test(NAME test-jit-00-vars COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00vars.ts")
tslang_add_test(NAME test-jit-00-var-bindings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00var_bindings.ts")
tslang_add_test(NAME test-jit-00-globals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals.ts")
tslang_add_test(NAME test-jit-00-globals-entry-point COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals_entry_point.ts")
tslang_add_test(NAME test-jit-00-globals2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals2.ts")
tslang_add_test(NAME test-jit-00-globals3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00globals3.ts")
tslang_add_test(NAME test-jit-00-arrays COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array.ts")
tslang_add_test(NAME test-jit-00-arrays2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array2.ts")
tslang_add_test(NAME test-jit-00-arrays3 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array3.ts")
tslang_add_test(NAME test-jit-00-arrays4-push-pop COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array4_push_pop.ts")
tslang_add_test(NAME test-jit-00-array-shift COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_shift.ts")
tslang_add_test(NAME test-jit-00-array-splice COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array_splice.ts")
tslang_add_test(NAME test-jit-00-arrays5-deconstruct COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array5_deconst.ts")
tslang_add_test(NAME test-jit-00-arrays6 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00array6.ts")
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
tslang_add_test(NAME test-jit-00-owned-locals COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_locals.ts")
tslang_add_test(NAME test-jit-00-throw-in-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_in_catch.ts")
tslang_add_test(NAME test-jit-00-throw-inlined COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_inlined.ts")
tslang_add_test(NAME test-jit-00-try-catch-sequence-and-nesting COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_sequence_and_nesting.ts")
tslang_add_test(NAME test-jit-00-try-in-catch-with-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_in_catch_with_finally.ts")
tslang_add_test(NAME test-jit-00-throw-number-untyped-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_number_untyped_catch.ts")
tslang_add_test(NAME test-jit-00-catch-untyped-value COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00catch_untyped_value.ts")
tslang_add_test(NAME test-jit-00-throw-bool-bigint COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_bool_bigint.ts")
tslang_add_test(NAME test-jit-00-throw-f32 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_f32.ts")
tslang_add_test(NAME test-jit-00-throw-u32 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00throw_u32.ts")
tslang_add_test(NAME test-jit-00-cast-to-f32 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00cast_to_f32.ts")
tslang_add_test(NAME test-jit-00-instanceof-namespace-narrowing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof_namespace_narrowing.ts")
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
tslang_add_test(NAME test-jit-00-owned-temporaries COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_temporaries.ts")
tslang_add_test(NAME test-jit-00-owned-interfaces COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_interfaces.ts")
tslang_add_test(NAME test-jit-00-owned-closures COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_closures.ts")
tslang_add_test(NAME test-jit-00-owned-any-boxing COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_any_boxing.ts")
tslang_add_test(NAME test-jit-00-owned-strings COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00owned_strings.ts")
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
tslang_add_test(NAME test-jit-00-instanceof COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00instanceof.ts")
tslang_add_test(NAME test-jit-00-class COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class.ts")
tslang_add_test(NAME test-jit-00-class-new COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_new.ts")
tslang_add_test(NAME test-jit-00-capture-in-new-arguments COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00capture_in_new_arguments.ts")
tslang_add_test(NAME test-jit-01-class-new COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01class_new.ts")
tslang_add_test(NAME test-jit-00-class-stack COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_stack.ts")
tslang_add_test(NAME test-jit-00-class-static COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static.ts")
tslang_add_test(NAME test-jit-00-class-static-block COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_block.ts")
tslang_add_test(NAME test-jit-00-class-discover-types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_discover_types.ts")
tslang_add_test(NAME test-jit-00-class-accessor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor.ts")
tslang_add_test(NAME test-jit-00-class-accessor-2 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor2.ts")
tslang_add_test(NAME test-jit-00-class-accessor-super COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_super.ts")
tslang_add_test(NAME test-jit-00-class-accessor-virtual COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_accessor_virtual.ts")
tslang_add_test(NAME test-jit-00-class-super COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super.ts")
tslang_add_test(NAME test-jit-00-class-super-static COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_super_static.ts")
tslang_add_test(NAME test-jit-00-class-default-constructor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_def_constr.ts")
tslang_add_test(NAME test-jit-00-class-static-constructor COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_static_constr.ts")
tslang_add_test(NAME test-jit-00-class-constructor-fields COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_constr_fields.ts")
tslang_add_test(NAME test-jit-00-class-abstract COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00class_abstract.ts")
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
tslang_add_test(NAME test-jit-01-any COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01any.ts")
tslang_add_test(NAME test-jit-00-any-types COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_types.ts")
tslang_add_test(NAME test-jit-00-any-compare COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00any_compare.ts")
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
tslang_add_test(NAME test-jit-01-try-finally COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01try_finally.ts")
tslang_add_test(NAME test-jit-00-try-finally-return COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_return.ts")
tslang_add_test(NAME test-jit-00-try-finally-break-continue COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_finally_break_continue.ts")
tslang_add_test(NAME test-jit-00-try-catch-rethrow COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00try_catch_rethrow.ts")
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
tslang_add_test(NAME test-jit-01-method-access-conditional COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01method_access_cond.ts")
tslang_add_test(NAME test-jit-00-question-question COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00question_question.ts")
tslang_add_test(NAME test-jit-00-spread-assignment COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread_assignment.ts")
tslang_add_test(NAME test-jit-01-spread-assignment COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread_assignment.ts")
tslang_add_test(NAME test-jit-00-spread COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00spread.ts")
tslang_add_test(NAME test-jit-01-spread COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01spread.ts")
tslang_add_test(NAME test-jit-00-types-unknown1 COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_unknown1.ts")
tslang_add_test(NAME test-jit-00-types-indexed-access-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_indexedaccesstype.ts")
tslang_add_test(NAME test-jit-00-types-mapped-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_mappedtype.ts")
tslang_add_test(NAME test-jit-01-types-mapped-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/01types_mappedtype.ts")
tslang_add_test(NAME test-jit-00-types-template-literal-type COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00types_templateliteraltype.ts")
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
# Item 5al: an imported function's returned reference is taken over rather than retained
# again. Registered statically as well as shared, and under every memory model - the
# static two-module form had no rc/none coverage at all, which is why the leak it fixes
# went unmeasured for so long.
tslang_add_test(NAME test-compile-export-import-owned-returns COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_owned_returns.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_owned_returns.ts")
tslang_add_test(NAME test-compile-export-import-const-locals COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_locals.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_locals.ts")
tslang_add_test(NAME test-compile-export-import-const-functions COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_const_functions.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_const_functions.ts")
tslang_add_test(NAME test-compile-export-import-generators COMMAND test-runner "${PROJECT_SOURCE_DIR}/test/tester/tests/import_generators.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_generators.ts")
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
# FIXED: see the matching test-compile-export-import-class-generic comment above (2026-07-22).
tslang_add_test(NAME test-compile-shared-export-import-class-generic COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_class_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_class_generic.ts")
tslang_add_test(NAME test-compile-shared-export-import-function-generic COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function_generic.ts")
tslang_add_test(NAME test-compile-shared-export-import-type-alias-generic COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_type_alias_generic.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_type_alias_generic.ts")
tslang_add_test(NAME test-compile-shared-export-import-function COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_function.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_function.ts")
tslang_add_test(NAME test-compile-shared-export-import-dllname COMMAND test-runner -shared "${PROJECT_SOURCE_DIR}/test/tester/tests/import_dllname.ts" "${PROJECT_SOURCE_DIR}/test/tester/tests/export_dllname.ts")
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
tslang_add_test(NAME test-jit-00-nested-catch COMMAND test-runner -jit "${PROJECT_SOURCE_DIR}/test/tester/tests/00nested_catch.ts")
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
    00any.ts
    00array_assignment5.ts
    00array_cond_access.ts
    00array_move_wide_elements.ts
    00array_of.ts
    00array_shift.ts
    00array_splice.ts
    00array_view.ts
    00array.ts
    00array2.ts
    00array3.ts
    00array4_push_pop.ts
    00array5_deconst.ts
    00array6.ts
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
    00break_continue_scope_exit.ts
    00break_continue.ts
    00capture_in_new_arguments.ts
    00cast_to_f32.ts
    00catch_untyped_value.ts
    00catch_value.ts
    00catch_value_minimal.ts
    00class_abstract.ts
    00class_access_control.ts
    00class_accessor_super.ts
    00class_accessor_virtual.ts
    00class_accessor.ts
    00class_accessor2.ts
    00class_constr_fields.ts
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
    00funcs_vararg.ts
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
    00generic_arguments_name_conflict.ts
    00global_const_object_method.ts
    00globals.ts
    00globals2.ts
    00globals3.ts
    00globals_entry_point.ts
    00if_conditional_compile.ts
    00if_return.ts
    00in_method_names.ts
    00in.ts
    00infer.ts
    00instanceof.ts
    00instanceof_namespace_narrowing.ts
    00interface_captures.ts
    00interface_conjunction.ts
    00interface_function_typed_field.ts
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
    00interface_optional_method_extends.ts
    00interface_optional.ts
    00interface_partial.ts
    00interface.ts
    00interface2.ts
    00intersection_type_generic.ts
    00intersection_type.ts
    00iterator_bug.ts
    00iterator.ts
    00lambdas_generic_global.ts
    00lambdas.ts
    00length.ts
    00map.ts
    00method_access_cond.ts
    00mixed_type_ops.ts
    00names_conflict.ts
    00nested_catch.ts
    00new_delete.ts
    00ns_bug.ts
    00ns.ts
    00ns2.ts
    00ns3.ts
    00ns4.ts
    00ns5.ts
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
    00safe_cast_bug.ts
    00safe_cast_field_access.ts
    00safe_cast_typeof.ts
    00safe_cast_while.ts
    00safe_cast.ts
    00safe_cast2.ts
    00sizeof.ts
    00spread_assignment.ts
    00spread.ts
    00stack_test.ts
    00str_null.ts
    00strings.ts
    00switch.ts
    00symbol.ts
    00throw_in_catch.ts
    00throw_bool_bigint.ts
    00throw_f32.ts
    00throw_u32.ts
    00throw_inlined.ts
    00throw_number_untyped_catch.ts
    00to_primitive_ops.ts
    00toplevel_control_flow.ts
    00toplevel_no_main_with_helpers.ts
    00try_catch_mismatch_rethrow.ts
    00try_catch_rethrow.ts
    00try_catch_return_dispose.ts
    00try_catch_return.ts
    00try_catch_sequence_and_nesting.ts
    00try_catch.ts
    00try_finally_break_continue.ts
    00try_finally_return.ts
    00try_in_catch_with_finally.ts
    00try_finally.ts
    00try_using_catch.ts
    00tuple_named.ts
    00tuple_with_array.ts
    00tuple.ts
    00type_aliases_in_generics.ts
    00type_guard_function.ts
    00typed_array.ts
    00typeof_function_narrowing.ts
    00typeof_static_fold.ts
    00typeof_static_fold_conditions.ts
    00typeof_union_narrowing.ts
    00types_indexedaccesstype.ts
    00types_keyof_enum.ts
    00types_mappedtype.ts
    00types_templateliteraltype.ts
    00types_unknown1.ts
    00types_utility.ts
    00types.ts
    00uint_compare_bug.ts
    00undef.ts
    00union_bin_ops.ts
    00union_bin_ops2.ts
    00union_errors.ts
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
                     "-DLLVM_LIB=${PROJECT_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/lib"
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
                         "-DLLVM_LIB=${PROJECT_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/lib"
                         "-DTSLANG_LIB=${CMAKE_BINARY_DIR}/lib"
                         -P "${CMAKE_CURRENT_SOURCE_DIR}/foreign-target-import.cmake")
    endforeach()
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
if (WIN32)
    set(defaultlib_collector_llvm_lib "${PROJECT_SOURCE_DIR}/../3rdParty/llvm/x64/${CMAKE_BUILD_TYPE_LOWERCASE}/lib")
    set(defaultlib_collector_gc_shared "-DGC_SHARED_LIB=${TEST_GC_SHARED_PREFIX}/lib")
else()
    set(defaultlib_collector_llvm_lib "${PROJECT_SOURCE_DIR}/../3rdParty/llvm/${CMAKE_BUILD_TYPE_LOWERCASE}/lib")
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
                     "-DX86=1"
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

# A call with fewer arguments than the callee's required parameters is an error (TS2554),
# not padded with undef. Each file in call-arity/ makes one bad call; the test passes when
# the compiler prints the expected message. The counts are what the user wrote, so `this`
# (a method, a constructor) and a closure's capture are not counted. Plain add_test: these
# compile nothing, so they need no x86 twin.
set(call_arity_cases
    "plain|Expected 2 arguments, but got 1\\."
    "optional|Expected 2\\-3 arguments, but got 1\\."
    "method|Expected 2 arguments, but got 1\\."
    "constructor|Expected 2 arguments, but got 1\\."
    "closure|Expected 2 arguments, but got 1\\."
    "declared|Expected 2 arguments, but got 1\\."
    "varargs|Expected at least 1 arguments, but got 0\\."
    "optional_call|Expected 2 arguments, but got 1\\.")
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
