# Locate the tslang compiler and derive install prefix paths.
#
# Cache variables:
#   TSLANG_ROOT          - TypeScriptCompiler __build folder (same as hello-cmake)
#   CMAKE_TSLANG_COMPILER - full path to tslang (optional override)
#
# Output variables:
#   CMAKE_TSLANG_COMPILER
#   TSLANG_BIN_DIR       - directory containing the tslang binary
#   TSLANG_PREFIX        - tslang build prefix (parent of bin/ and lib/)

function(_tslang_try_prefix out_var base_dir)
    if(EXISTS "${base_dir}/bin/tslang" OR EXISTS "${base_dir}/bin/tslang.exe")
        list(APPEND ${out_var} "${base_dir}")
        set(${out_var} "${${out_var}}" PARENT_SCOPE)
    endif()
endfunction()

function(locate_tslang_compiler)
    if(CMAKE_TSLANG_COMPILER AND EXISTS "${CMAKE_TSLANG_COMPILER}")
        cmake_path(GET CMAKE_TSLANG_COMPILER PARENT_PATH _bin_dir)
        cmake_path(GET _bin_dir PARENT_PATH _prefix)
        set(TSLANG_BIN_DIR "${_bin_dir}" PARENT_SCOPE)
        set(TSLANG_PREFIX "${_prefix}" PARENT_SCOPE)
        return()
    endif()

    if(NOT TSLANG_ROOT)
        set(_default_root "${CMAKE_SOURCE_DIR}/../../../__build")
        if(EXISTS "${_default_root}")
            set(TSLANG_ROOT "${_default_root}" CACHE PATH
                "TypeScriptCompiler __build folder (contains tslang/, llvm/, gc/)")
        else()
            set(TSLANG_ROOT "" CACHE PATH
                "TypeScriptCompiler __build folder (contains tslang/, llvm/, gc/)")
        endif()
    endif()

    set(_prefixes)
    if(TSLANG_ROOT)
        if(WIN32)
            foreach(_preset IN ITEMS
                windows-msbuild-2026-release
                windows-msbuild-2022-release
                windows-msbuild-release)
                _tslang_try_prefix(_prefixes "${TSLANG_ROOT}/tslang/${_preset}")
            endforeach()
        else()
            foreach(_preset IN ITEMS
                linux-ninja-gcc-release
                linux-ninja-clang-release
                linux-ninja-gcc-debug
                linux-ninja-clang-debug
                ninja/release
                ninja/debug)
                _tslang_try_prefix(_prefixes "${TSLANG_ROOT}/tslang/${_preset}")
            endforeach()
        endif()
    endif()

    set(_hint_bins)
    foreach(_prefix IN LISTS _prefixes)
        list(APPEND _hint_bins "${_prefix}/bin")
    endforeach()

    find_program(_tslang_compiler
        NAMES tslang tslang.exe
        HINTS ${_hint_bins}
        DOC "TSLANG compiler")

    if(NOT _tslang_compiler)
        if(WIN32)
            message(FATAL_ERROR
                "tslang compiler not found.\n"
                "\n"
                "Build TypeScriptCompiler first (from the repo root):\n"
                "  prepare_3rdParty.bat\n"
                "  cd tslang && config_tslang_release.bat && build_tslang_release.bat\n"
                "  bin\\tslang.exe --install-default-lib\n"
                "\n"
                "Then configure this sample with:\n"
                "  cmake --preset default -DTSLANG_ROOT=C:/path/to/TypeScriptCompiler/__build\n"
                "\n"
                "Or put tslang on PATH, or pass:\n"
                "  -DCMAKE_TSLANG_COMPILER=C:/path/to/tslang.exe")
        else()
            message(FATAL_ERROR
                "tslang compiler not found.\n"
                "\n"
                "Build TypeScriptCompiler first (from the repo root):\n"
                "  ./prepare_3rdParty_release.sh\n"
                "  cd tslang && ./config_tslang_release.sh && ./build_tslang_release.sh\n"
                "  ./bin/tslang --install-default-lib\n"
                "\n"
                "Then configure this sample with:\n"
                "  cmake --preset default -DTSLANG_ROOT=/path/to/TypeScriptCompiler/__build\n"
                "\n"
                "Or put tslang on PATH, or pass:\n"
                "  -DCMAKE_TSLANG_COMPILER=/path/to/tslang")
        endif()
    endif()

    cmake_path(GET _tslang_compiler PARENT_PATH _bin_dir)
    cmake_path(GET _bin_dir PARENT_PATH _prefix)

    set(CMAKE_TSLANG_COMPILER "${_tslang_compiler}" PARENT_SCOPE)
    set(TSLANG_BIN_DIR "${_bin_dir}" PARENT_SCOPE)
    set(TSLANG_PREFIX "${_prefix}" PARENT_SCOPE)
endfunction()

function(setup_tslang_link_paths)
    if(NOT TSLANG_PREFIX)
        message(FATAL_ERROR "setup_tslang_link_paths: TSLANG_PREFIX is not set")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(_defaultlib_config "release")
    else()
        set(_defaultlib_config "debug")
    endif()

    set(_link_dirs
        "${TSLANG_BIN_DIR}"
        "${TSLANG_PREFIX}/lib"
        "${TSLANG_BIN_DIR}/defaultlib/lib/${_defaultlib_config}")

    if(TSLANG_ROOT)
        if(EXISTS "${TSLANG_ROOT}/gc/release")
            list(APPEND _link_dirs "${TSLANG_ROOT}/gc/release")
        endif()
        if(EXISTS "${TSLANG_ROOT}/llvm/release/lib")
            list(APPEND _link_dirs "${TSLANG_ROOT}/llvm/release/lib")
        endif()
    endif()

    include_directories("${TSLANG_BIN_DIR}/defaultlib")
    link_directories(${_link_dirs})
endfunction()
