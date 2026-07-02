# TSLang.cmake
# ----------------------------------------------------------------------------
# Helper module that teaches CMake how to compile TypeScript (*.ts) sources to
# native executables with the `tslang` compiler.
#
# It exposes a single function:
#
#     add_ts_executable(<target> <source.ts> [<source2.ts> ...]
#                       [OUTPUT_NAME <name>]
#                       [OPTIONS <extra tslang flags>])
#
# Configure the paths below either on the cmake command line, e.g.
#
#     cmake -DTSLANG_ROOT=C:/dev/TypeScriptCompiler/__build -B build
#
# or via the cache variables in CMakeLists.txt.
# ----------------------------------------------------------------------------

# --- Locate the tslang toolchain --------------------------------------------

# Root of a built TypeScriptCompiler tree (the `__build` folder). Used to derive
# sensible defaults for the individual paths if they are not set explicitly.
set(TSLANG_ROOT "" CACHE PATH "Root of the built TypeScriptCompiler tree (the __build folder)")

if(WIN32)
    set(_tslang_default_bin  "${TSLANG_ROOT}/tslang/windows-msbuild-release/bin")
    set(_tslang_default_lib  "${TSLANG_ROOT}/tslang/windows-msbuild-release/lib")
    set(_tslang_default_llvm "${TSLANG_ROOT}/llvm/msbuild/x64/release/Release/lib")
    set(_tslang_default_gc   "${TSLANG_ROOT}/gc/msbuild/x64/release/Release")
else()
    set(_tslang_default_bin  "${TSLANG_ROOT}/tslang/linux-ninja-gcc-release/bin")
    set(_tslang_default_lib  "${TSLANG_ROOT}/tslang/linux-ninja-gcc-release/lib")
    set(_tslang_default_llvm "${TSLANG_ROOT}/llvm/release/lib")
    set(_tslang_default_gc   "${TSLANG_ROOT}/gc/release")
endif()

find_program(TSLANG_EXECUTABLE
    NAMES tslang
    HINTS "${_tslang_default_bin}"
    DOC "Path to the tslang compiler executable")

set(TSLANG_LIB_PATH "${_tslang_default_lib}"  CACHE PATH "Folder with the TSLANG runtime libraries")
set(LLVM_LIB_PATH   "${_tslang_default_llvm}" CACHE PATH "Folder with the LLVM/MLIR libraries")
set(GC_LIB_PATH     "${_tslang_default_gc}"   CACHE PATH "Folder with the Boehm GC library")

if(NOT TSLANG_EXECUTABLE)
    message(FATAL_ERROR
        "tslang compiler not found.\n"
        "Set -DTSLANG_ROOT=<path to TypeScriptCompiler/__build> "
        "or -DTSLANG_EXECUTABLE=<full path to tslang>.")
endif()

# --- The public helper -------------------------------------------------------

function(add_ts_executable target)
    cmake_parse_arguments(ARG "" "OUTPUT_NAME" "OPTIONS" ${ARGN})

    set(sources ${ARG_UNPARSED_ARGUMENTS})
    if(NOT sources)
        message(FATAL_ERROR "add_ts_executable(${target}): no .ts source files given")
    endif()

    if(ARG_OUTPUT_NAME)
        set(out_name "${ARG_OUTPUT_NAME}")
    else()
        set(out_name "${target}")
    endif()

    set(out_exe "${CMAKE_CURRENT_BINARY_DIR}/${out_name}${CMAKE_EXECUTABLE_SUFFIX}")

    # tslang places its output next to the input by default; we point it at the
    # build directory with -o and let it emit a native executable.
    list(GET sources 0 main_source)
    get_filename_component(main_source_abs "${main_source}" ABSOLUTE)

    add_custom_command(
        OUTPUT  "${out_exe}"
        COMMAND ${CMAKE_COMMAND} -E env
                    "GC_LIB_PATH=${GC_LIB_PATH}"
                    "LLVM_LIB_PATH=${LLVM_LIB_PATH}"
                    "TSLANG_LIB_PATH=${TSLANG_LIB_PATH}"
                    "TSLANGEXEPATH=${_tslang_default_bin}"
                "${TSLANG_EXECUTABLE}"
                    --emit=exe --opt
                    ${ARG_OPTIONS}
                    -o "${out_exe}"
                    "${main_source_abs}"
        DEPENDS ${sources}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Compiling TypeScript ${main_source} -> ${out_name}${CMAKE_EXECUTABLE_SUFFIX}"
        VERBATIM)

    add_custom_target(${target} ALL DEPENDS "${out_exe}")
    set_target_properties(${target} PROPERTIES TS_OUTPUT "${out_exe}")
endfunction()
