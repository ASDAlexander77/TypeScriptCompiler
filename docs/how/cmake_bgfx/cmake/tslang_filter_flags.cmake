# Filter C++ CPU flags that linked targets (bx/bgfx) export via PUBLIC
# compile options. tslang does not accept those flags.
#
# Invoked as:
#   cmake -P tslang_filter_flags.cmake -- <compiler> <args...>
#
# Everything after the literal "--" argument is forwarded, with the
# blocked flags removed, to COMMAND.
#
# Note: CMAKE_ARGV<n> is indexed access (CMAKE_ARGC gives the count); there
# is no plain CMAKE_ARGV list variable.

set(_args)
set(_seen_dashdash FALSE)
math(EXPR _last "${CMAKE_ARGC} - 1")
foreach(_i RANGE 0 ${_last})
    set(_raw_arg "${CMAKE_ARGV${_i}}")
    if(NOT _seen_dashdash)
        if(_raw_arg STREQUAL "--")
            set(_seen_dashdash TRUE)
        endif()
        continue()
    endif()

    if(_raw_arg MATCHES "^-m(sse4\\.2|sse4\\.1|avx|avx2)$")
        continue()
    endif()
    list(APPEND _args "${_raw_arg}")
endforeach()

execute_process(COMMAND ${_args} RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "tslang compile failed (exit ${_result})")
endif()
