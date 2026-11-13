// printf, sprintf_s, scanf and the rest of stdio, as functions a compiled program can call.
//
// The UCRT has them only as inline functions in its headers, over __stdio_common_vfprintf and the
// like: its libraries define none of them (x64's libucrt has printf alone, x86's nothing). A C or C++
// caller gets its own copy from the header; code tslang compiles never sees the header, so a
// `@varargs declare function printf` - or the sprintf_s a number's conversion to text calls - linked
// only when something else happened to bring a copy in. Microsoft's answer is
// legacy_stdio_definitions.lib, one more library in every link line.
//
// Instead, this file takes the address of each one, which makes the compiler emit the header's own
// definitions here, as linkable COMDATs (`selectany`), and the runtime every program links already
// carries them. The same kind of COMDAT a C or C++ object holds when it calls one: two copies are one,
// so a program whose other objects have a copy of their own links as it did. The file is its own
// member of the archive: the linker takes it only for a function nothing else defined.
#if defined(_WIN32)

#include <stdarg.h>
#include <stdio.h>
#include <wchar.h>

#define TSLANG_STDIO(name, type) reinterpret_cast<const void *>(static_cast<type>(&name))

extern "C" const void *const __tslang_stdio_definitions[] = {
    // printf
    TSLANG_STDIO(printf, int (*)(const char *, ...)),
    TSLANG_STDIO(fprintf, int (*)(FILE *, const char *, ...)),
    TSLANG_STDIO(sprintf, int (*)(char *, const char *, ...)),
    TSLANG_STDIO(snprintf, int (*)(char *, size_t, const char *, ...)),
    TSLANG_STDIO(_snprintf, int (*)(char *, size_t, const char *, ...)),
    TSLANG_STDIO(_scprintf, int (*)(const char *, ...)),
    TSLANG_STDIO(vprintf, int (*)(const char *, va_list)),
    TSLANG_STDIO(vfprintf, int (*)(FILE *, const char *, va_list)),
    TSLANG_STDIO(vsprintf, int (*)(char *, const char *, va_list)),
    TSLANG_STDIO(vsnprintf, int (*)(char *, size_t, const char *, va_list)),
    TSLANG_STDIO(_vsnprintf, int (*)(char *, size_t, const char *, va_list)),
    TSLANG_STDIO(_vscprintf, int (*)(const char *, va_list)),
    TSLANG_STDIO(printf_s, int (*)(const char *, ...)),
    TSLANG_STDIO(fprintf_s, int (*)(FILE *, const char *, ...)),
    TSLANG_STDIO(sprintf_s, int (*)(char *, size_t, const char *, ...)),
    TSLANG_STDIO(_snprintf_s, int (*)(char *, size_t, size_t, const char *, ...)),
    TSLANG_STDIO(vprintf_s, int (*)(const char *, va_list)),
    TSLANG_STDIO(vfprintf_s, int (*)(FILE *, const char *, va_list)),
    TSLANG_STDIO(vsprintf_s, int (*)(char *, size_t, const char *, va_list)),
    TSLANG_STDIO(vsnprintf_s, int (*)(char *, size_t, size_t, const char *, va_list)),
    TSLANG_STDIO(_vsnprintf_s, int (*)(char *, size_t, size_t, const char *, va_list)),

    // scanf
    TSLANG_STDIO(scanf, int (*)(const char *, ...)),
    TSLANG_STDIO(fscanf, int (*)(FILE *, const char *, ...)),
    TSLANG_STDIO(sscanf, int (*)(const char *, const char *, ...)),
    TSLANG_STDIO(vscanf, int (*)(const char *, va_list)),
    TSLANG_STDIO(vfscanf, int (*)(FILE *, const char *, va_list)),
    TSLANG_STDIO(vsscanf, int (*)(const char *, const char *, va_list)),
    TSLANG_STDIO(scanf_s, int (*)(const char *, ...)),
    TSLANG_STDIO(fscanf_s, int (*)(FILE *, const char *, ...)),
    TSLANG_STDIO(sscanf_s, int (*)(const char *, const char *, ...)),
    TSLANG_STDIO(vscanf_s, int (*)(const char *, va_list)),
    TSLANG_STDIO(vfscanf_s, int (*)(FILE *, const char *, va_list)),
    TSLANG_STDIO(vsscanf_s, int (*)(const char *, const char *, va_list)),

    // wprintf
    TSLANG_STDIO(wprintf, int (*)(const wchar_t *, ...)),
    TSLANG_STDIO(fwprintf, int (*)(FILE *, const wchar_t *, ...)),
    TSLANG_STDIO(swprintf, int (*)(wchar_t *, size_t, const wchar_t *, ...)),
    TSLANG_STDIO(_snwprintf, int (*)(wchar_t *, size_t, const wchar_t *, ...)),
    TSLANG_STDIO(vwprintf, int (*)(const wchar_t *, va_list)),
    TSLANG_STDIO(vfwprintf, int (*)(FILE *, const wchar_t *, va_list)),
    TSLANG_STDIO(vswprintf, int (*)(wchar_t *, size_t, const wchar_t *, va_list)),
    TSLANG_STDIO(_vsnwprintf, int (*)(wchar_t *, size_t, const wchar_t *, va_list)),
    TSLANG_STDIO(wprintf_s, int (*)(const wchar_t *, ...)),
    TSLANG_STDIO(fwprintf_s, int (*)(FILE *, const wchar_t *, ...)),
    TSLANG_STDIO(swprintf_s, int (*)(wchar_t *, size_t, const wchar_t *, ...)),
    TSLANG_STDIO(vwprintf_s, int (*)(const wchar_t *, va_list)),
    TSLANG_STDIO(vfwprintf_s, int (*)(FILE *, const wchar_t *, va_list)),
    TSLANG_STDIO(vswprintf_s, int (*)(wchar_t *, size_t, const wchar_t *, va_list)),

    // wscanf
    TSLANG_STDIO(wscanf, int (*)(const wchar_t *, ...)),
    TSLANG_STDIO(fwscanf, int (*)(FILE *, const wchar_t *, ...)),
    TSLANG_STDIO(swscanf, int (*)(const wchar_t *, const wchar_t *, ...)),
    TSLANG_STDIO(vwscanf, int (*)(const wchar_t *, va_list)),
    TSLANG_STDIO(vfwscanf, int (*)(FILE *, const wchar_t *, va_list)),
    TSLANG_STDIO(vswscanf, int (*)(const wchar_t *, const wchar_t *, va_list)),
    TSLANG_STDIO(wscanf_s, int (*)(const wchar_t *, ...)),
    TSLANG_STDIO(fwscanf_s, int (*)(FILE *, const wchar_t *, ...)),
    TSLANG_STDIO(swscanf_s, int (*)(const wchar_t *, const wchar_t *, ...)),
    TSLANG_STDIO(vwscanf_s, int (*)(const wchar_t *, va_list)),
    TSLANG_STDIO(vfwscanf_s, int (*)(FILE *, const wchar_t *, va_list)),
    TSLANG_STDIO(vswscanf_s, int (*)(const wchar_t *, const wchar_t *, va_list)),
};

#endif
