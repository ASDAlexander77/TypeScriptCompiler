#ifndef TYPESCRIPT_UNCAUGHTEXCEPTION_H_
#define TYPESCRIPT_UNCAUGHTEXCEPTION_H_

// What an exception nothing caught threw, as text: the JIT (tslang/jit.cpp) and an executable's runtime
// (TypeScriptAsyncRuntime/UncaughtException.cpp) both say "uncaught exception: <this>". On Windows it reads
// the exception record of a throw (_CxxThrowException's); on Linux the exception std::terminate() is called
// for, which the C++ runtime has begun to catch.

#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))
#define TSLANG_DESCRIBES_UNCAUGHT 1

#ifdef _WIN32
#include <windows.h>
#else
#include <cxxabi.h>
#include <exception>
#include <typeinfo>
#endif

#include <cctype>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace typescript
{

// A thrown number as the program would print it (Number::toString, as ConvertLogic's
// __ts_number_to_string): the fewest digits that read back as the same double, plain decimal for
// a decimal exponent n in -6 < n <= 21, `d.ddde+N` / `e-N` otherwise. `%.17g` said `0.1` as
// 0.10000000000000001.
inline void uncaughtNumberText(char *out, size_t size, double value)
{
    if (value != value)
    {
        snprintf(out, size, "NaN");
        return;
    }

    if (value == 0)
    {
        snprintf(out, size, "0");
        return;
    }

    auto *sign = value < 0 ? "-" : "";
    auto magnitude = value < 0 ? -value : value;
    if (magnitude > DBL_MAX)
    {
        snprintf(out, size, "%sInfinity", sign);
        return;
    }

    char scientific[32];
    int decimals = 0;
    for (; decimals < 16; decimals++)
    {
        snprintf(scientific, sizeof(scientific), "%.*e", decimals, magnitude);
        if (strtod(scientific, nullptr) == magnitude)
        {
            break;
        }
    }

    if (decimals == 16)
    {
        snprintf(scientific, sizeof(scientific), "%.*e", decimals, magnitude);
    }

    char digits[32];
    snprintf(digits, sizeof(digits), "%c%.*s", scientific[0], decimals, scientific + 2);
    int k = decimals + 1;
    int n = atoi(strchr(scientific, 'e') + 1) + 1;
    static const char zeros[] = "000000000000000000000";
    if (k <= n && n <= 21)
    {
        snprintf(out, size, "%s%s%.*s", sign, digits, n - k, zeros);
    }
    else if (0 < n && n <= 21)
    {
        snprintf(out, size, "%s%.*s.%s", sign, n, digits, digits + n);
    }
    else if (-6 < n && n <= 0)
    {
        snprintf(out, size, "%s0.%.*s%s", sign, -n, zeros, digits);
    }
    else
    {
        snprintf(out, size, "%s%c%s%se%+d", sign, digits[0], k > 1 ? "." : "", digits + 1, n - 1);
    }
}

#ifdef _WIN32
// The thrown value, read from the exception record of a throw (_CxxThrowException's): the object, and its
// ThrowInfo, whose first catchable type names its type (`.N` a number, `.PEAD` a string, `.PEAVName@@` a class).
// Read while the object is alive, in the filter; a record not shaped so is left undescribed.
inline void describeUncaught(EXCEPTION_POINTERS *pointers, char *uncaughtDescription, size_t size)
{
    uncaughtDescription[0] = '\0';
    __try
    {
        auto *record = pointers->ExceptionRecord;
        if (record->NumberParameters < 4)
        {
            return;
        }

        auto *object = reinterpret_cast<char *>(record->ExceptionInformation[1]);
        auto *throwInfo = reinterpret_cast<int32_t *>(record->ExceptionInformation[2]);
        auto *imageBase = reinterpret_cast<char *>(record->ExceptionInformation[3]);
        if (!object || !throwInfo || !imageBase)
        {
            return;
        }

        auto *catchableTypes = reinterpret_cast<int32_t *>(imageBase + throwInfo[3]);
        auto count = catchableTypes[0];
        if (count < 1)
        {
            return;
        }

        // a type descriptor: the type_info vtable, a spare pointer, then the name
        auto typeName = [&](int index) {
            auto *catchableType = reinterpret_cast<int32_t *>(imageBase + catchableTypes[1 + index]);
            return reinterpret_cast<const char *>(imageBase + catchableType[1] + 2 * sizeof(void *));
        };

        auto *name = typeName(0);
        if (strcmp(name, ".PEAD") == 0)
        {
            auto *text = *reinterpret_cast<const char **>(object);
            snprintf(uncaughtDescription, size, "%s", text ? text : "null");
        }
        else if (strcmp(name, ".N") == 0)
        {
            uncaughtNumberText(uncaughtDescription, size, *reinterpret_cast<double *>(object));
        }
        else if (strcmp(name, ".H") == 0)
        {
            snprintf(uncaughtDescription, size, "%d", *reinterpret_cast<int32_t *>(object));
        }
        else if (strcmp(name, "._N") == 0)
        {
            snprintf(uncaughtDescription, size, "%s", *reinterpret_cast<bool *>(object) ? "true" : "false");
        }
        else if (strcmp(name, "._J") == 0)
        {
            snprintf(uncaughtDescription, size, "%lldn", *reinterpret_cast<long long *>(object));
        }
        else if (strncmp(name, ".PEAV", 5) == 0)
        {
            // a class: its name, up to the `@@` that ends it
            auto *className = name + 5;
            auto *end = strstr(className, "@@");
            auto length = end ? static_cast<int>(end - className) : static_cast<int>(strlen(className));
            snprintf(uncaughtDescription, size, "%.*s", length, className);
            // an Error (core.ts): after the vtable, `message?: string` - the text, and whether it is set
            for (int i = 0; i < count; i++)
            {
                if (strcmp(typeName(i), ".PEAVError@@") != 0)
                {
                    continue;
                }

                auto *instance = *reinterpret_cast<char **>(object);
                auto *message = instance ? *reinterpret_cast<const char **>(instance + sizeof(void *)) : nullptr;
                auto isSet = instance && *reinterpret_cast<bool *>(instance + 2 * sizeof(void *));
                if (isSet && message)
                {
                    auto used = strlen(uncaughtDescription);
                    snprintf(uncaughtDescription + used, size - used, ": %s", message);
                }

                break;
            }
        }
    }
    __except (EXCEPTION_EXECUTE_HANDLER)
    {
        uncaughtDescription[0] = '\0';
    }
}
#else
} // namespace typescript

// the vtable of a class's type_info with one base (a class that extends another), which libstdc++ and
// libc++abi both define
extern "C" void *_ZTVN10__cxxabiv120__si_class_type_infoE[];

namespace typescript
{

// The exception std::terminate() was called for, when a throw found no catch: the C++ runtime has begun
// to catch it, so it is the current exception. Its type_info names its type (`d` a number, `PKc` a string,
// `P7Failure` a class, thrown as a pointer, see LLVMRTTIHelperVCLinuxConst.h). The Itanium layouts are read
// as they are - tslang is built without RTTI: a pointer's type_info is {vtable, name, flags, pointee}, a
// class's with one base {vtable, name, base}. False when there is no current exception.
inline bool describeCurrentException(char *uncaughtDescription, size_t size)
{
    uncaughtDescription[0] = '\0';
    auto *type = abi::__cxa_current_exception_type();
    if (!type)
    {
        return false;
    }

    // an exception_ptr holds the thrown object's address (libstdc++ and libc++ alike)
    auto exception = std::current_exception();
    auto *object = *reinterpret_cast<char **>(&exception);
    if (!object)
    {
        return true;
    }

    auto *name = type->name();
    if (strcmp(name, "PKc") == 0 || strcmp(name, "Pc") == 0)
    {
        auto *text = *reinterpret_cast<const char **>(object);
        snprintf(uncaughtDescription, size, "%s", text ? text : "null");
    }
    else if (strcmp(name, "d") == 0)
    {
        uncaughtNumberText(uncaughtDescription, size, *reinterpret_cast<double *>(object));
    }
    else if (strcmp(name, "f") == 0)
    {
        uncaughtNumberText(uncaughtDescription, size, *reinterpret_cast<float *>(object));
    }
    else if (strcmp(name, "i") == 0)
    {
        snprintf(uncaughtDescription, size, "%d", *reinterpret_cast<int32_t *>(object));
    }
    else if (strcmp(name, "j") == 0)
    {
        snprintf(uncaughtDescription, size, "%u", *reinterpret_cast<uint32_t *>(object));
    }
    else if (strcmp(name, "b") == 0)
    {
        snprintf(uncaughtDescription, size, "%s", *reinterpret_cast<bool *>(object) ? "true" : "false");
    }
    else if (strcmp(name, "x") == 0)
    {
        snprintf(uncaughtDescription, size, "%lldn", *reinterpret_cast<long long *>(object));
    }
    else if (name[0] == 'P' && isdigit(static_cast<unsigned char>(name[1])))
    {
        // a class: P<length><name>
        char *className;
        auto length = strtol(name + 1, &className, 10);
        snprintf(uncaughtDescription, size, "%.*s", static_cast<int>(length), className);

        // an Error (core.ts) among it and its bases: after the vtable, `message?: string` - the text, and
        // whether it is set
        auto *classType = *reinterpret_cast<const std::type_info *const *>(reinterpret_cast<const char *>(type) +
                                                                          2 * sizeof(void *) + sizeof(void *));
        while (classType)
        {
            if (strcmp(classType->name(), "5Error") == 0)
            {
                auto *instance = *reinterpret_cast<char **>(object);
                auto *message = instance ? *reinterpret_cast<const char **>(instance + sizeof(void *)) : nullptr;
                auto isSet = instance && *reinterpret_cast<bool *>(instance + 2 * sizeof(void *));
                if (isSet && message)
                {
                    auto used = strlen(uncaughtDescription);
                    snprintf(uncaughtDescription + used, size - used, ": %s", message);
                }

                break;
            }

            auto *vtable = *reinterpret_cast<void *const *>(classType);
            if (vtable != static_cast<void *>(_ZTVN10__cxxabiv120__si_class_type_infoE + 2))
            {
                break;
            }

            classType = *reinterpret_cast<const std::type_info *const *>(reinterpret_cast<const char *>(classType) +
                                                                         2 * sizeof(void *));
        }
    }

    return true;
}
#endif

} // namespace typescript

#endif // _WIN32 || Linux

#endif // TYPESCRIPT_UNCAUGHTEXCEPTION_H_
