#ifndef TYPESCRIPT_UNCAUGHTEXCEPTION_H_
#define TYPESCRIPT_UNCAUGHTEXCEPTION_H_

// What an exception nothing caught threw, as text: the JIT (tslang/jit.cpp) and an executable's runtime
// (TypeScriptAsyncRuntime/UncaughtException.cpp) both say "uncaught exception: <this>". Windows only: it
// reads the exception record of a throw (_CxxThrowException's).

#ifdef _WIN32
#include <windows.h>

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

} // namespace typescript

#endif // _WIN32

#endif // TYPESCRIPT_UNCAUGHTEXCEPTION_H_
