#ifndef TYPES_H
#define TYPES_H

#include "config.h"
#include "undefined.h"

namespace ts
{
    struct LineAndCharacter {

        LineAndCharacter() = default;

        /** 0-based. */
        number line;
        /*
            * 0-based. This value denotes the character position in line and is different from the 'column' because of tab characters.
            */
        number character;
    };

    struct DiagnosticMessageStore
    {
        DiagnosticMessageStore() = default;
        DiagnosticMessageStore(int code, DiagnosticCategory category, string label, string message) : code(code), category(category), label(label), message(message) {};
        DiagnosticMessageStore(undefined_t) : category{DiagnosticCategory::Undefined} {}

        int code;
        DiagnosticCategory category;
        string label;
        string message;

        bool operator !()
        {
            return category == DiagnosticCategory::Undefined;
        }
    };

} // namespace ts

#endif // ENUMS_H